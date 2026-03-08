import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from langchain_core.tools import BaseTool


def make_code_grep_tools(workspace_root: Path) -> list[BaseTool]:
    @tool
    def code_grep(
        pattern: str,
        language: Optional[str] = None,
        path: Optional[str] = None,
        rule: Optional[str] = None,
        rewrite: Optional[str] = None,
        apply_rewrite: Optional[bool] = None,
        max_results: Optional[int] = None,
        json_output: Optional[bool] = None,
    ) -> str:
        """Structural code search and rewrite using ast-grep. Searches and transforms code based on AST (Abstract Syntax Tree) patterns rather than plain text, enabling syntax-aware matching across 20+ programming languages.

Unlike Grep which does text/regex matching, code_grep understands code structure. It matches syntax nodes, not strings, so it won't match inside comments or strings and handles whitespace/formatting differences automatically.

## Meta Variables (Wildcards)
- `$VAR`: matches any single AST node (like regex `.`)
- `$$$ARGS`: matches zero or more AST nodes (like regex `.*`)
- `$_`: non-capturing wildcard (each occurrence matches independently)
- Same-name meta variables enforce equality: `$A == $A` matches `x == x` but not `x == y`

## Parameters
- pattern: AST pattern to search for
- language: Programming language (javascript, typescript, python, rust, go, java, c, cpp, etc.)
- path: Directory or file to search in (defaults to workspace root)
- rule: YAML rule string for complex multi-condition searches. When provided, pattern is ignored.
- rewrite: Replacement string using meta variables from the pattern. Use with pattern for simple rewrites.
- apply_rewrite: If true, applies the rewrite to files (default: false, only shows diff preview)
- max_results: Limit number of matches returned
- json_output: If true, return results as structured JSON

## Search Examples

Find all console.log calls (any number of arguments):
  pattern="console.log($$$ARGS)", language="javascript"

Find Python function definitions:
  pattern="def $FUNC($$$PARAMS):", language="python"

Find React useState hooks:
  pattern="const [$STATE, $SETTER] = useState($$$)", language="typescript"

Find Python type hints with Optional:
  pattern="Optional[$TYPE]", language="python"

Find async functions in JavaScript:
  pattern="async function $NAME($$$PARAMS) { $$$ }", language="javascript"

Find Python imports from a specific module:
  pattern="from $MODULE import $$$NAMES", language="python"

Find Rust unsafe blocks:
  pattern="unsafe { $$$ }", language="rust"

Find Go error handling patterns:
  pattern="if $ERR != nil { $$$ }", language="go"

Find Java class declarations:
  pattern="class $NAME extends $PARENT { $$$ }", language="java"

## Rewrite Examples (pattern + rewrite)

Rename a function across codebase:
  pattern="oldFunction($$$ARGS)", rewrite="newFunction($$$ARGS)", language="python"

Modernize Python type hints (Optional[X] -> X | None):
  pattern="Optional[$TYPE]", rewrite="$TYPE | None", language="python"

Convert print to logging:
  pattern="print($$$ARGS)", rewrite="logger.info($$$ARGS)", language="python"

Swap assertEqual argument order:
  pattern="self.assertEqual($ACTUAL, $EXPECTED)", rewrite="self.assertEqual($EXPECTED, $ACTUAL)", language="python"

Remove console.log statements:
  pattern="console.log($$$)", rewrite="", language="javascript"

## YAML Rule Examples (for complex structural queries)

Find async functions containing await:
  rule="id: async-await\\nlanguage: javascript\\nrule:\\n  all:\\n    - pattern: async function $NAME($$$) { $$$ }\\n    - has:\\n        pattern: await $EXPR"

Find functions longer than a threshold (by having many statements):
  rule="id: long-func\\nlanguage: python\\nrule:\\n  pattern: |\\n    def $F($$$):\\n        $$$BODY\\n  has:\\n    pattern: $$$STMTS\\n    stopBy: end"

YAML rules with `fix` key can also perform rewrites:
  rule="id: modernize\\nlanguage: python\\nrule:\\n  pattern: Optional[$T]\\nfix: $T | None"
"""
        search_path = str(workspace_root)
        if path:
            candidate = (workspace_root / path).resolve()
            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                return "Error: path is outside the workspace root."
            search_path = str(candidate)

        sg_bin = _find_sg_binary()
        if sg_bin is None:
            return (
                "Error: ast-grep (sg) is not installed. "
                "Install it with: brew install ast-grep (macOS) or cargo install ast-grep --locked"
            )

        use_json = json_output or False

        if rule:
            return _run_rule_search(
                sg_bin, rule, search_path, max_results,
                apply_fix=apply_rewrite or False, use_json=use_json,
            )
        return _run_pattern_search(
            sg_bin, pattern, language, search_path, max_results,
            rewrite=rewrite, apply_rewrite=apply_rewrite or False,
            use_json=use_json,
        )

    return [code_grep]


def _find_sg_binary() -> Optional[str]:
    for name in ("sg", "ast-grep"):
        try:
            subprocess.run(
                [name, "--version"],
                capture_output=True,
                timeout=5,
            )
            return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _run_pattern_search(
    sg_bin: str,
    pattern: str,
    language: Optional[str],
    search_path: str,
    max_results: Optional[int],
    rewrite: Optional[str] = None,
    apply_rewrite: bool = False,
    use_json: bool = False,
) -> str:
    cmd = [sg_bin, "run", "--pattern", pattern]
    if language:
        cmd.extend(["--lang", language])
    if rewrite is not None:
        cmd.extend(["--rewrite", rewrite])
        if apply_rewrite:
            cmd.append("--update-all")
    if use_json and not rewrite:
        cmd.extend(["--json", "compact"])
    cmd.append(search_path)

    result = _execute_sg(cmd, max_results)

    if rewrite and not apply_rewrite and "No matches" not in result:
        result = f"[PREVIEW] The following changes would be applied. Set apply_rewrite=true to apply:\n\n{result}"

    return result


def _run_rule_search(
    sg_bin: str,
    rule: str,
    search_path: str,
    max_results: Optional[int],
    apply_fix: bool = False,
    use_json: bool = False,
) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(rule)
        rule_path = f.name

    try:
        cmd = [sg_bin, "scan", "--inline-rules", rule_path]
        if apply_fix:
            cmd.append("--update-all")
        if use_json and not apply_fix:
            cmd.extend(["--json", "compact"])
        cmd.append(search_path)
        return _execute_sg(cmd, max_results)
    finally:
        Path(rule_path).unlink(missing_ok=True)


def _execute_sg(cmd: list[str], max_results: Optional[int]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Search timed out after 30 seconds."

    output = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if not output:
        if stderr:
            return f"Error: {stderr}"
        return "No matches found."

    if max_results is not None and max_results > 0:
        blocks = output.split("\n\n")
        output = "\n\n".join(blocks[:max_results])

    if len(output) > 30000:
        output = output[:30000] + "\n\n... (output truncated)"

    return output
