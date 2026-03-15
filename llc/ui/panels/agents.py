from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static

from llc.ui.state import WorkerRow


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(limit - 3, 0)] + "..."


def _status_tag(status: str) -> str:
    normalized = status.strip().lower()
    if normalized in {"running", "restarting", "terminating"}:
        return "RUN "
    if normalized in {"completed", "done"}:
        return "DONE"
    if normalized in {"failed", "error", "stuck"}:
        return "ERR "
    if normalized in {"review", "waiting"}:
        return "WAIT"
    return "IDLE"


class AgentsPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("AGENTS", classes="panel-title")
        with VerticalScroll(id="agents_scroll"):
            yield Static("", id="agents_body")

    def body(self) -> Static:
        return self.query_one("#agents_body", Static)

    def render_workers(self, workers: list[WorkerRow], sub_agent_mode_enabled: bool) -> None:
        if not sub_agent_mode_enabled:
            self.body().update("sub-agent mode is off\n\nrun /enable sub-agent-mode")
            return
        if not workers:
            self.body().update("no sub-agents yet\n\nrun /subagent {task}")
            return

        active = [worker for worker in workers if worker.is_active]
        inactive = [worker for worker in workers if not worker.is_active]
        active = sorted(active, key=lambda worker: worker.updated_at, reverse=True)
        inactive = sorted(inactive, key=lambda worker: worker.updated_at, reverse=True)
        lines: list[str] = []
        lines.append("orchestrator")
        lines.append(f"  workers {len(active):>2} active / {len(workers):>2} total")
        lines.append("")

        if active:
            lines.append("active workers")
            lines.append("id         st    task")
            for worker in active[:10]:
                objective = worker.current_task or worker.goal or "-"
                activity = worker.current_activity or worker.activity_detail or "working"
                tool_hint = worker.last_tool_name or "-"
                lines.append(
                    f"{worker.short_id[:9]:<9}  {_status_tag(worker.status):<4}  "
                    f"{_preview(objective, 22)}"
                )
                lines.append(f"           now  {_preview(activity, 30)}")
                lines.append(
                    f"           tool {_preview(tool_hint, 30)}  calls:{worker.tool_calls:>3}"
                )
                lines.append("")
        else:
            lines.append("active workers: none")
            lines.append("")

        if inactive:
            lines.append("recently finished")
            lines.append("id         st    summary")
            for worker in inactive[:8]:
                summary = worker.activity_detail or worker.current_activity or worker.goal or "-"
                lines.append(
                    f"{worker.short_id[:9]:<9}  {_status_tag(worker.status):<4}  "
                    f"{_preview(summary, 24)}"
                )
                lines.append(f"           output chars: {worker.output_chars}")
                lines.append("")
        self.body().update("\n".join(lines).strip())
