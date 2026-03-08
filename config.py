import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
import yaml

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.yaml"


def _load_system_prompt(path: Path = SYSTEM_PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"System prompt required at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    prompt = data.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"prompts/system_prompt.yaml must contain a non-empty 'system_prompt' key")
    return prompt.strip()


class Settings(BaseModel, frozen=True):
    model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    firecrawl_api_key: str | None = None
    workspace_root: Path = Field(default_factory=lambda: Path.cwd().resolve())
    system_prompt: str = Field(default_factory=_load_system_prompt)
    shell_timeout: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        )
