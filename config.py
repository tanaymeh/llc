import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = (
    "You are a local coding assistant. Use tools only when needed, keep answers concise, "
    "and show your final answer clearly."
)


class Settings(BaseModel, frozen=True):
    model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    workspace_root: Path = Field(default_factory=lambda: Path.cwd().resolve())
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    shell_timeout: int = 30

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
        )
