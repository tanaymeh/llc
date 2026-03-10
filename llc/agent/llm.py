from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from llc.config import Settings


def _is_openrouter_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    return "openrouter.ai" in base_url.lower()


def build_chat_model(settings: Settings, model_name: str | None = None):
    selected_model = model_name or settings.model_name
    if _is_openrouter_base_url(settings.openai_base_url):
        return ChatOpenRouter(
            model=selected_model,
            temperature=0,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            stream_usage=True,
        )
    return ChatOpenAI(
        model=selected_model,
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        stream_usage=True,
    )
