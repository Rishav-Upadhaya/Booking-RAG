from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_model() -> ChatOpenAI:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
    return ChatOpenAI(
        model=settings.openrouter_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        temperature=0,
        stream_usage=True,
        profile={"max_input_tokens": 32768},
    )
