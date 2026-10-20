"""Chat model construction. The only module that knows which provider generates answers."""

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr

from backend.app.config import Settings


def build_llm(settings: Settings) -> BaseChatModel | None:
    """Return the Groq chat model, or None when no key is configured (uploads still work).

    Temperature 0: answers must stay grounded in the excerpts, not creative.
    Construction makes no network call, so a bad key only surfaces on the first query.
    """
    if settings.groq_api_key is None:
        return None
    return ChatGroq(
        model=settings.groq_model,
        api_key=SecretStr(settings.groq_api_key),
        temperature=0,
        reasoning_effort=settings.groq_reasoning_effort,
        max_tokens=settings.groq_max_tokens,
        timeout=settings.groq_timeout_s,
        max_retries=2,
    )
