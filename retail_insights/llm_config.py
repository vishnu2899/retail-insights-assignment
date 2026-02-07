import os
from typing import Any

from langchain_openai import ChatOpenAI


def get_llm_client() -> Any:
    """Return a configured LLM client.

    Uses an OpenAI-compatible Chat API via langchain_openai.

    Priority of credentials/endpoints:
    - If OPENROUTER_API_KEY is set, call OpenRouter at https://openrouter.ai/api/v1.
    - Else, fall back to OPENAI_API_KEY and the default OpenAI endpoint.
    """

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY environment variables are set."
        )

    # Default to OpenRouter if OPENROUTER_API_KEY is present, otherwise OpenAI.
    if os.getenv("OPENROUTER_API_KEY"):
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    else:
        base_url = os.getenv("OPENAI_BASE_URL")  # None -> use OpenAI default
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    llm = ChatOpenAI(
        model=model,
        temperature=0.1,
        openai_api_key=api_key,
        base_url=base_url,
    )
    return llm
