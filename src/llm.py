"""LLM provider factory — returns a LangChain chat model from env config."""
import os
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

PROVIDER_DEFAULTS = {
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-8",
}


def get_llm(temperature: float = 0) -> BaseChatModel:
    """Return a LangChain chat model based on LLM_PROVIDER / LLM_MODEL env vars."""
    provider = os.environ.get("LLM_PROVIDER", "openai").lower().strip()
    model = os.environ.get("LLM_MODEL", PROVIDER_DEFAULTS.get(provider, ""))

    if not model:
        raise ValueError(f"Unknown provider '{provider}'. Set LLM_PROVIDER to 'openai' or 'anthropic'.")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Supported values: 'openai', 'anthropic'."
    )
