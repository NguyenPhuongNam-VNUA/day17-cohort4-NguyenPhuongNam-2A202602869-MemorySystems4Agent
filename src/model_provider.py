from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents and evaluation runtime.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize provider names and map aliases."""
    val = value.strip().lower()
    mapping = {
        "openai": "openai",
        "oai": "openai",
        "custom": "custom",
        "openai-compatible": "custom",
        "local": "custom",
        "vllm": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "anthorpic": "anthropic",  # Common typo handling
        "ollama": "ollama",
        "openrouter": "openrouter",
    }
    if val in mapping:
        return mapping[val]

    for k, v in mapping.items():
        if k in val:
            return v
    return val


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config.model_name or "gpt-4o-mini",
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("CUSTOM_API_KEY") or "EMPTY"
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        return ChatOpenAI(
            model=config.model_name or "default-model",
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config.model_name or "gemini-1.5-flash",
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=config.model_name or "claude-3-5-haiku-latest",
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=config.model_name or "llama3.2:latest",
            temperature=config.temperature,
            base_url=base_url,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        return ChatOpenRouter(
            model=config.model_name or "meta-llama/llama-3.1-8b-instruct",
            temperature=config.temperature,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported provider: '{config.provider}'")
