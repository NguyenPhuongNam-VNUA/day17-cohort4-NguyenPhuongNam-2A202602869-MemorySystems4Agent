from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig instance."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables from .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "650"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    main_provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    main_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    main_temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Provider-specific credentials
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")

    model_config = ProviderConfig(
        provider=main_provider,
        model_name=main_model,
        temperature=main_temp,
        api_key=api_key,
        base_url=base_url,
    )

    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", main_provider))
    judge_model_name = os.getenv("JUDGE_MODEL", main_model)
    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model_config,
        judge_model=judge_config,
    )
