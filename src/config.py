from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _build_provider_from_env(prefix: str, default_provider: str, default_model: str) -> ProviderConfig:
    provider = os.getenv(f"{prefix}_PROVIDER", default_provider)
    model_name = os.getenv(f"{prefix}_MODEL", default_model)
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0.0"))
    api_key = os.getenv(f"{prefix}_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL")
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    load_dotenv(root / ".env")

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "2000"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    model = _build_provider_from_env("LLM", "openai", "gpt-4o-mini")
    judge_model = _build_provider_from_env("JUDGE_LLM", "openai", "gpt-4o-mini")

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model,
        judge_model=judge_model,
    )
