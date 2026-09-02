"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide settings, populated from env vars or .env file."""

    # LLM provider — defaults to Anthropic (Claude)
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="anthropic", alias="ACHERON_LLM_PROVIDER"
    )

    # Generic LLM settings (used as overrides or for OpenAI provider)
    llm_api_key: str = Field(default="", alias="ACHERON_LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="ACHERON_LLM_BASE_URL")
    llm_model: str = Field(default="", alias="ACHERON_LLM_MODEL")

    # Anthropic-native LLM settings
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="ANTHROPIC_MODEL")
    anthropic_max_tokens: int = Field(default=4096, alias="ANTHROPIC_MAX_TOKENS")

    # Embeddings
    embedding_api_key: str = Field(default="", alias="ACHERON_EMBEDDING_API_KEY")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="ACHERON_EMBEDDING_MODEL")

    # NCBI / PubMed
    ncbi_api_key: str = Field(default="", alias="NCBI_API_KEY")
    ncbi_email: str = Field(default="", alias="NCBI_EMAIL")
    ncbi_tool: str = Field(default="AcheronNexus", alias="NCBI_TOOL")

    # Paths
    data_dir: Path = Field(default=_PROJECT_ROOT / "data", alias="ACHERON_DATA_DIR")
    vectorstore_dir: Path = Field(
        default=_PROJECT_ROOT / "data" / "vectorstore", alias="ACHERON_VECTORSTORE_DIR"
    )

    # Server
    host: str = Field(default="127.0.0.1", alias="ACHERON_HOST")
    port: int = Field(default=8000, alias="ACHERON_PORT")

    # Live retrieval thresholds
    live_min_chunks: int = Field(
        default=3, alias="ACHERON_LIVE_MIN_CHUNKS",
        description="If local retrieval returns fewer chunks, trigger live fetch",
    )
    live_min_score: float = Field(
        default=0.35, alias="ACHERON_LIVE_MIN_SCORE",
        description="If best local relevance score is below this, trigger live fetch",
    )
    live_max_results: int = Field(
        default=10, alias="ACHERON_LIVE_MAX_RESULTS",
        description="Max papers to fetch per source during live retrieval",
    )
    live_persist_threshold: float = Field(
        default=0.6, alias="ACHERON_LIVE_PERSIST_THRESHOLD",
        description="Persist live-fetched papers with relevance above this score",
    )

    # Logging
    log_level: str = Field(default="INFO", alias="ACHERON_LOG_LEVEL")

    # Science-First Mode flags
    science_first_mode: bool = Field(
        default=True, alias="ACHERON_SCIENCE_FIRST",
        description="Enforce strict evidence tagging and planarian-first defaults",
    )
    fetch_mode: str = Field(
        default="off", alias="ACHERON_FETCH_MODE",
        description="Live fetch: 'off' (generate queries only) or 'on' (auto-fetch)",
    )
    organism_strict: str = Field(
        default="planarian", alias="ACHERON_ORGANISM_STRICT",
        description="Organism scope: 'planarian' (strict) or 'any' (cross-species)",
    )
    max_speculation: int = Field(
        default=3, alias="ACHERON_MAX_SPECULATION",
        description="Maximum number of speculative hypotheses (0-3)",
    )
    evidence_mode: str = Field(
        default="verified", alias="ACHERON_EVIDENCE_MODE",
        description=(
            "Evidence policy: 'verified' (strict, default)"
            " or 'discovery' (heuristic extrapolation allowed)"
        ),
    )

    # Nexus-Interface settings
    interface_port: int = Field(default=8100, alias="NEXUS_INTERFACE_PORT")
    interface_host: str = Field(default="0.0.0.0", alias="NEXUS_INTERFACE_HOST")
    whisper_model: str = Field(
        default="base", alias="NEXUS_WHISPER_MODEL",
        description="Whisper model size: tiny, base, small, medium, large-v3",
    )
    whisper_device: str = Field(
        default="auto", alias="NEXUS_WHISPER_DEVICE",
        description="Whisper device: auto, cpu, cuda",
    )
    piper_model_path: str = Field(
        default="", alias="NEXUS_PIPER_MODEL",
        description="Path to Piper ONNX voice model (empty = disabled)",
    )
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", alias="ELEVENLABS_VOICE_ID",
        description="ElevenLabs voice ID (default: Rachel)",
    )
    nexus_session_path: str = Field(
        default="data/nexus_session.json", alias="NEXUS_SESSION_PATH",
        description="Path to persistent session memory file",
    )

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    # Derived — active model name regardless of provider
    @property
    def active_model(self) -> str:
        """Return the model name for the currently selected provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        return self.llm_model or "gpt-4o"

    @property
    def compute_available(self) -> bool:
        """True when the selected provider has an API key configured."""
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return bool(self.llm_api_key)

    # Derived LLM settings (provider-aware defaults)
    @property
    def resolved_llm_api_key(self) -> str:
        """Return the API key for the active provider.

        For anthropic: uses ANTHROPIC_API_KEY directly (no ACHERON_LLM_API_KEY needed).
        For openai: uses ACHERON_LLM_API_KEY, falling back to OPENAI_API_KEY.
        """
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        if self.llm_api_key:
            return self.llm_api_key
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def resolved_llm_model(self) -> str:
        """Return the model, defaulting based on provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        if self.llm_model:
            return self.llm_model
        return "gpt-4o"

    @property
    def resolved_llm_base_url(self) -> str:
        """Return the base URL, defaulting based on provider."""
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "anthropic":
            return "https://api.anthropic.com"
        return "https://api.openai.com/v1"

    # Derived paths
    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "papers" / "pdf"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "papers" / "raw"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "papers" / "metadata"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    if not hasattr(get_settings, "_instance"):
        get_settings._instance = Settings()  # type: ignore[attr-defined]
    return get_settings._instance  # type: ignore[attr-defined]
