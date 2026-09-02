"""Tests for configuration."""

from pathlib import Path

from acheron.config import Settings, get_settings


def test_default_settings():
    s = Settings()
    assert s.llm_provider == "anthropic"
    assert s.resolved_llm_model == "claude-sonnet-5"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.port == 8000
    assert isinstance(s.data_dir, Path)


def test_openai_provider_defaults():
    s = Settings(llm_provider="openai", llm_model="", llm_base_url="")
    assert s.resolved_llm_model == "gpt-4o"
    assert s.resolved_llm_base_url == "https://api.openai.com/v1"


def test_anthropic_provider_defaults():
    s = Settings(llm_provider="anthropic")
    assert s.resolved_llm_model == "claude-sonnet-5"
    assert s.resolved_llm_base_url == "https://api.anthropic.com"


def test_explicit_model_overrides_provider_default():
    s = Settings(llm_provider="anthropic", anthropic_model="claude-3-haiku-20240307")
    assert s.resolved_llm_model == "claude-3-haiku-20240307"
    assert s.active_model == "claude-3-haiku-20240307"


def test_derived_paths():
    s = Settings()
    assert s.pdf_dir == s.data_dir / "papers" / "pdf"
    assert s.raw_dir == s.data_dir / "papers" / "raw"
    assert s.metadata_dir == s.data_dir / "papers" / "metadata"


def test_get_settings_cached():
    # Clear the cache
    if hasattr(get_settings, "_instance"):
        delattr(get_settings, "_instance")
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
