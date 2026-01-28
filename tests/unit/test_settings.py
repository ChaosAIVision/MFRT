"""
Unit tests for configuration loading and settings management.

Tests the Settings class from chaos_auto_prompt.config.settings
including environment variable loading, validation, and defaults.
"""

import os
import pytest
from typing import Dict
from pydantic import ValidationError

from chaos_auto_prompt.config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Test default values for settings."""

    def test_default_server_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default server configuration values."""
        # Set only required env vars
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        # Clear potentially conflicting env vars that might be in the conda environment
        for env_key in ["host", "port"]:
            monkeypatch.delenv(env_key, raising=False)

        settings = Settings()

        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.reload is True
        assert settings.workers == 1

    def test_default_cors_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default CORS configuration."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:4357" in settings.cors_origins
        assert settings.cors_allow_credentials is True
        assert settings.cors_allow_methods == ["*"]
        assert settings.cors_allow_headers == ["*"]

    def test_default_model_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default model configuration."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        assert settings.default_model == "gpt-4o"
        assert settings.default_temperature == 0.7
        assert settings.default_max_tokens == 4096

    def test_default_optimization_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default optimization settings."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        assert settings.optimization_threshold == 4.0
        assert settings.max_optimization_iterations == 3
        assert settings.default_context_size == 128000
        assert settings.batch_size_tokens == 32000
        assert settings.safety_margin == 1000

    def test_default_budget_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default budget settings."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        assert settings.default_budget == 10.0
        assert settings.budget_warning_threshold == 0.9

    def test_default_provider_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default provider-specific settings."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        # OpenAI defaults
        assert settings.openai_default_model == "gpt-4o"
        assert settings.openai_timeout == 60
        assert settings.openai_max_retries == 3

        # Google defaults
        assert settings.google_default_model == "gemini-2.5-flash"
        assert settings.google_timeout == 60
        assert settings.google_max_retries == 3

    def test_default_delimiter_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default token delimiter settings."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        assert settings.start_delim == "{"
        assert settings.end_delim == "}"

    def test_default_meta_prompt_settings(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test default meta-prompt settings."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])

        settings = Settings()

        assert settings.meta_prompt_template is None
        assert settings.coding_agent_meta_prompt_template is None
        assert settings.save_meta_prompt_debug is False
        assert settings.meta_prompt_debug_path == "metaprompt_debug.txt"


class TestSettingsValidation:
    """Test settings validation and error handling."""

    def test_missing_openai_api_key(self, monkeypatch):
        """Test that missing OpenAI API key raises validation error."""
        # Delete both lowercase and uppercase versions (pydantic-settings is case-insensitive)
        monkeypatch.delenv("openai_api_key", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("google_api_key", "test-google-key")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "openai_api_key" in str(exc_info.value).lower()

    def test_missing_google_api_key(self, monkeypatch):
        """Test that missing Google API key raises validation error."""
        # Delete both lowercase and uppercase versions (pydantic-settings is case-insensitive)
        monkeypatch.delenv("google_api_key", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("openai_api_key", "test-openai-key")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "google_api_key" in str(exc_info.value).lower()

    def test_invalid_port_type(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that invalid port type raises validation error."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])
        monkeypatch.setenv("port", "invalid")

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_temperature_range(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that invalid temperature value is handled."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])
        monkeypatch.setenv("default_temperature", "2.5")  # Above typical range

        # Settings should accept it (validation is delegated to providers)
        settings = Settings()
        assert settings.default_temperature == 2.5

    def test_negative_max_tokens(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that negative max_tokens is accepted (provider validates)."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])
        monkeypatch.setenv("default_max_tokens", "-100")

        # Settings accepts it, validation happens at provider level
        settings = Settings()
        assert settings.default_max_tokens == -100

    def test_extra_fields_are_ignored(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that extra environment variables are ignored."""
        for key in ["openai_api_key", "google_api_key"]:
            if key in mock_env_vars:
                monkeypatch.setenv(key, mock_env_vars[key])
        monkeypatch.setenv("UNKNOWN_FIELD", "some_value")

        # Should not raise error due to extra="ignore"
        settings = Settings()
        assert not hasattr(settings, "UNKNOWN_FIELD")

    def test_case_insensitive_env_vars(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that environment variables are case-insensitive."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

        settings = Settings()
        assert settings.openai_api_key == "test-openai-key"
        assert settings.google_api_key == "test-google-key"


class TestSettingsEnvironmentLoading:
    """Test loading settings from environment variables."""

    def test_load_api_keys_from_env(self, monkeypatch):
        """Test loading API keys from environment."""
        monkeypatch.setenv("openai_api_key", "sk-test-123")
        monkeypatch.setenv("google_api_key", "google-test-456")

        settings = Settings()
        assert settings.openai_api_key == "sk-test-123"
        assert settings.google_api_key == "google-test-456"

    def test_load_server_config_from_env(self, monkeypatch):
        """Test loading server configuration from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("host", "127.0.0.1")
        monkeypatch.setenv("port", "9000")
        monkeypatch.setenv("workers", "4")

        settings = Settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.workers == 4

    def test_load_model_config_from_env(self, monkeypatch):
        """Test loading model configuration from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("default_model", "gpt-4-turbo")
        monkeypatch.setenv("default_temperature", "0.5")
        monkeypatch.setenv("default_max_tokens", "8192")

        settings = Settings()
        assert settings.default_model == "gpt-4-turbo"
        assert settings.default_temperature == 0.5
        assert settings.default_max_tokens == 8192

    def test_load_budget_config_from_env(self, monkeypatch):
        """Test loading budget configuration from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("default_budget", "50.0")
        monkeypatch.setenv("budget_warning_threshold", "0.8")

        settings = Settings()
        assert settings.default_budget == 50.0
        assert settings.budget_warning_threshold == 0.8

    def test_load_optimization_config_from_env(self, monkeypatch):
        """Test loading optimization configuration from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("optimization_threshold", "5.0")
        monkeypatch.setenv("max_optimization_iterations", "5")
        monkeypatch.setenv("batch_size_tokens", "64000")

        settings = Settings()
        assert settings.optimization_threshold == 5.0
        assert settings.max_optimization_iterations == 5
        assert settings.batch_size_tokens == 64000

    def test_load_provider_config_from_env(self, monkeypatch):
        """Test loading provider-specific configuration from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("openai_default_model", "gpt-4-turbo")
        monkeypatch.setenv("openai_timeout", "120")
        monkeypatch.setenv("google_default_model", "gemini-2.5-pro")
        monkeypatch.setenv("google_timeout", "90")

        settings = Settings()
        assert settings.openai_default_model == "gpt-4-turbo"
        assert settings.openai_timeout == 120
        assert settings.google_default_model == "gemini-2.5-pro"
        assert settings.google_timeout == 90

    def test_load_meta_prompt_config_from_env(self, monkeypatch):
        """Test loading meta-prompt configuration from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("meta_prompt_template", "Custom meta prompt template")
        monkeypatch.setenv("save_meta_prompt_debug", "true")
        monkeypatch.setenv("meta_prompt_debug_path", "/tmp/debug.txt")

        settings = Settings()
        assert settings.meta_prompt_template == "Custom meta prompt template"
        assert settings.save_meta_prompt_debug is True
        assert settings.meta_prompt_debug_path == "/tmp/debug.txt"

    def test_load_cors_origins_as_list(self, monkeypatch):
        """Test loading CORS origins from environment (comma-separated if string)."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        # Note: This would need custom parsing in Settings to work as comma-separated
        # For now, testing that it can be set programmatically
        settings = Settings(cors_origins=["http://example.com", "http://test.com"])
        assert "http://example.com" in settings.cors_origins
        assert "http://test.com" in settings.cors_origins

    def test_load_delimiters_from_env(self, monkeypatch):
        """Test loading token delimiters from environment."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("start_delim", "[")
        monkeypatch.setenv("end_delim", "]")

        settings = Settings()
        assert settings.start_delim == "["
        assert settings.end_delim == "]"


class TestGetSettings:
    """Test the get_settings() cached singleton function."""

    def test_get_settings_returns_singleton(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that get_settings returns cached instance."""
        for key, value in mock_env_vars.items():
            monkeypatch.setenv(key, value)

        # Import fresh to get the function, not the settings instance
        import sys
        if 'chaos_auto_prompt.config.settings' in sys.modules:
            del sys.modules['chaos_auto_prompt.config.settings']

        from chaos_auto_prompt.config.settings import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_cache_clear(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that settings cache can be cleared."""
        import sys

        for key, value in mock_env_vars.items():
            monkeypatch.setenv(key, value)

        # Import fresh
        if 'chaos_auto_prompt.config.settings' in sys.modules:
            del sys.modules['chaos_auto_prompt.config.settings']
        from chaos_auto_prompt.config.settings import get_settings

        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()

        # Should be different instances after cache clear
        assert settings1 is not settings2
        # But should have same values
        assert settings1.openai_api_key == settings2.openai_api_key

    def test_get_settings_with_env_change(self, mock_env_vars: Dict[str, str], monkeypatch):
        """Test that get_settings respects environment changes after cache clear."""
        import sys

        for key, value in mock_env_vars.items():
            monkeypatch.setenv(key, value)

        # Import fresh
        if 'chaos_auto_prompt.config.settings' in sys.modules:
            del sys.modules['chaos_auto_prompt.config.settings']
        from chaos_auto_prompt.config.settings import get_settings

        settings1 = get_settings()
        original_key = settings1.openai_api_key

        # Change environment
        monkeypatch.setenv("openai_api_key", "new-openai-key")
        get_settings.cache_clear()

        settings2 = get_settings()
        assert settings2.openai_api_key == "new-openai-key"
        assert settings2.openai_api_key != original_key


class TestSettingsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_cors_origins_list(self, monkeypatch):
        """Test settings with empty CORS origins list."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        settings = Settings(cors_origins=[])
        assert settings.cors_origins == []

    def test_zero_budget(self, monkeypatch):
        """Test settings with zero budget limit."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("default_budget", "0")

        settings = Settings()
        assert settings.default_budget == 0.0

    def test_very_large_max_tokens(self, monkeypatch):
        """Test settings with very large max_tokens value."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("default_max_tokens", "10000000")

        settings = Settings()
        assert settings.default_max_tokens == 10000000

    def test_boolean_reload_setting(self, monkeypatch):
        """Test boolean reload setting from various string formats."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        # Test "true" string
        monkeypatch.setenv("reload", "true")
        settings = Settings()
        assert settings.reload is True

        # Test "false" string
        monkeypatch.setenv("reload", "false")
        settings = Settings()
        assert settings.reload is False

    def test_log_level_values(self, monkeypatch):
        """Test various log level values."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            monkeypatch.setenv("log_level", level)
            settings = Settings()
            assert settings.log_level == level

    def test_log_format_values(self, monkeypatch):
        """Test various log format values."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        for fmt in ["json", "text"]:
            monkeypatch.setenv("log_format", fmt)
            settings = Settings()
            assert settings.log_format == fmt

    def test_unicode_in_api_keys(self, monkeypatch):
        """Test that API keys can contain unicode characters."""
        monkeypatch.setenv("openai_api_key", "sk-test-∆-key")
        monkeypatch.setenv("google_api_key", "google-λ-key")

        settings = Settings()
        assert "∆" in settings.openai_api_key
        assert "λ" in settings.google_api_key
