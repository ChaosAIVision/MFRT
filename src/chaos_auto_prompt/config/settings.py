"""
Configuration management using pydantic-settings.

All configuration is loaded from environment variables.
No hardcoded values - production-ready approach.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =============================================================================
    # API Keys
    # =============================================================================
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI base URL")
    google_api_key: str = Field(..., description="Google AI API key")

    # =============================================================================
    # Server Configuration
    # =============================================================================
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=True, description="Enable auto-reload")
    workers: int = Field(default=1, description="Number of worker processes")

    # =============================================================================
    # CORS Configuration
    # =============================================================================
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:4357"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow credentials")
    cors_allow_methods: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="Allowed headers")

    # =============================================================================
    # Default Model Settings
    # =============================================================================
    default_model: str = Field(default="gpt-4o", description="Default model to use")
    default_temperature: float = Field(default=0.7, description="Default temperature")
    default_max_tokens: int = Field(default=4096, description="Default max tokens")

    # =============================================================================
    # Optimization Settings
    # =============================================================================
    optimization_threshold: float = Field(default=4.0, description="Optimization convergence threshold")
    max_optimization_iterations: int = Field(default=3, description="Max optimization iterations")
    default_context_size: int = Field(default=128000, description="Default context window size")
    batch_size_tokens: int = Field(default=32000, description="Batch size in tokens")
    safety_margin: int = Field(default=1000, description="Token safety margin")

    # =============================================================================
    # Budget Settings
    # =============================================================================
    default_budget: float = Field(default=10.0, description="Default budget limit")
    budget_warning_threshold: float = Field(default=0.9, description="Budget warning threshold (0-1)")

    # =============================================================================
    # Token Delimiters
    # =============================================================================
    start_delim: str = Field(default="{", description="Template variable start delimiter")
    end_delim: str = Field(default="}", description="Template variable end delimiter")

    # =============================================================================
    # Provider Settings
    # =============================================================================
    # OpenAI
    openai_default_model: str = Field(default="gpt-4o", description="OpenAI default model")
    openai_timeout: int = Field(default=60, description="OpenAI request timeout")
    openai_max_retries: int = Field(default=3, description="OpenAI max retries")

    # Google AI
    google_default_model: str = Field(default="gemini-2.5-flash", description="Google default model")
    google_timeout: int = Field(default=60, description="Google request timeout")
    google_max_retries: int = Field(default=3, description="Google max retries")

    # =============================================================================
    # Logging
    # =============================================================================
    log_level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    log_format: str = Field(default="json", description="Log format (json, text)")

    # =============================================================================
    # Meta-Prompt Settings
    # =============================================================================
    # These allow customization of meta-prompt templates via environment variables
    # If not set, defaults from the MetaPrompt class will be used
    meta_prompt_template: str | None = Field(
        default=None,
        description="Custom meta-prompt template for general prompt optimization"
    )
    coding_agent_meta_prompt_template: str | None = Field(
        default=None,
        description="Custom meta-prompt template for coding agent optimization"
    )
    save_meta_prompt_debug: bool = Field(
        default=False,
        description="Whether to save generated meta-prompts to disk for debugging"
    )
    meta_prompt_debug_path: str = Field(
        default="metaprompt_debug.txt",
        description="Path to save debug meta-prompt output"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    Call this function to get the current settings.

    Returns:
        Settings: The application settings
    """
    return Settings()


# Export for convenience
settings = get_settings()
