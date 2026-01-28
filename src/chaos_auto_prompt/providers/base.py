"""
Base provider interface for AI model integrations.

This module provides the abstract base class for all AI providers,
ensuring consistent interface across OpenAI, Google, and other providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """
    Describes what a model can do.

    Attributes:
        supports_text: Whether the model supports text generation
        supports_images: Whether the model supports image inputs
        supports_grounding: Whether the model supports grounded generation
        max_tokens: Maximum tokens the model can handle
        cost_per_1k_tokens: Cost per 1000 tokens (optional)
    """

    supports_text: bool = True
    supports_images: bool = False
    supports_grounding: bool = False
    max_tokens: Optional[int] = None
    cost_per_1k_tokens: Optional[float] = None


@dataclass
class ProviderConfig:
    """
    Configuration for a provider instance.

    Attributes:
        api_key: API key for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        model: Default model to use
        temperature: Default sampling temperature
        max_tokens: Default maximum tokens to generate
    """

    api_key: str
    timeout: int = 60
    max_retries: int = 3
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ProviderError(Exception):
    """Base exception for provider errors."""

    pass


class ProviderTimeoutError(ProviderError):
    """Exception raised when a provider request times out."""

    pass


class ProviderRateLimitError(ProviderError):
    """Exception raised when rate limit is exceeded."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Exception raised when authentication fails."""

    pass


class BaseProvider(ABC):
    """
    Abstract base class for AI model providers.

    This class defines the interface that all providers must implement,
    including error handling, retry logic, and configuration management.

    Providers should inherit from this class and implement the abstract methods.
    """

    def __init__(
        self,
        config: Optional[ProviderConfig] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the base provider.

        Args:
            config: Provider configuration object (takes precedence)
            api_key: API key for authentication (if config not provided)
            timeout: Request timeout in seconds (if config not provided)
            max_retries: Maximum retry attempts (if config not provided)
            **kwargs: Additional provider-specific configuration
        """
        self.settings = get_settings()

        # Use config object if provided, otherwise build from parameters
        if config is not None:
            self.config = config
        else:
            # Get default values from settings
            provider_name = self._get_provider_name()
            default_timeout = self._get_setting_value(
                f"{provider_name}_timeout", "openai_timeout", default=60
            )
            default_max_retries = self._get_setting_value(
                f"{provider_name}_max_retries", "openai_max_retries", default=3
            )
            default_api_key = self._get_setting_value(
                f"{provider_name}_api_key", "openai_api_key", default=""
            )

            self.config = ProviderConfig(
                api_key=api_key or default_api_key,
                timeout=timeout or default_timeout,
                max_retries=max_retries or default_max_retries,
                **kwargs,
            )

        # Validate configuration
        if not self.config.api_key:
            raise ProviderAuthenticationError(
                f"API key is required for {self.__class__.__name__}. "
                f"Set {self._get_provider_name()}_api_key environment variable."
            )

        logger.debug(
            f"Initialized {self.__class__.__name__} with timeout={self.config.timeout}, "
            f"max_retries={self.config.max_retries}"
        )

    def _get_provider_name(self) -> str:
        """
        Get the provider name for settings lookup.

        Returns:
            Provider name (e.g., 'openai', 'google')
        """
        class_name = self.__class__.__name__.lower()
        # Remove 'provider' suffix if present
        if "provider" in class_name:
            class_name = class_name.replace("provider", "")
        return class_name

    def _get_setting_value(
        self, primary_key: str, fallback_key: str, default: Any = None
    ) -> Any:
        """
        Get a setting value with fallback.

        Args:
            primary_key: Primary setting key to look up
            fallback_key: Fallback setting key if primary doesn't exist
            default: Default value if neither setting exists

        Returns:
            Setting value or default
        """
        if hasattr(self.settings, primary_key):
            return getattr(self.settings, primary_key)
        if hasattr(self.settings, fallback_key):
            return getattr(self.settings, fallback_key)
        return default

    @abstractmethod
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text response from the model.

        This is the main method that providers must implement.
        It should handle the actual API call to the provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (uses default if not specified)
            temperature: Sampling temperature (uses default if not specified)
            max_tokens: Maximum tokens to generate (uses default if not specified)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            ProviderError: On generation errors
            ProviderTimeoutError: On timeout
            ProviderRateLimitError: On rate limit exceeded
            ProviderAuthenticationError: On authentication failure
        """
        pass

    @abstractmethod
    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get capabilities for a specific model.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities object describing what the model can do
        """
        pass

    @abstractmethod
    def list_available_models(self) -> List[str]:
        """
        List all available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    async def generate_text_with_retry(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with automatic retry logic.

        This method wraps generate_text with retry logic for transient failures.
        It handles rate limiting, timeouts, and temporary network issues.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response

        Raises:
            ProviderError: If all retries are exhausted
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"Generation attempt {attempt + 1}/{self.config.max_retries}"
                )
                return await asyncio.wait_for(
                    self.generate_text(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    ),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError as e:
                last_error = ProviderTimeoutError(
                    f"Request timed out after {self.config.timeout}s"
                )
                logger.warning(
                    f"Attempt {attempt + 1} timed out: {last_error}"
                )
            except ProviderAuthenticationError as e:
                # Don't retry authentication errors
                logger.error(f"Authentication error: {e}")
                raise
            except ProviderRateLimitError as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1} hit rate limit: {e}"
                )
                # Exponential backoff for rate limits
                if attempt < self.config.max_retries - 1:
                    backoff = 2**attempt
                    logger.info(f"Backing off for {backoff}s before retry")
                    await asyncio.sleep(backoff)
            except ProviderError as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1} failed with provider error: {e}"
                )
                # Small backoff for other errors
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                last_error = ProviderError(f"Unexpected error: {e}")
                logger.warning(
                    f"Attempt {attempt + 1} failed with unexpected error: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(1)

        # All retries exhausted
        logger.error(f"All {self.config.max_retries} attempts failed")
        raise last_error or ProviderError("All retry attempts failed")

    async def generate_with_grounding(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with grounding if supported.

        Grounding provides citations or evidence for generated content.
        Only works if the model supports it.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response with grounding

        Raises:
            ValueError: If model doesn't support grounding
            ProviderError: On generation errors
        """
        model = model or self.config.model
        if not model:
            raise ValueError("Model must be specified for grounding")

        capabilities = self.get_model_capabilities(model)
        if not capabilities.supports_grounding:
            raise ValueError(
                f"Model {model} does not support grounding. "
                f"Use a model with grounding capabilities."
            )

        # Default implementation - subclasses should override
        return await self.generate_text_with_retry(messages, model, **kwargs)

    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate message format.

        Args:
            messages: List of message dictionaries to validate

        Raises:
            ValueError: If messages are invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' field")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(
                    f"Message {i} has invalid role: {msg['role']}. "
                    "Must be 'system', 'user', or 'assistant'"
                )

    def get_default_model(self) -> str:
        """
        Get the default model for this provider.

        Returns:
            Default model identifier

        Raises:
            ValueError: If no default model is configured
        """
        if self.config.model:
            return self.config.model

        # Try to get from settings
        provider_name = self._get_provider_name()
        default_model = self._get_setting_value(
            f"{provider_name}_default_model", "default_model"
        )

        if not default_model:
            raise ValueError(
                f"No default model configured for {self.__class__.__name__}"
            )

        return default_model

    def __repr__(self) -> str:
        """String representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model}, "
            f"timeout={self.config.timeout}, "
            f"max_retries={self.config.max_retries})"
        )
