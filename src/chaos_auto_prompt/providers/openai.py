"""
OpenAI provider implementation.

This module provides the OpenAIProvider class for interacting with OpenAI's API,
including GPT-4, GPT-4o, GPT-3.5-turbo, and o1/o3 series models.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI
from openai import OpenAIError, APITimeoutError, RateLimitError, AuthenticationError

from .base import (
    BaseProvider,
    ModelCapabilities,
    ProviderError,
    ProviderTimeoutError,
    ProviderRateLimitError,
    ProviderAuthenticationError,
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation using the openai library.

    Supports GPT-4, GPT-4o, GPT-3.5-turbo, and o1/o3 series models
    with streaming capabilities and comprehensive error handling.
    """

    # Model capabilities mapping
    _MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
        "gpt-4": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=False,
            max_tokens=8192,
            cost_per_1k_tokens=0.03,
        ),
        "gpt-4-turbo": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=False,
            max_tokens=128000,
            cost_per_1k_tokens=0.01,
        ),
        "gpt-4o": ModelCapabilities(
            supports_text=True,
            supports_images=True,
            supports_grounding=False,
            max_tokens=128000,
            cost_per_1k_tokens=0.005,
        ),
        "gpt-4o-mini": ModelCapabilities(
            supports_text=True,
            supports_images=True,
            supports_grounding=False,
            max_tokens=128000,
            cost_per_1k_tokens=0.00015,
        ),
        "gpt-3.5-turbo": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=False,
            max_tokens=16385,
            cost_per_1k_tokens=0.0005,
        ),
        "o1": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=False,
            max_tokens=200000,
            cost_per_1k_tokens=0.015,
        ),
        "o3": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=False,
            max_tokens=200000,
            cost_per_1k_tokens=0.015,
        ),
    }

    def __init__(
        self,
        config: Optional["ProviderConfig"] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            config: Provider configuration object (takes precedence)
            api_key: OpenAI API key (if config not provided)
            timeout: Request timeout in seconds (if config not provided)
            max_retries: Maximum retry attempts (if config not provided)
            **kwargs: Additional provider-specific configuration
        """
        super().__init__(config, api_key, timeout, max_retries, **kwargs)

        self.settings = get_settings()

        # Initialize the async OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

        # Set default model from settings if not in config
        if not self.config.model:
            self.config.model = self.settings.openai_default_model

        logger.info(
            f"OpenAIProvider initialized with model={self.config.model}, "
            f"timeout={self.config.timeout}s, max_retries={self.config.max_retries}"
        )

    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Generate text response from OpenAI model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (uses default if not specified)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Generated text response

        Raises:
            ProviderError: On generation errors
            ProviderTimeoutError: On timeout
            ProviderRateLimitError: On rate limit exceeded
            ProviderAuthenticationError: On authentication failure
        """
        # Validate messages
        self.validate_messages(messages)

        # Use provided model or default
        model_name = model or self.config.model
        if not model_name:
            raise ProviderError("No model specified and no default model configured")

        # Check model capabilities
        capabilities = self.get_model_capabilities(model_name)
        if not capabilities.supports_text:
            raise ProviderError(f"Model {model_name} does not support text generation")

        try:
            logger.debug(f"Generating text with model={model_name}, messages={len(messages)}")

            # Prepare parameters
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            if stream:
                return await self._stream_response(params)
            else:
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content or ""

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise ProviderAuthenticationError(
                f"OpenAI API authentication failed: {e}"
            ) from e

        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit error: {e}")
            raise ProviderRateLimitError(
                f"OpenAI API rate limit exceeded: {e}"
            ) from e

        except APITimeoutError as e:
            logger.warning(f"OpenAI timeout error: {e}")
            raise ProviderTimeoutError(
                f"OpenAI API request timed out: {e}"
            ) from e

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ProviderError(f"OpenAI API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI provider: {e}")
            raise ProviderError(f"Unexpected error: {e}") from e

    async def _stream_response(self, params: Dict[str, Any]) -> str:
        """
        Handle streaming response from OpenAI.

        Args:
            params: API parameters

        Returns:
            Complete generated text
        """
        full_response = ""
        async with self.client.chat.completions.create(**params) as response:
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Yield content if this is used as an async generator
        return full_response

    async def generate_text_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate text response as an async stream.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        # Validate messages
        self.validate_messages(messages)

        # Use provided model or default
        model_name = model or self.config.model
        if not model_name:
            raise ProviderError("No model specified and no default model configured")

        try:
            params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            stream = await self.client.chat.completions.create(**params, stream=True)
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except AuthenticationError as e:
            logger.error(f"OpenAI authentication error: {e}")
            raise ProviderAuthenticationError(
                f"OpenAI API authentication failed: {e}"
            ) from e

        except RateLimitError as e:
            logger.warning(f"OpenAI rate limit error: {e}")
            raise ProviderRateLimitError(
                f"OpenAI API rate limit exceeded: {e}"
            ) from e

        except APITimeoutError as e:
            logger.warning(f"OpenAI timeout error: {e}")
            raise ProviderTimeoutError(
                f"OpenAI API request timed out: {e}"
            ) from e

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ProviderError(f"OpenAI API error: {e}") from e

        except Exception as e:
            logger.error(f"Unexpected error in OpenAI provider: {e}")
            raise ProviderError(f"Unexpected error: {e}") from e

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get capabilities for a specific OpenAI model.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities object describing what the model can do
        """
        # Return capabilities if known
        if model in self._MODEL_CAPABILITIES:
            return self._MODEL_CAPABILITIES[model]

        # Return default capabilities for unknown models
        logger.warning(f"Unknown model {model}, returning default capabilities")
        return ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=False,
            max_tokens=None,
            cost_per_1k_tokens=None,
        )

    def list_available_models(self) -> List[str]:
        """
        List all available OpenAI models.

        Returns:
            List of model identifiers supported by this provider
        """
        return list(self._MODEL_CAPABILITIES.keys())

    async def generate_with_grounding(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with grounding if supported.

        Note: OpenAI models do not support grounding natively.
        This method will raise ValueError.

        Args:
            messages: List of message dictionaries
            model: Model identifier
            **kwargs: Additional parameters

        Returns:
            Generated text response

        Raises:
            ValueError: OpenAI models do not support grounding
        """
        model_name = model or self.config.model
        capabilities = self.get_model_capabilities(model_name)

        if not capabilities.supports_grounding:
            raise ValueError(
                f"Model {model_name} does not support grounding. "
                f"OpenAI models do not have native grounding capabilities. "
                f"Use Google provider for grounded generation."
            )

        # This should never be reached, but just in case
        return await self.generate_text_with_retry(messages, model, **kwargs)

    def get_default_model(self) -> str:
        """
        Get the default model for OpenAI provider.

        Returns:
            Default model identifier
        """
        if self.config.model:
            return self.config.model
        return self.settings.openai_default_model

    def __repr__(self) -> str:
        """String representation of the OpenAI provider."""
        return (
            f"OpenAIProvider("
            f"model={self.config.model}, "
            f"timeout={self.config.timeout}, "
            f"max_retries={self.config.max_retries})"
        )
