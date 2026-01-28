"""
Google AI provider implementation using Gemini models.

This module provides integration with Google's Gemini models through
the google-genai library, supporting text generation with optional streaming.
"""

import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .base import (
    BaseProvider,
    ModelCapabilities,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """
    Google AI provider using Gemini models.

    Supports Gemini 2.5 and 1.5 series models with text generation
    and optional streaming capabilities.
    """

    # Model capabilities mapping
    _MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
        "gemini-2.5-flash": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=True,
            max_tokens=128000,
        ),
        "gemini-2.5-pro": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=True,
            max_tokens=128000,
        ),
        "gemini-1.5-flash": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=True,
            max_tokens=128000,
        ),
        "gemini-1.5-pro": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=True,
            max_tokens=128000,
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
        Initialize the Google provider.

        Args:
            config: Provider configuration object (takes precedence)
            api_key: Google API key (if config not provided)
            timeout: Request timeout in seconds (if config not provided)
            max_retries: Maximum retry attempts (if config not provided)
            **kwargs: Additional provider-specific configuration
        """
        super().__init__(config, api_key, timeout, max_retries, **kwargs)

        # Initialize Google AI client
        try:
            self.client = genai.Client(api_key=self.config.api_key)
            logger.info("Google AI client initialized successfully")
        except Exception as e:
            raise ProviderAuthenticationError(
                f"Failed to initialize Google AI client: {e}"
            )

        # Set default model from settings or kwargs
        if not self.config.model:
            default_model = kwargs.get(
                "model", self.settings.google_default_model
            )
            self.config.model = default_model

        logger.debug(
            f"GoogleProvider initialized with model={self.config.model}, "
            f"timeout={self.config.timeout}s, "
            f"max_retries={self.config.max_retries}"
        )

    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text response using Gemini models.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (uses default if not specified)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters including:
                - stream: bool - Enable streaming (default: False)

        Returns:
            Generated text response

        Raises:
            ProviderAuthenticationError: On authentication failure
            ProviderTimeoutError: On request timeout
            ProviderRateLimitError: On rate limit exceeded
            ProviderError: On other generation errors
        """
        # Validate messages
        self.validate_messages(messages)

        # Use provided model or default
        model_name = model or self.config.model
        if not model_name:
            raise ProviderError("Model must be specified")

        # Check model capabilities
        capabilities = self.get_model_capabilities(model_name)
        if not capabilities.supports_text:
            raise ProviderError(f"Model {model_name} does not support text generation")

        # Format messages for Gemini API
        prompt_text = self._format_messages(messages)

        # Check if streaming is requested
        stream = kwargs.get("stream", False)

        try:
            logger.debug(f"Generating text with model={model_name}, stream={stream}")

            # Build generation config
            config = types.GenerateContentConfig(
                temperature=temperature,
            )

            # Generate content
            if stream:
                return await self._generate_streaming(
                    model_name, prompt_text, config
                )
            else:
                return await self._generate_non_streaming(
                    model_name, prompt_text, config
                )

        except genai.errors.APIError as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                raise ProviderAuthenticationError(f"Google authentication failed: {e}")
            elif "429" in str(e) or "quota" in str(e).lower():
                raise ProviderRateLimitError(f"Google rate limit exceeded: {e}")
            else:
                raise ProviderError(f"Google API error: {e}")

        except Exception as e:
            if "timeout" in str(e).lower():
                raise ProviderTimeoutError(f"Request timed out: {e}")
            raise ProviderError(f"Unexpected error during generation: {e}")

    async def _generate_non_streaming(
        self,
        model: str,
        content: str,
        config: types.GenerateContentConfig,
    ) -> str:
        """
        Generate text without streaming.

        Args:
            model: Model identifier
            content: Formatted prompt content
            config: Generation configuration

        Returns:
            Generated text response
        """
        response = self.client.models.generate_content(
            model=model,
            contents=content,
            config=config,
        )

        if not response or not response.text:
            raise ProviderError("Empty response from Google API")

        return response.text

    async def _generate_streaming(
        self,
        model: str,
        content: str,
        config: types.GenerateContentConfig,
    ) -> str:
        """
        Generate text with streaming.

        Args:
            model: Model identifier
            content: Formatted prompt content
            config: Generation configuration

        Returns:
            Generated text response (concatenated from all chunks)
        """
        full_response = []

        for chunk in self.client.models.generate_content_stream(
            model=model,
            contents=content,
            config=config,
        ):
            if chunk.text:
                full_response.append(chunk.text)

        result = "".join(full_response)

        if not result:
            raise ProviderError("Empty response from Google API (streaming)")

        return result

    async def generate_with_grounding(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with Google Search grounding.

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
        model_name = model or self.config.model
        if not model_name:
            raise ValueError("Model must be specified for grounding")

        # Verify model supports grounding
        capabilities = self.get_model_capabilities(model_name)
        if not capabilities.supports_grounding:
            raise ValueError(
                f"Model {model_name} does not support grounding. "
                f"Use a model with grounding capabilities."
            )

        # Validate messages
        self.validate_messages(messages)

        # Format messages for Gemini API
        prompt_text = self._format_messages(messages)

        try:
            # Configure Google Search tool for grounding
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            config = types.GenerateContentConfig(tools=[grounding_tool])

            logger.debug(f"Generating text with grounding using model={model_name}")

            # Generate content with grounding
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt_text,
                config=config,
            )

            if not response or not response.text:
                raise ProviderError("Empty response from Google API (grounding)")

            return response.text

        except genai.errors.APIError as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                raise ProviderAuthenticationError(f"Google authentication failed: {e}")
            elif "429" in str(e) or "quota" in str(e).lower():
                raise ProviderRateLimitError(f"Google rate limit exceeded: {e}")
            else:
                raise ProviderError(f"Google API error (grounding): {e}")

        except Exception as e:
            if "timeout" in str(e).lower():
                raise ProviderTimeoutError(f"Request timed out: {e}")
            raise ProviderError(f"Unexpected error during grounded generation: {e}")

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get capabilities for a specific model.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities object describing what the model can do
        """
        return self._MODEL_CAPABILITIES.get(
            model,
            ModelCapabilities(supports_text=True, supports_images=False),
        )

    def list_available_models(self) -> List[str]:
        """
        List all available models for this provider.

        Returns:
            List of model identifiers
        """
        return list(self._MODEL_CAPABILITIES.keys())

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemini prompt format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted prompt string for Gemini API
        """
        formatted_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
            else:
                # Unknown role, default to user
                formatted_parts.append(f"{role}: {content}")

        return "\n\n".join(formatted_parts)
