"""
Shared fixtures and configuration for unit tests.

This module provides common test fixtures used across all unit tests.
"""

import os
import pytest
from typing import Dict, List
from unittest.mock import Mock, AsyncMock
import pandas as pd

from chaos_auto_prompt.config.settings import Settings
from chaos_auto_prompt.providers.base import ProviderConfig, ModelCapabilities
from chaos_auto_prompt.interfaces.token_counter import TokenCounter, TiktokenCounter


@pytest.fixture
def mock_env_vars() -> Dict[str, str]:
    """Mock environment variables for testing."""
    return {
        "openai_api_key": "test-openai-key-12345",
        "google_api_key": "test-google-key-67890",
        "host": "127.0.0.1",
        "port": "8001",
        "default_model": "gpt-4o",
        "default_temperature": "0.7",
        "default_max_tokens": "4096",
        "log_level": "DEBUG",
    }


@pytest.fixture
def test_settings(mock_env_vars: Dict[str, str], monkeypatch) -> Settings:
    """Provide Settings instance with test environment variables."""
    for key, value in mock_env_vars.items():
        monkeypatch.setenv(key, value)
    return Settings()


@pytest.fixture
def mock_provider_config() -> ProviderConfig:
    """Mock provider configuration for testing."""
    return ProviderConfig(
        api_key="test-api-key",
        timeout=30,
        max_retries=2,
        model="test-model",
        temperature=0.5,
        max_tokens=1000,
    )


@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """Sample message list for testing provider generation."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with a task?"},
    ]


@pytest.fixture
def sample_text() -> str:
    """Sample text for token counting tests."""
    return "This is a sample text for token counting. It contains multiple sentences and various words to test the token counter functionality."


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample DataFrame for testing dataset operations."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "text": [
            "Short text",
            "Medium length text with more words",
            "This is a much longer text entry that contains many more words and should count significantly more tokens when processed",
            "Another medium text entry for testing purposes",
            "Final short text",
        ],
        "category": ["A", "B", "A", "C", "B"],
    })


@pytest.fixture
def token_counter() -> TokenCounter:
    """Provide a TiktokenCounter instance for testing."""
    return TiktokenCounter(encoding_name="cl100k_base")


@pytest.fixture
def mock_token_counter() -> Mock:
    """Mock token counter with predictable behavior."""
    counter = Mock(spec=TokenCounter)
    counter.count_tokens.return_value = 10
    counter.count_dataframe_tokens.return_value = [10, 20, 30, 40, 50]
    counter.estimate_tokens.return_value = 10
    return counter


@pytest.fixture
def sample_model_capabilities() -> Dict[str, ModelCapabilities]:
    """Sample model capabilities for testing."""
    return {
        "gpt-4o": ModelCapabilities(
            supports_text=True,
            supports_images=True,
            supports_grounding=False,
            max_tokens=128000,
            cost_per_1k_tokens=0.005,
        ),
        "gemini-2.5-flash": ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_grounding=True,
            max_tokens=1000000,
            cost_per_1k_tokens=0.000075,
        ),
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_resp = Mock()
    mock_resp.choices = [Mock()]
    mock_resp.choices[0].message.content = "This is a generated response from the model."
    return mock_resp


@pytest.fixture
def mock_google_response():
    """Mock Google API response."""
    mock_resp = Mock()
    mock_resp.text = "This is a generated response from the Gemini model."
    return mock_resp


@pytest.fixture
def invalid_messages() -> List[Dict[str, str]]:
    """Invalid message list for testing validation."""
    return [
        {"role": "invalid", "content": "This has an invalid role"},
        {"content": "This is missing a role"},
    ]


@pytest.fixture
def empty_messages() -> List[Dict[str, str]]:
    """Empty message list for testing validation."""
    return []


@pytest.fixture
def large_dataframe() -> pd.DataFrame:
    """Large DataFrame for performance testing."""
    data = {
        "id": list(range(1, 1001)),
        "text": ["Sample text entry with some content"] * 1000,
        "metadata": ["metadata"] * 1000,
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_async_openai_client():
    """Mock async OpenAI client."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="Mocked response"))]
    ))
    return client


@pytest.fixture
def mock_google_client():
    """Mock Google AI client."""
    client = Mock()
    client.models = Mock()
    client.models.generate_content = Mock(return_value=Mock(
        text="Mocked Google response"
    ))
    client.models.generate_content_stream = Mock(return_value=[
        Mock(text="Mocked "),
        Mock(text="Google "),
        Mock(text="streaming "),
        Mock(text="response"),
    ])
    return client


@pytest.fixture(autouse=True, scope="function")
def clean_environment(monkeypatch):
    """Clean environment before each test to ensure clean state."""
    # Clear potentially conflicting environment variables
    # These can interfere with Settings defaults
    # Note: Pydantic-settings is case-insensitive, so check both cases
    for key in ['host', 'HOST', 'port', 'PORT', 'host_alias']:
        monkeypatch.delenv(key, raising=False)

    yield
