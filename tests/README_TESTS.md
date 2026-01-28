# Unit Tests for chaos-auto-prompt

Comprehensive unit test suite for the chaos-auto-prompt core components.

## Test Structure

```
tests/
├── unit/
│   ├── conftest.py              # Shared fixtures and test configuration
│   ├── test_settings.py         # Configuration loading tests
│   ├── test_pricing.py          # Pricing calculator tests
│   ├── test_meta_prompt.py      # Meta-prompt construction tests
│   ├── test_dataset_splitter.py # Dataset batching tests
│   └── test_providers.py        # Provider implementation tests
└── pytest.ini                   # Pytest configuration
```

## Running Tests

### Run All Unit Tests

```bash
# From project root
cd chaos-auto-prompt
pytest tests/unit/

# With verbose output
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ --cov=chaos_auto_prompt --cov-report=html
```

### Run Specific Test Files

```bash
# Test settings only
pytest tests/unit/test_settings.py -v

# Test pricing only
pytest tests/unit/test_pricing.py -v

# Test providers only
pytest tests/unit/test_providers.py -v
```

### Run Specific Test Cases

```bash
# Run specific test class
pytest tests/unit/test_settings.py::TestSettingsValidation -v

# Run specific test method
pytest tests/unit/test_pricing.py::TestPricingCalculator::test_add_usage -v

# Run tests matching pattern
pytest tests/unit/ -k "budget" -v
```

### Run Tests with Marks

```bash
# Run only fast tests (exclude slow)
pytest tests/unit/ -m "not slow" -v

# Run only unit tests
pytest tests/unit/ -m "unit" -v
```

## Test Coverage

The test suite covers:

### Configuration (test_settings.py)
- Environment variable loading
- Default value validation
- Settings caching
- Type validation
- Edge cases (empty values, unicode, etc.)

### Pricing (test_pricing.py)
- Cost calculation for all models
- Budget tracking
- Usage summaries
- Edge cases (zero budget, large tokens, etc.)
- Provider cost comparisons

### Meta-Prompt (test_meta_prompt.py)
- Template construction
- Variable formatting
- Debug mode
- Ruleset mode
- Special character handling

### Dataset Splitter (test_dataset_splitter.py)
- Token-aware batching
- Batch estimation
- Safety margin handling
- Large dataset performance
- Edge cases (empty data, NaN values, etc.)

### Providers (test_providers.py)
- OpenAI provider implementation
- Google provider implementation
- Error handling and retries
- Authentication
- Rate limiting
- Timeouts
- Concurrent requests

## Test Fixtures

Shared fixtures are defined in `conftest.py`:

- `mock_env_vars` - Mock environment variables
- `test_settings` - Settings instance with test config
- `mock_provider_config` - Mock provider configuration
- `sample_messages` - Sample message list for testing
- `sample_dataframe` - Sample DataFrame for testing
- `token_counter` - Real token counter instance
- `mock_token_counter` - Mock token counter
- `sample_model_capabilities` - Sample model capabilities
- `mock_openai_response` - Mock OpenAI API response
- `mock_google_response` - Mock Google API response
- And more...

## Writing New Tests

### Test File Template

```python
"""
Tests for [component name].
"""

import pytest
from chaos_auto_prompt.[module] import [ClassName]


class Test[ClassName]:
    """Test [ClassName] functionality."""

    def test_something(self, fixture_name):
        """Test description."""
        # Arrange
        instance = ClassName()

        # Act
        result = instance.method()

        # Assert
        assert result == expected_value


class Test[ClassName]EdgeCases:
    """Test edge cases and special scenarios."""

    def test_edge_case(self):
        """Test edge case description."""
        # Test implementation
        pass
```

### Best Practices

1. **Use descriptive test names** - `test_calculate_cost_with_both_input_and_output_tokens`
2. **Follow Arrange-Act-Assert pattern** - Clear structure in each test
3. **Use fixtures** - Shared setup in conftest.py
4. **Test edge cases** - Empty inputs, None values, large numbers, etc.
5. **Mock external dependencies** - API calls, file I/O, etc.
6. **Test error handling** - Exception cases and validation
7. **Keep tests independent** - Each test should work in isolation

## Continuous Integration

To run tests in CI:

```yaml
# Example GitHub Actions workflow
- name: Run unit tests
  run: |
    cd chaos-auto-prompt
    pip install -e ".[test]"
    pytest tests/unit/ -v --cov=chaos_auto_prompt --cov-report=xml
```

## Troubleshooting

### Import Errors

If you get import errors, ensure the package is installed:

```bash
cd chaos-auto-prompt
pip install -e .
```

### Environment Variables

Tests require API keys to be set (can be fake values for tests):

```bash
export OPENAI_API_KEY=test-key
export GOOGLE_API_KEY=test-key
```

### Async Tests

Make sure pytest-asyncio is installed:

```bash
pip install pytest-asyncio
```

## Test Statistics

- **Total test files**: 5
- **Total test cases**: 150+
- **Code coverage target**: 80%+
- **Test execution time**: < 30 seconds

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Add fixtures to conftest.py if needed
4. Update this README if adding new test files
5. Maintain or improve code coverage

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Mock Documentation](https://docs.pytest.org/en/stable/unittest.html)
- [Asyncio Testing with Pytest](https://pytest-asyncio.readthedocs.io/)
