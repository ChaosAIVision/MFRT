# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**chaos-auto-prompt** is a production-grade prompt optimization SDK with FastAPI REST API. It uses meta-prompting and evaluation-based optimization to improve LLM prompts through iterative natural language feedback rather than numerical scores.

## Architecture

### Core Components

**Optimizers** (`src/chaos_auto_prompt/optimizers/`)
- `PromptLearningOptimizer`: Main optimization engine using meta-prompt approach
- `MetaPrompt`: Constructs meta-prompts for LLM-based optimization
- Supports both general prompt optimization and specialized coding agent optimization

**Providers** (`src/chaos_auto_prompt/providers/`)
- `BaseProvider`: Abstract interface for AI model integrations
- `OpenAIProvider`: OpenAI API integration (supports custom base URLs)
- `GoogleProvider`: Google AI (Gemini) integration
- All providers implement retry logic, timeout handling, and error recovery

**Configuration** (`src/chaos_auto_prompt/config/`)
- `Settings`: Environment-based configuration using pydantic-settings
- `pricing.py`: Cost tracking and budget management for all supported models
- All configuration loaded from `.env` file (no hardcoded values)

**Core Utilities** (`src/chaos_auto_prompt/core/`)
- `DatasetSplitter`: Token-aware batching for large datasets
- `PricingCalculator`: Real-time cost tracking with budget limits

**Evaluators** (`src/chaos_auto_prompt/evaluators/`)
- `ClassificationEvaluator`: Classification task evaluation
- `BaseEvaluator`: Abstract interface for custom evaluators

**Interfaces** (`src/chaos_auto_prompt/interfaces/`)
- `TokenCounter`: Abstract token counting interface
- `TiktokenCounter`: OpenAI tiktoken-based counting
- `ApproximateCounter`: Fallback character-based counting

### API Layer

**FastAPI Application** (`api/`)
- `main.py`: Production-ready FastAPI app with CORS, exception handlers, structured logging
- `routes/optimize.py`: Prompt optimization endpoints
- `routes/health.py`: Health check and system status
- `routes/huggingface.py`: HuggingFace dataset integration
- All routes under `/api/v1` prefix

## Development Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install package in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install all optional dependencies
pip install -e ".[all]"
```

### Running the API

```bash
# Method 1: Using run script (recommended)
python scripts/run_api.py

# Method 2: Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Method 3: Using python -m
python -m api.main
```

**Access Points:**
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### Testing

```bash
# Run all unit tests
pytest tests/unit/

# Run with verbose output
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=src/chaos_auto_prompt --cov-report=html

# Run specific test file
pytest tests/unit/test_pricing.py -v

# Run specific test class
pytest tests/unit/test_settings.py::TestSettingsValidation -v

# Run specific test method
pytest tests/unit/test_pricing.py::TestPricingCalculator::test_add_usage -v

# Run tests matching pattern
pytest tests/unit/ -k "budget" -v
```

### Code Quality

```bash
# Format code
black . --line-length 100

# Lint code
ruff check .

# Type checking
mypy src/chaos_auto_prompt/
```

## Configuration

All configuration is via environment variables in `.env` file:

**Required:**
```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

**API Provider Configuration:**
```bash
OPENAI_BASE_URL=https://api.openai.com/v1  # Custom base URL support
OPENAI_DEFAULT_MODEL=gpt-4o
GOOGLE_DEFAULT_MODEL=gemini-2.5-flash
```

**Server Configuration:**
```bash
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json         # json or text
```

**Optimization Settings:**
```bash
DEFAULT_MODEL=gpt-4o
DEFAULT_TEMPERATURE=0.7
MAX_OPTIMIZATION_ITERATIONS=3
OPTIMIZATION_THRESHOLD=4.0
DEFAULT_CONTEXT_SIZE=128000
BATCH_SIZE_TOKENS=32000
SAFETY_MARGIN=1000
```

**Budget Settings:**
```bash
DEFAULT_BUDGET=10.0                 # USD
BUDGET_WARNING_THRESHOLD=0.9        # 90% warning
```

## Key Design Patterns

### Meta-Prompt Optimization

The core optimization uses a two-stage meta-prompt approach:

1. **Construction Phase**: Define problem model (entities, state variables, actions, constraints)
2. **Reasoning Phase**: Generate solution based on the constructed model

This is implemented in `optimizers/meta_prompt.py` with two templates:
- `DEFAULT_META_PROMPT_TEMPLATE`: General prompt optimization
- `DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE`: Specialized for coding agents

### Provider Pattern

All AI providers implement `BaseProvider` interface:
- `generate()`: Single text generation
- `batch_generate()`: Batch processing with concurrency
- Automatic retry with exponential backoff
- Timeout handling
- Rate limit detection

### Budget Management

`PricingCalculator` tracks costs in real-time:
- Model-specific pricing from `config/pricing.py`
- Budget limit enforcement
- Warning thresholds
- Usage summaries per model

### Token-Aware Batching

`DatasetSplitter` splits large datasets into token-based batches:
- Respects model context limits
- Adds safety margins
- Estimates batch count before processing
- Handles variable-length inputs

## Exception Hierarchy

- `OptimizationError`: Optimization process failures
- `DatasetError`: Dataset validation/processing errors
- `ProviderError`: Base AI provider error
  - `ProviderTimeoutError`: Request timeout
  - `ProviderRateLimitError`: Rate limit exceeded
  - `ProviderAuthenticationError`: Authentication failure

API returns structured errors with appropriate HTTP status codes:
- 400: Dataset errors
- 422: Optimization errors
- 503: Provider errors
- 500: Unexpected errors

## Template Variables

Prompts use `{variable}` syntax for template variables:
- Extracted via regex: `r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"`
- Must be present in dataset columns
- Validated before optimization
- Preserved during meta-prompt optimization

## API Request Flow

1. Request arrives at FastAPI endpoint
2. Pydantic models validate request body
3. Provider initialized based on model choice
4. PricingCalculator tracks costs
5. DatasetSplitter creates batches if needed
6. PromptLearningOptimizer runs iterations
7. MetaPrompt constructs optimization prompts
8. Provider generates improved prompts
9. Evaluator scores results
10. Response includes optimized prompt, metrics, and usage

## Testing Strategy

- **Unit Tests** (`tests/unit/`): Core component testing with mocks
- **Fixtures** (`tests/unit/conftest.py`): Shared test data and mocks
- **Coverage Target**: 80%+
- **Test Execution**: < 30 seconds

Key test areas:
- Settings validation and environment loading
- Pricing calculations for all models
- Meta-prompt template construction
- Token counting and batch splitting
- Provider error handling and retries

## Project Structure

```
chaos-auto-prompt/
├── src/chaos_auto_prompt/
│   ├── optimizers/         # Optimization engines
│   │   ├── prompt_optimizer.py
│   │   └── meta_prompt.py
│   ├── providers/          # AI provider integrations
│   │   ├── base.py
│   │   ├── openai.py
│   │   └── google.py
│   ├── evaluators/         # Evaluation systems
│   │   └── classification.py
│   ├── core/               # Core utilities
│   │   ├── dataset_splitter.py
│   │   └── pricing.py
│   ├── config/             # Configuration
│   │   ├── settings.py
│   │   └── pricing.py
│   ├── interfaces/         # Abstract interfaces
│   │   ├── evaluator.py
│   │   └── token_counter.py
│   └── utils/              # Utilities
├── api/                    # FastAPI application
│   ├── main.py
│   ├── models.py
│   └── routes/
│       ├── optimize.py
│       ├── health.py
│       └── huggingface.py
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── examples/               # Example usage scripts
├── scripts/                # Utility scripts
└── prompts/                # Prompt templates
```

## Common Development Tasks

### Adding a New Provider

1. Create new provider in `src/chaos_auto_prompt/providers/`
2. Inherit from `BaseProvider`
3. Implement `generate()` and `batch_generate()`
4. Add provider config in `config/settings.py`
5. Add pricing info in `config/pricing.py`
6. Write unit tests in `tests/unit/test_providers.py`

### Adding a New Evaluator

1. Create evaluator in `src/chaos_auto_prompt/evaluators/`
2. Inherit from `BaseEvaluator`
3. Implement `evaluate()` method
4. Add to API models if exposing via REST API
5. Write unit tests

### Adding a New API Endpoint

1. Define Pydantic models in `api/models.py`
2. Create route in `api/routes/`
3. Import and include router in `api/main.py`
4. Update OpenAPI documentation strings
5. Test with interactive docs at `/docs`
