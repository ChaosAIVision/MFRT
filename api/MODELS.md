# API Models Documentation

This document describes the Pydantic models defined in `api/models.py` for the chaos-auto-prompt API.

## Overview

The models provide type-safe request validation and response serialization for the prompt optimization API. All models use Pydantic v2 syntax with proper validation, field descriptions, and type hints.

## Model Categories

### 1. Base Models

#### `Message`
Represents a chat message with role and content.

**Fields:**
- `role` (str): Message role - "system", "user", or "assistant"
- `content` (str): Message content

**Validation:** Role must be one of the allowed values.

#### `ModelCapabilities`
Describes what a model can do.

**Fields:**
- `supports_text` (bool): Whether model supports text generation
- `supports_images` (bool): Whether model supports image inputs
- `supports_grounding` (bool): Whether model supports grounded generation
- `max_tokens` (int | None): Maximum token limit
- `cost_per_1k_tokens` (float | None): Cost per 1000 tokens in USD

#### `PricingInfo`
Pricing information for a model.

**Fields:**
- `input_price` (float): Price per 1M input tokens in USD (must be > 0)
- `output_price` (float): Price per 1M output tokens in USD (must be > 0)
- `model_name` (str): Model identifier

### 2. Request Models

#### `DatasetInput`
Dataset input for optimization, supporting multiple formats.

**Fields:**
- `data` (str | List[Dict] | Dict): Dataset data
  - JSON string
  - List of row dictionaries
  - Column-oriented dictionary
- `format` (str | None): Format hint ("json", "list", "columns")

**Validation:** Format must be one of the allowed values if provided.

#### `PromptFormat`
Prompt format specification.

**Fields:**
- `prompt` (str | List[Message]): The prompt content
- `type` (str | None): Prompt type ("string" or "messages")

#### `OptimizeRequest` ⭐ Main Request Model
Request for prompt optimization.

**Fields:**
- `prompt` (str | List[Message]): The prompt to optimize
- `dataset` (DatasetInput | List[Dict] | str): Dataset with examples and feedback
- `output_column` (str): Name of column with LLM outputs
- `feedback_columns` (List[str]): Columns with feedback data (min_length=1)
- `model` (str | None): Model to use (default: from settings)
- `provider` (str | None): Provider name
- `budget` (float | None): Maximum budget in USD (must be > 0)
- `context_size` (int | None): Context window size in tokens (must be > 0)
- `ruleset` (str | None): Initial ruleset for coding agent mode
- `verbose` (bool): Enable verbose logging (default: False)
- `validation_only` (bool): Only validate inputs (default: False)

#### `ProviderRequest`
Base request for provider operations.

**Fields:**
- `api_key` (str | None): API key for authentication
- `model` (str | None): Model identifier
- `temperature` (float | None): Sampling temperature (0-2)
- `max_tokens` (int | None): Maximum tokens to generate

#### `OpenAIRequest`
OpenAI-specific request extending ProviderRequest.

**Additional Fields:**
- `top_p` (float | None): Nucleus sampling (0-1)
- `frequency_penalty` (float | None): Frequency penalty (-2.0 to 2.0)
- `presence_penalty` (float | None): Presence penalty (-2.0 to 2.0)

#### `GoogleRequest`
Google AI-specific request extending ProviderRequest.

**Additional Fields:**
- `top_k` (int | None): Top-k sampling parameter
- `top_p` (float | None): Nucleus sampling (0-1)

### 3. Response Models

#### `UsageSummary`
Token and cost usage summary.

**Fields:**
- `total_cost` (float): Total cost in USD (≥ 0)
- `total_input_tokens` (int): Input tokens used (≥ 0)
- `total_output_tokens` (int): Output tokens used (≥ 0)
- `total_tokens` (int): Total tokens (≥ 0)
- `budget_limit` (float): Budget limit in USD (≥ 0)
- `remaining_budget` (float): Remaining budget in USD (≥ 0)
- `budget_usage_percentage` (float): Percentage of budget used (0-100)

#### `IterationResult`
Result of a single optimization iteration.

**Fields:**
- `iteration` (int): Iteration number (≥ 1)
- `prompt` (str | List[Message]): Optimized prompt from this iteration
- `cost` (float): Cost of this iteration in USD (≥ 0)
- `input_tokens` (int): Input tokens used (≥ 0)
- `output_tokens` (int): Output tokens used (≥ 0)
- `feedback` (str | None): Feedback summary

#### `OptimizeResponse` ⭐ Main Response Model
Response from prompt optimization.

**Fields:**
- `optimized_prompt` (str | List[Message]): The optimized prompt
- `original_prompt` (str | List[Message]): Original prompt before optimization
- `cost` (float): Total optimization cost in USD (≥ 0)
- `iterations` (int): Number of iterations (≥ 0)
- `usage` (UsageSummary): Token and cost summary
- `iterations_details` (List[IterationResult] | None): Detailed iteration results
- `model` (str): Model used
- `provider` (str | None): Provider used
- `success` (bool): Whether optimization was successful
- `message` (str | None): Optional message or warning

#### `ErrorResponse`
Standard error response.

**Fields:**
- `error` (str): Error message
- `detail` (str | None): Detailed error information
- `status_code` (int): HTTP status code (100-599)
- `error_type` (str): Error type (validation, provider, budget, dataset, internal)
- `request_id` (str | None): Request tracking ID

#### `ValidationErrorResponse`
Extended error response with field-level validation errors.

**Additional Fields:**
- `fields` (Dict[str, str]): Field-level validation errors

#### `HealthResponse`
Health check response.

**Fields:**
- `status` (str): Health status ("healthy", "degraded", "unhealthy")
- `version` (str): API version
- `environment` (str): Environment name
- `uptime` (float): Server uptime in seconds (≥ 0)
- `services` (Dict[str, str]): Status of external services

#### `ModelInfo`
Model information response.

**Fields:**
- `name` (str): Model identifier
- `provider` (str): Provider name
- `capabilities` (ModelCapabilities): Model capabilities
- `pricing` (PricingInfo): Pricing information
- `context_size` (int): Context window size in tokens (> 0)

#### `ModelsListResponse`
Response for listing available models.

**Fields:**
- `models` (List[ModelInfo]): List of available models
- `count` (int): Number of models (≥ 0)
- `default_model` (str): Default model name

#### `ValidationResponse`
Response for input validation.

**Fields:**
- `valid` (bool): Whether inputs are valid
- `errors` (List[str]): Validation errors
- `warnings` (List[str]): Validation warnings
- `estimated_cost` (float | None): Estimated optimization cost
- `estimated_tokens` (int | None): Estimated token usage

### 4. Budget Models

#### `BudgetCheckRequest`
Request to check if operation is within budget.

**Fields:**
- `model` (str): Model to use
- `input_tokens` (int): Estimated input tokens (≥ 0)
- `output_tokens` (int): Estimated output tokens (≥ 0, default: 0)
- `budget_limit` (float | None): Budget limit to check

#### `BudgetCheckResponse`
Response for budget check.

**Fields:**
- `within_budget` (bool): Whether operation is within budget
- `estimated_cost` (float): Estimated cost in USD (≥ 0)
- `remaining_budget` (float): Remaining budget after operation (≥ 0)
- `budget_limit` (float): Current budget limit (≥ 0)
- `would_exceed` (bool): Whether operation would exceed budget

### 5. Config Models

#### `ProviderConfig`
Provider configuration.

**Fields:**
- `api_key` (str): API key for authentication
- `timeout` (int): Request timeout in seconds (1-600, default: 60)
- `max_retries` (int): Maximum retry attempts (0-10, default: 3)
- `default_model` (str | None): Default model to use

## Usage Examples

### Creating an Optimization Request

```python
from api.models import OptimizeRequest

# String prompt
request = OptimizeRequest(
    prompt="You are a helpful assistant. Answer: {question}",
    dataset=[
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "feedback": "correct"
        }
    ],
    output_column="answer",
    feedback_columns=["feedback"],
    model="gpt-4o",
    budget=5.0
)
```

### Message-Based Prompt

```python
from api.models import OptimizeRequest, Message

request = OptimizeRequest(
    prompt=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Answer: {question}")
    ],
    dataset=[...],
    output_column="answer",
    feedback_columns=["feedback"]
)
```

### Creating a Response

```python
from api.models import OptimizeResponse, UsageSummary

response = OptimizeResponse(
    optimized_prompt="You are an expert assistant...",
    original_prompt="You are a helpful assistant...",
    cost=0.0234,
    iterations=2,
    usage=UsageSummary(
        total_cost=0.0234,
        total_input_tokens=1250,
        total_output_tokens=890,
        total_tokens=2140,
        budget_limit=5.0,
        remaining_budget=4.9766,
        budget_usage_percentage=0.468
    ),
    model="gpt-4o",
    success=True
)
```

### Error Response

```python
from api.models import ErrorResponse

error = ErrorResponse(
    error="Budget exceeded",
    detail="Cost would be $10.50, exceeding budget of $5.00",
    status_code=400,
    error_type="budget"
)
```

### Validation

All models perform automatic validation:

```python
from pydantic import ValidationError

try:
    # This will fail - feedback_columns requires at least 1 item
    request = OptimizeRequest(
        prompt="Test",
        dataset=[],
        output_column="output",
        feedback_columns=[]  # Error: min_length=1
    )
except ValidationError as e:
    print(e.errors())
```

### JSON Serialization

```python
# Convert to JSON
json_str = request.model_dump_json()

# Convert to dict
request_dict = request.model_dump()

# Parse from JSON
new_request = OptimizeRequest(**request_dict)
```

## Validation Rules

1. **Numeric Constraints:**
   - Costs must be ≥ 0
   - Tokens must be ≥ 0
   - Budget must be > 0 when provided
   - Percentages: 0-100
   - Temperature: 0-2
   - HTTP codes: 100-599

2. **String Constraints:**
   - Message role must be "system", "user", or "assistant"
   - Error types: validation, provider, budget, dataset, internal
   - Health status: healthy, degraded, unhealthy

3. **Array Constraints:**
   - feedback_columns must have at least 1 item

## Best Practices

1. **Always use Field() descriptions** for clear API documentation
2. **Provide Optional defaults** where appropriate
3. **Use proper type hints** (str, not String; int, not Integer)
4. **Add validators** for complex validation logic
5. **Export all models** in `__all__` for clean imports
6. **Use model_dump()** (not dict()) for Pydantic v2
7. **Use model_dump_json()** for JSON serialization

## Integration with FastAPI

These models integrate seamlessly with FastAPI:

```python
from fastapi import FastAPI
from api.models import OptimizeRequest, OptimizeResponse

app = FastAPI()

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_prompt(request: OptimizeRequest):
    # Request is automatically validated
    result = await run_optimization(request)
    # Response is automatically serialized
    return result
```

## Future Extensions

Consider adding:
- Streaming response models for real-time updates
- Batch optimization request models
- History tracking models
- Multi-dataset optimization models
- Custom evaluation metric models
