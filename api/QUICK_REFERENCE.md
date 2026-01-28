# API Models Quick Reference

Quick lookup guide for all models in `api/models.py`.

## Import Statement

```python
from api.models import (
    # Request models
    OptimizeRequest,
    ProviderRequest,
    OpenAIRequest,
    GoogleRequest,
    
    # Response models
    OptimizeResponse,
    ErrorResponse,
    HealthResponse,
    
    # Base models
    Message,
    ModelCapabilities,
    PricingInfo,
)
```

## Model Quick Reference

### Most Commonly Used Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `OptimizeRequest` | Main optimization request | `prompt`, `dataset`, `output_column`, `feedback_columns` |
| `OptimizeResponse` | Main optimization response | `optimized_prompt`, `cost`, `iterations`, `usage` |
| `ErrorResponse` | Error response | `error`, `status_code`, `error_type` |
| `HealthResponse` | Health check | `status`, `version`, `uptime` |
| `Message` | Chat message | `role`, `content` |

### Complete Model List

#### Requests
- `OptimizeRequest` - Prompt optimization
- `ProviderRequest` - Base provider request
- `OpenAIRequest` - OpenAI-specific
- `GoogleRequest` - Google-specific
- `DatasetInput` - Dataset input wrapper
- `PromptFormat` - Prompt format wrapper
- `BudgetCheckRequest` - Budget validation

#### Responses
- `OptimizeResponse` - Optimization result
- `ErrorResponse` - Standard error
- `ValidationErrorResponse` - Validation errors
- `HealthResponse` - Health status
- `ModelInfo` - Model information
- `ModelsListResponse` - List models
- `ValidationResponse` - Validation result
- `BudgetCheckResponse` - Budget check result

#### Base/Support
- `Message` - Chat message
- `ModelCapabilities` - Model features
- `PricingInfo` - Pricing data
- `UsageSummary` - Token usage
- `IterationResult` - Iteration data
- `ProviderConfig` - Provider config

## Field Validation Rules

### Numeric Constraints
```python
budget: float         # > 0
cost: float           # >= 0
tokens: int           # >= 0
temperature: float    # 0-2
percentage: float     # 0-100
status_code: int      # 100-599
```

### String Constraints
```python
role: str             # "system" | "user" | "assistant"
error_type: str       # "validation" | "provider" | "budget" | "dataset" | "internal"
status: str           # "healthy" | "degraded" | "unhealthy"
format: str           # "json" | "list" | "columns"
```

### Array Constraints
```python
feedback_columns: List[str]  # min_length=1
```

## Common Patterns

### Pattern 1: Basic Optimization Request
```python
request = OptimizeRequest(
    prompt="You are a helpful assistant. {input}",
    dataset=[{"input": "...", "output": "...", "feedback": "..."}],
    output_column="output",
    feedback_columns=["feedback"],
)
```

### Pattern 2: With Budget Control
```python
request = OptimizeRequest(
    prompt=...,
    dataset=...,
    output_column=...,
    feedback_columns=[...],
    model="gpt-4o",
    budget=5.0,
)
```

### Pattern 3: Message-Based Prompt
```python
request = OptimizeRequest(
    prompt=[
        Message(role="system", content="..."),
        Message(role="user", content="..."),
    ],
    ...,
)
```

### Pattern 4: Error Response
```python
error = ErrorResponse(
    error="Budget exceeded",
    detail="Cost $10.50 > budget $5.00",
    status_code=400,
    error_type="budget",
)
```

### Pattern 5: Optimization Response
```python
response = OptimizeResponse(
    optimized_prompt="...",
    original_prompt="...",
    cost=0.0234,
    iterations=2,
    usage=UsageSummary(...),
    model="gpt-4o",
    success=True,
)
```

## Validation Examples

### Valid Request
```python
✓ OptimizeRequest(
    prompt="Test {var}",
    dataset=[{"var": "x", "out": "y", "fb": "good"}],
    output_column="out",
    feedback_columns=["fb"],
  )
```

### Invalid Requests
```python
✗ feedback_columns=[]  # min_length=1
✗ budget=-5.0          # must be > 0
✗ temperature=3.0      # must be 0-2
✗ role="bot"           # must be system/user/assistant
```

## Pydantic Methods

### Serialization
```python
# To dict
data = request.model_dump()

# To JSON
json_str = request.model_dump_json()

# To dict with aliases
data = request.model_dump(by_alias=True)
```

### Validation
```python
from pydantic import ValidationError

try:
    request = OptimizeRequest(**data)
except ValidationError as e:
    # e.errors() -> list of error dicts
    # e.error_count() -> number of errors
```

### Schema
```python
# JSON schema
schema = OptimizeRequest.model_json_schema()

# Model fields
fields = OptimizeRequest.model_fields
```

## FastAPI Integration

### Route Definition
```python
@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    # Request is auto-validated
    result = await optimize_prompt(request)
    return result  # Auto-serialized to response_model
```

### Error Handling
```python
from fastapi import HTTPException

if exceeds_budget:
    raise HTTPException(
        status_code=400,
        detail=ErrorResponse(
            error="Budget exceeded",
            status_code=400,
            error_type="budget"
        ).model_dump()
    )
```

### Validation Endpoint
```python
@app.post("/api/validate")
async def validate(request: OptimizeRequest):
    # Just validate, don't process
    return {"valid": True}
```

## Type Hints Quick Reference

```python
# Basic types
str, int, float, bool

# Optional
Optional[str]  # str | None

# Union
Union[str, List[Message]]  # str | List[Message]

# Collections
List[str]
Dict[str, str]
List[Dict[str, Any]]

# Model references
Optional[UsageSummary]
List[IterationResult]
```

## Common Workflows

### Workflow 1: Validate → Optimize → Return
```python
@app.post("/optimize")
async def optimize(request: OptimizeRequest):
    # 1. Validate (automatic)
    # 2. Optimize
    result = await optimizer.optimize(...)
    # 3. Return response (automatic serialization)
    return OptimizeResponse(...)
```

### Workflow 2: Check Budget First
```python
@app.post("/optimize")
async def optimize(request: OptimizeRequest):
    # Check budget
    budget_req = BudgetCheckRequest(...)
    budget_resp = check_budget(budget_req)
    
    if not budget_resp.within_budget:
        raise HTTPException(...)
    
    # Proceed with optimization
    return await optimize_optimize(...)
```

### Workflow 3: Validation Only
```python
@app.post("/validate")
async def validate(request: OptimizeRequest):
    # Just validate inputs
    return ValidationResponse(
        valid=True,
        estimated_cost=...,
    )
```

## Tips

1. **Always use Field() for descriptions** - helps with auto-documentation
2. **Export in __all__** - clean imports
3. **Use model_dump() not dict()** - Pydantic v2
4. **Leverage Union types** - flexible input formats
5. **Add validators sparingly** - Pydantic does most validation
6. **Use Optional with defaults** - better UX
7. **Document complex fields** - help users understand

## Debugging

### Check Validation Errors
```python
try:
    request = OptimizeRequest(**data)
except ValidationError as e:
    print(e.errors())
    # [
    #   {
    #     'loc': ('feedback_columns',),
    #     'type': 'value_error.list.min_items',
    #     'msg': 'List should have at least 1 item after validation, not 0'
    #   },
    # ]
```

### Inspect Model Fields
```python
for name, field in OptimizeRequest.model_fields.items():
    print(f"{name}: {field.annotation}")
    print(f"  required: {field.is_required()}")
    print(f"  default: {field.default}")
```

---

**For detailed documentation, see `MODELS.md`**  
**For implementation details, see `IMPLEMENTATION_SUMMARY.md`**
