# All Optimization APIs - Before/After Metrics

## ✅ Updated APIs

Tất cả các API optimize đã được cập nhật với before/after metrics:

1. **`POST /api/v1/optimize`** - Standard optimization
2. **`POST /api/optimize/huggingface`** - HuggingFace dataset optimization

---

## 📊 Unified Output Structure

Tất cả APIs đều return cấu trúc giống nhau:

```json
{
  "optimized_prompt": "...",
  "original_prompt": "...",

  "metrics_before": {
    "quality": {
      "distribution": {"poor": 5, "good": 5}
    }
  },

  "metrics_after": {
    "quality": {
      "distribution": {"poor": 1, "good": 9}
    }
  },

  "improvement": {
    "quality": {
      "positive_count_before": 5,
      "positive_count_after": 9,
      "improvement": 4,
      "before_positive_pct": 50.0,
      "after_positive_pct": 90.0
    }
  },

  "usage": {
    "total_cost": 0.0012,
    "total_tokens": 5200
  }
}
```

---

## API 1: Standard Optimize

### Endpoint
```
POST /api/v1/optimize
```

### Request Example
```json
{
  "prompt": "You are a helpful assistant. Answer: {question}",
  "dataset": [
    {
      "question": "What is 2+2?",
      "answer": "Four",
      "quality": "poor"
    },
    {
      "question": "Capital of France?",
      "answer": "Paris",
      "quality": "good"
    }
  ],
  "output_column": "answer",
  "feedback_columns": ["quality"],
  "evaluators": [
    {
      "type": "classification",
      "feedback_column": "quality",
      "model": "gpt-4o-mini",
      "prompt_template": "Evaluate: {question} -> {answer}",
      "choices": {"excellent": 2, "good": 1, "poor": 0}
    }
  ],
  "model": "gpt-4o-mini",
  "budget": 1.0
}
```

### Response
```json
{
  "success": true,
  "optimized_prompt": "You are a highly accurate assistant...",
  "original_prompt": "You are a helpful assistant. Answer: {question}",

  "metrics_before": {
    "quality": {
      "distribution": {"poor": 5, "good": 5}
    }
  },

  "metrics_after": {
    "quality": {
      "distribution": {"poor": 1, "good": 9}
    }
  },

  "improvement": {
    "quality": {
      "improvement": 4,
      "before_positive_pct": 50.0,
      "after_positive_pct": 90.0
    }
  },

  "cost": 0.0008,
  "usage": {
    "total_cost": 0.0008,
    "total_tokens": 3500
  }
}
```

---

## API 2: HuggingFace Optimize

### Endpoint
```
POST /api/optimize/huggingface
```

### Request Example
```json
{
  "dataset_name": "user/my-dataset",
  "system_prompt_column": "system_prompt",
  "input_column": "input",
  "output_column": "output",
  "evaluators": [
    {
      "type": "classification",
      "feedback_column": "quality",
      "model": "gpt-4o-mini",
      "prompt_template": "Evaluate: {input} -> {output}",
      "choices": {"good": 1, "poor": 0}
    }
  ],
  "max_samples": 10,
  "budget": 1.0
}
```

### Response
```json
{
  "success": true,
  "initial_prompt": "Answer the question: {input}",
  "optimized_prompt": "You are a professional assistant...",

  "dataset_info": {
    "name": "user/my-dataset",
    "num_samples": 10
  },

  "metrics_before": {
    "quality": {
      "distribution": {"poor": 6, "good": 4}
    }
  },

  "metrics_after": {
    "quality": {
      "distribution": {"poor": 2, "good": 8}
    }
  },

  "improvement": {
    "quality": {
      "improvement": 4,
      "before_positive_pct": 40.0,
      "after_positive_pct": 80.0
    }
  },

  "usage_summary": {
    "total_cost": 0.0012,
    "total_tokens": 5200
  }
}
```

---

## Workflow Comparison

### Standard API (`/api/v1/optimize`)
```
1. Receive dataset with existing outputs
2. Run evaluators → metrics_before
3. Optimize prompt
4. Re-generate outputs with optimized prompt
5. Re-evaluate → metrics_after
6. Calculate improvement
7. Return results
```

### HuggingFace API (`/api/optimize/huggingface`)
```
1. Load dataset from HuggingFace Hub
2. Extract system_prompt from dataset
3. Run evaluators on existing outputs → metrics_before
4. Optimize prompt
5. Re-generate outputs with optimized prompt
6. Re-evaluate → metrics_after
7. Calculate improvement
8. Return results
```

**Key Difference:** HuggingFace API loads dataset from Hub, Standard API uses provided dataset.

---

## Metrics Interpretation

### Categorical Metrics (quality, correctness, etc.)

```json
{
  "quality": {
    "before_distribution": {"poor": 5, "good": 5},
    "after_distribution": {"poor": 1, "good": 9},
    "positive_count_before": 5,    // Số lượng "good" trước
    "positive_count_after": 9,     // Số lượng "good" sau
    "improvement": 4,              // Tăng 4 samples
    "before_positive_pct": 50.0,   // 50% tốt trước
    "after_positive_pct": 90.0     // 90% tốt sau
  }
}
```

**Positive Labels Auto-Detected:**
- `good`, `excellent`, `correct`, `yes`, `true`, `positive`

### Numeric Metrics (scores, ratings, etc.)

```json
{
  "accuracy_score": {
    "before_mean": 0.65,
    "after_mean": 0.88,
    "absolute_change": 0.23,      // Tăng 0.23
    "percent_change": 35.38       // Tăng 35.38%
  }
}
```

---

## When Metrics are Calculated

### metrics_before (BASELINE)
- Calculated from **existing outputs** in dataset
- Outputs were generated with **original/initial prompt**
- Represents **current quality** before optimization

### metrics_after (RESULTS)
- Calculated from **new outputs**
- Outputs are **re-generated** using **optimized prompt**
- Represents **improved quality** after optimization

### improvement
- Direct comparison of before vs after
- Shows absolute and percentage improvements
- Helps determine if optimization was successful

---

## Cost Tracking

Both APIs track full costs:

```json
"usage": {
  "total_cost": 0.0012,          // Total USD spent
  "total_tokens": 5200,          // All tokens used
  "total_input_tokens": 3500,
  "total_output_tokens": 1700,
  "budget_limit": 1.0,
  "remaining_budget": 0.9988,
  "budget_usage_percentage": 0.12  // 0.12% of budget
}
```

**Costs include:**
1. Initial evaluation (metrics_before)
2. Prompt optimization
3. Re-generation of outputs
4. Re-evaluation (metrics_after)

---

## Example Use Cases

### 1. Chatbot Quality Improvement
```python
# Before: 50% good responses
# After: 90% good responses
# Improvement: +40% (+4 samples)
```

### 2. Translation Accuracy
```python
# Before: mean BLEU score = 0.65
# After: mean BLEU score = 0.88
# Improvement: +35.38%
```

### 3. Classification Correctness
```python
# Before: 70% correct classifications
# After: 95% correct classifications
# Improvement: +25% (+5 samples)
```

---

## API Differences

| Feature | Standard API | HuggingFace API |
|---------|-------------|-----------------|
| **Dataset Source** | Provided in request | HuggingFace Hub |
| **Columns** | Any column names | Fixed: system_prompt, input, output |
| **Response Field** | `original_prompt` | `initial_prompt` |
| **Usage Field** | `usage` | `usage_summary` |
| **Dataset Info** | Not included | Included with metadata |

---

## Testing Both APIs

### Test Standard API
```bash
curl -X POST http://localhost:8435/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d @standard_request.json
```

### Test HuggingFace API
```bash
curl -X POST http://localhost:8435/api/optimize/huggingface \
  -H "Content-Type: application/json" \
  -d @huggingface_request.json
```

---

## Quick Decision Guide

**Use Standard API when:**
- You have custom dataset format
- Dataset is already prepared locally
- Need full control over column names

**Use HuggingFace API when:**
- Dataset is on HuggingFace Hub
- Want to use public/shared datasets
- Following standard format (system_prompt, input, output)

---

## Common Patterns

### Pattern 1: Iterate until satisfied
```python
result = optimize_prompt(data)
while result["improvement"]["after_positive_pct"] < 80:
    # Adjust evaluator or add more examples
    result = optimize_prompt(updated_data)
```

### Pattern 2: A/B Testing
```python
result_v1 = optimize_prompt(data, prompt_v1)
result_v2 = optimize_prompt(data, prompt_v2)

if result_v2["improvement"]["after_positive_pct"] > result_v1["improvement"]["after_positive_pct"]:
    use_prompt = result_v2["optimized_prompt"]
```

### Pattern 3: Cost-Quality Tradeoff
```python
result = optimize_prompt(data)

improvement_pct = result["improvement"]["after_positive_pct"] - \
                  result["improvement"]["before_positive_pct"]
cost = result["usage"]["total_cost"]

if improvement_pct > 20 and cost < 0.01:
    print("✅ Good value!")
```

---

## Related Documentation

- `BEFORE_AFTER_METRICS.md` - Detailed metrics explanation
- `HUGGINGFACE_API_OUTPUT.md` - HuggingFace API specifics
- `EVALUATOR_PATTERN.md` - Evaluator usage guide
