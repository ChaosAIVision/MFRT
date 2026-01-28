# Evaluator Pattern Guide

## Overview

The evaluator pattern allows automatic generation of feedback columns using LLM-as-a-Judge approach. Instead of providing pre-computed feedback, evaluators analyze your model outputs and generate structured feedback automatically.

## Two Approaches

### 1. Direct Feedback (Original)
Dataset already has feedback:
```python
{
    "question": "What is 2+2?",
    "answer": "4",
    "feedback": "Perfect answer"  # Pre-computed
}
```

### 2. Evaluator-Based (New)
Feedback generated automatically:
```python
{
    "question": "What is 2+2?",
    "answer": "4"
    # No feedback - evaluator will generate it
}
```

## Usage

### Python SDK

```python
from chaos_auto_prompt.optimizers import PromptLearningOptimizer
from chaos_auto_prompt.evaluators import ClassificationEvaluator

# Create evaluator
evaluator = ClassificationEvaluator(
    feedback_column="correctness",
    model="gpt-4o",
    prompt_template="""
        Evaluate: {output}
        Return JSON: {"correctness": "correct" or "incorrect", "explanation": "..."}
    """,
    choices={"correct": 1, "incorrect": 0}
)

# Create optimizer
optimizer = PromptLearningOptimizer(
    prompt="Your prompt: {input}",
    model_choice="gpt-4o"
)

# Generate feedback
dataset_with_feedback, feedback_cols = await optimizer.run_evaluators(
    dataset=dataset,
    evaluators=[evaluator]
)

# Optimize
optimized = await optimizer.optimize(
    dataset=dataset_with_feedback,
    output_column="output",
    feedback_columns=feedback_cols
)
```

### REST API

```bash
curl -X POST http://localhost:8000/api/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Answer: {question}",
    "dataset": [
      {"question": "What is 2+2?", "answer": "4"}
    ],
    "output_column": "answer",
    "evaluators": [{
      "type": "classification",
      "feedback_column": "correctness",
      "model": "gpt-4o",
      "prompt_template": "Evaluate: {answer}...",
      "choices": {"correct": 1, "incorrect": 0}
    }],
    "model": "gpt-4o",
    "budget": 1.0
  }'
```

## Evaluator Types

### ClassificationEvaluator

Classifies outputs into categories (correct/incorrect, good/bad, etc.)

**Parameters:**
- `feedback_column`: Primary feedback column name (e.g., "correctness")
- `model`: LLM model for evaluation (e.g., "gpt-4o")
- `prompt_template`: Evaluation prompt with {placeholders}
- `choices`: Label-to-score mapping (e.g., {"correct": 1, "incorrect": 0})
- `include_explanation`: Generate explanation column (default: True)

**Generated columns:**
- Primary feedback column (e.g., "correctness")
- "explanation" (if `include_explanation=True`)

## Examples

See `examples/` directory:
- `evaluator_example.py` - Python SDK usage
- `api_evaluator_example.py` - REST API usage

## Comparison with Old Repo

**Old repo (`/prompt-learning`):**
```python
# Uses phoenix.evals evaluators
from phoenix.evals import ClassificationEvaluator, LLM

evaluator = ClassificationEvaluator(
    "correctness",
    LLM(provider="openai", model="gpt-4o"),
    prompt_template,
    choices={"correct": 1, "incorrect": 0}
)

dataset, _ = optimizer.run_evaluators(
    dataset,
    evaluators=[evaluator],
    feedback_columns=["correctness", "explanation"]
)
```

**New repo (this one):**
```python
# Uses built-in evaluators (no phoenix dependency)
from chaos_auto_prompt.evaluators import ClassificationEvaluator

evaluator = ClassificationEvaluator(
    feedback_column="correctness",
    model="gpt-4o",
    prompt_template=prompt_template,
    choices={"correct": 1, "incorrect": 0}
)

dataset, _ = await optimizer.run_evaluators(
    dataset,
    evaluators=[evaluator],
    feedback_columns=["correctness", "explanation"]
)
```

Key differences:
1. No external dependencies (phoenix.evals)
2. Async/await support
3. Direct provider integration
4. REST API support
5. Pydantic validation

## When to Use Each Approach

**Direct Feedback:**
- You have human feedback
- Feedback is domain-specific
- Small datasets
- Maximum control

**Evaluator-Based:**
- Large datasets (automation)
- Standard evaluation criteria
- No human feedback available
- Reproducible evaluations
