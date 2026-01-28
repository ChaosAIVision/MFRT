# BEAD Plan: 2-Stage Evaluator (Construction + Reasoning)

## Component Purpose

### What It Does
Tạo evaluator mới đánh giá predictions theo 2 giai đoạn riêng biệt:
- **Stage 1**: Construction quality (entities, state variables, actions, constraints có đầy đủ không?)
- **Stage 2**: Reasoning correctness (reasoning path có dẫn đến kết luận đúng không?)

### Why It's Needed
Hiện tại chỉ có overall correctness score. Cần separate scores để:
- Biết prompt optimization cải thiện phần nào (construction vs reasoning)
- Debug tại sao accuracy giảm/tăng
- Provide clearer feedback cho meta-prompt

### Where It Fits
Core evaluation component - sẽ thay thế simple correctness check trong test.

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| input_text | str | Yes | Article/problem text | "Crosby scores..." | Non-empty |
| prediction | str | Yes | Model output | "True" | Non-empty |
| groundtruth | str | Yes | Expected output | "True" | Non-empty |
| construction | str | No | Extracted construction | XML from Phase 1 | Valid XML |
| reasoning | str | No | Extracted reasoning | XML from Phase 2 | Valid XML |

### Input Source
- Test dataset predictions
- Groundtruth labels from dataset
- (Optional) Construction/reasoning extractions

### Input Validation Rules
1. All text fields must be non-empty strings
2. Groundtruth must match expected format

## Output Specification

### Output Structure
```python
{
    "construction_score": 0.8,  # 0-1 score
    "reasoning_score": 0.9,     # 0-1 score
    "overall_correct": True,     # Boolean
    "construction_feedback": "Missing state variable X",
    "reasoning_feedback": "Correct logical flow",
    "metadata": {
        "entities_found": 4,
        "constraints_found": 3
    }
}
```

### Success Response
- construction_score: Quality of problem decomposition (0-1)
- reasoning_score: Correctness of reasoning path (0-1)
- Detailed feedback for each stage

### Error Response
- ValueError if inputs invalid
- Returns default scores (0.0) if evaluation fails

### Output Consumers
- test_optimization.py - reports separate metrics
- PromptLearningOptimizer - uses feedback for optimization

## Dependencies & Build Order

### Depends On (Must Build First)
1. **Component 1** - Analyze Flow
   - Status: ⏳ Not Started
   - Uses: Understanding of evaluation mechanism

### Depended On By (Build These After)
1. **Component 5** - Update Test
   - Will use this evaluator instead of simple comparison

### Build Priority
- Priority: P1 (High)
- Suggested build order: #2 out of 5
- Blocking: Component 5

## Component Relationships

### Data Flow
```
[Prediction + Groundtruth]
     ↓ (evaluate)
[2-Stage Evaluator]
     ↓ (produces)
[Construction Score + Reasoning Score + Feedback]
     ↓ (used by)
[Test Reporting & Meta-Prompt Optimization]
```

### Interaction Pattern
- Synchronous evaluation
- Batch processing support
- Returns structured scores

## Implementation Plan

### Complexity Assessment
- Complexity Level: Medium
- Estimated Effort: 3-4 hours
- Risk Level: Medium

### Technical Approach
1. Create new class `TwoStageEvaluator` inheriting from `BaseEvaluator`
2. Implement `evaluate_construction()` method:
   - Uses ConstructionExtractor if construction XML provided
   - Scores based on completeness (entities, state, actions, constraints)
   - Returns 0-1 score + feedback
3. Implement `evaluate_reasoning()` method:
   - Checks if reasoning leads to correct conclusion
   - Validates logical flow
   - Returns 0-1 score + feedback
4. Implement `evaluate()` method combining both stages
5. Add to `src/chaos_auto_prompt/evaluators/two_stage.py`

### Key Algorithms/Patterns
- **Construction scoring**: Count entities/state/actions/constraints → completeness ratio
- **Reasoning scoring**: Extract final prediction from reasoning → compare with groundtruth
- **Feedback generation**: Identify missing elements, logical gaps

### Technology Stack
- Language: Python 3.11
- Libraries: pandas, re (for pattern matching)
- Base class: BaseEvaluator
- Uses: ConstructionExtractor from utils

## Edge Cases & Risks

### Edge Cases to Handle
1. Missing construction/reasoning (fallback to simple comparison)
2. Malformed XML extraction
3. Ambiguous reasoning paths
4. Empty predictions

### Potential Risks
1. **Risk**: Construction scoring may be too strict
   - **Impact**: Medium
   - **Mitigation**: Use threshold-based scoring, not binary

2. **Risk**: Reasoning evaluation may miss subtle errors
   - **Impact**: Medium
   - **Mitigation**: Extract final conclusion explicitly, compare strictly

## Testing Strategy

### Test Cases Planned
1. **Normal Case**: Complete construction + correct reasoning
   - Input: Valid construction XML, correct reasoning
   - Expected: construction_score=1.0, reasoning_score=1.0

2. **Partial Construction**: Missing some elements
   - Input: Construction with 3/4 sections
   - Expected: construction_score=0.75

3. **Wrong Reasoning**: Incorrect conclusion
   - Input: Valid construction, wrong final answer
   - Expected: construction_score=1.0, reasoning_score=0.0

4. **Fallback Case**: No construction/reasoning XML
   - Input: Just prediction vs groundtruth
   - Expected: Simple comparison, both scores based on correctness

### Integration Tests
- Test with actual milestone-classification dataset
- Verify scores correlate with actual quality

### Performance Tests
- Should process 100 samples in < 5 seconds

## Missing Information

### Questions to Answer Before Implementation
1. How to weight construction vs reasoning in overall score?
   - Answer: Average them (50/50) or make configurable
2. What threshold for "good enough" construction?
   - Answer: >= 0.75 (3/4 sections present)

## Notes
- May need LLM assistance for complex reasoning evaluation
- Can start simple (rule-based) then enhance with LLM if needed
- Consider adding confidence scores

---

## Implementation Report

### Status: ✅ COMPLETED

### What Was Implemented

1. **TwoStageEvaluator class** (`evaluators/two_stage.py`):
   - Inherits from BaseEvaluator
   - Implements separate construction and reasoning evaluation
   - Configurable weights for each stage (default 50/50)
   - Returns granular scores and feedback

2. **Construction Evaluation** (`evaluate_construction()` method):
   - Uses ConstructionExtractor to parse construction XML
   - Counts 4 sections: entities, state variables, actions, constraints
   - Score = (sections_present / 4.0)
   - Generates feedback listing missing sections

3. **Reasoning Evaluation** (`evaluate_reasoning()` method):
   - Compares prediction with groundtruth
   - Normalizes values (handles True/False, yes/no, 1/0 variations)
   - Score = 1.0 if correct, 0.0 if incorrect
   - Detects structured reasoning flow (Step 1, Step 2, Final)

4. **Overall Scoring**:
   - Weighted average: `overall = construction_weight * C + reasoning_weight * R`
   - Default weights: 50% construction, 50% reasoning
   - Auto-normalizes if weights don't sum to 1.0

### Code Location

**File**: `src/chaos_auto_prompt/evaluators/two_stage.py` (370 lines)

**Key Methods**:
- `evaluate()`: Main async evaluation method (returns DataFrame + feedback columns)
- `evaluate_construction()`: Construction quality scoring
- `evaluate_reasoning()`: Reasoning correctness scoring
- `_normalize_value()`: Value normalization for comparison
- `_format_combined_feedback()`: Combines both stages into readable feedback

### Output Schema

```python
{
    "construction_score": 0.75,  # 0-1 (3/4 sections present)
    "construction_feedback": "Construction incomplete. Missing: constraints",
    "reasoning_score": 1.0,  # 0-1 (correct prediction)
    "reasoning_feedback": "Reasoning correct. Predicted: True, Expected: True",
    "overall_score": 0.875,  # Weighted average
    "overall_correct": True,  # Boolean match
    "two_stage_feedback": "Overall Score: 0.88 | Construction: 0.75 - ... | Reasoning: 1.00 - ..."
}
```

### Testing

Created `tests/unit/test_two_stage_evaluator.py` with **11 test cases**:

✅ All tests passed (11/11):
1. `test_evaluate_correct_prediction` - Correct prediction scoring
2. `test_evaluate_incorrect_prediction` - Incorrect prediction scoring
3. `test_evaluate_with_construction` - Full construction evaluation
4. `test_evaluate_partial_construction` - Incomplete construction (0.5 score)
5. `test_evaluate_no_construction` - Missing construction (0.0 score)
6. `test_evaluate_with_reasoning` - Reasoning text analysis
7. `test_normalize_value` - Value normalization (True/true/yes/1 → "true")
8. `test_overall_score_calculation` - Weighted average calculation
9. `test_weight_normalization` - Auto-normalize weights to sum 1.0
10. `test_feedback_columns_list` - Returns all 7 feedback columns
11. `test_batch_evaluation` - Multi-row DataFrame processing

### Design Decisions

1. **Rule-based evaluation**: Started simple without LLM (fast, deterministic)
2. **Configurable weights**: Allow different importance for construction vs reasoning
3. **Graceful degradation**: Works with or without construction/reasoning columns
4. **Comprehensive feedback**: 7 feedback columns for detailed analysis
5. **Value normalization**: Handles common boolean variations automatically

### Integration

Added to `evaluators/__init__.py`:
```python
from chaos_auto_prompt.evaluators.two_stage import TwoStageEvaluator
__all__ = ["ClassificationEvaluator", "TwoStageEvaluator"]
```

### Next Steps

Component 2 is complete. Ready to use in:
- **Component 5**: Update test_optimization.py with 2-stage evaluation
- **Prompt Optimization**: Use construction/reasoning feedback for better meta-prompting
