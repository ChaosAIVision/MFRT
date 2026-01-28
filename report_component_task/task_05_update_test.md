# BEAD Plan: Update Test with 2-Stage Evaluation

## Component Purpose

### What It Does
Modify `test_optimization.py` để:
- Use TwoStageEvaluator instead of simple comparison
- Report construction score vs reasoning score separately
- Show clear before/after metrics for both stages

### Why It's Needed
Clear visibility into what optimization improves: construction quality or reasoning correctness.

## Input Specification

### Input Parameters
- Same dataset (milestone-classification)
- Predictions from initial vs optimized prompt

## Output Specification

### Output Structure
```
📊 INITIAL PROMPT PERFORMANCE:
   Construction Score: 0.75 (avg)
   Reasoning Score: 0.82 (avg)
   Overall Accuracy: 72.0%

📊 OPTIMIZED PROMPT PERFORMANCE:
   Construction Score: 0.88 (avg)  ← +0.13 improvement!
   Reasoning Score: 0.85 (avg)     ← +0.03 improvement!
   Overall Accuracy: 85.0%          ← +13% improvement!
```

## Dependencies

### Depends On
1. Component 2 - TwoStageEvaluator
2. Component 4 - Auto-Fill (to ensure complete prompts)

## Implementation Plan

### Technical Approach
1. Import TwoStageEvaluator
2. Replace direct comparison with evaluator call
3. Track construction_scores and reasoning_scores lists
4. Compute averages for reporting
5. Show improvement breakdown

### Key Changes
```python
# Before
correct = prediction == groundtruth

# After
result = evaluator.evaluate(input_text, prediction, groundtruth)
construction_scores.append(result['construction_score'])
reasoning_scores.append(result['reasoning_score'])
```

## Testing Strategy

1. Run updated test with shuffled dataset
2. Verify separate scores reported
3. Confirm total accuracy matches overall correctness
