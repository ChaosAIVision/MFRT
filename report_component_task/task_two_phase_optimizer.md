# BEAD Plan: TwoPhaseOptimizer

## Component Purpose

### What It Does
Orchestrates the complete 2-phase training pipeline:
- **Phase 1 Training**: Optimize construction prompt until 90% similarity achieved
- **Phase 2 Training**: Optimize reasoning prompt based on answer correctness and path quality

Coordinates all components, manages iteration flow, and implements the training logic specified in AGENTS.md.

### Why It's Needed
This is the **main entry point** for the 2-phase optimization system. It implements the complex training flow:
1. Phase 1: Tune construction prompt → Evaluate with ConstructionSimilarityEvaluator → Repeat until 90%+
2. Phase 2: Use fixed construction → Tune reasoning prompt → Evaluate with ReasoningPathEvaluator → Update path databases → Repeat

Without this orchestrator, the individual components can't work together as a cohesive training system.

### Where It Fits
This is the **top-level component** that uses all other components. It's the API/interface that external code calls to run 2-phase optimization.

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| dataset | List[Dict] | Yes | Training dataset | [{"problem": ..., "answer": ..., "construction": ...}] | Non-empty list |
| model_name | str | Yes | LLM model to use | "gemini-2.5-flash" | Valid model |
| max_iterations_phase1 | int | No | Max Phase 1 iterations | 10 | Default: 10 |
| max_iterations_phase2 | int | No | Max Phase 2 iterations | 20 | Default: 20 |
| phase1_threshold | float | No | Construction similarity threshold | 0.90 | Default: 0.90 |
| save_results_path | str | No | Where to save outputs | "results.json" | Valid path |

### Input Source
- **dataset**: From HuggingFace or local JSON file
- **model_name**: Configuration or user input
- **Other params**: Configuration with sensible defaults

### Input Validation Rules
1. Dataset must have required fields: `problem`, `groundtruth_answer`, `groundtruth_construction`
2. Dataset must have at least 10 examples (train/test split)
3. model_name must be supported by existing providers
4. Thresholds must be in range (0.0, 1.0]

## Output Specification

### Output Structure
```python
{
    "phase1_results": {
        "final_construction_prompt": "Analyze the following problem...",
        "iterations": 8,
        "final_similarity": 0.92,
        "threshold_achieved": True,
        "iteration_history": [
            {
                "iteration": 1,
                "avg_similarity": 0.65,
                "passed": False
            },
            ...
        ]
    },
    "phase2_results": {
        "final_reasoning_prompt": "Using only the model defined above...",
        "iterations": 15,
        "final_accuracy": 0.87,
        "good_paths_count": 12,
        "bad_paths_count": 8,
        "iteration_history": [
            {
                "iteration": 1,
                "train_accuracy": 0.72,
                "test_accuracy": 0.68
            },
            ...
        ]
    },
    "outputs": [
        {
            "problem": "...",
            "construction": {...},
            "reasoning": "...",
            "predicted_answer": "42",
            "groundtruth_answer": "42",
            "correct": True
        }
    ],
    "metadata": {
        "total_iterations": 23,
        "total_time_seconds": 450,
        "total_cost_usd": 0.45,
        "model_used": "gemini-2.5-flash"
    }
}
```

### Success Response
- `phase1_results`: Complete Phase 1 training results
- `phase2_results`: Complete Phase 2 training results
- `outputs`: All LLM outputs with evaluations
- `metadata`: Run statistics

### Error Response
```python
{
    "error": "Phase 1 failed to reach 90% threshold after 10 iterations",
    "phase1_results": {...},  # Partial results
    "phase2_results": None,
    "metadata": {...}
}
```

### Output Consumers
- **API Endpoints** - Return to API callers
- **Logging/Monitoring** - Track optimization metrics
- **File Storage** - Save results to JSON
- **Downstream Analysis** - Compare optimization runs

## Dependencies & Build Order

### Depends On (Must Build First)
1. **XMLOutputParser** - Parse LLM outputs
   - Status: ⏳ Not Started
2. **ConstructionExtractor** - Extract construction elements
   - Status: ⏳ Not Started
3. **ConstructionSimilarityEvaluator** - Evaluate Phase 1
   - Status: ⏳ Not Started
4. **ReasoningPathEvaluator** - Evaluate Phase 2
   - Status: ⏳ Not Started
5. **TwoPhaseMetaPrompt** - Generate meta-prompts
   - Status: ⏳ Not Started

### Depended On By (Build These After)
None - This is the top-level component

### Build Priority
- Priority: **Normal (P2)**
- Suggested build order: **#6 out of 6** (last)
- Blocking: Nothing (leaf component)

## Component Relationships

### Data Flow
```
[TwoPhaseOptimizer] (orchestrator)
     ↓
[TwoPhaseMetaPrompt] → [LLM Provider] → [XMLOutputParser]
     ↓                                          ↓
[ConstructionExtractor]                  [Raw Output]
     ↓
[ConstructionSimilarityEvaluator] → Feedback → [TwoPhaseMetaPrompt]
     ↓
[Phase 1 Complete] → [Phase 2 Start]
     ↓
[ReasoningPathEvaluator] → Update Databases → [TwoPhaseMetaPrompt]
     ↓
[Final Results]
```

### Interaction Pattern
- **Stateful** - Maintains training state across iterations
- **Iterative** - Loops until thresholds met or max iterations
- **Sequential Phases** - Phase 2 only starts after Phase 1 success

## Implementation Plan

### Complexity Assessment
- Complexity Level: **Complex**
- Estimated Effort: 12-16 hours
- Risk Level: **High** (integrates all components, complex control flow)

### Technical Approach

**Phase 1: Construction Optimization**
1. Initialize with base construction prompt
2. For each iteration (max 10):
   a. Generate meta-prompt with current examples
   b. Run LLM on all dataset samples
   c. Parse outputs with XMLOutputParser
   d. Extract constructions with ConstructionExtractor
   e. Evaluate with ConstructionSimilarityEvaluator
   f. Calculate average similarity across dataset
   g. If avg_similarity >= 0.90: exit Phase 1
   h. Else: Update meta-prompt with feedback, add good examples
3. Save best construction prompt

**Phase 2: Reasoning Optimization**
1. Initialize good_paths=[], bad_paths=[]
2. For each iteration (max 20):
   a. Generate Phase 2 meta-prompt with current paths
   b. Run LLM with fixed construction from Phase 1
   c. Parse reasoning with XMLOutputParser
   d. Evaluate with ReasoningPathEvaluator
   e. Update good_paths/bad_paths based on recommendations
   f. Calculate train/test accuracy
   g. Log iteration results
3. Save final reasoning prompt and path databases

**Error Handling**
- Catch provider errors, retry with exponential backoff
- Save partial results if interrupted
- Validate outputs at each stage

### Key Algorithms/Patterns
- **Orchestrator Pattern**: Coordinates multiple components
  - Why: Clean separation of concerns
- **Iterative Refinement**: Gradient-free optimization loop
  - Why: Meta-prompt optimization is non-differentiable
- **Early Stopping**: Exit when thresholds met
  - Why: Save computation costs
- **State Persistence**: Save state after each iteration
  - Why: Resume if crashed, track progress
- **Train/Test Split**: 70/30 split for evaluation
  - Why: Prevent overfitting, measure generalization

### Technology Stack
- Language/Framework: Python 3.11+
- Libraries needed:
  - All component dependencies (sentence-transformers, etc.)
  - `json` - State persistence
  - `logging` - Detailed logging
  - `time` - Track duration
- External services: LLM providers (via existing provider classes)

## Edge Cases & Risks

### Edge Cases to Handle
1. **Phase 1 never reaches 90%** - Max iterations exceeded
2. **All examples fail in first iteration** - No good examples to learn from
3. **LLM refuses to follow format** - Doesn't use XML tags
4. **Dataset too small** - Not enough examples for train/test split
5. **Construction varies each run** - Non-deterministic LLM
6. **Cost exceeds budget** - Too many iterations

### Potential Risks
1. **Risk**: Phase 1 stuck at 85% similarity, can't reach 90%
   - **Impact**: Critical - Phase 2 can't start
   - **Mitigation**: Adaptive threshold (lower after N iterations), manual review at 85%+

2. **Risk**: Good paths database polluted with false positives
   - **Impact**: High - Phase 2 learns wrong patterns
   - **Mitigation**: Require multiple confirmations before adding to good paths

3. **Risk**: Expensive to run (many LLM calls)
   - **Impact**: Medium - cost concerns
   - **Mitigation**: Use cheaper model for development, budget tracking, caching

4. **Risk**: Long runtime (hours for large datasets)
   - **Impact**: Medium - slow iteration cycles
   - **Mitigation**: Batch processing, parallel LLM calls, smaller dev dataset

### Error Scenarios
1. **Scenario**: LLM API down in middle of Phase 1
   - Expected behavior: Save state, retry with backoff
   - Fallback: Allow resume from last completed iteration

2. **Scenario**: Output parsing fails for 50% of examples
   - Expected behavior: Log failures, continue with successful parses
   - Fallback: If >80% fail, abort and report error

## Testing Strategy

### Test Cases Planned
1. **Integration Test**: Full Phase 1 + Phase 2 run
   - Input: Small dataset (20 examples)
   - Expected output: Complete results with both phases

2. **Unit Test**: Phase 1 early stopping
   - Input: Mock evaluator that returns >90% on iteration 3
   - Expected output: Phase 1 exits after 3 iterations

3. **Unit Test**: Phase 2 path database updates
   - Input: Mock evaluator with specific recommendations
   - Expected output: Paths added/removed correctly

4. **Error Test**: Max iterations exceeded
   - Input: Mock evaluator never reaches threshold
   - Expected output: Error with partial results

5. **Edge Test**: Empty dataset
   - Input: dataset=[]
   - Expected output: Validation error

6. **Edge Test**: All Phase 2 answers wrong
   - Input: Mock LLM always wrong
   - Expected output: All paths go to bad_paths, no good_paths

### Integration Tests
- Test with real Gemini API on small dataset
- Test with real dataset from HuggingFace
- Test state persistence (save/resume)
- Test cost tracking accuracy

### Performance Tests
- Benchmark: Complete run < 10 minutes for 50 examples
- Parallel processing: Test batch LLM calls
- Memory: Handle 1000 examples without memory issues

## Missing Information

### Questions to Answer Before Implementation
1. **Train/test split ratio**: 70/30? 80/20? K-fold?
   - **Decision needed**: Affects evaluation reliability
   - **Impact**: Overfitting vs sample size tradeoff

2. **Batch size**: How many examples to process at once?
   - **Decision needed**: 1 (sequential)? 10 (batched)? All (parallel)?
   - **Impact**: Speed vs API rate limits

3. **State saving frequency**: Every iteration? Only checkpoints?
   - **Decision needed**: Balance between robustness and I/O overhead
   - **Impact**: Recovery capability

4. **Adaptive thresholds**: Should Phase 1 threshold decrease if stuck?
   - **Decision needed**: Yes/No, by how much
   - **Impact**: Ability to complete training

5. **Cost budget**: What's the max acceptable cost per run?
   - **Decision needed**: $1? $10? No limit?
   - **Impact**: Early stopping logic

### Information Needed
1. **Real dataset** - Need actual training data
   - Source: AGENTS.md references dataset
   - Impact: Can't test full pipeline without this

2. **Expected performance** - What accuracy is good?
   - Source: User expectations
   - Impact: Success criteria unclear

3. **API rate limits** - Provider throttling
   - Source: Provider documentation
   - Impact: Batching strategy

## Notes
- Implement comprehensive logging (DEBUG level for development)
- Use tqdm or similar for progress bars (user feedback)
- Save intermediate outputs to JSON files (debugging)
- Consider adding visualization (similarity trends, accuracy curves)
- Implement graceful shutdown (SIGINT handler)
- Add dry-run mode (validate without LLM calls)
- Version control for path databases (git-friendly JSON)
- Consider adding A/B testing (compare prompt variations)
- Monitor token usage per iteration (cost tracking)
- Add telemetry (optional, privacy-respecting)
- Keep optimizer stateless where possible (functional programming style)
- Consider adding callback hooks for custom logic
