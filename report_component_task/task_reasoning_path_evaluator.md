# BEAD Plan: ReasoningPathEvaluator

## Component Purpose

### What It Does
Evaluates reasoning logic paths from Phase 2 (reasoning phase). Maintains a database of:
- **Good logic paths** - Reasoning patterns that lead to correct answers
- **Bad logic paths** - Reasoning patterns that lead to wrong answers
- **Groundtruth logic paths** - Reference reasoning from dataset

Compares LLM's reasoning against these three categories and provides feedback for meta-prompt improvement.

### Why It's Needed
AGENTS.md specifies complex Phase 2 evaluation logic:
- If answer is **correct** but reasoning path differs from groundtruth → Consider adding to good paths
- If answer is **wrong** and reasoning not in good/bad/groundtruth paths → Add to bad paths
- If answer is **wrong** but reasoning matches good paths → Review and fix good paths

This component automates this logic for iterative prompt optimization.

### Where It Fits
This is the **Phase 2 evaluator** - used during reasoning phase training. It sits after XMLOutputParser (extracts `<think>` content) and provides feedback to TwoPhaseOptimizer.

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| reasoning_text | str | Yes | Extracted reasoning from `<think>` tag | "Step 1: Analyze entities..." | Non-empty string |
| predicted_answer | str | Yes | LLM's final answer | "42" or "Option A" | Any string |
| groundtruth_answer | str | Yes | Correct answer | "42" | Any string |
| groundtruth_reasoning | str | No | Reference reasoning path | "First calculate..." | Optional |
| good_paths_db | List[str] | No | Known good reasoning patterns | [...] | Default: [] |
| bad_paths_db | List[str] | No | Known bad reasoning patterns | [...] | Default: [] |

### Input Source
- **reasoning_text**: From XMLOutputParser (extracted `<think>` content)
- **predicted_answer**: From XMLOutputParser or final line of reasoning
- **groundtruth_answer**: From dataset
- **groundtruth_reasoning**: From dataset (if available)
- **good_paths_db / bad_paths_db**: From persistent storage (JSON/DB)

### Input Validation Rules
1. reasoning_text must be at least 20 characters (minimum viable reasoning)
2. predicted_answer and groundtruth_answer must not be empty
3. Answers are compared case-insensitive, whitespace-trimmed

## Output Specification

### Output Structure
```python
{
    "answer_correct": True,
    "reasoning_quality": "good",  # "good", "bad", "novel_correct", "novel_incorrect", "needs_review"
    "similarity_to_groundtruth": 0.85,
    "similarity_to_good_paths": 0.72,
    "similarity_to_bad_paths": 0.15,
    "recommendation": "add_to_good_paths",  # or "add_to_bad_paths", "review_good_paths", "keep_existing", "no_action"
    "matched_path_type": "groundtruth",  # or "good", "bad", "none"
    "matched_path_index": 3,
    "feedback": "Correct answer with novel reasoning approach. Consider adding to good paths database.",
    "metadata": {
        "reasoning_length": 450,
        "computation_time_ms": 120
    }
}
```

### Success Response
- `answer_correct`: bool - Whether predicted == groundtruth
- `reasoning_quality`: str - Quality assessment
- `similarity_to_*`: float (0-1) - Similarity scores to each path type
- `recommendation`: str - What action to take with this reasoning
- `feedback`: str - Human-readable feedback for prompt improvement

### Error Response
```python
{
    "answer_correct": False,
    "reasoning_quality": "unknown",
    "error": "Failed to extract answer from reasoning text",
    "recommendation": "no_action",
    "metadata": {}
}
```

### Output Consumers
- **TwoPhaseOptimizer** - Uses recommendations to update path databases
- **TwoPhaseMetaPrompt** - Uses good/bad paths for few-shot examples
- **Logging/Metrics** - Track reasoning quality over iterations

## Dependencies & Build Order

### Depends On (Must Build First)
1. **XMLOutputParser** - Extracts reasoning text from `<think>` tags
   - Uses: Parsed reasoning content
   - Status: ⏳ Not Started

### Depended On By (Build These After)
1. **TwoPhaseMetaPrompt** - Uses good/bad paths in prompts
2. **TwoPhaseOptimizer** - Uses evaluation results for Phase 2 training

### Build Priority
- Priority: **High (P1)**
- Suggested build order: **#3 out of 6** (parallel with ConstructionSimilarityEvaluator)
- Blocking: TwoPhaseMetaPrompt, TwoPhaseOptimizer

## Component Relationships

### Data Flow
```
[XMLOutputParser] + [Dataset Groundtruth] + [Path Databases]
     ↓ (provides: reasoning_text, answers, reference_paths)
[ReasoningPathEvaluator]
     ↓ (provides: quality_assessment, recommendation)
[TwoPhaseOptimizer] + [TwoPhaseMetaPrompt]
     ↓ (updates: good_paths_db, bad_paths_db)
[Persistent Storage]
```

### Interaction Pattern
- **Synchronous** - Direct evaluation call
- **Stateful** - Maintains path databases across evaluations
- Timing requirements: < 300ms per evaluation

## Implementation Plan

### Complexity Assessment
- Complexity Level: **Complex**
- Estimated Effort: 10-14 hours
- Risk Level: **High** (complex decision logic, semantic similarity required)

### Technical Approach
1. **Step 1**: Check answer correctness (simple string comparison)
2. **Step 2**: Calculate semantic similarity to groundtruth reasoning (if available)
3. **Step 3**: Calculate similarity to all good paths (max similarity)
4. **Step 4**: Calculate similarity to all bad paths (max similarity)
5. **Step 5**: Apply decision logic from AGENTS.md:
   ```
   IF answer_correct:
       IF reasoning similar to groundtruth (>0.8): return "keep_existing"
       ELIF reasoning similar to good_paths (>0.7): return "keep_existing"
       ELSE: return "add_to_good_paths" (novel correct reasoning)
   ELSE:
       IF reasoning similar to bad_paths (>0.7): return "keep_existing" (known bad)
       ELIF reasoning similar to good_paths (>0.7): return "review_good_paths" (good path led to wrong answer!)
       ELIF reasoning similar to groundtruth (>0.7): return "review_groundtruth" (groundtruth led to wrong answer!)
       ELSE: return "add_to_bad_paths" (novel bad reasoning)
   ```
6. **Step 6**: Generate detailed feedback based on decision
7. **Step 7**: Return structured evaluation result

### Key Algorithms/Patterns
- **Sentence Embeddings** (sentence-transformers) for semantic similarity
  - Why: Reasoning paths are long text, need semantic comparison
  - Model: `all-MiniLM-L6-v2` or `paraphrase-multilingual-MiniLM-L12-v2` (Vietnamese support)
- **Cosine Similarity** between reasoning embeddings
  - Why: Standard metric for text similarity
- **Max Pooling** over path databases (take highest similarity)
  - Why: Find best matching path in database
- **Threshold-based Decision Tree** for recommendations
  - Why: Clear, interpretable logic as specified in AGENTS.md

### Technology Stack
- Language/Framework: Python 3.11+
- Libraries needed:
  - `sentence-transformers` (2.3.1) - Semantic text similarity
  - `numpy` - Vector operations
  - `typing` - Type hints
  - `json` - Path database persistence
- External services: None (all local computation)

## Edge Cases & Risks

### Edge Cases to Handle
1. **Empty path databases** - No good/bad paths yet (first iteration)
2. **Ambiguous similarity** - Reasoning matches both good and bad paths
3. **Very short reasoning** - Only 1-2 sentences
4. **No groundtruth reasoning** - Only have groundtruth answer
5. **Multiple valid reasoning paths** - Different approaches to same answer
6. **Numerical answers** - "42" vs "42.0" vs "forty-two"

### Potential Risks
1. **Risk**: Semantic similarity threshold is too strict/lenient
   - **Impact**: High - affects path database quality
   - **Mitigation**: Make thresholds configurable, calibrate with real data

2. **Risk**: Path databases grow too large (performance degradation)
   - **Impact**: Medium - slow similarity computation
   - **Mitigation**: Cluster similar paths, keep only representative examples

3. **Risk**: False positive "review_good_paths" (one wrong answer invalidates good path)
   - **Impact**: Medium - might remove valid reasoning
   - **Mitigation**: Require multiple failures before removing from good paths

4. **Risk**: Multilingual reasoning (Vietnamese + English mixed)
   - **Impact**: Medium - affects similarity accuracy
   - **Mitigation**: Use multilingual sentence transformer model

### Error Scenarios
1. **Scenario**: Cannot extract answer from reasoning text
   - Expected behavior: Return error, recommendation = "no_action"
   - Fallback: Try regex patterns for common answer formats

2. **Scenario**: Reasoning matches both good and bad paths with high similarity
   - Expected behavior: Prioritize bad path match (safer), flag for manual review
   - Fallback: Log conflict for analysis

## Testing Strategy

### Test Cases Planned
1. **Normal Case**: Correct answer, similar to groundtruth reasoning
   - Input: answer_correct=True, similarity_to_groundtruth=0.9
   - Expected output: `reasoning_quality="good"`, `recommendation="keep_existing"`

2. **Normal Case**: Correct answer, novel reasoning (not in databases)
   - Input: answer_correct=True, all similarities < 0.5
   - Expected output: `reasoning_quality="novel_correct"`, `recommendation="add_to_good_paths"`

3. **Normal Case**: Wrong answer, matches known bad path
   - Input: answer_correct=False, similarity_to_bad_paths=0.8
   - Expected output: `reasoning_quality="bad"`, `recommendation="keep_existing"`

4. **Critical Case**: Wrong answer, matches good path (needs review!)
   - Input: answer_correct=False, similarity_to_good_paths=0.85
   - Expected output: `reasoning_quality="needs_review"`, `recommendation="review_good_paths"`

5. **Edge Case**: Empty path databases (first iteration)
   - Input: good_paths_db=[], bad_paths_db=[]
   - Expected output: Recommend adding to appropriate database

6. **Edge Case**: Ambiguous - matches both good and bad
   - Input: similarity_to_good=0.75, similarity_to_bad=0.72
   - Expected output: Flag for manual review, prioritize based on answer correctness

7. **Edge Case**: Numerical answer variations
   - Input: predicted="42", groundtruth="42.0"
   - Expected output: answer_correct=True (normalize numbers)

### Integration Tests
- Test with XMLOutputParser: Full pipeline from raw LLM output to evaluation
- Test with TwoPhaseOptimizer: Verify path database updates work correctly
- Test with real dataset: Use actual reasoning examples

### Performance Tests
- Benchmark: < 300ms per evaluation (including similarity computation)
- Stress test: 1000 evaluations with growing path databases (up to 100 paths each)
- Model loading: Cache sentence-transformer model (load once)
- Path database: Test with 500+ paths (realistic after many iterations)

## Missing Information

### Questions to Answer Before Implementation
1. **Similarity thresholds**: What values for "similar to" checks?
   - **Decision needed**: 0.7? 0.8? 0.9? Affects sensitivity
   - **Impact**: Core decision logic depends on this

2. **Answer normalization**: How to compare numerical/textual answers?
   - **Decision needed**: String exact match? Fuzzy match? Parse numbers?
   - **Impact**: Affects answer_correct accuracy

3. **Path pruning**: When to remove paths from databases?
   - **Decision needed**: Never? After X iterations? Based on usage?
   - **Impact**: Database growth and quality

4. **Conflicting signals**: What if good path leads to wrong answer multiple times?
   - **Decision needed**: Remove after N failures? Keep with warning?
   - **Impact**: Database quality and trust

### Information Needed
1. **Real reasoning examples** - Need actual LLM reasoning outputs
   - Source: Run test prompts through Gemini
   - Impact: Can't calibrate similarity without real data

2. **Groundtruth reasoning paths** - From dataset
   - Source: Dataset referenced in AGENTS.md
   - Impact: Can't test groundtruth comparison without this

3. **Expected path database size** - How many good/bad paths after training?
   - Source: User estimation or similar systems
   - Impact: Affects performance optimization strategy

## Notes
- Start with simple exact string matching, then add semantic similarity
- Log all "review_*" recommendations for manual inspection
- Consider adding confidence scores for recommendations
- Path databases should be versioned (track changes over iterations)
- May need to implement path clustering to manage database size
- Consider adding "explanation" feature: why this reasoning is good/bad
- Keep evaluation deterministic - cache embeddings for same text
- Vietnamese support is important - use multilingual models
