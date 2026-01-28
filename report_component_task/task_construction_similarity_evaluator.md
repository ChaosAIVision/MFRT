# BEAD Plan: ConstructionSimilarityEvaluator

## Component Purpose

### What It Does
Measures similarity between LLM-generated construction and groundtruth construction. Returns a similarity score (0-1) to determine if Phase 1 optimization is successful (>90% threshold required).

### Why It's Needed
AGENTS.md specifies: "Phase 1 training requires generated construction to be 90% similar to groundtruth construction." We need automated similarity measurement to:
1. Evaluate Phase 1 optimization progress
2. Decide when to move to Phase 2
3. Provide feedback for meta-prompt improvement

### Where It Fits
This is the **Phase 1 evaluator** - used exclusively during construction phase training. It sits after ConstructionExtractor and provides feedback to TwoPhaseOptimizer.

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| generated_construction | Dict | Yes | Extracted construction from LLM | {"entities": [...], ...} | Must have 4 keys |
| groundtruth_construction | Dict | Yes | Expected construction | {"entities": [...], ...} | Must have 4 keys |
| weights | Dict | No | Section importance weights | {"entities": 0.25, ...} | Sum to 1.0 |

### Input Source
- **generated_construction**: Output from ConstructionExtractor (LLM's construction)
- **groundtruth_construction**: From dataset (pre-defined correct construction)
- **weights**: From configuration or use defaults (0.25 each for 4 sections)

### Input Validation Rules
1. Both constructions must have all 4 sections: entities, state_variables, actions, constraints
2. Weights must sum to 1.0 (if provided)
3. Each section must be a list or dict (not null)

## Output Specification

### Output Structure
```python
{
    "overall_similarity": 0.92,
    "passed_threshold": True,
    "threshold": 0.90,
    "section_scores": {
        "entities": 0.95,
        "state_variables": 0.88,
        "actions": 0.91,
        "constraints": 0.94
    },
    "details": {
        "entities": {
            "matched": ["user", "task"],
            "missing": ["deadline"],
            "extra": ["priority"]
        },
        "state_variables": {
            "matched_count": 7,
            "missing_count": 1,
            "extra_count": 0
        },
        "actions": {
            "matched_count": 3,
            "partial_matches": 1,
            "missing_count": 0
        },
        "constraints": {
            "semantic_similarity_avg": 0.94
        }
    },
    "feedback": "Good construction. State variables section needs improvement (88%).",
    "metadata": {
        "evaluation_method": "hybrid",
        "computation_time_ms": 45
    }
}
```

### Success Response
- `overall_similarity`: float (0-1) - Weighted average of section scores
- `passed_threshold`: bool - True if >= 90%
- `section_scores`: Dict[str, float] - Individual section similarities
- `details`: Detailed breakdown for each section
- `feedback`: Human-readable improvement suggestions

### Error Response
```python
{
    "overall_similarity": 0.0,
    "passed_threshold": False,
    "error": "Invalid construction format: missing 'entities' section",
    "section_scores": {},
    "details": {}
}
```

### Output Consumers
- **TwoPhaseOptimizer** - Decides if Phase 1 is complete
- **TwoPhaseMetaPrompt** - Uses feedback to improve construction prompts
- **Logging/Metrics** - Track optimization progress

## Dependencies & Build Order

### Depends On (Must Build First)
1. **ConstructionExtractor** - Provides structured construction format
   - Uses: Extracted construction dict
   - Status: ⏳ Not Started

### Depended On By (Build These After)
1. **TwoPhaseMetaPrompt** - Uses similarity feedback for prompt improvement
2. **TwoPhaseOptimizer** - Uses pass/fail signal for training flow

### Build Priority
- Priority: **High (P1)**
- Suggested build order: **#3 out of 6**
- Blocking: TwoPhaseMetaPrompt, TwoPhaseOptimizer

## Component Relationships

### Data Flow
```
[ConstructionExtractor] + [Groundtruth Dataset]
     ↓ (provides: generated_construction, groundtruth_construction)
[ConstructionSimilarityEvaluator]
     ↓ (provides: similarity_score, feedback)
[TwoPhaseOptimizer] + [TwoPhaseMetaPrompt]
```

### Interaction Pattern
- **Synchronous** - Direct evaluation call
- **Request-Response** pattern
- Timing requirements: < 200ms per evaluation (not on critical path)

## Implementation Plan

### Complexity Assessment
- Complexity Level: **Complex**
- Estimated Effort: 8-12 hours
- Risk Level: **High** (defining "90% similarity" is subjective)

### Technical Approach
1. **Step 1**: Entities similarity - Set-based Jaccard index
   - Formula: `|intersection| / |union|`
2. **Step 2**: State variables similarity - Structural matching
   - Compare names (fuzzy match), types, and possible values
3. **Step 3**: Actions similarity - Complex structural comparison
   - Match action names (fuzzy)
   - Compare preconditions/effects as text sets
4. **Step 4**: Constraints similarity - Semantic text similarity
   - Use sentence embeddings (e.g., sentence-transformers)
   - Calculate cosine similarity between constraint pairs
5. **Step 5**: Weighted average with default weights (0.25 each)
6. **Step 6**: Generate detailed feedback for improvements

### Key Algorithms/Patterns
- **Jaccard Similarity** for entities (set comparison)
  - Why: Simple, interpretable, good for entity lists
- **Fuzzy String Matching** (fuzzywuzzy or rapidfuzz) for names
  - Why: LLM might use slight variations ("task_status" vs "taskStatus")
- **Sentence Embeddings** (sentence-transformers) for constraints
  - Why: Constraints are natural language, need semantic comparison
- **Weighted Average** for overall score
  - Why: Allows tuning importance of each section

### Technology Stack
- Language/Framework: Python 3.11+
- Libraries needed:
  - `rapidfuzz` (3.6.1) - Fast fuzzy string matching
  - `sentence-transformers` (2.3.1) - Semantic text similarity
  - `numpy` - Vector operations
  - `typing` - Type hints
- External services: None (all local computation)

## Edge Cases & Risks

### Edge Cases to Handle
1. **Empty sections** - One or both constructions missing a section
2. **Different granularity** - LLM provides more/fewer entities than groundtruth
3. **Synonym usage** - "user" vs "person", "task" vs "job"
4. **Order differences** - Same entities but different order
5. **Extra details** - LLM adds extra state variables not in groundtruth
6. **Partial matches** - Action mostly correct but preconditions slightly off

### Potential Risks
1. **Risk**: 90% threshold is too strict or too lenient
   - **Impact**: High - affects training effectiveness
   - **Mitigation**: Make threshold configurable, experiment with real data

2. **Risk**: Semantic similarity for constraints is unreliable
   - **Impact**: Medium - might accept bad constraints or reject good ones
   - **Mitigation**: Use ensemble of methods (embedding + keyword + fuzzy), tune thresholds

3. **Risk**: Slow computation due to sentence embeddings
   - **Impact**: Low - evaluation not on critical path
   - **Mitigation**: Cache embeddings, use lightweight model (all-MiniLM-L6-v2)

### Error Scenarios
1. **Scenario**: Groundtruth missing or malformed
   - Expected behavior: Return error, don't evaluate
   - Fallback: Log error, skip this example

2. **Scenario**: Generated construction has extra sections
   - Expected behavior: Ignore extra sections, evaluate only required 4
   - Fallback: Log warning about extra content

## Testing Strategy

### Test Cases Planned
1. **Normal Case**: 100% match
   - Input: Identical generated and groundtruth constructions
   - Expected output: `overall_similarity = 1.0, passed_threshold = True`

2. **Normal Case**: 90% match (threshold boundary)
   - Input: 1-2 missing entities, rest perfect
   - Expected output: `overall_similarity = 0.90, passed_threshold = True`

3. **Edge Case**: 89% match (just below threshold)
   - Input: Slightly more differences
   - Expected output: `overall_similarity = 0.89, passed_threshold = False`

4. **Edge Case**: Synonym usage
   - Input: Generated uses "user", groundtruth uses "person"
   - Expected output: High similarity (fuzzy matching works)

5. **Edge Case**: Different order
   - Input: Same entities but different order
   - Expected output: 100% similarity (order doesn't matter for sets)

6. **Error Case**: Missing section
   - Input: Generated construction missing "actions"
   - Expected output: Error or very low score

7. **Complex Case**: Partial action match
   - Input: Action name correct, preconditions slightly different
   - Expected output: Partial credit (e.g., 0.7-0.8 for that action)

### Integration Tests
- Test with ConstructionExtractor: Full pipeline from raw text to similarity score
- Test with real dataset: Use actual groundtruth examples from training data
- Test with TwoPhaseOptimizer: Verify pass/fail signals work correctly

### Performance Tests
- Benchmark: < 200ms per evaluation
- Stress test: 1000 evaluations without memory leaks
- Model loading: Cache sentence-transformer model (load once)

## Missing Information

### Questions to Answer Before Implementation
1. **Synonym handling**: Should we use a predefined synonym list or rely on fuzzy matching?
   - **Decision needed**: Affects accuracy and complexity

2. **Partial credit**: How to score partial matches (e.g., action name correct but effects wrong)?
   - **Decision needed**: Linear interpolation? Step function? Custom logic?

3. **Constraint weighting**: Are all constraints equally important?
   - **Decision needed**: Affects constraints section scoring

4. **Vietnamese support**: Should semantic similarity work for Vietnamese constraints?
   - **Decision needed**: Impacts model choice (multilingual sentence-transformers)

### Information Needed
1. **Real groundtruth examples** - Need actual dataset with constructions
   - Source: Dataset referenced in AGENTS.md
   - Impact: Can't calibrate similarity thresholds without real data

2. **Example "90% similar" pairs** - User's definition of 90%
   - Source: Ask user or create examples for validation
   - Impact: Risk of misaligned similarity definition

3. **Error tolerance** - Which differences are acceptable?
   - Source: User feedback after initial implementation
   - Impact: Affects strictness of matching

## Notes
- Start with simple exact matching, then add fuzzy matching, then semantic embeddings
- Log detailed similarity breakdowns for debugging and calibration
- Consider adding visualization of differences (side-by-side comparison)
- May need to tune section weights based on real optimization results
- Keep evaluation deterministic - same inputs should give same score
- Consider adding an "explain" mode that shows why similarity is X%
