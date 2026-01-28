# BEAD Plan: TwoPhaseMetaPrompt

## Component Purpose

### What It Does
Constructs meta-prompts for the 2-phase optimization system:
- **Phase 1 Meta-Prompt**: For construction phase optimization (entities, state variables, actions, constraints)
- **Phase 2 Meta-Prompt**: For reasoning phase optimization (with good/bad path examples)

Dynamically updates meta-prompts based on evaluation feedback, adding good examples and filtering bad patterns.

### Why It's Needed
AGENTS.md specifies a structured 2-phase prompt format:
- Phase 1: "Analyze the problem. Define: (1) entities, (2) state variables, (3) actions, (4) constraints. Do not propose a solution yet."
- Phase 2: "Using only the model defined above, generate a step-by-step solution plan."

The meta-prompt builder must incorporate:
- Few-shot examples from good reasoning paths
- Negative examples from bad reasoning paths
- Feedback from evaluators to improve prompts iteratively

### Where It Fits
This component is used by **TwoPhaseOptimizer** to generate optimized prompts for each training iteration. It receives feedback from both evaluators and updates the meta-prompt accordingly.

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| phase | str | Yes | Which phase to build prompt for | "construction" or "reasoning" | Must be valid phase |
| task_description | str | Yes | The problem/task to solve | "Schedule meeting for 3 people" | Non-empty string |
| good_construction_examples | List[Dict] | No | Examples of good constructions | [{...}] | For phase 1 |
| good_reasoning_paths | List[str] | No | Examples of good reasoning | ["Step 1..."] | For phase 2 |
| bad_reasoning_paths | List[str] | No | Examples of bad reasoning | ["Wrong: ..."] | For phase 2 |
| current_construction | Dict | No | Construction to use for Phase 2 | {...} | Required for phase 2 |
| feedback | str | No | Improvement suggestions | "Focus more on constraints" | Optional |

### Input Source
- **task_description**: From dataset (problem statement)
- **good_construction_examples**: From previous successful iterations
- **good_reasoning_paths**: From ReasoningPathEvaluator database
- **bad_reasoning_paths**: From ReasoningPathEvaluator database
- **current_construction**: From Phase 1 output (for Phase 2)
- **feedback**: From ConstructionSimilarityEvaluator or ReasoningPathEvaluator

### Input Validation Rules
1. phase must be "construction" or "reasoning"
2. task_description must not be empty
3. For Phase 2: current_construction must be provided
4. Example lists should not exceed 10 items (keep prompts manageable)

## Output Specification

### Output Structure
```python
{
    "meta_prompt": "Analyze the following problem. First, explicitly define...\n\nExample 1:\n...",
    "phase": "construction",
    "components": {
        "instruction": "Analyze the following problem...",
        "examples": ["Example 1: ...", "Example 2: ..."],
        "constraints": ["Keep it concise (500 words)", "Use bullet points"],
        "output_format": "Wrap in <construction> tags"
    },
    "metadata": {
        "num_good_examples": 3,
        "num_bad_examples": 2,
        "includes_feedback": True,
        "prompt_length": 1250
    }
}
```

### Success Response
- `meta_prompt`: str - Complete ready-to-use prompt
- `phase`: str - Which phase this prompt is for
- `components`: Dict - Breakdown of prompt parts
- `metadata`: Dict - Information about prompt construction

### Error Response
```python
{
    "meta_prompt": "",
    "error": "Cannot build Phase 2 prompt without current_construction",
    "phase": "reasoning",
    "metadata": {}
}
```

### Output Consumers
- **TwoPhaseOptimizer** - Uses meta_prompt to call LLM
- **Logging** - Track prompt evolution over iterations
- **Debugging** - Inspect prompt components

## Dependencies & Build Order

### Depends On (Must Build First)
1. **ConstructionSimilarityEvaluator** - Provides construction feedback
   - Uses: Feedback for improving Phase 1 prompts
   - Status: ⏳ Not Started

2. **ReasoningPathEvaluator** - Provides good/bad reasoning paths
   - Uses: Path databases for Phase 2 prompts
   - Status: ⏳ Not Started

### Depended On By (Build These After)
1. **TwoPhaseOptimizer** - Uses generated meta-prompts

### Build Priority
- Priority: **Medium (P1)**
- Suggested build order: **#4 out of 6**
- Blocking: TwoPhaseOptimizer

## Component Relationships

### Data Flow
```
[ConstructionSimilarityEvaluator] + [ReasoningPathEvaluator]
     ↓ (provides: feedback, good_examples, bad_examples)
[TwoPhaseMetaPrompt]
     ↓ (provides: meta_prompt)
[TwoPhaseOptimizer]
     ↓ (sends to: LLM Provider)
[LLM Output]
```

### Interaction Pattern
- **Synchronous** - Direct prompt construction
- **Stateful** - Accumulates examples across iterations
- Timing requirements: < 50ms (simple string construction)

## Implementation Plan

### Complexity Assessment
- Complexity Level: **Medium**
- Estimated Effort: 6-8 hours
- Risk Level: **Medium** (prompt quality affects all downstream results)

### Technical Approach
1. **Step 1**: Define base prompt templates for each phase
   - Phase 1 template: Construction instructions with format spec
   - Phase 2 template: Reasoning instructions with construction context
2. **Step 2**: Add few-shot examples (good paths)
   - Select top 3-5 most representative examples
   - Format with clear delimiters
3. **Step 3**: Add negative examples (bad paths) with "Avoid:" prefix
   - Show what NOT to do
   - Explain why it's bad
4. **Step 4**: Incorporate feedback from evaluators
   - Add specific guidance based on similarity scores
   - Highlight weak areas
5. **Step 5**: Format output requirements (`<construction>` or `<think>` tags)
6. **Step 6**: Add length constraints (500 words for construction)
7. **Step 7**: Assemble final meta-prompt

### Key Algorithms/Patterns
- **Template Pattern**: Use Jinja2 or f-strings for prompt templates
  - Why: Flexible, readable, maintainable
- **Example Selection**: Pick diverse, high-quality examples (not just most recent)
  - Why: Better coverage of problem space
- **Negative Examples**: Show 1-2 bad examples per 3-5 good examples
  - Why: Balance - too many negatives confuse the model
- **Dynamic Feedback Insertion**: Conditionally add feedback sections
  - Why: Only include relevant guidance

### Technology Stack
- Language/Framework: Python 3.11+
- Libraries needed:
  - `jinja2` (3.1.2) - Template rendering (optional, can use f-strings)
  - `typing` - Type hints
- External services: None

## Edge Cases & Risks

### Edge Cases to Handle
1. **No examples available** - First iteration, no good paths yet
2. **Too many examples** - Database has 100+ good paths
3. **Conflicting examples** - Good paths show different approaches
4. **Very long examples** - Some reasoning paths are 2000+ words
5. **Multilingual content** - Examples in Vietnamese
6. **Empty feedback** - Evaluators don't provide specific suggestions

### Potential Risks
1. **Risk**: Prompt becomes too long (context limit exceeded)
   - **Impact**: High - LLM might truncate or fail
   - **Mitigation**: Limit examples, truncate long paths, monitor total token count

2. **Risk**: Examples are too similar (not diverse enough)
   - **Impact**: Medium - model overfits to specific patterns
   - **Mitigation**: Cluster examples, pick from different clusters

3. **Risk**: Bad examples confuse the model instead of helping
   - **Impact**: Medium - model might reproduce bad patterns
   - **Mitigation**: Frame clearly as "AVOID:", use sparingly (1-2 max)

4. **Risk**: Feedback contradicts base instructions
   - **Impact**: Medium - model gets conflicting signals
   - **Mitigation**: Validate feedback consistency before adding

### Error Scenarios
1. **Scenario**: Phase 2 prompt requested but no construction provided
   - Expected behavior: Return error
   - Fallback: Use empty construction with warning

2. **Scenario**: All examples are too long (each >1000 words)
   - Expected behavior: Truncate to first 200 words
   - Fallback: Use only 1 example instead of 3

## Testing Strategy

### Test Cases Planned
1. **Normal Case**: Phase 1 prompt with 3 good examples
   - Input: phase="construction", 3 good_construction_examples
   - Expected output: Properly formatted prompt with examples

2. **Normal Case**: Phase 2 prompt with good/bad paths
   - Input: phase="reasoning", current_construction, 5 good + 2 bad paths
   - Expected output: Prompt with construction context and path examples

3. **Edge Case**: First iteration (no examples)
   - Input: Empty example lists
   - Expected output: Base prompt without examples (still valid)

4. **Edge Case**: Too many examples (15 good paths)
   - Input: 15 good_reasoning_paths
   - Expected output: Select top 5, include in prompt

5. **Edge Case**: Very long examples (2000 words each)
   - Input: Long reasoning paths
   - Expected output: Truncated examples or fewer examples

6. **Error Case**: Invalid phase name
   - Input: phase="invalid"
   - Expected output: Error with clear message

7. **Complex Case**: Feedback integration
   - Input: feedback="Focus more on constraints section"
   - Expected output: Prompt includes feedback guidance

### Integration Tests
- Test with TwoPhaseOptimizer: Verify generated prompts work with LLM
- Test with real LLM: Send generated prompts to Gemini, verify output format
- Test iterative improvement: Track prompt changes across 5 iterations

### Performance Tests
- Benchmark: < 50ms per prompt generation
- Token count: < 4000 tokens per prompt (safe for most models)
- Memory: Handle 100+ examples in database efficiently

## Missing Information

### Questions to Answer Before Implementation
1. **Example selection strategy**: Random? Most recent? Highest quality? Diverse clustering?
   - **Decision needed**: Affects example representativeness
   - **Impact**: Model learning quality

2. **Prompt token budget**: What's the max token count for meta-prompts?
   - **Decision needed**: 2000? 4000? 8000?
   - **Impact**: How many examples to include

3. **Feedback format**: Free text? Structured suggestions? Scoring?
   - **Decision needed**: Affects how feedback is incorporated
   - **Impact**: Prompt clarity

4. **Vietnamese vs English**: Should prompts be in Vietnamese?
   - **Decision needed**: Language choice
   - **Impact**: Model instruction following

### Information Needed
1. **Base prompt templates** - Need initial Phase 1 and Phase 2 templates
   - Source: Refine from AGENTS.md specification
   - Impact: Foundation for all meta-prompts

2. **Example format** - How should examples be formatted?
   - Source: Test with LLM to see what works best
   - Impact: Model learning effectiveness

3. **Token counter** - Need to measure prompt length
   - Source: Use tiktoken or approximation
   - Impact: Prevent context overflow

## Notes
- Start with simple f-string templates, upgrade to Jinja2 if complexity grows
- Log all generated prompts for reproducibility and debugging
- Consider A/B testing different prompt formats
- Version meta-prompts (track which prompt version produced which results)
- Keep base templates separate from examples (easier to maintain)
- Consider adding temperature/sampling guidance in prompt
- May need separate templates for different problem types (classification, reasoning, etc.)
- Prompt evolution should be gradual - don't change everything at once
