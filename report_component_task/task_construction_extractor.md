# BEAD Plan: ConstructionExtractor

## Component Purpose

### What It Does
Extracts and structures the 4 core elements from construction phase output:
1. Entities (relevant objects/agents)
2. State Variables (attributes that change over time)
3. Actions (operations with preconditions and effects)
4. Constraints (rules and limitations)

### Why It's Needed
Phase 1 optimization requires measuring similarity between generated construction and groundtruth. To do this, we need structured extraction of construction elements, not just raw text comparison.

### Where It Fits
This component sits between **XMLOutputParser** (provides raw construction text) and **ConstructionSimilarityEvaluator** (needs structured elements for comparison).

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| construction_text | str | Yes | Extracted construction phase content | "Entities: user, task\nState..." | Non-empty string |
| extraction_format | str | No | Output format preference | "dict" or "json" | Default: "dict" |

### Input Source
- Output from **XMLOutputParser** after extracting `<construction>` content
- Raw construction text following the format specified in meta-prompt

### Input Validation Rules
1. Text must contain at least 50 characters (minimum viable construction)
2. Text should mention at least one of: "entities", "state", "actions", "constraints" (case-insensitive)
3. Must be valid UTF-8 text

## Output Specification

### Output Structure
```python
{
    "entities": ["user", "task", "deadline", "status"],
    "state_variables": [
        {
            "name": "task_status",
            "type": "enum",
            "possible_values": ["pending", "in_progress", "completed"]
        },
        {
            "name": "user_availability",
            "type": "boolean"
        }
    ],
    "actions": [
        {
            "name": "assign_task",
            "preconditions": ["user is available", "task is pending"],
            "effects": ["task status = in_progress", "user availability = false"]
        }
    ],
    "constraints": [
        "One user can only work on one task at a time",
        "Task must be assigned before it can be completed"
    ],
    "metadata": {
        "extraction_confidence": 0.95,
        "missing_sections": [],
        "extraction_method": "regex_parser"
    }
}
```

### Success Response
- `entities`: List[str] - Extracted entity names
- `state_variables`: List[Dict] - State variables with types and possible values
- `actions`: List[Dict] - Actions with preconditions and effects
- `constraints`: List[str] - Constraint statements
- `metadata.extraction_confidence`: float (0-1) - How confident the extraction is
- `metadata.missing_sections`: List[str] - Which sections were not found

### Error Response
```python
{
    "entities": [],
    "state_variables": [],
    "actions": [],
    "constraints": [],
    "error": "Failed to extract any construction elements",
    "metadata": {
        "extraction_confidence": 0.0,
        "missing_sections": ["entities", "state_variables", "actions", "constraints"]
    }
}
```

### Output Consumers
- **ConstructionSimilarityEvaluator** - Compares extracted elements with groundtruth
- **TwoPhaseMetaPrompt** - Uses examples for few-shot prompting
- **TwoPhaseOptimizer** - Logs extraction quality

## Dependencies & Build Order

### Depends On (Must Build First)
1. **XMLOutputParser** - Needed to extract construction text from tags
   - Uses: Parsed `<construction>` content
   - Status: ⏳ Not Started

### Depended On By (Build These After)
1. **ConstructionSimilarityEvaluator** - Needs structured elements for comparison

### Build Priority
- Priority: **High (P1)**
- Suggested build order: **#2 out of 6**
- Blocking: ConstructionSimilarityEvaluator, TwoPhaseMetaPrompt

## Component Relationships

### Data Flow
```
[XMLOutputParser]
     ↓ (provides: construction_text)
[ConstructionExtractor]
     ↓ (provides: structured_elements)
[ConstructionSimilarityEvaluator]
```

### Interaction Pattern
- **Synchronous** - Direct function call
- **Request-Response** pattern
- Timing requirements: < 100ms per extraction

## Implementation Plan

### Complexity Assessment
- Complexity Level: **Medium**
- Estimated Effort: 6-8 hours
- Risk Level: **Medium** (LLM output format can vary)

### Technical Approach
1. **Step 1**: Define regex patterns for each section (entities, state_variables, actions, constraints)
2. **Step 2**: Parse entities as comma-separated list
3. **Step 3**: Parse state variables with type detection (look for "type:", "enum:", "boolean", etc.)
4. **Step 4**: Parse actions with structured preconditions/effects (look for "->" or "=>" symbols)
5. **Step 5**: Parse constraints as bullet points or numbered list
6. **Step 6**: Calculate extraction confidence based on how many sections found
7. **Step 7**: Return structured dict

### Key Algorithms/Patterns
- **Section Detection**: Regex for headers like "Entities:", "(1) relevant entities", "State Variables:", etc.
  - Why: LLM might use slight variations in formatting
- **Bullet Point Parsing**: Extract lines starting with "-", "*", "•", or numbers
  - Why: Common list formats in LLM outputs
- **Key-Value Extraction**: For state variables (name: type: values)
  - Why: Structured format specified in meta-prompt
- **Fallback to NLP**: If regex fails, use simple sentence splitting
  - Why: Robustness against format variations

### Technology Stack
- Language/Framework: Python 3.11+
- Libraries needed:
  - `re` (built-in) - Regex parsing
  - `typing` - Type hints
  - `pydantic` (optional) - Data validation
- External services: None

## Edge Cases & Risks

### Edge Cases to Handle
1. **Missing sections** - Some constructions might not have all 4 elements
2. **Different formatting** - LLM might use bullets, numbers, or prose
3. **Multilingual content** - Construction might be in Vietnamese
4. **Nested structures** - Actions might have sub-actions
5. **Ambiguous parsing** - Unclear where one section ends and next begins
6. **Empty sections** - Header present but no content

### Potential Risks
1. **Risk**: LLM uses completely different format than expected
   - **Impact**: High
   - **Mitigation**: Start with flexible regex, add fallback parsers, log unparseable formats for manual review

2. **Risk**: Extraction confidence is unreliable
   - **Impact**: Medium
   - **Mitigation**: Test confidence calculation with diverse examples, calibrate thresholds

3. **Risk**: Actions with complex preconditions/effects
   - **Impact**: Medium
   - **Mitigation**: Use simple text extraction first, enhance later if needed

### Error Scenarios
1. **Scenario**: Text has construction content but no clear section headers
   - Expected behavior: Attempt heuristic parsing (look for keywords)
   - Fallback: Return low confidence with partial extraction

2. **Scenario**: Malformed state variable definitions
   - Expected behavior: Extract what's parseable, mark others as "unknown type"
   - Fallback: Store raw text in metadata for manual inspection

## Testing Strategy

### Test Cases Planned
1. **Normal Case**: Well-formatted construction
   - Input:
   ```
   Entities: user, task, deadline
   State Variables:
   - task_status: enum [pending, in_progress, completed]
   - user_availability: boolean
   Actions:
   - assign_task
     Preconditions: user is available, task is pending
     Effects: task status becomes in_progress
   Constraints:
   - One user per task
   ```
   - Expected output: Full structured dict with all 4 sections

2. **Edge Case**: Missing actions section
   - Input: Only entities, state variables, constraints
   - Expected output: Partial extraction, actions=[], missing_sections=["actions"]

3. **Edge Case**: Different formatting (numbered list)
   - Input: "(1) Entities: ..." instead of "Entities:"
   - Expected output: Successfully extract despite format variation

4. **Error Case**: Completely unstructured text
   - Input: Prose paragraph without clear sections
   - Expected output: Low confidence, attempt keyword-based extraction

5. **Edge Case**: Vietnamese text
   - Input: "Thực thể: người dùng, nhiệm vụ..."
   - Expected output: Extract successfully with Vietnamese keywords

6. **Edge Case**: Nested action definitions
   - Input: Actions with sub-steps
   - Expected output: Flatten or preserve structure (TBD)

### Integration Tests
- Test with XMLOutputParser: Full pipeline from raw LLM output to structured elements
- Test with ConstructionSimilarityEvaluator: Verify extracted format is compatible

### Performance Tests
- Benchmark: < 100ms for 500-word construction
- Stress test: Handle 5000-word construction text
- Load test: 100 extractions/second

## Missing Information

### Questions to Answer Before Implementation
1. **Vietnamese support**: Should we support Vietnamese section headers?
   - **Decision needed**: Yes/No, impacts regex patterns

2. **Action structure**: Flatten nested actions or preserve hierarchy?
   - **Decision needed**: Affects data structure design

3. **Confidence threshold**: What confidence level is acceptable?
   - **Decision needed**: For flagging low-quality extractions

### Information Needed
1. **Real groundtruth examples** - Need to see actual construction groundtruth format
   - Source: AGENTS.md mentions groundtruth, need dataset examples
   - Impact: Can't design similarity comparison without knowing target format

2. **LLM output samples** - Real outputs from Gemini with construction tags
   - Source: Run test prompts
   - Impact: Need to see format variations to build robust parser

## Notes
- Consider using a formal grammar parser (like pyparsing) if regex becomes too complex
- May need to iteratively improve based on real LLM outputs
- Extraction confidence calculation is critical for identifying poor constructions early
- Keep parsing logic modular - separate parser for each section type
