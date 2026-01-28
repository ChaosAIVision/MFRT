# BEAD Plan: XMLOutputParser

## Component Purpose

### What It Does
Extracts content from XML-style tags (`<construction>` and `<think>`) in LLM output text.

### Why It's Needed
The 2-phase meta-prompt system requires LLMs to output structured responses with construction phase in `<construction>` tags and reasoning phase in `<think>` tags. We need reliable parsing to separate these phases for evaluation.

### Where It Fits
This is the **foundation component** - all other components depend on this parser to extract phase-specific content from raw LLM outputs.

## Input Specification

### Input Parameters
| Parameter | Type | Required | Description | Example | Validation |
|-----------|------|----------|-------------|---------|------------|
| text | str | Yes | Raw LLM output text | "...\<construction\>entities: ...\</construction\>..." | Non-empty string |
| tag_name | str | Yes | XML tag to extract | "construction" or "think" | Must be valid tag name |

### Input Source
- Raw text output from LLM providers (OpenAI, Google)
- Comes from provider's `generate()` method response

### Input Validation Rules
1. Text must not be empty
2. Tag name must be alphanumeric + underscore/hyphen only
3. Text should contain at least one occurrence of the requested tag

## Output Specification

### Output Structure
```python
{
    "success": True,
    "content": "extracted content here",
    "metadata": {
        "tag_found": True,
        "num_occurrences": 1,
        "extraction_method": "regex"
    }
}
```

### Success Response
- `success`: bool - Whether extraction succeeded
- `content`: str - Extracted text content (stripped of tags and whitespace)
- `metadata.tag_found`: bool - Whether tag was found
- `metadata.num_occurrences`: int - How many times tag appeared
- `metadata.extraction_method`: str - Parsing method used ("regex", "xml_parser", etc.)

### Error Response
```python
{
    "success": False,
    "content": "",
    "error": "Tag 'construction' not found in text",
    "metadata": {
        "tag_found": False,
        "num_occurrences": 0
    }
}
```

### Output Consumers
- **ConstructionExtractor** - Uses extracted construction content
- **ReasoningPathEvaluator** - Uses extracted think/reasoning content
- **TwoPhaseOptimizer** - Uses both for separate evaluation

## Dependencies & Build Order

### Depends On (Must Build First)
None - This is a foundation component

### Depended On By (Build These After)
1. **ConstructionExtractor** - Needs parsed construction content
2. **ReasoningPathEvaluator** - Needs parsed reasoning content
3. All other components indirectly

### Build Priority
- Priority: **Critical (P0)**
- Suggested build order: **#1 out of 6**
- Blocking: All other components

## Component Relationships

### Data Flow
```
[LLM Provider Output (raw text)]
     ↓ (provides: full response with XML tags)
[XMLOutputParser]
     ↓ (provides: extracted construction, extracted reasoning)
[ConstructionExtractor] + [ReasoningPathEvaluator]
```

### Interaction Pattern
- **Synchronous** - Direct function call
- **Request-Response** pattern
- No timing requirements (fast operation)

## Implementation Plan

### Complexity Assessment
- Complexity Level: **Simple**
- Estimated Effort: 2-4 hours
- Risk Level: **Low** (well-defined parsing task)

### Technical Approach
1. **Step 1**: Use regex for simple XML tag extraction (handles malformed XML)
2. **Step 2**: Fallback to Python's xml.etree if regex fails
3. **Step 3**: Handle edge cases (nested tags, multiple occurrences, unclosed tags)
4. **Step 4**: Return structured dict with success status and metadata

### Key Algorithms/Patterns
- **Regex pattern**: `r'<{tag}>(.*?)</{tag}>'` with `re.DOTALL` flag
  - Why: Handles newlines and multi-line content
- **Fallback to xml.etree.ElementTree**: For well-formed XML
  - Why: More robust for complex nested structures
- **Strategy pattern**: Try regex first, fallback to XML parser if needed

### Technology Stack
- Language/Framework: Python 3.11+
- Libraries needed:
  - `re` (built-in) - Regex parsing
  - `xml.etree.ElementTree` (built-in) - XML fallback
  - `typing` - Type hints
- External services: None

## Edge Cases & Risks

### Edge Cases to Handle
1. **Empty input** - Return error with clear message
2. **Tag not found** - Return success=False with metadata
3. **Multiple occurrences** - Take first occurrence, log warning
4. **Nested tags** - Extract outermost content
5. **Unclosed tags** - Handle gracefully with partial extraction
6. **Malformed XML** - Regex should still work
7. **Tag with attributes** - `<construction type="v1">` should still parse

### Potential Risks
1. **Risk**: LLM might not always use exact tag format
   - **Impact**: Medium
   - **Mitigation**: Use flexible regex, try multiple patterns

2. **Risk**: Content might contain escaped XML characters
   - **Impact**: Low
   - **Mitigation**: Unescape XML entities after extraction

### Error Scenarios
1. **Scenario**: Text contains `<construction>` but no closing tag
   - Expected behavior: Try to extract until end of text or next tag
   - Fallback: Return error with partial content

2. **Scenario**: Empty tag `<construction></construction>`
   - Expected behavior: Return success=True with empty content
   - Fallback: N/A (this is valid)

## Testing Strategy

### Test Cases Planned
1. **Normal Case**: Well-formed single tag
   - Input: `"Text before <construction>entities: user, task</construction> text after"`
   - Expected output: `{"success": True, "content": "entities: user, task", ...}`

2. **Edge Case**: Multiple occurrences
   - Input: `"<construction>v1</construction> and <construction>v2</construction>"`
   - Expected output: Extract first occurrence, log warning

3. **Edge Case**: Nested tags
   - Input: `"<construction><inner>nested</inner></construction>"`
   - Expected output: Extract full content including inner tags

4. **Error Case**: Tag not found
   - Input: `"No tags here"`
   - Expected output: `{"success": False, "error": "Tag 'construction' not found", ...}`

5. **Edge Case**: Unclosed tag
   - Input: `"<construction>incomplete"`
   - Expected output: Attempt partial extraction or return error

6. **Edge Case**: Tag with attributes
   - Input: `"<construction type='v1'>content</construction>"`
   - Expected output: Extract "content" successfully

### Integration Tests
- Test with ConstructionExtractor: Verify extracted content is properly formatted
- Test with real LLM outputs: Use actual Gemini/GPT responses

### Performance Tests
- Benchmark: < 1ms for typical 500-word responses
- Stress test: Handle 10,000 character texts
- Load test: 1000 extractions/second

## Missing Information

### Questions to Answer Before Implementation
1. Should we preserve whitespace/newlines in extracted content?
   - **Decision needed**: Strip or preserve?
2. How to handle multiple occurrences - first, last, all?
   - **Decision needed**: Current plan is "first + warning"

### Information Needed
1. **Real LLM output examples** from Gemini with construction/think tags
   - Source: Run a test prompt through current system
   - Impact: Can't test with realistic data without this

## Notes
- Consider using `lxml` library if built-in xml.etree is insufficient
- May want to add caching if same text is parsed multiple times
- Consider adding validation for tag content structure (optional)
- Keep implementation simple - don't over-engineer for future requirements
