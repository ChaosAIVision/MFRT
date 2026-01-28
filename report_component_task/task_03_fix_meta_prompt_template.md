# BEAD Plan: Fix Meta-Prompt Template

## Component Purpose

### What It Does
Sửa `DEFAULT_META_PROMPT_TEMPLATE` để LLM generate complete prompts với actual examples thay vì placeholders.

### Why It's Needed
Hiện tại meta-prompt output có `{examples1}`, `{examples2}` → confuse LLM → worse performance.
Cần instruct LLM: "Include the actual example text, not placeholders"

### Where It Fits
Core optimization fix - directly improves meta-prompt quality.

## Input Specification

### Input Parameters
- Current template in `meta_prompt.py`
- Examples from training data

### Input Source
- `DEFAULT_META_PROMPT_TEMPLATE` constant
- Training examples passed via `construct_content()`

## Output Specification

### Output Structure
Modified template that produces complete prompts:
```
Example 1:
Article: [ACTUAL ARTICLE TEXT HERE]
Output: True

Example 2:
Article: [ACTUAL ARTICLE TEXT HERE]
Output: False
```

Instead of:
```
Example 1:
Article: {examples1}
...
```

## Dependencies & Build Order

### Depends On
1. Component 1 - Analyze Flow (understanding template mechanism)

### Depended On By
1. Component 4 - Auto-Fill (fallback if template fix incomplete)

## Implementation Plan

### Technical Approach
1. Locate `DEFAULT_META_PROMPT_TEMPLATE` in `meta_prompt.py`
2. Add explicit instruction: "Do NOT use placeholders like {examples1}. Include actual example text."
3. Modify example formatting in template
4. Test with actual optimization run

### Key Changes
- Line ~42-46: Add anti-placeholder instruction
- Emphasize: "Copy the actual example text verbatim"

## Testing Strategy

1. Run optimization with new template
2. Check output has no `{exampleN}` placeholders
3. Verify examples are actual training data
4. Verify output format matches original prompt (True/False not categories)

---

## Implementation Report - Update 2

### Additional Fix: Preserve Output Format

**Issue Discovered**: Meta-prompt was allowing LLM to change output format from binary (True/False) to multi-category classification (breaking_record/team_victory/individual_achievement/tournament_announcement).

**Root Cause**: Line 55 said "copy paste the exact return instructions" but wasn't strong enough to prevent LLM from "improving" the output schema.

**Fix Applied** (`meta_prompt.py:56-63`):

Added explicit section:
```
CRITICAL: PRESERVE OUTPUT FORMAT FROM ORIGINAL PROMPT
- You MUST copy the exact output format and return instructions from the original prompt.
- DO NOT change the output schema, categories, or classification structure.
- If the original prompt asks for binary output (True/False), your optimized prompt MUST also output True/False.
- If the original prompt asks for specific categories, keep those exact categories - do NOT create new ones.
- DO NOT add additional classification steps, confidence scores, or reasoning fields unless they were in the original prompt.
- Example INCORRECT: Original asks for "True or False" → Optimized asks for "breaking_record/team_victory/individual_achievement" (WRONG!)
- Example CORRECT: Original asks for "True or False" → Optimized asks for "True or False" (CORRECT!)
```

**Why Critical**:
- User's prompt expects binary output (milestone: True/False)
- Optimized prompt was creating 4-way classification → breaks evaluation
- Changing output schema invalidates all testing/metrics
- LLM needs explicit constraint to NOT "improve" the output format

**Total Changes in Component 3**:
1. Anti-placeholder instructions (lines 44-50) ✅
2. Preserve output format instructions (lines 56-63) ✅
