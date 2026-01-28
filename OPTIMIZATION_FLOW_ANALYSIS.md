# Optimization Flow Analysis

## Current Optimization Flow

### Step-by-Step Process:

```
1. User calls optimizer.optimize(dataset, output_column, feedback_columns)
   ↓
2. PromptLearningOptimizer.optimize() (line 286)
   ↓
3. Load and validate dataset
   ↓
4. Extract template variables from prompt (e.g. {input})
   ↓
5. Split dataset into batches (token-aware)
   ↓
6. FOR EACH BATCH:
   ├─ 6a. MetaPrompt.construct_content() (line 334)
   │   ├─ Select template (DEFAULT_META_PROMPT_TEMPLATE or CODING_AGENT)
   │   ├─ Replace {baseline_prompt} with current prompt
   │   ├─ Build examples section from batch_df:
   │   │   Example 0
   │   │   Data: [values for template_variables]
   │   │   Output: <LLM output>
   │   │   Feedback: <correctness>, <explanation>
   │   ├─ Replace {examples} with examples string
   │   ├─ Replace {annotations} (if any)
   │   └─ Return complete meta-prompt text
   │
   ├─ 6b. Check budget limit
   │
   ├─ 6c. Send meta-prompt to LLM (line 358)
   │   messages = [{"role": "user", "content": meta_prompt_content}]
   │   response_text = await provider.generate_text(messages)
   │
   ├─ 6d. Track costs
   │
   └─ 6e. Update optimized_prompt_content = response_text (line 376)
       ⚠️ THIS IS WHERE ISSUE HAPPENS - raw LLM output assigned directly
   ↓
7. Return optimized_prompt_content
```

## Example Handling Mechanism

### How Examples Are Passed to Meta-Prompt:

**In `construct_content()` (line 193-216)**:
```python
examples = ""
for ind, row in batch_df.iterrows():
    row_dict = row.to_dict()
    output_value = row_dict[output_column]
    
    current_example = f"""
        Example {str(ind)}
        
        Data for baseline prompt: {[row_dict[temp_var] for temp_var in template_variables]}
        
        LLM Output using baseline prompt: {output_value}
        
        Output level feedback:
    """
    # ... add feedback columns ...
    examples += current_example
```

**Then replaced in template**:
```python
content = content.replace("{examples}", examples)
```

### What LLM Receives:

```
BELOW ARE THE EXAMPLES USING THE ABOVE PROMPT
************* start example data *************

Example 0
Data for baseline prompt: ['Crosby scores 58th goal...']
LLM Output using baseline prompt: True
Output level feedback:
  correctness: incorrect
  explanation: Predicted True, expected False

Example 1
Data for baseline prompt: ['Betting scandal...']
...

************* end example data *************
```

## Where Placeholders Come From

### Root Cause Analysis:

**The meta-prompt template (line 39-49) instructs**:
```
FINAL INSTRUCTIONS
Iterate on the original prompt with a new prompt...

A common best practice is to add guidelines and the most helpful few shot examples.

Note: Make sure to include the variables from the original prompt, which are wrapped 
in curly brackets (e.g. {var}).
```

**LLM interprets this as**:
- "I should add few-shot examples"
- "I should use placeholders for flexibility"
- **LLM generates**: `Article: {examples1}` thinking it's creating a TEMPLATE

**But we want**: `Article: [actual article text]` - a COMPLETE prompt

### The Issue:

1. Template says: "add few shot examples"
2. Template says: "use curly brackets for variables"
3. LLM thinks: "I'll create example placeholders {examples1}, {examples2}"
4. Code at line 376: `optimized_prompt_content = response_text` - takes it as-is
5. **NO post-processing** to fill these placeholders!

## Issues Identified

### Issue 1: Ambiguous Template Instructions
**Location**: `meta_prompt.py` line 42-46
**Problem**: Template tells LLM to use `{var}` for variables, causing confusion
**Impact**: LLM creates placeholder-based examples instead of complete examples

### Issue 2: No Post-Processing
**Location**: `prompt_optimizer.py` line 376
**Problem**: Raw LLM output assigned directly without placeholder detection/filling
**Impact**: Placeholders like `{examples1}` remain in final prompt

### Issue 3: No Validation
**Problem**: No check for unexpected placeholders in optimized prompt
**Impact**: Silent failure - prompt looks OK but performs worse

## Recommendations

### Fix 1: Modify Meta-Prompt Template (HIGH PRIORITY)

**Add explicit anti-placeholder instruction**:
```diff
FINAL INSTRUCTIONS
...
A common best practice is to add guidelines and the most helpful few shot examples.

+ IMPORTANT: When adding few-shot examples, include the ACTUAL example text from the 
+ training data above. Do NOT create placeholders like {example1}, {example2}, etc.
+ Copy the real article/input text verbatim into your examples.

Note: The ONLY curly brackets should be the template variables from the original 
prompt (like {input}). Do not add any other curly bracket placeholders.
```

### Fix 2: Add Auto-Fill Logic (MEDIUM PRIORITY)

**After line 376 in `prompt_optimizer.py`**:
```python
# Update optimized prompt
if ruleset:
    ruleset = response_text
else:
    optimized_prompt_content = response_text
    
    # Auto-fill any {exampleN} placeholders
    optimized_prompt_content = self._auto_fill_example_placeholders(
        optimized_prompt_content, 
        batch
    )
```

**New method**:
```python
def _auto_fill_example_placeholders(self, prompt: str, train_df: pd.DataFrame) -> str:
    """Fill {example1}, {example2}, etc. with actual training examples."""
    import re
    pattern = r'\{example(\d+)\}'
    matches = re.findall(pattern, prompt)
    
    for match in matches:
        idx = int(match) - 1
        if idx < len(train_df) and 'input' in train_df.columns:
            example_text = train_df.iloc[idx]['input']
            prompt = prompt.replace(f'{{example{match}}}', example_text)
    
    return prompt
```

### Fix 3: Add Validation (LOW PRIORITY)

**After auto-fill**:
```python
# Warn about unexpected placeholders
unexpected = re.findall(r'\{(?!' + '|'.join(self.template_variables) + r'\})[^}]+\}', 
                        optimized_prompt_content)
if unexpected:
    print(f"⚠️  Warning: Unexpected placeholders found: {unexpected}")
```

## Flow Diagrams

### Current Flow (WITH BUG):
```
Initial Prompt: "Return True/False for milestone. Article: {input}"
         ↓
Meta-Prompt: "Here are examples... improve this prompt"
         ↓
LLM Output: "Add examples: Article: {example1} → True..."
         ↓
TAKEN AS-IS (line 376)
         ↓
Optimized Prompt: HAS PLACEHOLDERS {example1}, {example2}
         ↓
Used for predictions → CONFUSES LLM → WORSE PERFORMANCE
```

### Fixed Flow (AFTER FIX):
```
Initial Prompt: "Return True/False for milestone. Article: {input}"
         ↓
Meta-Prompt: "Include ACTUAL examples, not placeholders"
         ↓
LLM Output: "Add examples: Article: [Crosby scores...] → True..."
         ↓
Auto-Fill (fallback): Replaces any {exampleN} found
         ↓
Optimized Prompt: COMPLETE with real examples
         ↓
Used for predictions → BETTER PERFORMANCE
```

## Summary

**Root Cause**: 
- Meta-prompt template ambiguously instructs LLM about curly brackets
- LLM creates placeholder-based examples {example1}, {example2}
- No post-processing fills these placeholders

**Impact**:
- Optimized prompts have unfilled placeholders
- LLM sees "{example1}" literally → confusion → worse performance
- Observed: 72% → 55% accuracy drop

**Solution**:
1. Fix template with explicit instruction: "use ACTUAL example text"
2. Add auto-fill fallback to replace any {exampleN}
3. Add validation to warn about unexpected placeholders

**Priority**:
- Fix #1 (Template): HIGH - solves at source
- Fix #2 (Auto-fill): MEDIUM - safety net
- Fix #3 (Validation): LOW - nice to have
