# Intent-Aware Construction System - Design Document

## Problem Statement

Current 2-phase system (Construction → Reasoning) lacks **user intent understanding**.

**Current Flow**:
```
Input → Construction (entities, state, actions, constraints) → Reasoning → Prediction
         ❌ Missing: What does "milestone" mean to THIS user?
```

**Issue**: Construction generates generic elements without grounding in user's specific definition.

## Solution: 3-Phase System with Intent

### Phase 0: Intent Extraction
**Goal**: Understand what the user means by the classification concept.

**Input**:
- System prompt: "classify this article contain milestone or not"
- Optional: Few-shot examples with labels

**Process**:
Extract intent by analyzing:
1. **System prompt keywords**: "milestone", "achievement", "record"
2. **Labeled examples pattern**: What do True samples have in common?
3. **Domain context**: Sports, business, technology, etc.

**Output**:
```xml
<intent>
  <concept>milestone</concept>
  <definition>
    A milestone in sports is a significant achievement that marks:
    - Breaking a record (personal or league-wide)
    - Reaching a historic numeric milestone (e.g., 50th goal, 1000th point)
    - First-time accomplishments
    - Championship victories
  </definition>
  <positive_indicators>
    - Keywords: "record", "historic", "first time", "championship"
    - Numeric achievements: "50th", "100th", "1000th"
    - Superlatives: "fastest", "most", "youngest"
  </positive_indicators>
  <negative_indicators>
    - Routine game results without records
    - Contract signings (unless historic value)
    - Injuries, trades, announcements
  </negative_indicators>
  <boundary_cases>
    - Consecutive wins: Only if breaking a streak record
    - High scores: Only if personal/team record
  </boundary_cases>
</intent>
```

### Phase 1: Intent-Grounded Construction
**Modification**: Construction must reference intent.

**Before** (generic):
```
(1) Entities: Article, Milestone, Record
(2) State Variables: contains_milestone: boolean
(3) Actions: classify_as_milestone
(4) Constraints: Binary output
```

**After** (intent-grounded):
```
<construction>
  <intent_reference>
    Based on user's definition: "milestone = record-breaking achievement or historic numeric milestone"
  </intent_reference>

  (1) Relevant Entities:
      - Article (input text)
      - Achievement (event described)
      - Record (previous best, if mentioned)
      - Milestone_Type (breaking_record | numeric_milestone | first_time | championship)

  (2) State Variables:
      - article_content: text
      - achievement_type: enum (record_break, numeric_milestone, championship, other)
      - has_record_keyword: boolean (checks "record", "historic", "first")
      - has_numeric_milestone: boolean (checks "50th", "100th", etc.)
      - matches_user_intent: boolean (does this fit user's milestone definition?)
      - contains_milestone: boolean (final classification)

  (3) Possible Actions:
      - extract_achievement_type(article) → achievement_type
      - check_record_keywords(article) → has_record_keyword
      - check_numeric_milestones(article) → has_numeric_milestone
      - evaluate_against_intent(achievement, user_intent) → matches_user_intent
      - classify_as_milestone
        Preconditions: matches_user_intent == True AND
                      (has_record_keyword OR has_numeric_milestone)

  (4) Constraints:
      - Classification MUST align with user's intent definition
      - If article doesn't match any positive_indicators from intent, classify as False
      - If article matches negative_indicators from intent, classify as False
      - Binary output only (True/False)
</construction>
```

### Phase 2: Intent-Aware Reasoning
**Modification**: Reasoning must validate against intent.

**Structure**:
```xml
<think>
  Step 1: Recall user intent
  - User defines milestone as: [extract from Phase 0]

  Step 2: Extract article facts
  - Event: [what happened?]
  - Keywords found: [record, historic, first, etc.]
  - Numeric indicators: [50th goal, 1000th point, etc.]

  Step 3: Match facts against intent definition
  - Does this event match positive_indicators? [Yes/No + reasoning]
  - Does this event match negative_indicators? [Yes/No + reasoning]
  - Boundary case check: [if applicable]

  Step 4: Apply construction logic
  - has_record_keyword: [True/False]
  - has_numeric_milestone: [True/False]
  - matches_user_intent: [True/False based on Step 3]

  Step 5: Final decision
  - Classification: [True/False]
  - Confidence: [high/medium/low]
  - Reasoning: This [does/does not] match user's intent because [specific alignment with definition]
</think>
```

## Implementation Changes

### 1. Add IntentExtractor Component

**File**: `src/chaos_auto_prompt/utils/intent_extractor.py`

```python
class IntentExtractor:
    """
    Extract user intent from system prompt and examples.
    """

    @staticmethod
    async def extract_intent(
        system_prompt: str,
        examples: pd.DataFrame,
        provider: BaseProvider
    ) -> Dict[str, Any]:
        """
        Extract user's definition of the classification concept.

        Args:
            system_prompt: User's classification instruction
            examples: Few-shot examples with labels
            provider: LLM provider for intent analysis

        Returns:
            {
                "concept": str,  # e.g., "milestone"
                "definition": str,
                "positive_indicators": List[str],
                "negative_indicators": List[str],
                "boundary_cases": List[str]
            }
        """
```

### 2. Modify MetaPrompt Template

**File**: `src/chaos_auto_prompt/optimizers/meta_prompt.py`

Add intent section to construction prompt:

```python
INTENT_AWARE_CONSTRUCTION_TEMPLATE = """
PHASE 0: USER INTENT
{intent_definition}

PHASE 1: CONSTRUCTION
Given the user's intent above, create a construction that GROUNDS all elements in this specific definition.

Your construction MUST:
1. Reference the user's intent explicitly in each section
2. Design state variables that check alignment with intent
3. Define actions with preconditions based on intent indicators
4. Add constraints that enforce intent compliance

[Rest of construction instructions...]
"""
```

### 3. Update TwoStageEvaluator

**File**: `src/chaos_auto_prompt/evaluators/two_stage.py`

Add intent alignment scoring:

```python
def evaluate_construction(self, construction_text: Optional[str], intent: Optional[Dict] = None):
    """
    Evaluate construction with intent alignment check.
    """

    # Existing completeness check
    sections_score = sections_present / 4.0

    # NEW: Intent alignment check
    if intent:
        intent_score = self._evaluate_intent_alignment(construction_text, intent)
        # Weight: 50% completeness + 50% intent alignment
        final_score = 0.5 * sections_score + 0.5 * intent_score
    else:
        final_score = sections_score

    return {"score": final_score, ...}

def _evaluate_intent_alignment(self, construction: str, intent: Dict) -> float:
    """
    Score how well construction references user intent.

    Checks:
    - Does construction mention the concept definition?
    - Do state variables check intent indicators?
    - Do action preconditions use intent criteria?
    - Do constraints enforce intent compliance?
    """
```

### 4. Create IntentAwarePromptOptimizer

**File**: `src/chaos_auto_prompt/optimizers/intent_aware_optimizer.py`

```python
class IntentAwarePromptOptimizer(PromptLearningOptimizer):
    """
    Prompt optimizer with intent extraction phase.
    """

    async def optimize(self, dataset, system_prompt, ...):
        # Phase 0: Extract intent
        intent = await self.intent_extractor.extract_intent(
            system_prompt=system_prompt,
            examples=dataset.head(5),  # Use first 5 for intent analysis
            provider=self.provider
        )

        # Phase 1: Generate intent-grounded construction
        construction = await self._generate_construction_with_intent(
            problem=row['input'],
            intent=intent
        )

        # Phase 2: Generate intent-aware reasoning
        reasoning = await self._generate_reasoning_with_intent(
            construction=construction,
            intent=intent
        )

        # Evaluate with intent alignment
        evaluation = await self.evaluator.evaluate(
            ...,
            intent=intent  # Pass intent for scoring
        )
```

## Summary of Changes

### Files to Modify:
1. ✅ `src/chaos_auto_prompt/utils/intent_extractor.py` (NEW)
2. ✅ `src/chaos_auto_prompt/optimizers/meta_prompt.py` (MODIFY)
3. ✅ `src/chaos_auto_prompt/evaluators/two_stage.py` (MODIFY)
4. ✅ `src/chaos_auto_prompt/optimizers/intent_aware_optimizer.py` (NEW)

### Evaluation Changes:
- Construction score = 50% completeness + 50% intent alignment
- Reasoning score = 50% correctness + 50% intent compliance
- New feedback: "Construction missing intent reference" vs "Construction well-grounded in user intent"

## Example Output Comparison

### Before (No Intent):
```
Construction Score: 1.0 (all 4 sections present)
But: Generic entities like "Article, Milestone" - doesn't understand user's specific definition
```

### After (Intent-Aware):
```
Construction Score: 0.95
- Completeness: 1.0 (4/4 sections)
- Intent Alignment: 0.9 (references user's definition, uses intent indicators)
Feedback: "Construction well-grounded in user's milestone definition (record-breaking + numeric milestones)"
```

## Benefits

1. ✅ **Context-Specific**: Construction tailored to user's exact definition
2. ✅ **Explainable**: Clear why something is/isn't a milestone per user's intent
3. ✅ **Transferable**: Same framework works for any classification task
4. ✅ **Evaluatable**: Can measure intent alignment objectively

## Next Steps

1. Implement IntentExtractor
2. Update meta-prompt templates
3. Modify TwoStageEvaluator scoring
4. Create intent-aware optimizer
5. Test with milestone dataset
6. Compare results: generic vs intent-aware construction
