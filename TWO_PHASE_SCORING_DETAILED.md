# 2-Phase Evaluation Scoring - Detailed Specification

## Overview

2-phase evaluation scores **construction quality** and **reasoning correctness** separately to identify where optimization should focus.

---

## Phase 1: Construction Quality Score

### What is Construction?

Construction decomposes a problem into 4 formal elements:

```xml
<construction>
  <intent_reference>
    User's intent: "Milestone = record-breaking achievement or historic numeric milestone"
  </intent_reference>

  (1) Relevant Entities:
      - Article (input)
      - Achievement (event)
      - Record (previous best)
      - Milestone_Type (enum: breaking_record | numeric_milestone | championship)

  (2) State Variables:
      - article_content: text
      - achievement_type: enum
      - has_record_keyword: boolean
      - has_numeric_milestone: boolean
      - matches_user_intent: boolean

  (3) Possible Actions:
      - check_record_keywords(article) → has_record_keyword
      - check_numeric_milestones(article) → has_numeric_milestone
      - evaluate_against_intent(achievement, intent) → matches_user_intent
      - classify_as_milestone
        Preconditions: matches_user_intent == True AND (has_record_keyword OR has_numeric_milestone)

  (4) Constraints:
      - Binary output only (True/False)
      - Must align with user's intent definition
      - If no positive_indicators found, classify as False
</construction>
```

---

## Construction Score Breakdown (0-1 scale)

### Component 1: Completeness Score (0.25 weight)

**What it measures**: Are all 4 required sections present?

**Calculation**:
```python
sections_found = 0
if has_entities: sections_found += 1
if has_state_variables: sections_found += 1
if has_actions: sections_found += 1
if has_constraints: sections_found += 1

completeness_score = sections_found / 4.0
```

**Examples**:
- All 4 sections present → 1.0
- Missing constraints → 0.75 (3/4)
- Only entities + state → 0.5 (2/4)
- Only entities → 0.25 (1/4)
- No sections → 0.0

**Feedback**:
- Score 1.0: "Construction complete. Found 4 entities, 5 state vars, 3 actions, 2 constraints"
- Score 0.75: "Construction incomplete. Missing: constraints"
- Score 0.5: "Construction incomplete. Missing: actions, constraints"

---

### Component 2: Intent Alignment Score (0.25 weight)

**What it measures**: Does construction reference and use user's intent?

**Checks**:
1. **Intent Reference Check** (0.3 weight):
   - Does construction have `<intent_reference>` section?
   - Does it mention the concept definition?
   - Example: "Based on user's definition: milestone = record-breaking achievement"

2. **Intent-Grounded Entities** (0.2 weight):
   - Do entities relate to user's intent?
   - Good: "Record, Achievement, Milestone_Type" (specific to milestone intent)
   - Bad: "Article, Data, Object" (too generic)

3. **Intent Indicators in State Variables** (0.25 weight):
   - Do state variables check intent criteria?
   - Good: `has_record_keyword: boolean`, `matches_user_intent: boolean`
   - Bad: `data: text`, `result: boolean` (no intent alignment)

4. **Intent-Based Preconditions** (0.25 weight):
   - Do action preconditions use intent indicators?
   - Good: "Preconditions: matches_user_intent == True AND has_record_keyword"
   - Bad: "Preconditions: data exists" (generic)

**Calculation**:
```python
intent_alignment_score = (
    0.3 * has_intent_reference +
    0.2 * entity_specificity +
    0.25 * state_var_alignment +
    0.25 * precondition_intent_use
)
```

**Examples**:
- Perfect intent grounding → 1.0
- Has intent reference but weak entity/state alignment → 0.6
- No intent reference, generic elements → 0.2

**Feedback**:
- Score 1.0: "Construction well-grounded in user's intent. All elements reference milestone definition."
- Score 0.6: "Construction mentions intent but state variables are too generic."
- Score 0.2: "Construction missing intent reference. Elements not aligned with user's definition."

---

### Component 3: Richness Score (0.25 weight)

**What it measures**: How detailed and specific is the construction?

**Metrics**:
1. **Entity Count** (threshold: ≥ 3 meaningful entities)
   - Too few (1-2): Score 0.3
   - Good (3-5): Score 0.8
   - Rich (6+): Score 1.0

2. **State Variable Detail** (threshold: ≥ 4 variables)
   - Minimal (1-2): Score 0.3
   - Adequate (3-4): Score 0.7
   - Rich (5+): Score 1.0

3. **Action Specificity**:
   - Generic actions (classify, decide): Score 0.4
   - Specific actions with preconditions: Score 0.8
   - Detailed actions with effects: Score 1.0

4. **Constraint Precision**:
   - Generic (binary output): Score 0.4
   - Specific (must check X before Y): Score 0.8
   - Precise with validation rules: Score 1.0

**Calculation**:
```python
richness_score = (
    0.25 * entity_richness +
    0.25 * state_var_richness +
    0.25 * action_specificity +
    0.25 * constraint_precision
)
```

**Examples**:
- Rich construction (6 entities, 5 states, detailed actions) → 0.9
- Adequate construction (3 entities, 3 states, basic actions) → 0.6
- Minimal construction (2 entities, 2 states, generic actions) → 0.3

**Feedback**:
- Score 0.9: "Rich construction with 6 entities, 5 state variables, and detailed action preconditions."
- Score 0.6: "Adequate construction but could be more detailed (only 3 entities, generic actions)."
- Score 0.3: "Minimal construction lacking detail (2 entities, 2 state variables)."

---

### Component 4: Logical Coherence Score (0.25 weight)

**What it measures**: Do all parts work together logically?

**Checks**:
1. **Entity-State Alignment** (0.3 weight):
   - Do state variables actually describe entities?
   - Example: If entity is "Record", should have state var "record_broken: boolean"

2. **Action-State Connection** (0.3 weight):
   - Do actions modify or check state variables?
   - Example: Action "check_record_keywords()" → updates "has_record_keyword"

3. **Constraint Feasibility** (0.2 weight):
   - Can constraints be checked with available state variables?
   - Example: Constraint "Must have record keyword" ← needs "has_record_keyword" state

4. **Precondition Validity** (0.2 weight):
   - Do preconditions reference existing state variables?
   - Example: "Preconditions: has_record_keyword == True" ← valid if state exists

**Calculation**:
```python
coherence_score = (
    0.3 * entity_state_alignment +
    0.3 * action_state_connection +
    0.2 * constraint_feasibility +
    0.2 * precondition_validity
)
```

**Examples**:
- Fully coherent (all parts connected) → 1.0
- Mostly coherent (minor gaps) → 0.7
- Incoherent (disconnected parts) → 0.3

**Feedback**:
- Score 1.0: "Construction logically coherent. All entities, states, actions, and constraints are properly connected."
- Score 0.7: "Construction mostly coherent but action preconditions reference undefined state variables."
- Score 0.3: "Construction incoherent. State variables don't relate to entities, actions are disconnected."

---

## Final Construction Score

```python
construction_score = (
    0.25 * completeness_score +
    0.25 * intent_alignment_score +
    0.25 * richness_score +
    0.25 * logical_coherence_score
)
```

**Weight justification**:
- **Completeness (25%)**: Must have all sections
- **Intent Alignment (25%)**: CRITICAL - must match user's definition
- **Richness (25%)**: More detail = better problem understanding
- **Coherence (25%)**: Parts must work together logically

---

## Phase 2: Reasoning Correctness Score

### What is Reasoning?

Reasoning is the step-by-step logic that leads to a prediction:

```xml
<think>
  Step 1: Recall user intent
  - User defines milestone as: record-breaking achievement or historic numeric milestone

  Step 2: Extract article facts
  - Event: Player scores 58th goal, breaking team record
  - Keywords found: "scores", "58th", "breaking", "record"
  - Numeric indicators: "58th goal" (numeric milestone)

  Step 3: Match facts against intent definition
  - Does this match positive_indicators? YES (has "record", "58th")
  - Does this match negative_indicators? NO
  - Boundary case: N/A

  Step 4: Apply construction logic
  - has_record_keyword: True (found "breaking", "record")
  - has_numeric_milestone: True (found "58th")
  - matches_user_intent: True (both indicators present)

  Step 5: Final decision
  - Classification: True
  - Confidence: high
  - Reasoning: This is a milestone because it matches both positive indicators
               (record-breaking + numeric milestone) from user's intent definition
</think>
```

---

## Reasoning Score Breakdown (0-1 scale)

### Component 1: Correctness (0.5 weight)

**What it measures**: Is the final prediction correct?

**Calculation**:
```python
if prediction == groundtruth:
    correctness_score = 1.0
else:
    correctness_score = 0.0
```

**Binary**: Either correct (1.0) or incorrect (0.0)

**Feedback**:
- Score 1.0: "Reasoning correct. Predicted: True, Expected: True"
- Score 0.0: "Reasoning incorrect. Predicted: True, Expected: False"

---

### Component 2: Intent Compliance (0.25 weight)

**What it measures**: Does reasoning validate against user's intent?

**Checks**:
1. **Intent Recall** (0.3 weight):
   - Does Step 1 mention user's intent definition?
   - Good: "User defines milestone as record-breaking achievement"
   - Bad: Missing or generic

2. **Indicator Matching** (0.4 weight):
   - Does Step 3 check facts against intent's positive/negative indicators?
   - Good: "Matches positive_indicators: has 'record', '58th'"
   - Bad: No explicit matching

3. **Intent-Based Conclusion** (0.3 weight):
   - Does final decision reference intent alignment?
   - Good: "This matches user's intent because..."
   - Bad: "Classification: True" (no justification)

**Calculation**:
```python
intent_compliance_score = (
    0.3 * has_intent_recall +
    0.4 * checks_indicators +
    0.3 * intent_based_conclusion
)
```

**Examples**:
- Full compliance (all 3 checks) → 1.0
- Partial (intent recall + conclusion, no indicator check) → 0.6
- No compliance → 0.0

**Feedback**:
- Score 1.0: "Reasoning fully complies with user intent. Explicitly checks against positive/negative indicators."
- Score 0.6: "Reasoning mentions intent but doesn't explicitly match against indicators."
- Score 0.0: "Reasoning doesn't reference user's intent definition."

---

### Component 3: Structured Flow (0.25 weight)

**What it measures**: Does reasoning follow a clear step structure?

**Expected Structure**:
- Step 1: Recall user intent
- Step 2: Extract facts
- Step 3: Match against intent
- Step 4: Apply construction logic
- Step 5: Final decision

**Checks**:
```python
has_step1 = bool(re.search(r'step\s*1', reasoning, re.IGNORECASE))
has_step2 = bool(re.search(r'step\s*2', reasoning, re.IGNORECASE))
has_step3 = bool(re.search(r'step\s*3', reasoning, re.IGNORECASE))
has_final = bool(re.search(r'final|conclusion|decision', reasoning, re.IGNORECASE))

steps_found = sum([has_step1, has_step2, has_step3, has_final])
structured_flow_score = steps_found / 4.0
```

**Examples**:
- All 4 steps present → 1.0
- 3 steps present → 0.75
- 2 steps present → 0.5
- No structure → 0.0

**Feedback**:
- Score 1.0: "Reasoning follows structured flow (Step 1, Step 2, Step 3, Final)."
- Score 0.75: "Reasoning mostly structured but missing explicit Step 3."
- Score 0.0: "Reasoning lacks clear step structure."

---

## Final Reasoning Score

```python
reasoning_score = (
    0.5 * correctness_score +
    0.25 * intent_compliance_score +
    0.25 * structured_flow_score
)
```

**Weight justification**:
- **Correctness (50%)**: Most important - is the answer right?
- **Intent Compliance (25%)**: Does it align with user's definition?
- **Structured Flow (25%)**: Clear reasoning is better than implicit

---

## Overall Score

```python
overall_score = (
    construction_weight * construction_score +
    reasoning_weight * reasoning_score
)

# Default: 50/50 weighting
overall_score = 0.5 * construction_score + 0.5 * reasoning_score
```

---

## Complete Example

### Input

**System Prompt**: "classify this article contain milestone or not"

**Article**: "Player scores 58th goal, breaking team record in dominant victory"

**Groundtruth**: True

**Construction**:
```
<intent_reference>User's intent: milestone = record-breaking achievement</intent_reference>
(1) Entities: Article, Record, Achievement
(2) State Variables: has_record_keyword: boolean, matches_intent: boolean
(3) Actions: check_record_keywords(), Preconditions: has_record_keyword == True
(4) Constraints: Must align with user's intent
```

**Reasoning**:
```
Step 1: User defines milestone as record-breaking achievement
Step 2: Article mentions "breaking team record"
Step 3: Matches positive_indicators (has "record", "breaking")
Final: True (matches user's intent)
```

**Prediction**: True

---

### Scoring

#### Construction Score

**Completeness**: 1.0 (4/4 sections)
**Intent Alignment**: 0.8
  - Has intent reference (0.3): ✓
  - Intent-grounded entities (0.2): 0.7 (good but could be richer)
  - Intent indicators in state (0.25): ✓
  - Intent-based preconditions (0.25): ✓
  - Subtotal: 0.3×1.0 + 0.2×0.7 + 0.25×1.0 + 0.25×1.0 = 0.84

**Richness**: 0.6
  - Entity count (3): 0.7
  - State var count (2): 0.5
  - Action specificity: 0.6
  - Constraint precision: 0.5
  - Subtotal: 0.25×0.7 + 0.25×0.5 + 0.25×0.6 + 0.25×0.5 = 0.58

**Coherence**: 0.9
  - Entity-state alignment: 1.0
  - Action-state connection: 1.0
  - Constraint feasibility: 0.8
  - Precondition validity: 0.8
  - Subtotal: 0.3×1.0 + 0.3×1.0 + 0.2×0.8 + 0.2×0.8 = 0.92

**Final Construction Score**:
```
0.25 × 1.0 + 0.25 × 0.84 + 0.25 × 0.58 + 0.25 × 0.92 = 0.835
```

---

#### Reasoning Score

**Correctness**: 1.0 (Predicted True, Expected True)

**Intent Compliance**: 0.9
  - Intent recall (0.3): ✓ (mentions user's definition)
  - Indicator matching (0.4): ✓ (checks positive_indicators)
  - Intent-based conclusion (0.3): ✓ (references intent)
  - Subtotal: 0.3×1.0 + 0.4×1.0 + 0.3×1.0 = 1.0

**Structured Flow**: 0.75 (has Step 1, Step 2, Step 3, Final but missing Step 4)

**Final Reasoning Score**:
```
0.5 × 1.0 + 0.25 × 1.0 + 0.25 × 0.75 = 0.9375
```

---

### Overall Score

```
Overall = 0.5 × 0.835 + 0.5 × 0.9375 = 0.886
```

---

## Detailed Feedback Output

```python
{
    "construction_score": 0.835,
    "construction_feedback": "Construction: 0.84/1.0 | Completeness: 1.0 (4/4 sections) | Intent Alignment: 0.84 (well-grounded) | Richness: 0.58 (could be more detailed) | Coherence: 0.92 (logically connected)",

    "construction_breakdown": {
        "completeness": 1.0,
        "intent_alignment": 0.84,
        "richness": 0.58,
        "coherence": 0.92
    },

    "reasoning_score": 0.9375,
    "reasoning_feedback": "Reasoning: 0.94/1.0 | Correctness: 1.0 (Correct prediction) | Intent Compliance: 1.0 (fully aligned with user's definition) | Structured Flow: 0.75 (mostly structured)",

    "reasoning_breakdown": {
        "correctness": 1.0,
        "intent_compliance": 1.0,
        "structured_flow": 0.75
    },

    "overall_score": 0.886,
    "overall_correct": true,

    "two_stage_feedback": "Overall: 0.89/1.0 | Construction focuses on problem decomposition quality (intent alignment, richness, coherence). Reasoning focuses on prediction correctness and intent compliance."
}
```

---

## Key Insights

### Why Separate Scores Matter

**Scenario 1**: High construction, low reasoning
```
Construction: 0.9 (excellent problem decomposition)
Reasoning: 0.4 (wrong conclusion despite good setup)
→ Focus optimization on: Reasoning logic, not construction
```

**Scenario 2**: Low construction, high reasoning
```
Construction: 0.4 (poor problem decomposition)
Reasoning: 0.9 (somehow got right answer)
→ Focus optimization on: Construction quality (got lucky on reasoning)
```

**Scenario 3**: Both low
```
Construction: 0.3
Reasoning: 0.3
→ Major rework needed on both phases
```

**Scenario 4**: Both high
```
Construction: 0.9
Reasoning: 0.95
→ System working well, minor tuning only
```

---

## Comparison: Before vs After Intent-Aware

### Before (No Intent)

**Construction Score**: 0.75
- Completeness: 1.0 (4/4 sections)
- Intent Alignment: **0.0** (no intent reference)
- Richness: 0.6
- Coherence: 0.8
- **Total**: 0.25×1.0 + 0.25×0.0 + 0.25×0.6 + 0.25×0.8 = **0.60**

**Reasoning Score**: 0.5
- Correctness: 1.0
- Intent Compliance: **0.0** (no intent check)
- Structured Flow: 0.5
- **Total**: 0.5×1.0 + 0.25×0.0 + 0.25×0.5 = **0.625**

**Overall**: 0.5×0.60 + 0.5×0.625 = **0.613**

---

### After (Intent-Aware)

**Construction Score**: 0.84
- Completeness: 1.0
- Intent Alignment: **0.84** (well-grounded)
- Richness: 0.58
- Coherence: 0.92
- **Total**: **0.835**

**Reasoning Score**: 0.94
- Correctness: 1.0
- Intent Compliance: **1.0** (fully aligned)
- Structured Flow: 0.75
- **Total**: **0.9375**

**Overall**: 0.5×0.835 + 0.5×0.9375 = **0.886**

**Improvement**: +0.273 points (+44.5% relative improvement)

---

## Implementation in Code

```python
class TwoStageEvaluator:
    def evaluate_construction(self, construction_text, intent=None):
        # Component 1: Completeness
        completeness = self._score_completeness(construction_text)

        # Component 2: Intent Alignment (NEW)
        if intent:
            intent_alignment = self._score_intent_alignment(construction_text, intent)
        else:
            intent_alignment = 0.0  # Penalize if no intent

        # Component 3: Richness
        richness = self._score_richness(construction_text)

        # Component 4: Coherence
        coherence = self._score_coherence(construction_text)

        # Final score
        score = (
            0.25 * completeness +
            0.25 * intent_alignment +
            0.25 * richness +
            0.25 * coherence
        )

        return {
            "score": score,
            "breakdown": {
                "completeness": completeness,
                "intent_alignment": intent_alignment,
                "richness": richness,
                "coherence": coherence
            },
            "feedback": self._generate_construction_feedback(...)
        }

    def evaluate_reasoning(self, reasoning_text, prediction, groundtruth, intent=None):
        # Component 1: Correctness
        correctness = 1.0 if self._normalize(prediction) == self._normalize(groundtruth) else 0.0

        # Component 2: Intent Compliance (NEW)
        if intent:
            intent_compliance = self._score_intent_compliance(reasoning_text, intent)
        else:
            intent_compliance = 0.0

        # Component 3: Structured Flow
        structured_flow = self._score_structured_flow(reasoning_text)

        # Final score
        score = (
            0.5 * correctness +
            0.25 * intent_compliance +
            0.25 * structured_flow
        )

        return {
            "score": score,
            "breakdown": {
                "correctness": correctness,
                "intent_compliance": intent_compliance,
                "structured_flow": structured_flow
            },
            "feedback": self._generate_reasoning_feedback(...)
        }
```
