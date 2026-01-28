"""
Two-Stage Evaluator - Separate evaluation for Construction and Reasoning phases.

This evaluator provides granular feedback by scoring construction quality
and reasoning correctness independently, enabling better optimization insights.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from ..interfaces.evaluator import BaseEvaluator
from ..utils.construction_extractor import ConstructionExtractor

logger = logging.getLogger(__name__)


class TwoStageEvaluator(BaseEvaluator):
    """
    Evaluator that separately scores construction quality and reasoning correctness.

    Stage 1 (Construction): Evaluates 4 components (25% each)
        - Completeness: All 4 sections present?
        - Intent Alignment: References user's intent definition?
        - Richness: Detailed vs minimal problem decomposition?
        - Coherence: Parts logically connected?

    Stage 2 (Reasoning): Evaluates 3 components
        - Correctness (50%): Prediction matches groundtruth?
        - Intent Compliance (25%): Validates against user's intent?
        - Structured Flow (25%): Follows step structure?

    This two-stage approach helps identify whether optimization should focus on
    improving problem understanding (construction) or logical reasoning.
    """

    def __init__(
        self,
        feedback_column: str = "two_stage_feedback",
        construction_weight: float = 0.5,
        reasoning_weight: float = 0.5,
        groundtruth_column: str = "output",
        prediction_column: str = "prediction",
        construction_column: Optional[str] = None,
        reasoning_column: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize two-stage evaluator.

        Args:
            feedback_column: Primary feedback column name
            construction_weight: Weight for construction score (0-1)
            reasoning_weight: Weight for reasoning score (0-1)
            groundtruth_column: Column containing ground truth labels
            prediction_column: Column containing model predictions
            construction_column: Optional column with construction XML
            reasoning_column: Optional column with reasoning XML
            intent: Optional user intent definition for intent-aware scoring
        """
        super().__init__(feedback_column)
        self.construction_weight = construction_weight
        self.reasoning_weight = reasoning_weight
        self.groundtruth_column = groundtruth_column
        self.prediction_column = prediction_column
        self.construction_column = construction_column
        self.reasoning_column = reasoning_column
        self.intent = intent

        # Validate weights sum to 1.0
        if not abs((construction_weight + reasoning_weight) - 1.0) < 0.001:
            logger.warning(
                f"Weights don't sum to 1.0 ({construction_weight + reasoning_weight}). "
                "Normalizing..."
            )
            total = construction_weight + reasoning_weight
            self.construction_weight = construction_weight / total
            self.reasoning_weight = reasoning_weight / total

    async def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Evaluate dataframe with 2-stage scoring.

        Args:
            dataframe: DataFrame with predictions and optionally construction/reasoning
            **kwargs: Additional parameters (ignored)

        Returns:
            Tuple of (updated_dataframe, feedback_column_names)
        """
        df = dataframe.copy()

        # Initialize feedback columns
        construction_scores = []
        reasoning_scores = []
        overall_scores = []
        construction_feedbacks = []
        reasoning_feedbacks = []
        overall_correct_list = []

        for idx, row in df.iterrows():
            # Extract data
            prediction = str(row.get(self.prediction_column, ""))
            groundtruth = str(row.get(self.groundtruth_column, ""))
            construction_text = str(row.get(self.construction_column, "")) if self.construction_column else None
            reasoning_text = str(row.get(self.reasoning_column, "")) if self.reasoning_column else None

            # Evaluate construction
            construction_result = self.evaluate_construction(construction_text, self.intent)
            construction_scores.append(construction_result["score"])
            construction_feedbacks.append(construction_result["feedback"])

            # Evaluate reasoning
            reasoning_result = self.evaluate_reasoning(
                reasoning_text, prediction, groundtruth, self.intent
            )
            reasoning_scores.append(reasoning_result["score"])
            reasoning_feedbacks.append(reasoning_result["feedback"])

            # Overall score
            overall_score = (
                self.construction_weight * construction_result["score"] +
                self.reasoning_weight * reasoning_result["score"]
            )
            overall_scores.append(overall_score)

            # Overall correctness (based on prediction match)
            overall_correct = self._normalize_value(prediction) == self._normalize_value(groundtruth)
            overall_correct_list.append(overall_correct)

        # Add feedback columns
        df["construction_score"] = construction_scores
        df["construction_feedback"] = construction_feedbacks
        df["reasoning_score"] = reasoning_scores
        df["reasoning_feedback"] = reasoning_feedbacks
        df["overall_score"] = overall_scores
        df["overall_correct"] = overall_correct_list
        df[self.feedback_column] = df.apply(
            lambda row: self._format_combined_feedback(row), axis=1
        )

        feedback_columns = [
            "construction_score",
            "construction_feedback",
            "reasoning_score",
            "reasoning_feedback",
            "overall_score",
            "overall_correct",
            self.feedback_column,
        ]

        return df, feedback_columns

    def evaluate_construction(
        self, construction_text: Optional[str], intent: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate construction quality with 4 components (25% each).

        Components:
            1. Completeness (25%): All 4 sections present?
            2. Intent Alignment (25%): References user's intent?
            3. Richness (25%): Detailed vs minimal?
            4. Coherence (25%): Parts logically connected?

        Args:
            construction_text: Extracted construction from Phase 1
            intent: Optional user intent definition

        Returns:
            Dict with:
                - score: 0-1 score (weighted average of 4 components)
                - feedback: Detailed feedback string
                - breakdown: Individual component scores
                - metadata: Counts of each element found
        """
        if not construction_text or construction_text.strip() == "":
            return {
                "score": 0.0,
                "feedback": "No construction provided",
                "breakdown": {
                    "completeness": 0.0,
                    "intent_alignment": 0.0,
                    "richness": 0.0,
                    "coherence": 0.0,
                },
                "metadata": {
                    "entities_count": 0,
                    "state_variables_count": 0,
                    "actions_count": 0,
                    "constraints_count": 0,
                },
            }

        try:
            # Extract construction elements
            extracted = ConstructionExtractor.extract(construction_text)

            # Count elements
            entities_count = len(extracted.get("entities", []))
            state_vars_count = len(extracted.get("state_variables", []))
            actions_count = len(extracted.get("actions", []))
            constraints_count = len(extracted.get("constraints", []))

            # Component 1: Completeness (25%)
            completeness_score = self._score_completeness(
                entities_count, state_vars_count, actions_count, constraints_count
            )

            # Component 2: Intent Alignment (25%)
            intent_alignment_score = 0.0
            if intent:
                intent_alignment_score = self._score_intent_alignment(
                    construction_text, intent, extracted
                )

            # Component 3: Richness (25%)
            richness_score = self._score_richness(
                entities_count, state_vars_count, actions_count, constraints_count
            )

            # Component 4: Coherence (25%)
            coherence_score = self._score_coherence(construction_text, extracted)

            # Final weighted score
            score = (
                0.25 * completeness_score
                + 0.25 * intent_alignment_score
                + 0.25 * richness_score
                + 0.25 * coherence_score
            )

            # Generate feedback
            feedback_parts = [
                f"Completeness: {completeness_score:.2f}",
                f"Intent Alignment: {intent_alignment_score:.2f}",
                f"Richness: {richness_score:.2f}",
                f"Coherence: {coherence_score:.2f}",
            ]
            feedback = f"Construction: {score:.2f} | " + " | ".join(feedback_parts)

            return {
                "score": score,
                "feedback": feedback,
                "breakdown": {
                    "completeness": completeness_score,
                    "intent_alignment": intent_alignment_score,
                    "richness": richness_score,
                    "coherence": coherence_score,
                },
                "metadata": {
                    "entities_count": entities_count,
                    "state_variables_count": state_vars_count,
                    "actions_count": actions_count,
                    "constraints_count": constraints_count,
                    "extraction_confidence": extracted.get("metadata", {}).get(
                        "extraction_confidence", 0.0
                    ),
                },
            }

        except Exception as e:
            logger.warning(f"Construction extraction failed: {e}")
            return {
                "score": 0.0,
                "feedback": f"Construction extraction error: {str(e)}",
                "breakdown": {
                    "completeness": 0.0,
                    "intent_alignment": 0.0,
                    "richness": 0.0,
                    "coherence": 0.0,
                },
                "metadata": {
                    "entities_count": 0,
                    "state_variables_count": 0,
                    "actions_count": 0,
                    "constraints_count": 0,
                },
            }

    def evaluate_reasoning(
        self,
        reasoning_text: Optional[str],
        prediction: str,
        groundtruth: str,
        intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate reasoning with 3 components.

        Components:
            1. Correctness (50%): Prediction matches groundtruth?
            2. Intent Compliance (25%): Validates against user intent?
            3. Structured Flow (25%): Follows step structure?

        Args:
            reasoning_text: Extracted reasoning from Phase 2
            prediction: Model's final prediction
            groundtruth: Expected correct answer
            intent: Optional user intent definition

        Returns:
            Dict with:
                - score: 0-1 weighted score
                - feedback: Explanation of correctness
                - breakdown: Individual component scores
        """
        # Component 1: Correctness (50%)
        pred_normalized = self._normalize_value(prediction)
        truth_normalized = self._normalize_value(groundtruth)
        is_correct = pred_normalized == truth_normalized
        correctness_score = 1.0 if is_correct else 0.0

        # Component 2: Intent Compliance (25%)
        intent_compliance_score = 0.0
        if intent and reasoning_text and reasoning_text.strip():
            intent_compliance_score = self._score_intent_compliance(reasoning_text, intent)

        # Component 3: Structured Flow (25%)
        structured_flow_score = 0.0
        if reasoning_text and reasoning_text.strip():
            structured_flow_score = self._score_structured_flow(reasoning_text)

        # Final weighted score
        score = (
            0.5 * correctness_score
            + 0.25 * intent_compliance_score
            + 0.25 * structured_flow_score
        )

        # Generate feedback
        if is_correct:
            feedback = f"Reasoning: {score:.2f} | Correct (pred={prediction}, true={groundtruth})"
        else:
            feedback = f"Reasoning: {score:.2f} | Incorrect (pred={prediction}, true={groundtruth})"

        feedback += f" | Intent Compliance: {intent_compliance_score:.2f} | Flow: {structured_flow_score:.2f}"

        return {
            "score": score,
            "feedback": feedback,
            "breakdown": {
                "correctness": correctness_score,
                "intent_compliance": intent_compliance_score,
                "structured_flow": structured_flow_score,
            },
        }

    def _normalize_value(self, value: str) -> str:
        """
        Normalize value for comparison (handles True/False, yes/no, 1/0, etc.).

        Args:
            value: String value to normalize

        Returns:
            Normalized lowercase string
        """
        value_str = str(value).strip().lower()

        # Boolean mappings
        if value_str in ["true", "yes", "1", "correct"]:
            return "true"
        if value_str in ["false", "no", "0", "incorrect"]:
            return "false"

        return value_str

    # ==================== Construction Component Scorers ====================

    def _score_completeness(
        self, entities_count: int, state_vars_count: int, actions_count: int, constraints_count: int
    ) -> float:
        """
        Score construction completeness (0-1).

        Checks if all 4 required sections are present.

        Args:
            entities_count: Number of entities found
            state_vars_count: Number of state variables found
            actions_count: Number of actions found
            constraints_count: Number of constraints found

        Returns:
            Score: 0.0 (none) to 1.0 (all 4 sections present)
        """
        sections_present = sum([
            1 if entities_count > 0 else 0,
            1 if state_vars_count > 0 else 0,
            1 if actions_count > 0 else 0,
            1 if constraints_count > 0 else 0,
        ])
        return sections_present / 4.0

    def _score_intent_alignment(
        self, construction_text: str, intent: Dict[str, Any], extracted: Dict[str, Any]
    ) -> float:
        """
        Score intent alignment (0-1).

        Checks:
            1. Intent Reference (30%): Has <intent_reference> section?
            2. Intent-Grounded Entities (20%): Entities specific to user's concept?
            3. Intent Indicators in State (25%): State variables check intent criteria?
            4. Intent-Based Preconditions (25%): Action preconditions use intent?

        Args:
            construction_text: Raw construction text
            intent: User intent definition
            extracted: Extracted construction elements

        Returns:
            Score: 0.0 to 1.0
        """
        # Check 1: Intent Reference (30%)
        has_intent_reference = bool(
            re.search(r'<intent_reference>|user.*intent|user.*definition', construction_text, re.IGNORECASE)
        )
        intent_reference_score = 1.0 if has_intent_reference else 0.0

        # Check 2: Intent-Grounded Entities (20%)
        # entities is List[str]
        concept = intent.get("concept", "")
        entities = extracted.get("entities", [])
        entity_specificity = 0.0
        if entities and concept:
            # Check if any entity relates to the concept
            concept_mentioned = any(
                concept.lower() in str(entity).lower() for entity in entities
            )
            # Or if entities are specific (not just "Article", "Data", "Object")
            generic_entities = ["article", "data", "object", "input", "text"]
            has_specific_entities = any(
                str(entity).lower() not in generic_entities for entity in entities
            )
            if concept_mentioned or has_specific_entities:
                entity_specificity = 1.0
            else:
                entity_specificity = 0.5  # Some entities but generic

        # Check 3: Intent Indicators in State Variables (25%)
        # state_variables is List[Dict] with 'name' and 'type' keys
        state_vars = extracted.get("state_variables", [])
        state_var_alignment = 0.0
        if state_vars:
            # Check if state variables reference intent indicators
            positive_indicators = intent.get("positive_indicators", [])
            negative_indicators = intent.get("negative_indicators", [])
            all_indicators = positive_indicators + negative_indicators

            # Extract state variable names (handle both dict and str formats)
            state_names = []
            for var in state_vars:
                if isinstance(var, dict):
                    state_names.append(var.get('name', ''))
                else:
                    state_names.append(str(var))

            has_intent_state = any(
                any(indicator.lower() in name.lower() for indicator in all_indicators)
                for name in state_names
            )
            # Or has generic intent checking variables
            has_intent_check = any(
                "intent" in name.lower() or "matches" in name.lower() or "indicator" in name.lower()
                for name in state_names
            )
            if has_intent_state or has_intent_check:
                state_var_alignment = 1.0
            else:
                state_var_alignment = 0.3  # Has state vars but not intent-aligned

        # Check 4: Intent-Based Preconditions (25%)
        # actions is List[Dict] with 'name', 'preconditions', 'effects' keys
        actions = extracted.get("actions", [])
        precondition_intent_use = 0.0
        if actions:
            # Extract action text (handle both dict and str formats)
            action_texts = []
            for action in actions:
                if isinstance(action, dict):
                    action_texts.append(action.get('name', '') + ' ' + str(action.get('preconditions', '')))
                else:
                    action_texts.append(str(action))

            # Check if action preconditions mention intent-related state
            has_intent_precondition = any(
                "intent" in text.lower() or "matches" in text.lower() or "indicator" in text.lower()
                for text in action_texts
            )
            if has_intent_precondition:
                precondition_intent_use = 1.0
            else:
                precondition_intent_use = 0.4  # Has actions but not intent-based

        # Weighted score
        score = (
            0.3 * intent_reference_score
            + 0.2 * entity_specificity
            + 0.25 * state_var_alignment
            + 0.25 * precondition_intent_use
        )

        return score

    def _score_richness(
        self, entities_count: int, state_vars_count: int, actions_count: int, constraints_count: int
    ) -> float:
        """
        Score construction richness (0-1).

        Metrics:
            - Entity count: ≥3 good, ≥6 rich
            - State variable count: ≥4 adequate, ≥5 rich
            - Action count: ≥2 adequate, ≥3 rich
            - Constraint count: ≥2 adequate, ≥3 rich

        Args:
            entities_count: Number of entities
            state_vars_count: Number of state variables
            actions_count: Number of actions
            constraints_count: Number of constraints

        Returns:
            Score: 0.0 to 1.0
        """
        # Entity richness (25%)
        if entities_count >= 6:
            entity_richness = 1.0
        elif entities_count >= 3:
            entity_richness = 0.8
        elif entities_count >= 2:
            entity_richness = 0.5
        else:
            entity_richness = 0.3

        # State variable richness (25%)
        if state_vars_count >= 5:
            state_richness = 1.0
        elif state_vars_count >= 4:
            state_richness = 0.7
        elif state_vars_count >= 2:
            state_richness = 0.5
        else:
            state_richness = 0.3

        # Action richness (25%)
        if actions_count >= 3:
            action_richness = 1.0
        elif actions_count >= 2:
            action_richness = 0.8
        elif actions_count >= 1:
            action_richness = 0.5
        else:
            action_richness = 0.3

        # Constraint richness (25%)
        if constraints_count >= 3:
            constraint_richness = 1.0
        elif constraints_count >= 2:
            constraint_richness = 0.8
        elif constraints_count >= 1:
            constraint_richness = 0.5
        else:
            constraint_richness = 0.3

        # Weighted average
        score = (
            0.25 * entity_richness
            + 0.25 * state_richness
            + 0.25 * action_richness
            + 0.25 * constraint_richness
        )

        return score

    def _score_coherence(self, construction_text: str, extracted: Dict[str, Any]) -> float:
        """
        Score logical coherence (0-1).

        Checks:
            1. Entity-State Alignment (30%): State variables describe entities?
            2. Action-State Connection (30%): Actions reference state variables?
            3. Constraint Feasibility (20%): Constraints checkable with state?
            4. Precondition Validity (20%): Preconditions reference existing state?

        Args:
            construction_text: Raw construction text
            extracted: Extracted construction elements

        Returns:
            Score: 0.0 to 1.0
        """
        entities = extracted.get("entities", [])  # List[str]
        state_vars = extracted.get("state_variables", [])  # List[Dict]
        actions = extracted.get("actions", [])  # List[Dict]
        constraints = extracted.get("constraints", [])  # List[str]

        # Check 1: Entity-State Alignment (30%)
        entity_state_alignment = 0.8  # Default if we can't check
        if entities and state_vars:
            # Simple heuristic: do state variables mention entity keywords?
            entity_keywords = [str(e).lower().split()[0] for e in entities if e]

            # Extract state variable names
            state_names = []
            for var in state_vars:
                if isinstance(var, dict):
                    state_names.append(var.get('name', '').lower())
                else:
                    state_names.append(str(var).lower())

            state_mentions_entities = any(
                any(keyword in name for keyword in entity_keywords)
                for name in state_names
            )
            if state_mentions_entities:
                entity_state_alignment = 1.0
            else:
                entity_state_alignment = 0.6

        # Check 2: Action-State Connection (30%)
        action_state_connection = 0.8
        if actions and state_vars:
            # Extract state variable names (first part before ':')
            state_keywords = []
            for var in state_vars:
                if isinstance(var, dict):
                    name = var.get('name', '')
                    state_keywords.append(name.split(':')[0].lower())
                else:
                    state_keywords.append(str(var).split(':')[0].lower())

            # Extract action text
            action_texts = []
            for action in actions:
                if isinstance(action, dict):
                    action_texts.append(str(action.get('name', '')).lower())
                else:
                    action_texts.append(str(action).lower())

            actions_mention_state = any(
                any(keyword in text for keyword in state_keywords)
                for text in action_texts
            )
            if actions_mention_state:
                action_state_connection = 1.0
            else:
                action_state_connection = 0.6

        # Check 3: Constraint Feasibility (20%)
        constraint_feasibility = 0.8
        if constraints and state_vars:
            # Extract state variable names
            state_keywords = []
            for var in state_vars:
                if isinstance(var, dict):
                    name = var.get('name', '')
                    state_keywords.append(name.split(':')[0].lower())
                else:
                    state_keywords.append(str(var).split(':')[0].lower())

            constraints_mention_state = any(
                any(keyword in str(constraint).lower() for keyword in state_keywords)
                for constraint in constraints
            )
            if constraints_mention_state:
                constraint_feasibility = 1.0
            else:
                constraint_feasibility = 0.7

        # Check 4: Precondition Validity (20%)
        precondition_validity = 0.8
        if actions:
            # Check if preconditions exist and seem valid
            has_preconditions = False
            for action in actions:
                if isinstance(action, dict):
                    if action.get('preconditions'):
                        has_preconditions = True
                        break
                else:
                    if "precondition" in str(action).lower():
                        has_preconditions = True
                        break

            if has_preconditions:
                precondition_validity = 1.0
            else:
                precondition_validity = 0.6

        # Weighted score
        score = (
            0.3 * entity_state_alignment
            + 0.3 * action_state_connection
            + 0.2 * constraint_feasibility
            + 0.2 * precondition_validity
        )

        return score

    # ==================== Reasoning Component Scorers ====================

    def _score_intent_compliance(self, reasoning_text: str, intent: Dict[str, Any]) -> float:
        """
        Score intent compliance in reasoning (0-1).

        Checks:
            1. Intent Recall (30%): Mentions user's intent definition?
            2. Indicator Matching (40%): Checks against positive/negative indicators?
            3. Intent-Based Conclusion (30%): Final decision references intent?

        Args:
            reasoning_text: Reasoning text
            intent: User intent definition

        Returns:
            Score: 0.0 to 1.0
        """
        concept = intent.get("concept", "")
        definition = intent.get("definition", "")
        positive_indicators = intent.get("positive_indicators", [])
        negative_indicators = intent.get("negative_indicators", [])

        # Check 1: Intent Recall (30%)
        has_intent_recall = False
        if concept and concept.lower() in reasoning_text.lower():
            has_intent_recall = True
        # Or mentions "user defines" / "user's intent"
        if re.search(r"user.*defin|user.*intent|based on.*definition", reasoning_text, re.IGNORECASE):
            has_intent_recall = True
        intent_recall_score = 1.0 if has_intent_recall else 0.0

        # Check 2: Indicator Matching (40%)
        checks_indicators = False
        # Check if reasoning mentions positive or negative indicators
        all_indicators = positive_indicators + negative_indicators
        if all_indicators:
            indicators_mentioned = sum(
                1 for indicator in all_indicators
                if indicator.lower() in reasoning_text.lower()
            )
            if indicators_mentioned >= 2:
                checks_indicators = True
        # Or explicitly mentions "positive_indicators" / "negative_indicators"
        if re.search(r"positive.*indicator|negative.*indicator|matches.*indicator", reasoning_text, re.IGNORECASE):
            checks_indicators = True
        indicator_matching_score = 1.0 if checks_indicators else 0.0

        # Check 3: Intent-Based Conclusion (30%)
        has_intent_conclusion = False
        # Check final decision section
        final_section = reasoning_text.lower()
        if "final" in final_section or "conclusion" in final_section:
            # Does it reference intent?
            if "intent" in final_section or concept.lower() in final_section:
                has_intent_conclusion = True
        intent_conclusion_score = 1.0 if has_intent_conclusion else 0.0

        # Weighted score
        score = (
            0.3 * intent_recall_score
            + 0.4 * indicator_matching_score
            + 0.3 * intent_conclusion_score
        )

        return score

    def _score_structured_flow(self, reasoning_text: str) -> float:
        """
        Score structured flow (0-1).

        Checks for step structure:
            - Step 1 (25%)
            - Step 2 (25%)
            - Step 3 (25%)
            - Final/Conclusion (25%)

        Args:
            reasoning_text: Reasoning text

        Returns:
            Score: 0.0 to 1.0
        """
        has_step1 = bool(re.search(r'step\s*1', reasoning_text, re.IGNORECASE))
        has_step2 = bool(re.search(r'step\s*2', reasoning_text, re.IGNORECASE))
        has_step3 = bool(re.search(r'step\s*3', reasoning_text, re.IGNORECASE))
        has_final = bool(re.search(r'final|conclusion|decision', reasoning_text, re.IGNORECASE))

        steps_found = sum([has_step1, has_step2, has_step3, has_final])
        return steps_found / 4.0

    def _format_combined_feedback(self, row: pd.Series) -> str:
        """
        Format combined feedback from both stages.

        Args:
            row: DataFrame row with all feedback columns

        Returns:
            Formatted feedback string
        """
        construction_score = row.get("construction_score", 0.0)
        reasoning_score = row.get("reasoning_score", 0.0)
        overall_score = row.get("overall_score", 0.0)

        feedback_parts = [
            f"Overall Score: {overall_score:.2f}",
            f"Construction: {construction_score:.2f} - {row.get('construction_feedback', 'N/A')}",
            f"Reasoning: {reasoning_score:.2f} - {row.get('reasoning_feedback', 'N/A')}",
        ]

        return " | ".join(feedback_parts)

    def get_feedback_columns(self) -> List[str]:
        """
        Get list of all feedback columns this evaluator generates.

        Returns:
            List of column names
        """
        return [
            "construction_score",
            "construction_feedback",
            "reasoning_score",
            "reasoning_feedback",
            "overall_score",
            "overall_correct",
            self.feedback_column,
        ]


__all__ = ["TwoStageEvaluator"]
