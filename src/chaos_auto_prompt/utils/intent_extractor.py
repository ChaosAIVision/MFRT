"""
Intent Extractor - Extract user's concept definition from system prompt and examples.

This component analyzes the user's classification task to understand what the concept
(e.g., "milestone", "spam", "urgent") means in their specific context.
"""

import logging
import re
from typing import Dict, Any, List, Optional
import pandas as pd

from ..providers.base import BaseProvider

logger = logging.getLogger(__name__)


class IntentExtractor:
    """
    Extract user intent from system prompt and labeled examples.

    This is Phase 0 in the intent-aware 3-phase system.
    Understanding user intent enables grounded construction and reasoning.

    Example:
        User prompt: "classify this article contain milestone or not"
        Examples: 5 samples with True/False labels

        Output:
        {
            "concept": "milestone",
            "definition": "A significant achievement that marks record-breaking...",
            "positive_indicators": ["record", "historic", "first time", "50th"],
            "negative_indicators": ["routine game", "contract signing"],
            "boundary_cases": ["consecutive wins (only if record)"]
        }
    """

    def __init__(self, provider: BaseProvider, model: Optional[str] = None):
        """
        Initialize intent extractor.

        Args:
            provider: LLM provider for intent analysis
            model: Optional model override
        """
        self.provider = provider
        self.model = model

    async def extract_intent(
        self,
        system_prompt: str,
        examples: pd.DataFrame,
        output_column: str = "output",
        input_column: str = "input",
        max_examples: int = 10,
    ) -> Dict[str, Any]:
        """
        Extract user's definition of the classification concept.

        Args:
            system_prompt: User's classification instruction
            examples: DataFrame with labeled examples
            output_column: Column name for labels
            input_column: Column name for inputs
            max_examples: Maximum number of examples to analyze

        Returns:
            {
                "concept": str,  # e.g., "milestone"
                "definition": str,
                "positive_indicators": List[str],
                "negative_indicators": List[str],
                "boundary_cases": List[str],
                "confidence": float,  # 0-1
                "metadata": {
                    "examples_analyzed": int,
                    "true_count": int,
                    "false_count": int,
                }
            }

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")

        if examples is None or len(examples) == 0:
            raise ValueError("examples cannot be empty")

        if output_column not in examples.columns:
            raise ValueError(f"output_column '{output_column}' not found in examples")

        if input_column not in examples.columns:
            raise ValueError(f"input_column '{input_column}' not found in examples")

        # Extract concept from system prompt
        concept = self._extract_concept_name(system_prompt)

        # Sample examples (balanced if possible)
        sampled_examples = self._sample_balanced_examples(
            examples, output_column, max_examples
        )

        # Build intent extraction prompt
        intent_prompt = self._build_intent_prompt(
            system_prompt, sampled_examples, input_column, output_column
        )

        # Call LLM to extract intent
        logger.info(f"Extracting intent for concept: '{concept}'")
        logger.debug(f"Intent extraction prompt length: {len(intent_prompt)} chars")

        messages = [{"role": "user", "content": intent_prompt}]

        try:
            response = await self.provider.generate_text(
                messages=messages,
                temperature=0.3,  # Low temperature for consistent extraction
                model=self.model,
            )

            # Parse response
            intent = self._parse_intent_response(response, concept)

            # Add metadata
            intent["metadata"] = {
                "examples_analyzed": len(sampled_examples),
                "true_count": sampled_examples[output_column].sum(),
                "false_count": len(sampled_examples) - sampled_examples[output_column].sum(),
                "system_prompt": system_prompt,
            }

            logger.info(f"Intent extracted successfully for '{concept}'")
            logger.debug(f"Definition: {intent['definition'][:100]}...")

            return intent

        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            # Return fallback intent
            return self._create_fallback_intent(concept, system_prompt)

    def _extract_concept_name(self, system_prompt: str) -> str:
        """
        Extract the main concept name from system prompt.

        Examples:
            "classify this article contain milestone or not" → "milestone"
            "determine if email is spam" → "spam"
            "check if task is urgent" → "urgent"
        """
        prompt_lower = system_prompt.lower()

        # Common patterns
        patterns = [
            r'contain[s]?\s+(\w+)',  # "contains milestone"
            r'is\s+(\w+)',           # "is spam"
            r'classify.*?(\w+)',     # "classify milestone"
            r'determine.*?(\w+)',    # "determine spam"
            r'check.*?(\w+)',        # "check urgent"
        ]

        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                concept = match.group(1)
                # Filter out common words
                if concept not in ['a', 'an', 'the', 'this', 'that', 'if', 'or', 'not']:
                    return concept

        # Fallback: extract noun-like words
        words = re.findall(r'\b[a-z]{4,}\b', prompt_lower)
        if words:
            return words[0]

        return "concept"

    def _sample_balanced_examples(
        self,
        examples: pd.DataFrame,
        output_column: str,
        max_examples: int
    ) -> pd.DataFrame:
        """
        Sample examples with balanced True/False if possible.

        Args:
            examples: Full example set
            output_column: Label column
            max_examples: Maximum to sample

        Returns:
            Balanced sample
        """
        # Count True/False
        true_examples = examples[examples[output_column] == True]
        false_examples = examples[examples[output_column] == False]

        true_count = len(true_examples)
        false_count = len(false_examples)

        # Calculate balanced sample size
        samples_per_class = max_examples // 2

        # Sample from each class
        sampled_true = true_examples.head(min(samples_per_class, true_count))
        sampled_false = false_examples.head(min(samples_per_class, false_count))

        # Combine
        sampled = pd.concat([sampled_true, sampled_false], ignore_index=True)

        # If still under max, add more from larger class
        if len(sampled) < max_examples:
            remaining = max_examples - len(sampled)
            if true_count > samples_per_class:
                extra_true = true_examples.iloc[samples_per_class:samples_per_class + remaining]
                sampled = pd.concat([sampled, extra_true], ignore_index=True)
            elif false_count > samples_per_class:
                extra_false = false_examples.iloc[samples_per_class:samples_per_class + remaining]
                sampled = pd.concat([sampled, extra_false], ignore_index=True)

        return sampled

    def _build_intent_prompt(
        self,
        system_prompt: str,
        examples: pd.DataFrame,
        input_column: str,
        output_column: str
    ) -> str:
        """Build prompt for LLM to extract intent."""

        # Format examples
        examples_text = []
        for idx, row in examples.iterrows():
            label = "TRUE" if row[output_column] else "FALSE"
            input_text = str(row[input_column])[:200]  # Truncate long inputs
            examples_text.append(f"Example {idx + 1} (Label: {label}):\n{input_text}\n")

        examples_str = "\n".join(examples_text)

        prompt = f"""You are an intent analysis expert. Your task is to understand what the user means by their classification concept.

**User's Classification Task**:
"{system_prompt}"

**Labeled Examples**:
{examples_str}

**Your Task**:
Analyze the system prompt and examples to extract the user's specific definition of the classification concept.

Output your analysis in this exact XML format:

<intent>
  <concept>The main concept being classified (e.g., milestone, spam, urgent)</concept>
  <definition>A clear, specific definition of what this concept means to THIS user based on their examples. Be concrete and detailed.</definition>
  <positive_indicators>
    <indicator>Keyword, pattern, or characteristic that indicates TRUE label</indicator>
    <indicator>Another positive indicator</indicator>
    <indicator>...</indicator>
  </positive_indicators>
  <negative_indicators>
    <indicator>Keyword, pattern, or characteristic that indicates FALSE label</indicator>
    <indicator>Another negative indicator</indicator>
    <indicator>...</indicator>
  </negative_indicators>
  <boundary_cases>
    <case>Describe edge cases or ambiguous scenarios with clarification</case>
    <case>Another boundary case</case>
  </boundary_cases>
  <confidence>Your confidence in this intent extraction (0.0-1.0)</confidence>
</intent>

**Guidelines**:
1. Study BOTH the TRUE and FALSE examples to understand the distinction
2. Identify concrete keywords, numeric patterns, or structural features
3. Make the definition specific to this user's context, not generic
4. List 3-5 positive indicators and 3-5 negative indicators minimum
5. Consider boundary cases where classification might be ambiguous

Output ONLY the XML, nothing else."""

        return prompt

    def _parse_intent_response(self, response: str, concept_fallback: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured intent."""

        try:
            # Extract concept
            concept_match = re.search(r'<concept>(.*?)</concept>', response, re.DOTALL)
            concept = concept_match.group(1).strip() if concept_match else concept_fallback

            # Extract definition
            def_match = re.search(r'<definition>(.*?)</definition>', response, re.DOTALL)
            definition = def_match.group(1).strip() if def_match else "No definition extracted"

            # Extract positive indicators
            pos_indicators = []
            pos_section = re.search(r'<positive_indicators>(.*?)</positive_indicators>', response, re.DOTALL)
            if pos_section:
                indicators = re.findall(r'<indicator>(.*?)</indicator>', pos_section.group(1), re.DOTALL)
                pos_indicators = [ind.strip() for ind in indicators if ind.strip()]

            # Extract negative indicators
            neg_indicators = []
            neg_section = re.search(r'<negative_indicators>(.*?)</negative_indicators>', response, re.DOTALL)
            if neg_section:
                indicators = re.findall(r'<indicator>(.*?)</indicator>', neg_section.group(1), re.DOTALL)
                neg_indicators = [ind.strip() for ind in indicators if ind.strip()]

            # Extract boundary cases
            boundary_cases = []
            boundary_section = re.search(r'<boundary_cases>(.*?)</boundary_cases>', response, re.DOTALL)
            if boundary_section:
                cases = re.findall(r'<case>(.*?)</case>', boundary_section.group(1), re.DOTALL)
                boundary_cases = [case.strip() for case in cases if case.strip()]

            # Extract confidence
            conf_match = re.search(r'<confidence>(.*?)</confidence>', response, re.DOTALL)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1).strip())
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except:
                    confidence = 0.8
            else:
                confidence = 0.8

            return {
                "concept": concept,
                "definition": definition,
                "positive_indicators": pos_indicators,
                "negative_indicators": neg_indicators,
                "boundary_cases": boundary_cases,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Failed to parse intent response: {e}")
            return self._create_fallback_intent(concept_fallback, "")

    def _create_fallback_intent(self, concept: str, system_prompt: str) -> Dict[str, Any]:
        """Create fallback intent when extraction fails."""
        return {
            "concept": concept,
            "definition": f"Classification of '{concept}' based on user's criteria from: {system_prompt}",
            "positive_indicators": [f"contains {concept}", f"is {concept}", f"has {concept}"],
            "negative_indicators": [f"no {concept}", f"not {concept}", f"lacks {concept}"],
            "boundary_cases": ["Ambiguous cases require manual review"],
            "confidence": 0.3,  # Low confidence for fallback
            "metadata": {
                "examples_analyzed": 0,
                "true_count": 0,
                "false_count": 0,
                "fallback": True,
            }
        }


__all__ = ["IntentExtractor"]
