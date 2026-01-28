"""
Construction Extractor - Extract structured elements from construction phase output.

This module provides functionality to parse and structure the 4 core elements from
LLM construction phase output: entities, state variables, actions, and constraints.
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ConstructionExtractor:
    """
    Extractor for parsing construction phase output into structured elements.

    This extractor is designed for the 2-phase meta-prompt system where Phase 1
    (construction) requires extracting:
    1. Entities - Relevant objects/agents
    2. State Variables - Attributes that change over time
    3. Actions - Operations with preconditions and effects
    4. Constraints - Rules and limitations

    Features:
    - Flexible regex-based extraction (handles format variations)
    - Multiple section header patterns (numbered, bulleted, plain)
    - Confidence scoring based on extraction completeness
    - Comprehensive error handling and logging
    """

    # Section header patterns (case-insensitive, flexible)
    ENTITIES_PATTERNS = [
        r'\(1\)\s*relevant\s+entities?:?\s*(.+?)(?=\(2\)|\n\n|state|$)',
        r'entities?:?\s*(.+?)(?=state|actions?|constraints?|\n\n|$)',
        r'1\.\s*entities?:?\s*(.+?)(?=2\.|state|$)',
    ]

    STATE_PATTERNS = [
        r'\(2\)\s*state\s+variables?:?\s*(.+?)(?=\(3\)|\n\n|actions?|$)',
        r'state\s+variables?:?\s*(.+?)(?=actions?|constraints?|\n\n|$)',
        r'2\.\s*state\s+variables?:?\s*(.+?)(?=3\.|actions?|$)',
    ]

    ACTION_PATTERNS = [
        r'\(3\)\s*(?:possible\s+)?actions?:?\s*(.+?)(?=\(4\)|\n\n|constraints?|$)',
        r'(?:possible\s+)?actions?:?\s*(.+?)(?=constraints?|\n\n|$)',
        r'3\.\s*(?:possible\s+)?actions?:?\s*(.+?)(?=4\.|constraints?|$)',
    ]

    CONSTRAINT_PATTERNS = [
        r'\(4\)\s*constraints?:?\s*(.+?)$',
        r'constraints?:?\s*(.+?)$',
        r'4\.\s*constraints?:?\s*(.+?)$',
    ]

    @staticmethod
    def extract(construction_text: str) -> Dict[str, Any]:
        """
        Extract structured elements from construction phase text.

        Args:
            construction_text: Raw construction phase content from XMLOutputParser

        Returns:
            Dictionary containing:
                - entities: List[str] - Entity names
                - state_variables: List[Dict] - State variables with types
                - actions: List[Dict] - Actions with preconditions/effects
                - constraints: List[str] - Constraint statements
                - metadata: Dict - Extraction info (confidence, missing sections)
                - error: str (optional) - Error message if extraction failed

        Raises:
            ValueError: If construction_text is invalid (empty, too short)

        Example:
            >>> extractor = ConstructionExtractor()
            >>> result = extractor.extract(construction_text)
            >>> result["entities"]
            ['user', 'task', 'deadline']
        """
        # Input validation
        if not construction_text or not construction_text.strip():
            error_msg = "Construction text cannot be empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if len(construction_text.strip()) < 50:
            error_msg = f"Construction text too short ({len(construction_text)} chars). Minimum 50 characters."
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # Check if text mentions any construction elements
        text_lower = construction_text.lower()
        has_keywords = any(
            keyword in text_lower
            for keyword in ['entit', 'state', 'action', 'constraint']
        )

        if not has_keywords:
            error_msg = "Construction text does not mention entities, state, actions, or constraints"
            logger.warning(error_msg, extra={"text_preview": construction_text[:100]})
            raise ValueError(error_msg)

        logger.debug(
            f"Extracting construction elements from text",
            extra={"text_length": len(construction_text)}
        )

        # Extract each section
        entities = ConstructionExtractor._extract_entities(construction_text)
        state_variables = ConstructionExtractor._extract_state_variables(construction_text)
        actions = ConstructionExtractor._extract_actions(construction_text)
        constraints = ConstructionExtractor._extract_constraints(construction_text)

        # Calculate confidence
        sections_found = [
            bool(entities),
            bool(state_variables),
            bool(actions),
            bool(constraints)
        ]
        confidence = sum(sections_found) / 4.0

        missing_sections = []
        if not entities:
            missing_sections.append("entities")
        if not state_variables:
            missing_sections.append("state_variables")
        if not actions:
            missing_sections.append("actions")
        if not constraints:
            missing_sections.append("constraints")

        logger.info(
            f"Extracted construction elements",
            extra={
                "entities_count": len(entities),
                "state_variables_count": len(state_variables),
                "actions_count": len(actions),
                "constraints_count": len(constraints),
                "confidence": confidence,
                "missing_sections": missing_sections
            }
        )

        result = {
            "entities": entities,
            "state_variables": state_variables,
            "actions": actions,
            "constraints": constraints,
            "metadata": {
                "extraction_confidence": confidence,
                "missing_sections": missing_sections,
                "extraction_method": "regex_parser"
            }
        }

        # Add error if confidence is too low
        if confidence < 0.5:
            result["error"] = f"Low extraction confidence ({confidence:.2f}). Missing: {', '.join(missing_sections)}"

        return result

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extract entities from construction text."""
        entities = []

        for pattern in ConstructionExtractor.ENTITIES_PATTERNS:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                entities = ConstructionExtractor._parse_list_items(content)
                if entities:
                    logger.debug(f"Extracted {len(entities)} entities using pattern")
                    break

        return entities

    @staticmethod
    def _extract_state_variables(text: str) -> List[Dict[str, Any]]:
        """Extract state variables with types from construction text."""
        state_variables = []

        for pattern in ConstructionExtractor.STATE_PATTERNS:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                state_variables = ConstructionExtractor._parse_state_variables(content)
                if state_variables:
                    logger.debug(f"Extracted {len(state_variables)} state variables")
                    break

        return state_variables

    @staticmethod
    def _extract_actions(text: str) -> List[Dict[str, Any]]:
        """Extract actions with preconditions and effects from construction text."""
        actions = []

        for pattern in ConstructionExtractor.ACTION_PATTERNS:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                actions = ConstructionExtractor._parse_actions(content)
                if actions:
                    logger.debug(f"Extracted {len(actions)} actions")
                    break

        return actions

    @staticmethod
    def _extract_constraints(text: str) -> List[str]:
        """Extract constraints from construction text."""
        constraints = []

        for pattern in ConstructionExtractor.CONSTRAINT_PATTERNS:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                constraints = ConstructionExtractor._parse_list_items(content)
                if constraints:
                    logger.debug(f"Extracted {len(constraints)} constraints")
                    break

        return constraints

    @staticmethod
    def _parse_list_items(content: str) -> List[str]:
        """
        Parse comma-separated or bullet-pointed list items.

        Handles:
        - Comma-separated: "user, task, deadline"
        - Bullet points: "- user\n- task\n- deadline"
        - Numbered: "1. user\n2. task\n3. deadline"
        """
        items = []

        # Try comma-separated first
        if ',' in content and '\n' not in content[:50]:  # Likely comma-separated
            items = [item.strip() for item in content.split(',') if item.strip()]
        else:
            # Try line-by-line (bullets or numbers)
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Remove bullet points, numbers, etc.
                clean_line = re.sub(r'^[-*•\d]+[\.\):]?\s*', '', line).strip()
                if clean_line and len(clean_line) > 1:
                    items.append(clean_line)

        return items

    @staticmethod
    def _parse_state_variables(content: str) -> List[Dict[str, Any]]:
        """
        Parse state variables with type information.

        Formats handled:
        - task_status: enum [pending, in_progress, done]
        - user_availability: boolean
        - Simple list: task_status, user_availability
        """
        state_vars = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            # Remove bullet/number prefix
            line = re.sub(r'^[-*•\d]+[\.\):]?\s*', '', line).strip()

            # Try to parse structured format: name: type [values]
            structured_match = re.match(
                r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\w+)(?:\s*\[(.+?)\])?',
                line,
                re.IGNORECASE
            )

            if structured_match:
                name = structured_match.group(1).strip()
                var_type = structured_match.group(2).strip().lower()
                values_str = structured_match.group(3)

                var_dict = {
                    "name": name,
                    "type": var_type
                }

                if values_str:
                    # Parse possible values
                    values = [v.strip() for v in values_str.split(',') if v.strip()]
                    var_dict["possible_values"] = values

                state_vars.append(var_dict)
            else:
                # Simple format: just variable name
                simple_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if simple_match:
                    name = simple_match.group(1).strip()
                    state_vars.append({
                        "name": name,
                        "type": "unknown"
                    })

        return state_vars

    @staticmethod
    def _parse_actions(content: str) -> List[Dict[str, Any]]:
        """
        Parse actions with preconditions and effects.

        Formats handled:
        - Action: name
          Preconditions: condition1, condition2
          Effects: effect1, effect2
        - Simple list: action1, action2, action3
        """
        actions = []

        # Try to parse structured actions
        action_blocks = re.split(
            r'\n\s*(?=(?:action|name)[\s:]+)',
            content,
            flags=re.IGNORECASE
        )

        for block in action_blocks:
            if not block.strip():
                continue

            action_dict = {}

            # Extract action name
            name_match = re.search(
                r'(?:action|name)[\s:]+([^\n:]+)',
                block,
                re.IGNORECASE
            )
            if name_match:
                action_dict["name"] = name_match.group(1).strip()

                # Extract preconditions
                precond_match = re.search(
                    r'preconditions?[\s:]+(.+?)(?=effects?|$)',
                    block,
                    re.DOTALL | re.IGNORECASE
                )
                if precond_match:
                    precond_text = precond_match.group(1).strip()
                    action_dict["preconditions"] = ConstructionExtractor._parse_list_items(precond_text)

                # Extract effects
                effects_match = re.search(
                    r'effects?[\s:]+(.+?)$',
                    block,
                    re.DOTALL | re.IGNORECASE
                )
                if effects_match:
                    effects_text = effects_match.group(1).strip()
                    action_dict["effects"] = ConstructionExtractor._parse_list_items(effects_text)

                actions.append(action_dict)

        # If no structured actions found, try simple list
        if not actions:
            simple_items = ConstructionExtractor._parse_list_items(content)
            for item in simple_items:
                actions.append({
                    "name": item,
                    "preconditions": [],
                    "effects": []
                })

        return actions
