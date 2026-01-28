"""
Meta-prompt system for prompt optimization.

This module provides the MetaPrompt class which constructs meta-prompts
for LLM-based prompt optimization. It supports both general prompt optimization
and specialized coding agent optimization.
"""

from typing import List, Mapping, Union

import pandas as pd

from ..config.settings import settings


# Default meta-prompt templates
DEFAULT_META_PROMPT_TEMPLATE = """
You are an expert in prompt optimization. Given the original baseline prompt and the following associated metadata (such as model inputs, outputs, evaluation labels and explanations),
generate a revised version of the original prompt that would likely improve results with respect to the evaluation labels.
Your goal is to align the prompt with the feedback and evaluation criteria.

BELOW IS THE ORIGINAL BASELINE PROMPT
************* start prompt *************


{baseline_prompt}
************* end prompt *************

BELOW ARE THE EXAMPLES USING THE ABOVE PROMPT
************* start example data *************


{examples}
************* end example data *************

HERE ARE SOME ANNOTATIONS THAT MAY BE HELPFUL:
{annotations}

FINAL INSTRUCTIONS
Iterate on the original prompt (above) with a new prompt that will improve the results, based on the examples and feedback above.

A common best practice in prompt optimization is to add guidelines and the most helpful few shot examples.

CRITICAL INSTRUCTIONS FOR FEW-SHOT EXAMPLES:
- When adding few-shot examples to your improved prompt, you MUST include the ACTUAL example text from the training data shown above.
- DO NOT create placeholder variables like {example1}, {example2}, {examples1}, {examples2}, or any similar placeholders for examples.
- COPY the real input text verbatim from the "Data for baseline prompt" sections above into your few-shot examples.
- Your few-shot examples should contain the complete, actual text that will be shown to the LLM, not placeholders.
- Example of CORRECT format: "Article: Crosby scores 58th goal in dominant victory..." (actual text)
- Example of INCORRECT format: "Article: {example1}" or "Article: {examples1}" (placeholder - DO NOT DO THIS)

Note about template variables: Make sure to include the variables from the original prompt, which are wrapped in curly brackets (e.g.
{var}, {input}, {question}). These are the ONLY curly brackets that should appear in your optimized prompt. If you fail to include these variables, the LLM will not be able to access the required data.
Do not add any single or double brackets around anything other than the variables from the original prompt.

CRITICAL: PRESERVE OUTPUT FORMAT FROM ORIGINAL PROMPT
- You MUST copy the exact output format and return instructions from the original prompt.
- DO NOT change the output schema, categories, or classification structure.
- If the original prompt asks for binary output (True/False), your optimized prompt MUST also output True/False.
- If the original prompt asks for specific categories, keep those exact categories - do NOT create new ones.
- DO NOT add additional classification steps, confidence scores, or reasoning fields unless they were in the original prompt.
- Example INCORRECT: Original asks for "True or False" → Optimized asks for "breaking_record/team_victory/individual_achievement" (WRONG!)
- Example CORRECT: Original asks for "True or False" → Optimized asks for "True or False" (CORRECT!)

YOUR NEW PROMPT:
"""

# Intent-aware meta-prompt template (3-phase system)
INTENT_AWARE_META_PROMPT_TEMPLATE = """
You are an expert in prompt optimization with deep understanding of user intent.

PHASE 0: USER INTENT DEFINITION
************* start user intent *************
{intent_definition}
************* end user intent *************

The user has defined their classification concept as shown above. This is the foundation for all prompt optimization.
Your optimized prompt MUST align with this specific intent definition.

BELOW IS THE ORIGINAL BASELINE PROMPT
************* start prompt *************

{baseline_prompt}
************* end prompt *************

BELOW ARE THE EXAMPLES USING THE ABOVE PROMPT
************* start example data *************

{examples}
************* end example data *************

HERE ARE SOME ANNOTATIONS THAT MAY BE HELPFUL:
{annotations}

FINAL INSTRUCTIONS
Iterate on the original prompt (above) with a new prompt that will improve the results, based on the USER'S INTENT DEFINITION and the examples/feedback above.

CRITICAL: INTENT-GROUNDED OPTIMIZATION
Your optimized prompt MUST:

1. **Reference User Intent Explicitly**:
   - Start your prompt by restating the user's definition of the concept
   - Example: "Based on the definition: {concept} means {definition}"
   - Make it clear what criteria determine True vs False

2. **Use Intent Indicators**:
   - Incorporate the positive_indicators from user's intent
   - Incorporate the negative_indicators from user's intent
   - Example: "Look for these indicators: {positive_indicators}"
   - Example: "Avoid classifying if: {negative_indicators}"

3. **Handle Boundary Cases**:
   - Address the boundary_cases from user's intent
   - Provide clear guidance on edge cases
   - Example: "Note: {boundary_case_explanation}"

4. **Ground Few-Shot Examples in Intent**:
   - When adding few-shot examples, EXPLAIN why they match/don't match the user's intent
   - Example: "This is True because it contains {positive_indicator} which matches our definition"
   - DO NOT use placeholders like {{example1}} - use ACTUAL text from training data above

5. **Align Classification Logic with Intent**:
   - Decision criteria should directly reference user's definition
   - Example: "Classify as True if the input matches the definition: {definition}"

CRITICAL INSTRUCTIONS FOR FEW-SHOT EXAMPLES:
- You MUST include the ACTUAL example text from the training data shown above
- DO NOT create placeholder variables like {{example1}}, {{example2}}, {{examples1}}, {{examples2}}
- COPY the real input text verbatim from the "Data for baseline prompt" sections above
- For EACH example, explain how it aligns (or doesn't) with the user's intent
- Example CORRECT format:
  "Example 1 (True): 'Player scores 58th goal, breaking team record'
   → This matches our definition because it contains 'record-breaking' (positive indicator)"
- Example INCORRECT format:
  "Example 1: {{example1}}" (DO NOT DO THIS!)

PRESERVE OUTPUT FORMAT:
- You MUST copy the exact output format from the original prompt
- DO NOT change the output schema (if binary, keep binary; if categories, keep same categories)
- If original asks for True/False, optimized MUST also ask for True/False

PRESERVE TEMPLATE VARIABLES:
- Keep all variables from original prompt wrapped in curly brackets: {{var}}, {{input}}, {{question}}
- These are the ONLY curly brackets in your optimized prompt (besides few-shot example text)

YOUR NEW INTENT-GROUNDED PROMPT:
"""

DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE = """
You are an expert in coding agent prompt optimization.
Your goal is to improve the dynamic ruleset that guides the coding agent.

Process:
1. Carefully review the baseline prompt, the current dynamic ruleset, examples, and annotations.
2. Identify high-level issues in the baseline prompt and dynamic ruleset — focus on missing guidance, vague constraints, or areas where rules could be made more robust.
3. Revise the dynamic ruleset so it is stronger, more reliable, and generalizes well beyond the provided examples.

BELOW IS THE ORIGINAL BASELINE PROMPT WITH STATIC RULESET
************* start prompt *************

{baseline_prompt}
************* end prompt *************

BELOW IS THE CURRENT DYNAMIC RULESET (CHANGE THESE OR ADD NEW RULES)
************* start ruleset *************

{ruleset}
************* end ruleset *************

Now you will be given data examples that use the above prompt and ruleset. Each example consists of:
- problem_statement: the problem statement
- coding agent patch: a patch generated by the coding agent, which is supposed to fix the problem.
- ground truth patch: a ground truth solution/patch to the problem
- test patch: a test patch that the coding agent's output should pass, which directly addresses the issue in the problem statement
- pass_or_fail: either "pass" or "fail" indicating whether the coding agent's code changes passed the unit tests (indicates whether the coding agent's output is correct or incorrect)
- explanation: explanation of your reasoning: why/why not the coding agent's output is correct, why the coding agent may have taken that approach, and general improvement suggestions for the coding agent to improve its output.

BELOW ARE THE EXAMPLES USING THE ABOVE PROMPT AND RULESET
************* start example data *************

{examples}
************* end example data *************

FINAL INSTRUCTIONS
Iterate on the **dynamic ruleset only**. You may:
- Add new rules
- Edit or strengthen existing rules

Important constraints:
- Do **not** modify the static rules in the baseline prompt.
- Do **not** add rules that request user input, confirmations, or follow-up questions (e.g., `ask_followup_question`). The coding agent should always act autonomously.
- Keep the ruleset concise and relevant — avoid unnecessary rules that don't match the general types of problems that the coding agent is likely to encounter or overly specific rules that only patch the given examples.
- Remember that you are writing GENERAL rules. They should not be specific to the repositories or problems that you are given. They should be general rules that would improve the overall ability of the coding agent.
Output format:
- Return only the final, revised dynamic ruleset as a bullet-point list.
- Do not include any extra commentary, explanations, or text outside the ruleset.

New ruleset:
"""


class MetaPrompt:
    """
    Meta-prompt constructor for prompt optimization.

    This class constructs meta-prompts that guide an LLM to optimize
    a base prompt based on example data and feedback.

    Attributes:
        meta_prompt: The template for general prompt optimization
        rules_meta_prompt: The template for coding agent ruleset optimization
        start_delim: Template variable start delimiter (from settings)
        end_delim: Template variable end delimiter (from settings)
    """

    def __init__(
        self,
        meta_prompt: str | None = None,
        rules_meta_prompt: str | None = None,
    ):
        """
        Initialize the MetaPrompt constructor.

        Args:
            meta_prompt: Custom template for general prompt optimization.
                        If None, uses settings.meta_prompt_template or DEFAULT_META_PROMPT_TEMPLATE.
            rules_meta_prompt: Custom template for coding agent optimization.
                              If None, uses settings.coding_agent_meta_prompt_template or DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE.
        """
        # Use settings if available, otherwise fall back to defaults
        self.meta_prompt = meta_prompt or settings.meta_prompt_template or DEFAULT_META_PROMPT_TEMPLATE
        self.rules_meta_prompt = rules_meta_prompt or settings.coding_agent_meta_prompt_template or DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE
        self.start_delim = settings.start_delim
        self.end_delim = settings.end_delim

    def construct_content(
        self,
        batch_df: pd.DataFrame,
        prompt_to_optimize_content: str,
        template_variables: List[str],
        feedback_columns: List[str],
        output_column: str,
        annotations: List[str] | None = None,
        ruleset: str | None = None,
    ) -> str:
        """
        Construct a meta-prompt for optimizing a base prompt.

        This method builds a complete meta-prompt by combining the base prompt
        with example data, feedback, and optional annotations or rulesets.

        Args:
            batch_df: DataFrame containing example data with inputs, outputs, and feedback
            prompt_to_optimize_content: The base prompt to be optimized
            template_variables: List of template variable names in the base prompt
            feedback_columns: List of column names containing feedback/evaluation data
            output_column: Name of the column containing LLM outputs
            annotations: Optional list of annotation strings to include
            ruleset: Optional ruleset for coding agent optimization mode

        Returns:
            The constructed meta-prompt as a string

        Examples:
            >>> df = pd.DataFrame({
            ...     "input": ["example 1", "example 2"],
            ...     "output": ["result 1", "result 2"],
            ...     "feedback": ["good", "bad"]
            ... })
            >>> mp = MetaPrompt()
            >>> meta_prompt = mp.construct_content(
            ...     batch_df=df,
            ...     prompt_to_optimize_content="Process: {input}",
            ...     template_variables=["input"],
            ...     feedback_columns=["feedback"],
            ...     output_column="output"
            ... )
        """
        # Select the appropriate template based on mode
        if ruleset is not None:
            content = self.rules_meta_prompt
            content = content.replace("{ruleset}", ruleset)
        else:
            content = self.meta_prompt

        # Insert the baseline prompt
        content = content.replace("{baseline_prompt}", prompt_to_optimize_content)

        # Build examples section
        examples = ""
        for ind, row in batch_df.iterrows():
            row_dict = row.to_dict()
            output_value = row_dict[output_column]

            # Sanitize output value
            if output_value is not None and isinstance(output_value, str):
                output_value = output_value.replace(self.start_delim, " ").replace(
                    self.end_delim, " "
                )
            else:
                output_value = "None"

            # Build example based on mode
            if ruleset is None:
                current_example = f"""
                    Example {str(ind)}

                    Data for baseline prompt: {[row_dict[temp_var] for temp_var in template_variables]}

                    LLM Output using baseline prompt: {output_value}

                    Output level feedback:
                """
            else:
                current_example = f"""
                    Example {str(ind)}

                    coding agent patch: {output_value}
                """

            # Add feedback columns
            for feedback_column in feedback_columns:
                feedback_value = row_dict[feedback_column]
                if feedback_value is not None:
                    # Cast to string to handle integers and other types
                    feedback_value = str(feedback_value)
                    feedback_value = feedback_value.replace(self.start_delim, " ").replace(
                        self.end_delim, " "
                    )
                else:
                    feedback_value = "None"
                current_example += f"\n{feedback_column}: {feedback_value}"

            examples += current_example

        # Insert examples into template
        content = content.replace("{examples}", examples)

        # Add annotations if provided
        if annotations:
            content = content.replace("{annotations}", "\n".join(annotations))

        # Optional debug file writing
        if settings.save_meta_prompt_debug:
            try:
                with open(settings.meta_prompt_debug_path, "w") as f:
                    f.write(content)
            except (IOError, OSError) as e:
                # Silently fail if we can't write the debug file
                # This prevents the optimization from failing due to debug file issues
                pass

        return content

    def construct_intent_aware_content(
        self,
        batch_df: pd.DataFrame,
        prompt_to_optimize_content: str,
        template_variables: List[str],
        feedback_columns: List[str],
        output_column: str,
        intent: dict,
        annotations: List[str] | None = None,
    ) -> str:
        """
        Construct an intent-aware meta-prompt for optimization.

        This method builds a meta-prompt that includes the user's intent definition,
        ensuring the optimized prompt aligns with the user's specific criteria.

        Args:
            batch_df: DataFrame containing example data
            prompt_to_optimize_content: The base prompt to optimize
            template_variables: List of template variable names
            feedback_columns: List of feedback column names
            output_column: Name of output column
            intent: Intent dictionary from IntentExtractor with:
                - concept: str
                - definition: str
                - positive_indicators: List[str]
                - negative_indicators: List[str]
                - boundary_cases: List[str]
                - confidence: float
            annotations: Optional additional annotations

        Returns:
            Intent-aware meta-prompt string
        """
        # Use intent-aware template
        content = INTENT_AWARE_META_PROMPT_TEMPLATE

        # Format intent definition
        intent_text = self._format_intent_definition(intent)
        content = content.replace("{intent_definition}", intent_text)

        # Insert baseline prompt
        content = content.replace("{baseline_prompt}", prompt_to_optimize_content)

        # Build examples section (same as construct_content)
        examples = ""
        for ind, row in batch_df.iterrows():
            row_dict = row.to_dict()
            output_value = row_dict[output_column]

            # Sanitize output
            if output_value is not None and isinstance(output_value, str):
                output_value = output_value.replace(self.start_delim, " ").replace(
                    self.end_delim, " "
                )
            else:
                output_value = "None"

            current_example = f"""
                Example {str(ind)}

                Data for baseline prompt: {[row_dict[temp_var] for temp_var in template_variables]}

                LLM Output using baseline prompt: {output_value}

                Output level feedback:
            """

            # Add feedback
            for feedback_col in feedback_columns:
                if feedback_col in row_dict:
                    feedback_value = row_dict[feedback_col]
                    current_example += f"\n                    {feedback_col}: {feedback_value}"

            examples += current_example + "\n"

        content = content.replace("{examples}", examples)

        # Add annotations
        if annotations:
            annotations_text = "\n".join(annotations)
        else:
            annotations_text = "No additional annotations."

        content = content.replace("{annotations}", annotations_text)

        return content

    def _format_intent_definition(self, intent: dict) -> str:
        """
        Format intent dictionary into readable text for meta-prompt.

        Args:
            intent: Intent dictionary from IntentExtractor

        Returns:
            Formatted intent definition string
        """
        concept = intent.get("concept", "concept")
        definition = intent.get("definition", "No definition provided")
        positive_indicators = intent.get("positive_indicators", [])
        negative_indicators = intent.get("negative_indicators", [])
        boundary_cases = intent.get("boundary_cases", [])
        confidence = intent.get("confidence", 0.0)

        # Format as structured text
        intent_text = f"""
**Concept**: {concept}

**User's Definition**:
{definition}

**Positive Indicators** (classify as TRUE if present):
"""
        for ind in positive_indicators:
            intent_text += f"  - {ind}\n"

        intent_text += "\n**Negative Indicators** (classify as FALSE if present):\n"
        for ind in negative_indicators:
            intent_text += f"  - {ind}\n"

        if boundary_cases:
            intent_text += "\n**Boundary Cases** (edge cases requiring careful consideration):\n"
            for case in boundary_cases:
                intent_text += f"  - {case}\n"

        intent_text += f"\n**Intent Extraction Confidence**: {confidence:.2f}\n"

        return intent_text

    def format_template_with_vars(
        self,
        template: str,
        template_variables: List[str],
        variable_values: Mapping[str, Union[bool, int, float, str]],
    ) -> str:
        """
        Format a template string with variable values.

        This method replaces template variables (wrapped in delimiters)
        with their corresponding values.

        Args:
            template: The template string containing variables
            template_variables: List of variable names to replace
            variable_values: Mapping of variable names to values

        Returns:
            The formatted template with variables replaced

        Examples:
            >>> mp = MetaPrompt()
            >>> template = "Process: {input} with {context}"
            >>> result = mp.format_template_with_vars(
            ...     template=template,
            ...     template_variables=["input", "context"],
            ...     variable_values={"input": "data", "context": "background"}
            ... )
            >>> print(result)
            'Process: data with background'
        """
        formatted = template
        for template_var in template_variables:
            var_value = str(variable_values[template_var])
            if var_value is not None:
                var_value = var_value.replace(self.start_delim, " ")
                var_value = var_value.replace(self.end_delim, " ")
            formatted = formatted.replace(
                self.start_delim + template_var + self.end_delim,
                var_value,
            )
        return formatted
