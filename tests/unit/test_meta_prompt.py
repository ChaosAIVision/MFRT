"""
Unit tests for meta-prompt construction system.

Tests the MetaPrompt class from chaos_auto_prompt.optimizers.meta_prompt
including template construction, variable formatting, and debug mode.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from chaos_auto_prompt.optimizers.meta_prompt import (
    MetaPrompt,
    DEFAULT_META_PROMPT_TEMPLATE,
    DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE,
)


class TestMetaPromptInit:
    """Test MetaPrompt initialization."""

    def test_default_initialization(self, monkeypatch):
        """Test initialization with default templates."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()

        assert mp.meta_prompt == DEFAULT_META_PROMPT_TEMPLATE
        assert mp.rules_meta_prompt == DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE
        assert mp.start_delim == "{"
        assert mp.end_delim == "}"

    def test_custom_meta_prompt(self, monkeypatch):
        """Test initialization with custom meta-prompt template."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        custom_template = "Custom prompt template: {baseline_prompt}"
        mp = MetaPrompt(meta_prompt=custom_template)

        assert mp.meta_prompt == custom_template
        assert mp.rules_meta_prompt == DEFAULT_CODING_AGENT_META_PROMPT_TEMPLATE

    def test_custom_rules_meta_prompt(self, monkeypatch):
        """Test initialization with custom rules meta-prompt template."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        custom_template = "Custom rules template: {ruleset}"
        mp = MetaPrompt(rules_meta_prompt=custom_template)

        assert mp.meta_prompt == DEFAULT_META_PROMPT_TEMPLATE
        assert mp.rules_meta_prompt == custom_template

    def test_both_custom_templates(self, monkeypatch):
        """Test initialization with both custom templates."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        custom_meta = "Custom meta: {baseline_prompt}"
        custom_rules = "Custom rules: {ruleset}"

        mp = MetaPrompt(meta_prompt=custom_meta, rules_meta_prompt=custom_rules)

        assert mp.meta_prompt == custom_meta
        assert mp.rules_meta_prompt == custom_rules

    def test_template_from_settings(self, monkeypatch):
        """Test that templates can be loaded from settings."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("meta_prompt_template", "Settings template: {baseline_prompt}")
        monkeypatch.setenv("coding_agent_meta_prompt_template", "Settings rules: {ruleset}")

        mp = MetaPrompt()

        assert mp.meta_prompt == "Settings template: {baseline_prompt}"
        assert mp.rules_meta_prompt == "Settings rules: {ruleset}"

    def test_custom_overrides_settings(self, monkeypatch):
        """Test that custom templates override settings."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("meta_prompt_template", "Settings template")

        custom_template = "Custom template: {baseline_prompt}"
        mp = MetaPrompt(meta_prompt=custom_template)

        assert mp.meta_prompt == custom_template

    def test_custom_delimiters_from_settings(self, monkeypatch):
        """Test that custom delimiters can be set via settings."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("start_delim", "[")
        monkeypatch.setenv("end_delim", "]")

        mp = MetaPrompt()

        assert mp.start_delim == "["
        assert mp.end_delim == "]"


class TestConstructContent:
    """Test construct_content method."""

    def test_construct_content_basic(self, monkeypatch, sample_dataframe):
        """Test basic meta-prompt construction."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        baseline = "Process this: {text}"

        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content=baseline,
            template_variables=["text"],
            feedback_columns=["category"],
            output_column="text"
        )

        assert baseline in content
        assert "Process this:" in content
        assert "Example" in content
        assert "Short text" in content or "Medium length text" in content

    def test_construct_content_with_annotations(self, monkeypatch, sample_dataframe):
        """Test meta-prompt construction with annotations."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        baseline = "Classify: {text}"
        annotations = ["Focus on accuracy", "Consider edge cases"]

        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content=baseline,
            template_variables=["text"],
            feedback_columns=["category"],
            output_column="text",
            annotations=annotations
        )

        assert "Focus on accuracy" in content
        assert "Consider edge cases" in content

    def test_construct_content_ruleset_mode(self, monkeypatch, sample_dataframe):
        """Test meta-prompt construction in ruleset mode."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        baseline = "You are a coding agent."
        ruleset = "1. Write clean code\n2. Add tests"

        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content=baseline,
            template_variables=["text"],
            feedback_columns=["category"],
            output_column="text",
            ruleset=ruleset
        )

        assert baseline in content
        assert ruleset in content
        assert "coding agent patch:" in content

    def test_construct_content_multiple_feedback_columns(self, monkeypatch):
        """Test construction with multiple feedback columns."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "input": ["test input"],
            "output": ["test output"],
            "score": [0.8],
            "feedback": ["good"],
            "notes": ["well done"]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Process: {input}",
            template_variables=["input"],
            feedback_columns=["score", "feedback", "notes"],
            output_column="output"
        )

        assert "score: 0.8" in content
        assert "feedback: good" in content
        assert "notes: well done" in content

    def test_construct_content_sanitizes_delimiters(self, monkeypatch):
        """Test that output values are sanitized for template delimiters."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "input": ["test"],
            "output": ["This has {curly} braces"]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Process: {input}",
            template_variables=["input"],
            feedback_columns=[],
            output_column="output"
        )

        # Output should have delimiters replaced with spaces
        assert "This has  curly  braces" in content
        assert "This has {curly} braces" not in content

    def test_construct_content_with_none_values(self, monkeypatch):
        """Test handling of None values in dataframe."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "input": ["test1", None, "test3"],
            "output": ["out1", "out2", None],
            "feedback": [None, "good", None]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Process: {input}",
            template_variables=["input"],
            feedback_columns=["feedback"],
            output_column="output"
        )

        assert "None" in content  # Should show None for missing values

    def test_construct_content_empty_dataframe(self, monkeypatch):
        """Test construction with empty dataframe."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({"input": [], "output": []})

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Process: {input}",
            template_variables=["input"],
            feedback_columns=[],
            output_column="output"
        )

        # Should still have baseline prompt
        assert "Process: {input}" in content

    def test_construct_content_preserves_baseline(self, monkeypatch):
        """Test that baseline prompt is preserved in output."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        baseline = """This is a multi-line
baseline prompt with
{variable} placeholders"""

        df = pd.DataFrame({
            "variable": ["value1", "value2"],
            "output": ["out1", "out2"]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content=baseline,
            template_variables=["variable"],
            feedback_columns=[],
            output_column="output"
        )

        assert "multi-line" in content
        assert "baseline prompt" in content
        assert "{variable}" in content


class TestFormatTemplateWithVars:
    """Test format_template_with_vars method."""

    def test_format_single_variable(self, monkeypatch):
        """Test formatting template with single variable."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Process: {input}"
        variables = ["input"]
        values = {"input": "test data"}

        result = mp.format_template_with_vars(template, variables, values)

        assert result == "Process: test data"

    def test_format_multiple_variables(self, monkeypatch):
        """Test formatting template with multiple variables."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Process: {input} with context: {context}"
        variables = ["input", "context"]
        values = {"input": "data", "context": "background"}

        result = mp.format_template_with_vars(template, variables, values)

        assert result == "Process: data with context: background"

    def test_format_sanitizes_delimiters(self, monkeypatch):
        """Test that variable values are sanitized for delimiters."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Process: {input}"
        variables = ["input"]
        values = {"input": "data with {braces}"}

        result = mp.format_template_with_vars(template, variables, values)

        assert result == "Process: data with  braces "

    def test_format_with_custom_delimiters(self, monkeypatch):
        """Test formatting with custom delimiters."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("start_delim", "[")
        monkeypatch.setenv("end_delim", "]")

        mp = MetaPrompt()
        template = "Process: [input]"
        variables = ["input"]
        values = {"input": "data"}

        result = mp.format_template_with_vars(template, variables, values)

        assert result == "Process: data"

    def test_format_with_none_value(self, monkeypatch):
        """Test formatting with None value."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Process: {input}"
        variables = ["input"]
        values = {"input": None}

        result = mp.format_template_with_vars(template, variables, values)

        assert "Process: None" in result

    def test_format_with_numeric_values(self, monkeypatch):
        """Test formatting with numeric variable values."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Count: {count}, Score: {score}"
        variables = ["count", "score"]
        values = {"count": 42, "score": 3.14}

        result = mp.format_template_with_vars(template, variables, values)

        assert "Count: 42" in result
        assert "Score: 3.14" in result

    def test_format_with_boolean_values(self, monkeypatch):
        """Test formatting with boolean variable values."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Flag: {flag}"
        variables = ["flag"]
        values = {"flag": True}

        result = mp.format_template_with_vars(template, variables, values)

        assert "Flag: True" in result

    def test_format_partial_substitution(self, monkeypatch):
        """Test that only specified variables are substituted."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        template = "Process: {input}, ignore: {other}"
        variables = ["input"]  # Only substitute input
        values = {"input": "data", "other": "ignored"}

        result = mp.format_template_with_vars(template, variables, values)

        assert "Process: data" in result
        assert "{other}" in result  # Should not be substituted


class TestMetaPromptDebug:
    """Test debug file writing functionality."""

    def test_debug_mode_disabled_by_default(self, monkeypatch, sample_dataframe):
        """Test that debug mode is disabled by default."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()

        # Construct content - should not write file
        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content="Test: {text}",
            template_variables=["text"],
            feedback_columns=[],
            output_column="text"
        )

        # No file should be created
        assert not os.path.exists(mp.meta_prompt_debug_path)

    def test_debug_mode_enabled(self, monkeypatch, sample_dataframe):
        """Test debug mode writes to file."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("save_meta_prompt_debug", "true")

        # Use temp directory for debug file
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_path = os.path.join(tmpdir, "debug.txt")
            monkeypatch.setenv("meta_prompt_debug_path", debug_path)

            mp = MetaPrompt()

            content = mp.construct_content(
                batch_df=sample_dataframe,
                prompt_to_optimize_content="Test: {text}",
                template_variables=["text"],
                feedback_columns=[],
                output_column="text"
            )

            # File should be created
            assert os.path.exists(debug_path)

            # Check content
            with open(debug_path, 'r') as f:
                written_content = f.read()

            assert written_content == content

    def test_debug_mode_custom_path(self, monkeypatch, sample_dataframe):
        """Test debug mode with custom path."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("save_meta_prompt_debug", "true")

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_debug.txt")
            monkeypatch.setenv("meta_prompt_debug_path", custom_path)

            mp = MetaPrompt()

            content = mp.construct_content(
                batch_df=sample_dataframe,
                prompt_to_optimize_content="Test: {text}",
                template_variables=["text"],
                feedback_columns=[],
                output_column="text"
            )

            assert os.path.exists(custom_path)

    def test_debug_mode_handles_write_errors(self, monkeypatch, sample_dataframe):
        """Test that debug mode failures don't crash the method."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")
        monkeypatch.setenv("save_meta_prompt_debug", "true")
        # Set invalid path
        monkeypatch.setenv("meta_prompt_debug_path", "/nonexistent/dir/debug.txt")

        mp = MetaPrompt()

        # Should not raise exception
        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content="Test: {text}",
            template_variables=["text"],
            feedback_columns=[],
            output_column="text"
        )

        assert content is not None
        assert len(content) > 0


class TestMetaPromptEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_template_variables(self, monkeypatch, sample_dataframe):
        """Test construction with empty template variables."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()

        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content="Static prompt",
            template_variables=[],
            feedback_columns=[],
            output_column="text"
        )

        assert "Static prompt" in content

    def test_empty_feedback_columns(self, monkeypatch, sample_dataframe):
        """Test construction with no feedback columns."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()

        content = mp.construct_content(
            batch_df=sample_dataframe,
            prompt_to_optimize_content="Test: {text}",
            template_variables=["text"],
            feedback_columns=[],
            output_column="text"
        )

        assert "Example" in content

    def test_missing_column_in_dataframe(self, monkeypatch):
        """Test handling when feedback column doesn't exist."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "input": ["test"],
            "output": ["result"]
        })

        # This should raise KeyError for missing feedback column
        with pytest.raises(KeyError):
            mp.construct_content(
                batch_df=df,
                prompt_to_optimize_content="Test: {input}",
                template_variables=["input"],
                feedback_columns=["nonexistent_column"],
                output_column="output"
            )

    def test_large_dataframe(self, monkeypatch):
        """Test construction with large dataframe."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "text": [f"Sample text {i}" for i in range(100)],
            "output": [f"Output {i}" for i in range(100)]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Test: {text}",
            template_variables=["text"],
            feedback_columns=[],
            output_column="output"
        )

        # Should have many examples
        assert content.count("Example") >= 100

    def test_special_characters_in_values(self, monkeypatch):
        """Test handling of special characters in values."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "text": ["Text with\nnewlines\tand\rcarriage returns"],
            "output": ["Result"]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Test: {text}",
            template_variables=["text"],
            feedback_columns=[],
            output_column="output"
        )

        assert "newlines" in content or "Text with" in content

    def test_unicode_in_values(self, monkeypatch):
        """Test handling of unicode characters in values."""
        monkeypatch.setenv("openai_api_key", "test-key")
        monkeypatch.setenv("google_api_key", "test-key")

        mp = MetaPrompt()
        df = pd.DataFrame({
            "text": ["Text with emoji 🎉 and symbols ∑ ∆ ∂"],
            "output": ["Result"]
        })

        content = mp.construct_content(
            batch_df=df,
            prompt_to_optimize_content="Test: {text}",
            template_variables=["text"],
            feedback_columns=[],
            output_column="output"
        )

        # Unicode should be preserved
        assert "🎉" in content or "Text with" in content
