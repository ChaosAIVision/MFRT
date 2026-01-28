# Meta-Prompt System

## Overview

The Meta-Prompt System is a ported and enhanced version of the meta-prompt functionality from `prompt-learning/optimizer_sdk`. It provides tools for constructing meta-prompts that guide LLMs to optimize base prompts based on example data and feedback.

## Architecture

### Components

1. **MetaPrompt Class** (`optimizers/meta_prompt.py`)
   - Core class for constructing meta-prompts
   - Supports both general prompt optimization and coding agent optimization
   - Uses settings from `config/settings.py` for configuration

2. **Configuration** (`config/settings.py`)
   - Meta-prompt templates can be customized via environment variables
   - Debug output can be enabled/disabled
   - Delimiters are configurable

### Key Features

- **Two Optimization Modes**:
  - General prompt optimization: Improves prompts based on input/output examples and feedback
  - Coding agent optimization: Optimizes dynamic rulesets for coding agents

- **Template Variable Handling**:
  - Automatic detection and formatting of template variables
  - Configurable delimiters (default: `{` and `}`)
  - Sanitization of delimiter characters in values

- **Flexible Configuration**:
  - Templates can be customized via environment variables
  - Debug mode saves generated meta-prompts to disk
  - Settings are managed through pydantic-settings

## Installation

The meta-prompt system is part of the `chaos-auto-prompt` package. No additional installation is required beyond the main package dependencies.

```bash
cd chaos-auto-prompt
pip install -e .
```

## Configuration

### Environment Variables

Add these to your `.env` file to customize meta-prompt behavior:

```bash
# Meta-Prompt Templates (optional - uses defaults if not set)
# META_PROMPT_TEMPLATE="path/to/custom/template.txt"
# CODING_AGENT_META_PROMPT_TEMPLATE="path/to/coding_agent_template.txt"

# Debug Settings
SAVE_META_PROMPT_DEBUG=true
META_PROMPT_DEBUG_PATH=metaprompt_debug.txt

# Delimiters (default: { and })
START_DELIM="{"
END_DELIM="}"
```

## Usage

### Basic Prompt Optimization

```python
import pandas as pd
from chaos_auto_prompt.optimizers import MetaPrompt

# Prepare your data
batch_df = pd.DataFrame({
    "input_text": ["Great movie!", "Terrible experience"],
    "output": ["Positive", "Positive"],  # Second one is wrong
    "feedback": ["Correct", "Incorrect - should be Negative"],
    "explanation": ["Good", "Failed to detect negative sentiment"]
})

# Define your base prompt
base_prompt = """
Analyze the sentiment: {input_text}

Return: Positive, Negative, or Neutral
"""

# Create meta-prompt constructor
mp = MetaPrompt()

# Construct the meta-prompt
meta_prompt = mp.construct_content(
    batch_df=batch_df,
    prompt_to_optimize_content=base_prompt,
    template_variables=["input_text"],
    feedback_columns=["feedback", "explanation"],
    output_column="output",
    annotations=[
        "Pay attention to strong emotional words",
        "Consider context and intensity"
    ]
)

# Use the meta-prompt to get an improved prompt from an LLM
```

### Coding Agent Optimization

```python
import pandas as pd
from chaos_auto_prompt.optimizers import MetaPrompt

# Prepare coding agent data
batch_df = pd.DataFrame({
    "problem": ["Fix NPE in UserService"],
    "agent_patch": ["Added null check"],
    "ground_truth": ["Used Optional pattern"],
    "pass_or_fail": ["fail"],
    "explanation": ["Null check insufficient"]
})

# Define baseline and current ruleset
baseline_prompt = "Fix this bug: {problem}\nStatic rules: Write clean code"
current_ruleset = "- Analyze problem\n- Generate patch\n- Test it"

mp = MetaPrompt()

# Construct coding agent meta-prompt
meta_prompt = mp.construct_content(
    batch_df=batch_df,
    prompt_to_optimize_content=baseline_prompt,
    template_variables=["problem"],
    feedback_columns=["pass_or_fail", "explanation"],
    output_column="agent_patch",
    ruleset=current_ruleset,  # Enables coding agent mode
    annotations=["Focus on robust error handling"]
)
```

### Template Variable Formatting

```python
from chaos_auto_prompt.optimizers import MetaPrompt

template = "Process: {input} with context: {context}"

mp = MetaPrompt()

formatted = mp.format_template_with_vars(
    template=template,
    template_variables=["input", "context"],
    variable_values={
        "input": "user data",
        "context": "validation enabled"
    }
)

# Result: "Process: user data with context: validation enabled"
```

## API Reference

### MetaPrompt Class

#### Constructor

```python
MetaPrompt(
    meta_prompt: str | None = None,
    rules_meta_prompt: str | None = None
)
```

**Parameters:**
- `meta_prompt`: Custom template for general optimization (optional)
- `rules_meta_prompt`: Custom template for coding agent optimization (optional)

If not provided, templates are loaded from settings or defaults are used.

#### Methods

##### `construct_content`

```python
construct_content(
    batch_df: pd.DataFrame,
    prompt_to_optimize_content: str,
    template_variables: List[str],
    feedback_columns: List[str],
    output_column: str,
    annotations: List[str] | None = None,
    ruleset: str | None = None
) -> str
```

Constructs a complete meta-prompt for optimization.

**Parameters:**
- `batch_df`: DataFrame with example data
- `prompt_to_optimize_content`: The base prompt to optimize
- `template_variables`: List of variable names in the base prompt
- `feedback_columns`: Columns containing feedback/evaluation data
- `output_column`: Column with LLM outputs
- `annotations`: Optional list of annotation strings
- `ruleset`: Optional ruleset (enables coding agent mode)

**Returns:**
- Complete meta-prompt as a string

##### `format_template_with_vars`

```python
format_template_with_vars(
    template: str,
    template_variables: List[str],
    variable_values: Mapping[str, Union[bool, int, float, str]]
) -> str
```

Formats a template by replacing variables with values.

**Parameters:**
- `template`: Template string with variables
- `template_variables`: List of variable names to replace
- `variable_values`: Mapping of variable names to values

**Returns:**
- Formatted template with variables replaced

## Template Placeholders

### General Optimization Template

The default template supports these placeholders:

- `{baseline_prompt}`: The original prompt to optimize
- `{examples}`: Auto-generated from batch_df
- `{annotations}`: Optional annotation strings

### Coding Agent Template

The coding agent template supports these placeholders:

- `{baseline_prompt}`: The original prompt with static rules
- `{ruleset}`: The current dynamic ruleset to optimize
- `{examples}`: Auto-generated from batch_df
- `{annotations}`: Optional annotation strings

## Differences from Original

This port improves upon the original `prompt-learning/optimizer_sdk` implementation:

1. **Settings Integration**: Templates and configuration are managed through `config/settings.py`
2. **No Phoenix Dependencies**: Removed Phoenix-specific code
3. **Cleaner Imports**: Uses type hints and modern Python syntax
4. **Optional Debug Output**: File writing is controlled by settings, not hardcoded
5. **Better Documentation**: Comprehensive docstrings and examples
6. **Error Handling**: Graceful handling of debug file write failures

## Examples

See `examples/meta_prompt_example.py` for complete working examples of:

1. Basic prompt optimization with sentiment analysis
2. Coding agent ruleset optimization
3. Template variable formatting

Run the examples:

```bash
cd chaos-auto-prompt
python examples/meta_prompt_example.py
```

## Testing

The meta-prompt system can be tested with mock data:

```python
import pandas as pd
from chaos_auto_prompt.optimizers import MetaPrompt

# Create test data
test_df = pd.DataFrame({
    "input": ["test input"],
    "output": ["test output"],
    "feedback": ["test feedback"]
})

# Test basic functionality
mp = MetaPrompt()
result = mp.construct_content(
    batch_df=test_df,
    prompt_to_optimize_content="Test: {input}",
    template_variables=["input"],
    feedback_columns=["feedback"],
    output_column="output"
)

assert "Test: test input" in result
assert "test output" in result
assert "test feedback" in result
```

## Troubleshooting

### Issue: Templates not loading from settings

**Solution**: Ensure your `.env` file is in the correct location and variables are properly named. Check that `pydantic-settings` is installed.

### Issue: Debug file not being created

**Solution**: Set `SAVE_META_PROMPT_DEBUG=true` in your `.env` file. Check that the application has write permissions for the debug path.

### Issue: Variables not being replaced

**Solution**: Ensure your template variables match the column names in your DataFrame. Check that delimiters (`{` and `}`) match your settings.

## Contributing

When modifying the meta-prompt system:

1. Keep templates backwards compatible
2. Update this documentation for any API changes
3. Add examples for new features
4. Test with both optimization modes

## License

This component is part of the chaos-auto-prompt project. See the main project LICENSE for details.
