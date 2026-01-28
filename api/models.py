"""
API request and response models for chaos-auto-prompt.

This module defines Pydantic models for type-safe API communication,
including request validation, response serialization, and error handling.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Base Models
# =============================================================================

class Message(BaseModel):
    """
    A message in a chat conversation.

    Attributes:
        role: The role of the message sender (system, user, assistant)
        content: The content of the message
    """

    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that role is one of the allowed values."""
        allowed_roles = {"system", "user", "assistant"}
        if v not in allowed_roles:
            raise ValueError(f"role must be one of {allowed_roles}, got '{v}'")
        return v


class ModelCapabilities(BaseModel):
    """
    Model capabilities and constraints.

    Attributes:
        supports_text: Whether the model supports text generation
        supports_images: Whether the model supports image inputs
        supports_grounding: Whether the model supports grounded generation
        max_tokens: Maximum tokens the model can handle
        cost_per_1k_tokens: Cost per 1000 tokens in USD
    """

    supports_text: bool = Field(default=True, description="Whether model supports text generation")
    supports_images: bool = Field(default=False, description="Whether model supports image inputs")
    supports_grounding: bool = Field(default=False, description="Whether model supports grounding")
    max_tokens: Optional[int] = Field(default=None, description="Maximum token limit")
    cost_per_1k_tokens: Optional[float] = Field(
        default=None, description="Cost per 1000 tokens in USD"
    )


class PricingInfo(BaseModel):
    """
    Pricing information for a model.

    Attributes:
        input_price: Price per 1M input tokens in USD
        output_price: Price per 1M output tokens in USD
        model_name: Name of the model
    """

    input_price: float = Field(..., description="Price per 1M input tokens", gt=0)
    output_price: float = Field(..., description="Price per 1M output tokens", gt=0)
    model_name: str = Field(..., description="Model identifier")


# =============================================================================
# Request Models
# =============================================================================

class DatasetInput(BaseModel):
    """
    Dataset input for optimization.

    Supports multiple input formats for flexibility:
    - JSON string
    - List of dictionaries
    - Dictionary with column arrays

    Attributes:
        data: Dataset data (JSON string, list of dicts, or dict with columns)
        format: Data format hint ('json', 'list', 'columns')
    """

    data: Union[str, List[Dict[str, Any]], Dict[str, List[Any]]] = Field(
        ...,
        description="Dataset data as JSON string, list of records, or column-oriented dict",
    )
    format: Optional[str] = Field(
        default=None,
        description="Data format hint: 'json', 'list', or 'columns' (auto-detected if not provided)",
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate format if provided."""
        if v is not None:
            allowed_formats = {"json", "list", "columns"}
            if v not in allowed_formats:
                raise ValueError(f"format must be one of {allowed_formats}, got '{v}'")
        return v


class PromptFormat(BaseModel):
    """
    Prompt format specification.

    Supports both string prompts and message-based prompts.

    Attributes:
        prompt: The prompt as a string or list of messages
        type: Prompt type ('string' or 'messages')
    """

    prompt: Union[str, List[Message]] = Field(
        ..., description="Prompt as string or list of messages"
    )
    type: Optional[str] = Field(
        default=None,
        description="Prompt type: 'string' or 'messages' (auto-detected if not provided)",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate type if provided."""
        if v is not None:
            allowed_types = {"string", "messages"}
            if v not in allowed_types:
                raise ValueError(f"type must be one of {allowed_types}, got '{v}'")
        return v


class EvaluatorConfig(BaseModel):
    """
    Configuration for automatic feedback generation via evaluators.

    Attributes:
        type: Evaluator type ('classification', 'custom')
        feedback_column: Primary feedback column name (e.g., 'correctness')
        model: Model to use for evaluation
        prompt_template: Template for evaluation prompt (with {placeholders})
        choices: For classification evaluators, label-to-score mapping
        include_explanation: Whether to generate explanation column
    """

    type: str = Field(..., description="Evaluator type: 'classification'")
    feedback_column: str = Field(..., description="Primary feedback column name")
    model: str = Field(default="gpt-4o", description="Model for evaluation")
    prompt_template: str = Field(..., description="Evaluation prompt template")
    choices: Optional[Dict[str, int]] = Field(
        default=None,
        description="For classification: label-to-score mapping"
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to include explanation column"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate evaluator type."""
        allowed_types = {"classification"}
        if v not in allowed_types:
            raise ValueError(f"type must be one of {allowed_types}, got '{v}'")
        return v


class OptimizeRequest(BaseModel):
    """
    Request model for prompt optimization.

    This is the main request model for the optimization endpoint.
    It includes all parameters needed to optimize a prompt using
    natural language feedback.

    Attributes:
        prompt: The prompt to optimize (string or list of messages)
        dataset: Dataset with examples and feedback
        output_column: Name of column containing LLM outputs
        feedback_columns: List of column names with feedback/evaluation
        model: Model to use for optimization (default: from settings)
        provider: Provider to use ('openai', 'google', etc.)
        budget: Maximum budget in USD (default: from settings)
        context_size: Context window size in tokens (default: from model)
        ruleset: Optional initial ruleset for coding agent optimization
        verbose: Enable verbose logging
        validation_only: Only validate inputs without optimizing
    """

    prompt: Union[str, List[Message]] = Field(
        ...,
        description="The prompt to optimize as a string or list of messages",
    )
    dataset: Union[DatasetInput, List[Dict[str, Any]], str] = Field(
        ...,
        description="Dataset with examples, outputs, and feedback",
    )
    output_column: str = Field(
        ...,
        description="Name of column containing LLM outputs to evaluate",
    )
    feedback_columns: List[str] = Field(
        default=[],
        description="List of column names containing feedback/evaluation data (optional if evaluators provided)",
    )
    evaluators: Optional[List[EvaluatorConfig]] = Field(
        default=None,
        description="Optional evaluators for automatic feedback generation (alternative to pre-computed feedback)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for optimization (default: from settings)",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Provider to use ('openai', 'google', etc.)",
    )
    budget: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum budget in USD (default: from settings)",
    )
    context_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Context window size in tokens (default: from model)",
    )
    ruleset: Optional[str] = Field(
        default=None,
        description="Initial ruleset for coding agent optimization mode",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging for optimization process",
    )
    validation_only: bool = Field(
        default=False,
        description="Only validate inputs without running optimization",
    )


class ProviderRequest(BaseModel):
    """
    Base request model for provider-specific operations.

    Attributes:
        api_key: API key for the provider (optional, uses default if not provided)
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    """

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (uses default if not provided)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model identifier (uses default if not provided)",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2, uses default if not provided)",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens to generate (uses default if not provided)",
    )


class OpenAIRequest(ProviderRequest):
    """
    OpenAI-specific request model.

    Extends ProviderRequest with OpenAI-specific options.

    Attributes:
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
    """

    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0-1)",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty (-2.0 to 2.0)",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Presence penalty (-2.0 to 2.0)",
    )


class GoogleRequest(ProviderRequest):
    """
    Google AI-specific request model.

    Extends ProviderRequest with Google-specific options.

    Attributes:
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
    """

    top_k: Optional[int] = Field(
        default=None,
        gt=0,
        description="Top-k sampling parameter",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0-1)",
    )


# =============================================================================
# Response Models
# =============================================================================

class UsageSummary(BaseModel):
    """
    Token and cost usage summary.

    Attributes:
        total_cost: Total cost in USD
        total_input_tokens: Total input tokens used
        total_output_tokens: Total output tokens used
        total_tokens: Total tokens (input + output)
        budget_limit: Budget limit in USD
        remaining_budget: Remaining budget in USD
        budget_usage_percentage: Percentage of budget used
    """

    total_cost: float = Field(..., description="Total cost in USD", ge=0)
    total_input_tokens: int = Field(..., description="Total input tokens", ge=0)
    total_output_tokens: int = Field(..., description="Total output tokens", ge=0)
    total_tokens: int = Field(..., description="Total tokens (input + output)", ge=0)
    budget_limit: float = Field(..., description="Budget limit in USD", ge=0)
    remaining_budget: float = Field(..., description="Remaining budget in USD", ge=0)
    budget_usage_percentage: float = Field(
        ...,
        description="Percentage of budget used",
        ge=0,
        le=100,
    )


class IterationResult(BaseModel):
    """
    Result of a single optimization iteration.

    Attributes:
        iteration: Iteration number
        prompt: The optimized prompt from this iteration
        cost: Cost of this iteration in USD
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        feedback: Feedback summary (if available)
    """

    iteration: int = Field(..., description="Iteration number", ge=1)
    prompt: Union[str, List[Message]] = Field(
        ..., description="Optimized prompt from this iteration"
    )
    cost: float = Field(..., description="Cost of this iteration in USD", ge=0)
    input_tokens: int = Field(..., description="Input tokens used", ge=0)
    output_tokens: int = Field(..., description="Output tokens used", ge=0)
    feedback: Optional[str] = Field(
        default=None, description="Feedback summary (if available)"
    )


class OptimizeResponse(BaseModel):
    """
    Response model for prompt optimization.

    Contains the optimized prompt along with metadata about
    the optimization process.

    Attributes:
        optimized_prompt: The optimized prompt (same format as input)
        original_prompt: The original prompt before optimization
        cost: Total optimization cost in USD
        iterations: Number of optimization iterations
        usage: Token and cost usage summary
        iterations_details: Detailed results for each iteration
        model: Model used for optimization
        provider: Provider used for optimization
        success: Whether optimization was successful
        message: Optional message or warning
        metrics_before: Quality metrics BEFORE optimization (baseline)
        metrics_after: Quality metrics AFTER optimization (with new prompt)
        improvement: Improvement summary comparing before vs after
    """

    optimized_prompt: Union[str, List[Message]] = Field(
        ..., description="The optimized prompt"
    )
    original_prompt: Union[str, List[Message]] = Field(
        ..., description="The original prompt before optimization"
    )
    cost: float = Field(..., description="Total optimization cost in USD", ge=0)
    iterations: int = Field(..., description="Number of optimization iterations", ge=0)
    usage: UsageSummary = Field(..., description="Token and cost usage summary")
    iterations_details: Optional[List[IterationResult]] = Field(
        default=None, description="Detailed results for each iteration"
    )
    model: str = Field(..., description="Model used for optimization")
    provider: Optional[str] = Field(
        default=None, description="Provider used for optimization"
    )
    success: bool = Field(..., description="Whether optimization was successful")
    message: Optional[str] = Field(
        default=None, description="Optional message or warning"
    )
    metrics_before: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality metrics BEFORE optimization (baseline with original prompt)"
    )
    metrics_after: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality metrics AFTER optimization (with optimized prompt)"
    )
    improvement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Improvement summary (percentage changes, quality gains)"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    Attributes:
        error: Error message describing what went wrong
        detail: Detailed error information
        status_code: HTTP status code
        error_type: Type of error (validation, provider, budget, etc.)
        request_id: Request ID for tracking
    """

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code", ge=100, le=599)
    error_type: str = Field(
        ...,
        description="Type of error: validation, provider, budget, dataset, or internal",
    )
    request_id: Optional[str] = Field(
        default=None, description="Request ID for tracking"
    )


class ValidationErrorResponse(ErrorResponse):
    """
    Validation error response with field-level details.

    Attributes:
        fields: Dictionary of field names to error messages
    """

    fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Field-level validation errors",
    )


class HealthResponse(BaseModel):
    """
    Health check response model.

    Attributes:
        status: Health status ('healthy', 'degraded', 'unhealthy')
        version: API version
        environment: Environment name (dev, staging, production)
        uptime: Server uptime in seconds
        services: Status of external services
    """

    status: str = Field(
        ...,
        description="Health status: 'healthy', 'degraded', or 'unhealthy'",
    )
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    uptime: float = Field(..., description="Server uptime in seconds", ge=0)
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of external services (openai, google, etc.)",
    )


class ModelInfo(BaseModel):
    """
    Model information response.

    Attributes:
        name: Model name/identifier
        provider: Provider name
        capabilities: Model capabilities
        pricing: Pricing information
        context_size: Context window size in tokens
    """

    name: str = Field(..., description="Model name/identifier")
    provider: str = Field(..., description="Provider name")
    capabilities: ModelCapabilities = Field(..., description="Model capabilities")
    pricing: PricingInfo = Field(..., description="Pricing information")
    context_size: int = Field(..., description="Context window size in tokens", gt=0)


class ModelsListResponse(BaseModel):
    """
    Response model for listing available models.

    Attributes:
        models: List of available models
        count: Number of models
        default_model: Default model name
    """

    models: List[ModelInfo] = Field(..., description="List of available models")
    count: int = Field(..., description="Number of models", ge=0)
    default_model: str = Field(..., description="Default model name")


class ValidationResponse(BaseModel):
    """
    Response for input validation.

    Attributes:
        valid: Whether inputs are valid
        errors: List of validation errors (if any)
        warnings: List of validation warnings (if any)
        estimated_cost: Estimated cost for optimization
        estimated_tokens: Estimated token usage
    """

    valid: bool = Field(..., description="Whether inputs are valid")
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings",
    )
    estimated_cost: Optional[float] = Field(
        default=None,
        description="Estimated cost for optimization in USD",
    )
    estimated_tokens: Optional[int] = Field(
        default=None,
        description="Estimated token usage",
    )


class BudgetCheckRequest(BaseModel):
    """
    Request to check if operation is within budget.

    Attributes:
        model: Model to use
        input_tokens: Estimated input tokens
        output_tokens: Estimated output tokens
        budget_limit: Budget limit to check against
    """

    model: str = Field(..., description="Model to use")
    input_tokens: int = Field(..., description="Estimated input tokens", ge=0)
    output_tokens: int = Field(
        default=0,
        description="Estimated output tokens",
        ge=0,
    )
    budget_limit: Optional[float] = Field(
        default=None,
        description="Budget limit to check against (uses default if not provided)",
    )


class BudgetCheckResponse(BaseModel):
    """
    Response for budget check.

    Attributes:
        within_budget: Whether operation is within budget
        estimated_cost: Estimated cost in USD
        remaining_budget: Remaining budget after operation
        budget_limit: Current budget limit
        would_exceed: Whether operation would exceed budget
    """

    within_budget: bool = Field(
        ..., description="Whether operation is within budget"
    )
    estimated_cost: float = Field(..., description="Estimated cost in USD", ge=0)
    remaining_budget: float = Field(
        ..., description="Remaining budget after operation", ge=0
    )
    budget_limit: float = Field(..., description="Current budget limit", ge=0)
    would_exceed: bool = Field(..., description="Whether operation would exceed budget")


# =============================================================================
# Config Models
# =============================================================================

class ProviderConfig(BaseModel):
    """
    Provider configuration model.

    Attributes:
        api_key: API key for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        default_model: Default model to use
    """

    api_key: str = Field(..., description="API key for authentication")
    timeout: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts",
    )
    default_model: Optional[str] = Field(
        default=None,
        description="Default model to use",
    )


# =============================================================================
# HuggingFace Dataset Optimization Models
# =============================================================================

class HuggingFaceOptimizeRequest(BaseModel):
    """
    Request model for HuggingFace dataset-based prompt optimization.

    This endpoint loads a dataset from HuggingFace Hub and optimizes a prompt
    based on input-output pairs with optional evaluator-based feedback.

    Attributes:
        dataset_name: HuggingFace dataset name (e.g., 'squad', 'user/dataset')
        dataset_config: Optional dataset configuration name
        dataset_split: Dataset split to use (default: 'train')
        system_prompt_column: Column name containing system prompts (default: 'system_prompt')
        input_column: Column name containing inputs (default: 'input')
        output_column: Column name containing expected outputs (default: 'output')
        feedback_columns: Optional pre-existing feedback columns
        evaluators: Optional evaluators for automatic feedback generation
        model: Model to use for optimization (default: from settings)
        provider: Provider to use ('openai', 'google')
        budget: Maximum budget in USD
        max_samples: Maximum number of samples to use from dataset
        verbose: Enable verbose logging
    """

    dataset_name: str = Field(
        ...,
        description="HuggingFace dataset name (e.g., 'squad', 'user/my-dataset')"
    )
    dataset_config: Optional[str] = Field(
        default=None,
        description="Dataset configuration name (if applicable)"
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use (train, validation, test)"
    )
    system_prompt_column: str = Field(
        default="system_prompt",
        description="Column name containing system prompts"
    )
    input_column: str = Field(
        default="input",
        description="Column name containing inputs"
    )
    output_column: str = Field(
        default="output",
        description="Column name containing expected outputs"
    )
    feedback_columns: List[str] = Field(
        default=[],
        description="Pre-existing feedback column names (optional if evaluators provided)"
    )
    evaluators: Optional[List[EvaluatorConfig]] = Field(
        default=None,
        description="Optional evaluators for automatic feedback generation"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use for optimization (uses default if not provided)"
    )
    provider: Optional[str] = Field(
        default="openai",
        description="Provider to use: 'openai' or 'google'"
    )
    budget: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum budget in USD (uses default if not provided)"
    )
    max_samples: Optional[int] = Field(
        default=100,
        gt=0,
        le=10000,
        description="Maximum number of samples to use from dataset"
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )
    train_split: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of data for training (0.5-1.0). Set to 1.0 to use full dataset."
    )
    stratify: bool = Field(
        default=True,
        description="Whether to stratify train/test split by output column"
    )
    max_loops: Optional[int] = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of optimization loops (1-10)"
    )
    threshold: Optional[float] = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Target accuracy threshold to stop optimization early (0.0-1.0)"
    )


class HuggingFaceOptimizeResponse(BaseModel):
    """
    Response model for HuggingFace dataset optimization.

    Attributes:
        success: Whether optimization succeeded
        initial_prompt: Original system prompt from dataset
        optimized_prompt: Optimized prompt
        dataset_info: Information about the dataset used
        usage_summary: Token and cost usage
        iterations: Details of each optimization iteration
        metrics_before: Quality metrics BEFORE optimization (baseline on train set)
        metrics_after: Quality metrics AFTER optimization (on test set)
        improvement: Improvement summary comparing before vs after
        train_metrics: Metrics history on train set per iteration
        test_metrics: Metrics history on test set per iteration
        prompts_history: All prompts generated during optimization
        num_loops: Number of optimization loops executed
        stopped_reason: Why optimization stopped (max_loops, threshold_reached, or early_return)
    """

    success: bool = Field(..., description="Whether optimization succeeded")
    initial_prompt: str = Field(..., description="Original system prompt")
    optimized_prompt: str = Field(..., description="Optimized prompt")
    dataset_info: Dict[str, Any] = Field(
        ...,
        description="Dataset metadata (name, split, num_samples, columns)"
    )
    usage_summary: UsageSummary = Field(..., description="Token and cost usage")
    iterations: List[IterationResult] = Field(
        default=[],
        description="Optimization iteration details"
    )
    metrics_before: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality metrics BEFORE optimization (baseline with initial prompt on train set)"
    )
    metrics_after: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality metrics AFTER optimization (with optimized prompt on test set)"
    )
    improvement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Improvement summary (percentage changes, quality gains)"
    )
    train_metrics: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metrics history on train set for each iteration"
    )
    test_metrics: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metrics history on test set for each iteration"
    )
    prompts_history: Optional[List[str]] = Field(
        default=None,
        description="All prompts generated during optimization (initial + optimized per iteration)"
    )
    num_loops: Optional[int] = Field(
        default=None,
        description="Number of optimization loops executed"
    )
    stopped_reason: Optional[str] = Field(
        default=None,
        description="Why optimization stopped: 'max_loops', 'threshold_reached', or 'early_return'"
    )


# =============================================================================
# Export all models
# =============================================================================

__all__ = [
    # Base models
    "Message",
    "ModelCapabilities",
    "PricingInfo",
    # Request models
    "DatasetInput",
    "PromptFormat",
    "EvaluatorConfig",
    "OptimizeRequest",
    "HuggingFaceOptimizeRequest",
    "ProviderRequest",
    "OpenAIRequest",
    "GoogleRequest",
    # Response models
    "UsageSummary",
    "IterationResult",
    "OptimizeResponse",
    "HuggingFaceOptimizeResponse",
    "ErrorResponse",
    "ValidationErrorResponse",
    "HealthResponse",
    "ModelInfo",
    "ModelsListResponse",
    "ValidationResponse",
    "BudgetCheckRequest",
    "BudgetCheckResponse",
    # Config models
    "ProviderConfig",
]
