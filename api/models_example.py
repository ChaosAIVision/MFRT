"""
Example usage of API models for chaos-auto-prompt.

This file demonstrates how to use the Pydantic models defined in models.py
for request validation and response serialization.
"""

from typing import List
from api.models import (
    OptimizeRequest,
    OptimizeResponse,
    Message,
    DatasetInput,
    UsageSummary,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelCapabilities,
    PricingInfo,
)


def example_optimize_request():
    """Example: Create an optimization request."""
    
    # Example 1: String prompt
    request = OptimizeRequest(
        prompt="You are a helpful assistant. Answer: {question}",
        dataset=[
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "feedback": "correct"
            },
            {
                "question": "What is 2+2?",
                "answer": "5",
                "feedback": "incorrect - the answer is 4"
            }
        ],
        output_column="answer",
        feedback_columns=["feedback"],
        model="gpt-4o",
        budget=5.0,
    )
    print("✓ Created OptimizeRequest with string prompt")
    return request


def example_message_prompt():
    """Example: Create a request with message-based prompt."""
    
    request = OptimizeRequest(
        prompt=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Answer: {question}"),
        ],
        dataset={
            "data": [
                {"question": "What is AI?", "answer": "...", "feedback": "good"}
            ],
            "format": "list"
        },
        output_column="answer",
        feedback_columns=["feedback"],
    )
    print("✓ Created OptimizeRequest with message-based prompt")
    return request


def example_optimize_response():
    """Example: Create an optimization response."""
    
    response = OptimizeResponse(
        optimized_prompt="You are an expert assistant. Provide accurate answers: {question}",
        original_prompt="You are a helpful assistant. Answer: {question}",
        cost=0.0234,
        iterations=2,
        usage=UsageSummary(
            total_cost=0.0234,
            total_input_tokens=1250,
            total_output_tokens=890,
            total_tokens=2140,
            budget_limit=5.0,
            remaining_budget=4.9766,
            budget_usage_percentage=0.468,
        ),
        model="gpt-4o",
        provider="openai",
        success=True,
        message="Optimization completed successfully",
    )
    print("✓ Created OptimizeResponse")
    return response


def example_error_response():
    """Example: Create an error response."""
    
    error = ErrorResponse(
        error="Budget exceeded",
        detail="The optimization would cost $10.50, which exceeds the budget limit of $5.00",
        status_code=400,
        error_type="budget",
        request_id="req_12345",
    )
    print("✓ Created ErrorResponse")
    return error


def example_health_response():
    """Example: Create a health check response."""
    
    health = HealthResponse(
        status="healthy",
        version="1.0.0",
        environment="production",
        uptime=3600.5,
        services={
            "openai": "healthy",
            "google": "healthy",
            "database": "healthy",
        },
    )
    print("✓ Created HealthResponse")
    return health


def example_model_info():
    """Example: Create model information."""
    
    model_info = ModelInfo(
        name="gpt-4o",
        provider="openai",
        capabilities=ModelCapabilities(
            supports_text=True,
            supports_images=True,
            supports_grounding=False,
            max_tokens=128000,
            cost_per_1k_tokens=0.005,
        ),
        pricing=PricingInfo(
            input_price=2.50,
            output_price=10.0,
            model_name="gpt-4o",
        ),
        context_size=128000,
    )
    print("✓ Created ModelInfo")
    return model_info


def example_validation():
    """Example: Validate request data."""
    from pydantic import ValidationError
    
    try:
        # This will fail validation (empty feedback_columns)
        invalid_request = OptimizeRequest(
            prompt="Test prompt",
            dataset=[],
            output_column="output",
            feedback_columns=[],  # Empty list - should fail!
        )
    except ValidationError as e:
        print("✓ Validation caught error correctly:")
        print(f"  Error: {e.error_count()} validation error(s)")
        for error in e.errors():
            print(f"  - {error['loc']}: {error['msg']}")


def example_json_serialization():
    """Example: JSON serialization of models."""
    
    request = example_optimize_request()
    
    # Convert to JSON
    json_str = request.model_dump_json(indent=2)
    print("✓ JSON serialization:")
    print(json_str[:200] + "...")
    
    # Convert from JSON
    request_dict = request.model_dump()
    print("\n✓ Model to dict conversion:")
    print(f"  Keys: {list(request_dict.keys())[:5]}...")


if __name__ == "__main__":
    print("=== API Models Examples ===\n")
    
    print("1. Optimization Request:")
    example_optimize_request()
    
    print("\n2. Message-based Prompt:")
    example_message_prompt()
    
    print("\n3. Optimization Response:")
    example_optimize_response()
    
    print("\n4. Error Response:")
    example_error_response()
    
    print("\n5. Health Response:")
    example_health_response()
    
    print("\n6. Model Info:")
    example_model_info()
    
    print("\n7. Validation Example:")
    example_validation()
    
    print("\n8. JSON Serialization:")
    example_json_serialization()
    
    print("\n✅ All examples completed successfully!")
