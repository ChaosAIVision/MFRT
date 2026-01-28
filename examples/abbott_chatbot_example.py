"""
Abbott Milk Sales Chatbot - Dataset & Evaluation Flow

This example demonstrates:
1. Creating a dataset for Abbott milk sales chatbot
2. Using evaluator to automatically generate feedback
3. Optimizing the chatbot prompt based on feedback
"""

import asyncio
import pandas as pd
from chaos_auto_prompt.optimizers import PromptLearningOptimizer
from chaos_auto_prompt.evaluators import ClassificationEvaluator
from chaos_auto_prompt.config import get_settings

settings = get_settings()


# Dataset: 10 customer questions about Abbott milk products
# Real scenarios from Vietnamese market
dataset = pd.DataFrame([
    {
        "customer_question": "Sữa Abbott nào tốt cho trẻ 1 tuổi?",
        "chatbot_answer": "Abbott Grow có thể phù hợp cho bé 1 tuổi."
    },
    {
        "customer_question": "Giá sữa Ensure Gold bao nhiêu?",
        "chatbot_answer": "Khoảng 800,000 VND/hộp 850g."
    },
    {
        "customer_question": "Sữa Similac có tốt cho trẻ sơ sinh không?",
        "chatbot_answer": "Có, Similac được thiết kế đặc biệt cho trẻ sơ sinh với hệ tiêu hóa non yếu."
    },
    {
        "customer_question": "Tôi muốn mua sữa cho bà ngoại 70 tuổi, nên chọn loại nào?",
        "chatbot_answer": "Ensure Gold là lựa chọn tốt cho người cao tuổi."
    },
    {
        "customer_question": "Sữa PediaSure có giúp trẻ tăng cân không?",
        "chatbot_answer": "PediaSure được thiết kế đặc biệt để hỗ trợ tăng trưởng chiều cao và cân nặng cho trẻ biếng ăn."
    },
    {
        "customer_question": "Có giao hàng tận nhà không?",
        "chatbot_answer": "Vâng."
    },
    {
        "customer_question": "Sữa Abbott Grow và Similac khác nhau thế nào?",
        "chatbot_answer": "Similac cho trẻ 0-3 tuổi, Grow cho trẻ lớn hơn."
    },
    {
        "customer_question": "Tôi có thể đổi trả nếu con không uống được không?",
        "chatbot_answer": "Có thể đổi trả trong vòng 7 ngày nếu sản phẩm chưa mở nắp và còn nguyên tem."
    },
    {
        "customer_question": "Sữa Ensure có bị tiểu đường uống được không?",
        "chatbot_answer": "Ensure có dòng Ensure Diabetes Care dành riêng cho người tiểu đường với đường huyết ổn định."
    },
    {
        "customer_question": "Khuyến mãi gì trong tháng này?",
        "chatbot_answer": "Mua 2 tặng 1 cho tất cả dòng sữa Abbott."
    }
])

print("📊 Abbott Milk Chatbot Dataset")
print("=" * 80)
print(f"Total samples: {len(dataset)}")
print("\nSample questions:")
for i, row in dataset.head(3).iterrows():
    print(f"\n{i+1}. Q: {row['customer_question']}")
    print(f"   A: {row['chatbot_answer']}")
print("\n" + "=" * 80)


async def main():
    print("\n🤖 STEP 1: Creating Evaluator for Abbott Chatbot")
    print("=" * 80)

    # Create evaluator to assess chatbot quality
    evaluator = ClassificationEvaluator(
        feedback_column="quality",
        model=settings.openai_default_model,  # Use model from .env
        prompt_template="""
        Bạn là chuyên gia đánh giá chatbot bán hàng.

        Câu hỏi khách hàng: {customer_question}
        Câu trả lời chatbot: {chatbot_answer}

        Đánh giá chất lượng câu trả lời dựa trên:
        1. Độ chính xác thông tin về sản phẩm Abbott
        2. Tính chuyên nghiệp và thân thiện
        3. Độ chi tiết phù hợp
        4. Khả năng tư vấn bán hàng

        Trả về JSON:
        {{
            "quality": "excellent" hoặc "good" hoặc "poor",
            "explanation": "giải thích ngắn gọn tại sao đánh giá như vậy và gợi ý cải thiện"
        }}
        """,
        choices={"excellent": 2, "good": 1, "poor": 0},
        include_explanation=True
    )

    print("✅ Evaluator created")
    print(f"   - Model: {settings.openai_default_model}")
    print(f"   - Feedback columns: quality, explanation")
    print(f"   - Concurrency: 20 (parallel evaluation)")

    # Initial prompt (not very good)
    initial_prompt = """Bạn là chatbot bán sữa Abbott. Trả lời: {customer_question}"""

    print(f"\n📝 Initial Prompt: '{initial_prompt}'")

    # Create optimizer
    optimizer = PromptLearningOptimizer(
        prompt=initial_prompt,
        model_choice=settings.openai_default_model,  # Use model from .env
        budget_limit=2.0,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("🔍 STEP 2: Running Evaluator (Generating Feedback)")
    print("=" * 80)

    # Run evaluator to generate feedback
    dataset_with_feedback, feedback_cols = await optimizer.run_evaluators(
        dataset=dataset,
        evaluators=[evaluator]
    )

    print("\n✅ Feedback generated!")
    print(f"   Feedback columns: {feedback_cols}")

    # Show feedback results
    print("\n📊 Evaluation Results:")
    print("-" * 80)
    for i, row in dataset_with_feedback.head(5).iterrows():
        print(f"\n{i+1}. Q: {row['customer_question'][:50]}...")
        print(f"   Quality: {row['quality']}")
        print(f"   Explanation: {row['explanation'][:80]}...")

    # Quality distribution
    quality_counts = dataset_with_feedback['quality'].value_counts()
    print("\n📈 Quality Distribution:")
    for quality, count in quality_counts.items():
        print(f"   {quality}: {count} samples")

    print("\n" + "=" * 80)
    print("🚀 STEP 3: Optimizing Prompt")
    print("=" * 80)

    # Optimize prompt
    optimized_prompt = await optimizer.optimize(
        dataset=dataset_with_feedback,
        output_column="chatbot_answer",
        feedback_columns=feedback_cols
    )

    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)

    print(f"\n🔴 BEFORE (Initial Prompt):")
    print(f"   {initial_prompt}")

    print(f"\n🟢 AFTER (Optimized Prompt):")
    print(f"   {optimized_prompt}")

    print(f"\n💰 Cost Summary:")
    print(f"   Total cost: ${optimizer.pricing_calculator.get_total_cost():.4f}")
    print(f"   Budget used: {optimizer.pricing_calculator.get_total_cost() / optimizer.pricing_calculator.budget_limit * 100:.1f}%")
    print(f"   Remaining: ${optimizer.pricing_calculator.get_remaining_budget():.4f}")

    usage = optimizer.pricing_calculator.get_usage_summary()
    print(f"\n📈 Token Usage:")
    print(f"   Input tokens: {usage['total_input_tokens']:,}")
    print(f"   Output tokens: {usage['total_output_tokens']:,}")
    print(f"   Total tokens: {usage['total_tokens']:,}")

    print("\n" + "=" * 80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 80)

    # Test with new prompt
    print("\n🧪 STEP 4: Testing Optimized Prompt")
    print("=" * 80)

    test_question = "Sữa nào tốt nhất cho bé 2 tuổi biếng ăn?"
    print(f"\n❓ Test Question: {test_question}")
    print(f"\nNow you can use the optimized prompt to generate better responses!")

    return {
        "initial_prompt": initial_prompt,
        "optimized_prompt": optimized_prompt,
        "dataset_with_feedback": dataset_with_feedback,
        "cost": optimizer.pricing_calculator.get_total_cost(),
        "usage": usage
    }


if __name__ == "__main__":
    print("\n" + "🏥 ABBOTT MILK CHATBOT - PROMPT OPTIMIZATION ".center(80, "="))
    print("Using Evaluator Pattern with LLM-as-a-Judge\n")

    results = asyncio.run(main())

    print(f"\n💾 Results saved to memory. You can access:")
    print(f"   - results['initial_prompt']")
    print(f"   - results['optimized_prompt']")
    print(f"   - results['dataset_with_feedback']")
