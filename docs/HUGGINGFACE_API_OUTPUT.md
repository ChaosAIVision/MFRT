# HuggingFace Optimization API - Output Structure

## Endpoint
```
POST /api/optimize/huggingface
```

## Response Model: `HuggingFaceOptimizeResponse`

### Complete JSON Output Structure

```json
{
  "success": true,
  "initial_prompt": "Original system prompt from dataset",
  "optimized_prompt": "Improved prompt after optimization with detailed instructions and examples",

  "dataset_info": {
    "name": "user/my-dataset",
    "config": null,
    "split": "train",
    "num_samples": 10,
    "columns": ["system_prompt", "input", "output", "quality", "explanation"],
    "system_prompt_column": "system_prompt",
    "input_column": "input",
    "output_column": "output"
  },

  "usage_summary": {
    "total_cost": 0.0025,
    "total_input_tokens": 1500,
    "total_output_tokens": 500,
    "total_tokens": 2000,
    "budget_limit": 5.0,
    "remaining_budget": 4.9975,
    "budget_usage_percentage": 0.05
  },

  "iterations": [
    {
      "iteration": 1,
      "prompt": "Optimized prompt from iteration 1",
      "input_tokens": 750,
      "output_tokens": 250,
      "cost": 0.00125
    },
    {
      "iteration": 2,
      "prompt": "Further optimized prompt from iteration 2",
      "input_tokens": 750,
      "output_tokens": 250,
      "cost": 0.00125
    }
  ],

  "metrics": {
    "quality": {
      "distribution": {
        "excellent": 3,
        "good": 5,
        "poor": 2
      }
    },
    "explanation": {
      "distribution": {
        "clear and detailed": 6,
        "needs improvement": 4
      }
    }
  }
}
```

---

## Field Descriptions

### 1. **`success`** (boolean)
- Indicates whether optimization succeeded
- `true` = thành công, `false` = thất bại

**Example:**
```json
"success": true
```

---

### 2. **`initial_prompt`** (string)
- System prompt gốc từ dataset (từ cột `system_prompt_column`)
- Đây là prompt TRƯỚC KHI optimize

**Example:**
```json
"initial_prompt": "Bạn là chatbot bán sữa Abbott. Trả lời: {customer_question}"
```

---

### 3. **`optimized_prompt`** (string)
- System prompt SAU KHI optimize
- Thường dài hơn, chi tiết hơn, có ví dụ cụ thể
- Đây là kết quả chính của API

**Example:**
```json
"optimized_prompt": "Bạn là chatbot bán sữa Abbott chuyên nghiệp, thân thiện và tận tâm. Trả lời rõ ràng, chính xác và đầy đủ cho câu hỏi: {customer_question}.\n\n- Luôn cung cấp thông tin cụ thể về các loại sữa Abbott phù hợp...\n- Nếu câu hỏi về giá cả, cung cấp thông tin chi tiết...\n\nVí dụ 1:\nCustomer: \"Sữa Abbott nào cho bé 1 tuổi?\"\nChatbot: \"Chào bạn! Đối với trẻ 1 tuổi, Abbott Grow hoặc Similac...\""
```

---

### 4. **`dataset_info`** (object)
Thông tin về dataset đã sử dụng

#### Fields:
- **`name`** - Tên dataset trên HuggingFace Hub
- **`config`** - Dataset configuration (nếu có)
- **`split`** - Split đã dùng (train/validation/test)
- **`num_samples`** - Số lượng samples đã xử lý
- **`columns`** - Danh sách tất cả columns trong dataset (bao gồm feedback columns đã generate)
- **`system_prompt_column`** - Column chứa system prompt
- **`input_column`** - Column chứa input
- **`output_column`** - Column chứa expected output

**Example:**
```json
"dataset_info": {
  "name": "user/abbott-chatbot",
  "config": null,
  "split": "train",
  "num_samples": 10,
  "columns": ["system_prompt", "customer_question", "chatbot_answer", "quality", "explanation"],
  "system_prompt_column": "system_prompt",
  "input_column": "customer_question",
  "output_column": "chatbot_answer"
}
```

---

### 5. **`usage_summary`** (object)
Thống kê token usage và cost

#### Fields:
- **`total_cost`** (float) - Tổng chi phí (USD)
- **`total_input_tokens`** (int) - Tổng input tokens
- **`total_output_tokens`** (int) - Tổng output tokens
- **`total_tokens`** (int) - Tổng tokens (input + output)
- **`budget_limit`** (float) - Budget giới hạn (USD)
- **`remaining_budget`** (float) - Budget còn lại (USD)
- **`budget_usage_percentage`** (float) - % budget đã dùng (0-100)

**Example:**
```json
"usage_summary": {
  "total_cost": 0.0004,
  "total_input_tokens": 2233,
  "total_output_tokens": 383,
  "total_tokens": 2616,
  "budget_limit": 2.0,
  "remaining_budget": 1.9996,
  "budget_usage_percentage": 0.02
}
```

---

### 6. **`iterations`** (array)
Chi tiết từng iteration của quá trình optimization

#### Array of objects with fields:
- **`iteration`** (int) - Số thứ tự iteration (1, 2, 3...)
- **`prompt`** (string) - Prompt sau iteration này
- **`input_tokens`** (int) - Input tokens dùng trong iteration
- **`output_tokens`** (int) - Output tokens dùng trong iteration
- **`cost`** (float) - Chi phí iteration (USD)

**Example:**
```json
"iterations": [
  {
    "iteration": 1,
    "prompt": "First optimization attempt...",
    "input_tokens": 1200,
    "output_tokens": 300,
    "cost": 0.0002
  },
  {
    "iteration": 2,
    "prompt": "Second optimization attempt...",
    "input_tokens": 1033,
    "output_tokens": 83,
    "cost": 0.0002
  }
]
```

---

### 7. **`metrics`** (object, optional)
Quality metrics từ feedback columns

#### Structure:
- Key = tên feedback column
- Value = metrics cho column đó

**Numeric columns:** (quality scores, ratings, etc.)
```json
{
  "accuracy_score": {
    "mean": 0.85,
    "std": 0.12,
    "min": 0.5,
    "max": 1.0
  }
}
```

**Categorical columns:** (labels, classifications, etc.)
```json
{
  "quality": {
    "distribution": {
      "excellent": 3,
      "good": 5,
      "poor": 2
    }
  }
}
```

**Complete Example:**
```json
"metrics": {
  "quality": {
    "distribution": {
      "poor": 5,
      "good": 5
    }
  },
  "explanation": {
    "distribution": {
      "Câu trả lời thiếu chính xác...": 3,
      "Thông tin đầy đủ và chính xác": 4,
      "Câu trả lời đúng về mục đích...": 3
    }
  }
}
```

---

## Real Example from Abbott Chatbot

Đây là output thực tế từ example đã chạy:

```json
{
  "success": true,
  "initial_prompt": "Bạn là chatbot bán sữa Abbott. Trả lời: {customer_question}",
  "optimized_prompt": "Bạn là chatbot bán sữa Abbott chuyên nghiệp, thân thiện và tận tâm. Trả lời rõ ràng, chính xác và đầy đủ cho câu hỏi: {customer_question}.\n\n- Luôn cung cấp thông tin cụ thể về các loại sữa Abbott phù hợp với nhu cầu khách hàng, nêu bật các đặc điểm, lợi ích, độ tuổi phù hợp hoặc các dòng sản phẩm đặc biệt (ví dụ: dành cho trẻ sơ sinh, người cao tuổi, người tiểu đường, hỗ trợ tăng cân, v.v.).\n- Nếu câu hỏi liên quan đến giá cả, khuyến mãi hoặc chính sách, hãy cung cấp thông tin rõ ràng, chi tiết, có thể thêm các hướng dẫn mua hàng hoặc liên hệ hỗ trợ.\n- Tránh trả lời mập mờ, không đầy đủ hoặc chỉ theo dạng xác nhận chung chung.\n- Đạt tiêu chuẩn tư vấn thân thiện, chuyên nghiệp, gây tín nhiệm cao và khuyến khích khách hàng tương tác tiếp.\n\nDưới đây là ví dụ minh họa để hướng dẫn trả lời phù hợp:\n\nVí dụ 1:\nCustomer: \"Sữa Abbott nào phù hợp cho bé 1 tuổi?\"\nChatbot: \"Chào bạn! Đối với trẻ 1 tuổi, bạn có thể cân nhắc các dòng sữa Abbott như Abbott Grow hoặc Similac Milo, giúp hỗ trợ sự phát triển toàn diện, bổ sung dưỡng chất cần thiết cho trẻ trong giai đoạn này. Nếu bạn cần tư vấn chọn loại phù hợp hoặc có thêm câu hỏi, vui lòng cho biết nhé!\"\n\nVí dụ 2:\nCustomer: \"Giá sữa Ensure Gold là bao nhiêu?\"\nChatbot: \"Chào bạn! Hiện tại, giá sữa Ensure Gold khoảng 800,000 VND cho hộp 850g. Bạn cần biết thêm về chương trình khuyến mãi hoặc mua hàng, cứ liên hệ theo để được hỗ trợ tốt nhất nhé!\"\n\nLưu ý: Luôn diễn đạt thân thiện, rõ ràng, giúp khách hàng cảm thấy được hỗ trợ tận tình và chuyên nghiệp trong mọi câu trả lời.",
  "dataset_info": {
    "name": "local_dataset",
    "config": null,
    "split": "train",
    "num_samples": 10,
    "columns": ["customer_question", "chatbot_answer", "quality", "explanation"],
    "system_prompt_column": "system_prompt",
    "input_column": "customer_question",
    "output_column": "chatbot_answer"
  },
  "usage_summary": {
    "total_cost": 0.0004,
    "total_input_tokens": 2233,
    "total_output_tokens": 383,
    "total_tokens": 2616,
    "budget_limit": 2.0,
    "remaining_budget": 1.9996,
    "budget_usage_percentage": 0.02
  },
  "iterations": [],
  "metrics": {
    "quality": {
      "distribution": {
        "poor": 5,
        "good": 5
      }
    }
  }
}
```

---

## Key Takeaways

1. **Main output**: `optimized_prompt` - đây là kết quả chính bạn cần
2. **Cost tracking**: `usage_summary` - theo dõi chi phí và tokens
3. **Quality metrics**: `metrics` - đánh giá chất lượng responses
4. **Dataset info**: `dataset_info` - metadata về dataset đã dùng
5. **Comparison**: So sánh `initial_prompt` vs `optimized_prompt` để thấy sự cải thiện

---

## How to Use Output

### 1. Lấy optimized prompt:
```python
optimized_prompt = response["optimized_prompt"]
# Sử dụng prompt này trong production
```

### 2. Kiểm tra cost:
```python
cost = response["usage_summary"]["total_cost"]
if cost > budget_threshold:
    print("Warning: Over budget!")
```

### 3. Xem metrics:
```python
quality_dist = response["metrics"]["quality"]["distribution"]
print(f"Good: {quality_dist.get('good', 0)}")
print(f"Poor: {quality_dist.get('poor', 0)}")
```

### 4. Track iterations:
```python
for iteration in response["iterations"]:
    print(f"Iteration {iteration['iteration']}: ${iteration['cost']}")
```
