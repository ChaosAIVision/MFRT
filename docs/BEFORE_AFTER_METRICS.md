# HuggingFace API - Before/After Metrics Example

## New Output Structure với Before/After Comparison

```json
{
  "success": true,
  "initial_prompt": "Bạn là chatbot bán sữa Abbott. Trả lời: {customer_question}",
  "optimized_prompt": "Bạn là chatbot bán sữa Abbott chuyên nghiệp...",

  "metrics_before": {
    "quality": {
      "distribution": {
        "poor": 5,
        "good": 5
      }
    }
  },

  "metrics_after": {
    "quality": {
      "distribution": {
        "poor": 1,
        "good": 9
      }
    }
  },

  "improvement": {
    "quality": {
      "before_distribution": {"poor": 5, "good": 5},
      "after_distribution": {"poor": 1, "good": 9},
      "positive_count_before": 5,
      "positive_count_after": 9,
      "improvement": 4,
      "before_positive_pct": 50.0,
      "after_positive_pct": 90.0
    }
  }
}
```

---

## Giải thích chi tiết

### 1. **`metrics_before`** - Baseline (TRƯỚC optimize)

Đánh giá chất lượng outputs được tạo từ **INITIAL PROMPT**:

```json
"metrics_before": {
  "quality": {
    "distribution": {
      "poor": 5,   // 5/10 câu trả lời KÉM
      "good": 5    // 5/10 câu trả lời TỐT
    }
  }
}
```

**Timeline:**
```
Dataset có 10 samples
  ↓
Chatbot dùng INITIAL PROMPT trả lời 10 câu
  ↓
Evaluator đánh giá 10 outputs
  ↓
Kết quả: 50% good, 50% poor ← metrics_before
```

---

### 2. **`metrics_after`** - Results (SAU optimize)

Đánh giá chất lượng outputs được tạo từ **OPTIMIZED PROMPT**:

```json
"metrics_after": {
  "quality": {
    "distribution": {
      "poor": 1,   // Chỉ còn 1/10 câu KÉM
      "good": 9    // Tăng lên 9/10 câu TỐT
    }
  }
}
```

**Timeline:**
```
Optimizer tạo OPTIMIZED PROMPT
  ↓
Chatbot dùng OPTIMIZED PROMPT trả lời LẠI 10 câu
  ↓
Evaluator đánh giá lại 10 outputs mới
  ↓
Kết quả: 90% good, 10% poor ← metrics_after
```

---

### 3. **`improvement`** - So sánh Before vs After

Tổng hợp sự cải thiện:

```json
"improvement": {
  "quality": {
    "before_distribution": {"poor": 5, "good": 5},
    "after_distribution": {"poor": 1, "good": 9},

    "positive_count_before": 5,     // 5 câu tốt TRƯỚC
    "positive_count_after": 9,      // 9 câu tốt SAU
    "improvement": 4,               // Cải thiện +4 câu

    "before_positive_pct": 50.0,    // 50% TỐT trước
    "after_positive_pct": 90.0      // 90% TỐT sau
  }
}
```

**Ý nghĩa:**
- **improvement: +4** = Tăng 4 câu trả lời tốt (từ 5 → 9)
- **before_positive_pct: 50%** = Chất lượng ban đầu (baseline)
- **after_positive_pct: 90%** = Chất lượng sau optimize
- **Tổng cải thiện: +40%** (từ 50% → 90%)

---

## Visualization

### Before Optimization:
```
Initial Prompt: "Bạn là chatbot bán sữa Abbott. Trả lời: {customer_question}"

Quality Distribution:
┌────────────────────────────┐
│  ❌ Poor:  5 (50%)         │
│  ✅ Good:  5 (50%)         │
└────────────────────────────┘
```

### After Optimization:
```
Optimized Prompt: "Bạn là chatbot bán sữa Abbott chuyên nghiệp, thân thiện..."
(với hướng dẫn chi tiết + ví dụ cụ thể)

Quality Distribution:
┌────────────────────────────┐
│  ❌ Poor:  1 (10%)         │
│  ✅ Good:  9 (90%)         │
└────────────────────────────┘

📈 IMPROVEMENT: +40% (50% → 90%)
```

---

## Ví dụ với Numeric Metrics

Nếu feedback là điểm số (0-1):

### Before:
```json
"metrics_before": {
  "accuracy_score": {
    "mean": 0.65,
    "std": 0.25,
    "min": 0.2,
    "max": 1.0
  }
}
```

### After:
```json
"metrics_after": {
  "accuracy_score": {
    "mean": 0.88,
    "std": 0.12,
    "min": 0.6,
    "max": 1.0
  }
}
```

### Improvement:
```json
"improvement": {
  "accuracy_score": {
    "before_mean": 0.65,
    "after_mean": 0.88,
    "absolute_change": 0.23,
    "percent_change": 35.38
  }
}
```

**Ý nghĩa:**
- Accuracy tăng từ 65% → 88%
- Absolute change: +0.23 (tăng 23 điểm %)
- Percent change: +35.38% (tăng 35.38% so với baseline)

---

## API Workflow chi tiết

```
STEP 1: Load HuggingFace dataset
  Dataset columns: [system_prompt, input, output]
  ↓

STEP 2: Extract initial_prompt từ dataset
  initial_prompt = dataset[0]["system_prompt"]
  ↓

STEP 3: Create optimizer
  ↓

STEP 4: Run evaluators on EXISTING outputs
  Đánh giá outputs đã có trong dataset
  → Generate feedback columns (quality, explanation)
  ↓

STEP 5: Optimize prompt
  Dựa vào feedback → tạo optimized_prompt
  ↓

STEP 6: Calculate metrics_before ← BASELINE
  metrics_before = quality distribution của outputs cũ
  ↓

STEP 7: Re-generate outputs với OPTIMIZED prompt
  For each input:
    - Dùng optimized_prompt + input
    - Call LLM → new_output
  ↓

STEP 8: Re-run evaluators on NEW outputs
  Đánh giá lại new_outputs
  → Generate new feedback
  ↓

STEP 9: Calculate metrics_after ← RESULTS
  metrics_after = quality distribution của outputs mới
  ↓

STEP 10: Calculate improvement
  Compare metrics_before vs metrics_after
  ↓

STEP 11: Return response với đầy đủ metrics
```

---

## Real Example Output

```json
{
  "success": true,
  "initial_prompt": "Bạn là chatbot bán sữa Abbott. Trả lời: {customer_question}",
  "optimized_prompt": "Bạn là chatbot bán sữa Abbott chuyên nghiệp, thân thiện và tận tâm. Trả lời rõ ràng...",

  "dataset_info": {
    "name": "user/abbott-chatbot",
    "num_samples": 10,
    "columns": ["customer_question", "chatbot_answer", "quality", "explanation"]
  },

  "usage_summary": {
    "total_cost": 0.0012,
    "total_tokens": 5200,
    "budget_usage_percentage": 0.024
  },

  "metrics_before": {
    "quality": {
      "distribution": {"poor": 5, "good": 5}
    }
  },

  "metrics_after": {
    "quality": {
      "distribution": {"poor": 1, "good": 9}
    }
  },

  "improvement": {
    "quality": {
      "before_distribution": {"poor": 5, "good": 5},
      "after_distribution": {"poor": 1, "good": 9},
      "positive_count_before": 5,
      "positive_count_after": 9,
      "improvement": 4,
      "before_positive_pct": 50.0,
      "after_positive_pct": 90.0
    }
  }
}
```

---

## Key Metrics to Watch

### 1. **Improvement Number**
```
improvement.quality.improvement = +4
```
→ Tăng 4 câu trả lời tốt

### 2. **Percentage Improvement**
```
before: 50% → after: 90% = +40%
```
→ Cải thiện 40 điểm phần trăm

### 3. **Total Cost**
```
usage_summary.total_cost = $0.0012
```
→ Chi phí cho toàn bộ quá trình (optimize + re-generate + re-evaluate)

### 4. **Success Rate**
```
after_positive_pct = 90%
```
→ 90% responses đạt chất lượng tốt

---

## How to Use in Code

```python
response = requests.post("http://localhost:8435/optimize/huggingface", json=payload)
result = response.json()

# Get improvement
improvement = result["improvement"]["quality"]["improvement"]
print(f"Improved by {improvement} samples")

# Get percentage
before_pct = result["improvement"]["quality"]["before_positive_pct"]
after_pct = result["improvement"]["quality"]["after_positive_pct"]
print(f"Quality: {before_pct}% → {after_pct}% (+{after_pct - before_pct}%)")

# Check if worth it
if after_pct >= 80:
    print("✅ Optimized prompt is good enough for production!")
else:
    print("⚠️ May need more optimization")
```
