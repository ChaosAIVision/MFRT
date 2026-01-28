- Hiện tại quy trình thực hiện tôi không cảm thấy nỏ đúng đắn lắm bởi vì một số lý do sau :

1. Điểm thấp:

🎉 IMPROVEMENT: +-30 correct samples
📈 ACCURACY GAIN: 70.0% → 27.1% (+-42.9%)

📈 ITERATION HISTORY:
--------------------------------------------------------------------------------
Iteration 1:
  Train: 52/70 correct (74.3%)
  Test:  20/30 correct (66.7%)
Iteration 2:
  Train: 52/70 correct (74.3%)
  Test:  17/30 correct (56.7%)
Iteration 3:
  Train: 51/70 correct (72.9%)
  Test:  19/30 correct (63.3%)

Lý do bởi vì nó đang bị tối ưu hẹp , không phải general, rất dễ overfitting.

2. Blackbox optimizer 

Ở đây quy trình optimization nó không có quy tắc gì hết , cho nên nguyên tắc của nó gì cũng điền , case nào cũng chọn , vậy là không được , tôi muốn quy trình optimize phải rõ ràng step by step.


FORMAT tôi muốn.
"""


Cấu trúc Prompt 2 Giai đoạn
Phase 1: Model Construction (Xây dựng Model)
text
Analyze the following problem. First, explicitly define the problem model by listing:
(1) relevant entities,
(2) state variables,
(3) possible actions with preconditions and effects,
and (4) constraints.
Do not propose a solution yet.
Mục đích: Buộc LLM phải xây dựng một biểu diễn cấu trúc rõ ràng của bài toán trước khi giải quyết.
​

Analyze the following problem. First, explicitly define the problem model by listing:
(1) relevant entities,
(2) state variables,
(3) possible actions with preconditions and effects,
and (4) constraints.
Do not propose a solution yet.


Phase 2: Reasoning (Suy luận dựa trên Model)
text
Using only the model defined above, generate a step-by-step solution plan. 
Ensure that all actions respect the defined constraints and state transitions.
Mục đích: Thực hiện suy luận chỉ trong phạm vi model đã được định nghĩa, đảm bảo tuân thủ constraints.
​

Các Thành phần của Model
Khi xây dựng model, LLM cần định nghĩa rõ:
​

Entities: Các đối tượng hoặc agents liên quan (ví dụ: người, tài nguyên, địa điểm)

State Variables: Các thuộc tính có thể thay đổi theo thời gian (ví dụ: availability, location, status)

Actions: Các thao tác được phép với preconditions và effects

Constraints: Các ràng buộc cần tuân thủ

Using only the model defined above, generate a step-by-step solution plan. 
Ensure that all actions respect the defined constraints and state transitions.


"""


QUY TRÌNH TRAIN



Đầu tiên của pipeline là input vào model trích xuất (không phải model train) những thông tin.
1 là bài toán
2 là input
3 là groundtruth

Model dựa vào 3 thông tin để sinh ra 2 quy trình.

Nội dung phần contruction đặt trong thẻ

<contruction> Nội dung </construction>

Yêu cầu của phần này là . Nội dung tuân theo quy  quy  dinh nhưng phải ngắn gọn , viết theo gạch đầu dòng (khoảng 500 từ ) 

Nội dung phần reasoning:
- Từ yêu cầu
- Từ kết quả tính ra từ phần construction

Suy luận ra đáp án là gì 
<think> Nội dung phần think </think>


3. Kết quả trả về dạng json.

Lưu kết quả về json file.


Chia thành 2 giai đoạn training :

1. Training giai đoạn Phase 1: Model Construction (Xây dựng Model).
Tuning phần prompt cho contruction. (Chỉ được tác động phần prompt cho reasoning)
Yêu cầu là model tạo ra contruction phải giống 90 % so với model tạo ra construction đúng với groundtruth bằng cách prompt cải thiện cho model xác thực được 
đúng.
(1) relevant entities,
(2) state variables,
(3) possible actions with preconditions and effects,
and (4) constraints.

hãy yêu cầu model chỉ cần trả trong thẻ yêu cầu thôi đừng yêu cầu nó chạy thêm phần thẻ reasoning 

2 Training giai đoạn Phase 2. Reasoning.
Model sẽ trả về phần reasoning và kết quả , logic xử lý như sau:

Kết quả đúng.
Nếu model trả lời kết quả chính xác mà cách reasoning path khác với prompt và reasoning path logic khác với kết quả groundtruth vẫn được chấp nhận , xem xét add thêm logic reasoning này vào trong prompt hay không .

Kết quả sai

Có 3 thứ cần xét.
- logic path tốt trong prompt 

- logic path tệ trong prompt 

- logic path của ground truth


Nếu logic path của model cho kết quả ra sai mà không giống trong 3 cái này -> add vào logic path tệ.
Nếu logic path của model cho kết quả sai mà nằm trong logic path tốt hoặc nằm trong logic path của grouth truth , xem xét lại 2 cái đó và sửa cho đúng


Dùng 2 skill 
 
bead-method
production-flow

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
