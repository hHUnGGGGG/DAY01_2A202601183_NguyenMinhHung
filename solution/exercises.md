# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `[Câu trả lời của bạn]` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Mở rộng thử nghiệm trên 4 model (GPT-4o, GPT-4o-mini, Gemini 2.5 Flash, Claude Sonnet 4), quy luật rõ ràng: **temperature thấp (0.0–0.5) cho kết quả ổn định và lặp lại** — GPT-4o 3 lần liên đều nhắc Hang Sơn Đoòng, GPT-4o-mini luôn nói về 3.000 km bờ biển; **temperature cao (1.0–1.5) tăng độ đa dạng** — GPT-4o chuyển sang nói về đèn lồng Hội An, GPT-4o-mini đề cập 54 dân tộc, Claude thay đổi từ cà phê xuất khẩu sang hạt tiêu. Đáng chú ý, mỗi model có ngưỡng "sáng tạo" khác nhau: Claude đã bắt đầu thay đổi chủ đề từ temp=0.5, trong khi GPT-4o chỉ thực sự đổi ở temp=1.5. Gemini 2.5 Flash giữ giọng văn đồng nhất với cách mở đầu "Chắc chắn rồi!" ở mọi mức temperature.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt temperature ở mức **0.0 đến 0.2**. Qua thử nghiệm trên 4 model với cùng prompt, temp=0.0 cho kết quả nhất quán 100% (GPT-4o luôn nhắc Hang Sơn Đoòng, GPT-4o-mini luôn nhắc 3.000 km bờ biển). Chatbot hỗ trợ khách hàng cần: (1) **chính xác** — không bịa thông tin chính sách/giá cả, (2) **ổn định** — cùng câu hỏi phải cho cùng câu trả lời, (3) **tuân thủ tài liệu gốc** — không được hallucinate. Ngoài ra, với mục tiêu tiết kiệm chi phí cho 10K users/ngày, model rẻ như GPT-4o-mini ($6.30/ngày) hoặc Gemini 2.5 Flash ($6.30/ngày) kết hợp temperature thấp là lựa chọn tối ưu.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> **Kết quả tính toán thực tế** (10K users × 3 calls × 350 tokens output/ngày = 10.5M tokens/ngày):
> | Model | Chi phí/ngày | Chi phí/tháng | Tỷ lệ vs mini |
> |---|---|---|---|
> | GPT-4o | $105.00 | $3,150 | **16.7x** |
> | GPT-4o-mini | $6.30 | $189 | 1.0x |
> | Gemini 2.5 Flash | $6.30 | $189 | 1.0x |
> | Claude Sonnet 4 | $157.50 | $4,725 | 25.0x |
> | DeepSeek Chat | $29.40 | $882 | 4.7x |
>
> GPT-4o đắt hơn mini **16.7 lần**, nhưng Claude Sonnet 4 còn đắt hơn tới **25 lần**. Đáng chú ý, Gemini 2.5 Flash có giá **bằng đúng GPT-4o-mini** nhưng là model đa phương thức (multimodal).
> - **GPT-4o xứng đáng:** Các bài toán lập luận phức tạp, viết code, phân tích dữ liệu đa chiều — nơi chất lượng output quyết định giá trị kinh doanh.
> - **Nên dùng mini/Gemini Flash:** Tác vụ lặp lại nhiều (phân loại văn bản, tóm tắt, chatbot QA thông thường) — tiết kiệm gấp 16–25 lần chi phí mà chất lượng vẫn đủ dùng.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Thử nghiệm trên 4 model cho thấy **tất cả đều tuân thủ persona nhất quán**. Persona "giáo viên tiểu học" luôn dùng phép so sánh dễ hiểu: GPT-4o ví blockchain như "cuốn sổ tay đặc biệt", Claude như "cuốn sổ ma thuật", GPT-4o-mini như "cuốn sổ lớn". Persona "chuyên gia tài chính" thì mọi model đều dùng thuật ngữ chuyên môn: GPT-4o nhắc "Distributed Ledger Technology (DLT)", Claude dùng "mật mã học", "bất biến". Về độ dài, Claude Sonnet 4 cho câu trả lời dài nhất (giáo viên: 240 từ, chuyên gia: 275 từ) nhưng cũng chậm nhất (16–17s), trong khi GPT-4o-mini ngắn gọn nhất (170 từ) và nhanh nhất (3.52s). Kết luận: system prompt đóng vai trò "khung hành vi" ép model nhập vai nhất quán, ảnh hưởng mạnh đến từ vựng, độ phức tạp và cách diễn đạt — và hiệu ứng này hoạt động nhất quán trên mọi nhà cung cấp model (OpenAI, Google, Anthropic).

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Thử nghiệm trên 3 mẫu văn bản cùng nội dung (Việt–Anh) cho kết quả cụ thể:
> | Mẫu | Từ VI | Token VI | Từ EN | Token EN | Tỷ lệ VI/EN | Sai số ước lượng |
> |---|---|---|---|---|---|---|
> | Lịch sử Hà Nội | 82 | 101 | 59 | 71 | **1.42x** | 7.6% |
> | Công nghệ AI | 63 | 75 | 34 | 40 | **1.88x** | 10.7% |
> | Ẩm thực | 49 | 66 | 40 | 50 | **1.32x** | 1.0% |
>
> Trung bình tiếng Việt tốn **1.32–1.88 lần** nhiều token hơn tiếng Anh cùng nội dung. Sai số của công thức ước lượng thô (`từ/0.75`) dao động từ 1–10.7%. Nguyên nhân: bộ từ vựng BPE (Byte-Pair Encoding) của OpenAI được huấn luyện chủ yếu trên dữ liệu tiếng Anh, nên các từ tiếng Việt có dấu (ư, ơ, ă, ô, ế, ứ...) thường bị tách thành 2–3 token riêng lẻ, trong khi một từ tiếng Anh thường chỉ chiếm 1 token. Đặc biệt các đoạn văn có nhiều thuật ngữ chuyên ngành (mẫu "Công nghệ AI") có tỷ lệ chênh lệch cao nhất (1.88x).

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Đo đạc non-streaming trên 4 model cho thấy TTLT (Time-To-Last-Token) rất khác nhau: GPT-4o-mini nhanh nhất (2.13s), GPT-4o (2.34s), Gemini Flash (2.26s), và Claude Sonnet 4 chậm nhất (7.70s). Với Claude, người dùng phải chờ gần 8 giây mới thấy bất kỳ phản hồi nào — streaming sẽ giảm TTFT (Time-To-First-Token) xuống chỉ còn ~0.5–1s, giúp người dùng cảm thấy phản hồi tức thì. **Streaming quan trọng nhất** trong các ứng dụng tương tác trực tiếp (chatbot CLI, web UI, mobile app) và đặc biệt cần thiết khi dùng model chậm như Claude. **Non-streaming phù hợp hơn** với: xử lý batch ngầm (phân loại email, trích xuất dữ liệu), pipeline tự động cần toàn bộ output trước khi xử lý tiếp, hoặc API backend nội bộ không có người dùng trực tiếp chờ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff (delay = base_delay × 2^attempt) tăng thời gian chờ theo cấp số nhân: lần 1 chờ 0.1s, lần 2 chờ 0.2s, lần 3 chờ 0.4s... Khi API bị quá tải, điều này tạo ra hiệu ứng **"giãn dân tự nhiên"**: các client tự động rải đều request theo thời gian, giảm áp lực cho server. Ngược lại, nếu hàng nghìn client cùng retry với delay cố định (ví dụ luôn 1s), xảy ra **"Thundering Herd"** (đàn trâu giẫm đạp): server vừa phục hồi lại bị dội bom bởi hàng nghìn request đồng thời tại giây thứ 1, giây thứ 2... lặp lại vô hạn. Trong thực tế, nên kết hợp exponential backoff với **jitter** (thêm nhiễu ngẫu nhiên, ví dụ `delay = base_delay * 2^attempt + random(0, 0.5)`) để tránh các client vẫn vô tình retry cùng lúc.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **Persona:** "Bạn là CodeMaster, một lập trình viên Senior 15 năm kinh nghiệm. Hãy trả lời cực kỳ súc tích, đi thẳng vào giải pháp code và luôn dùng tiếng Việt."
> **Giải thích lựa chọn từ ngữ:**
> 1. **"cực kỳ súc tích"**: Thử nghiệm persona cho thấy Claude Sonnet 4 có xu hướng viết dài nhất (240–275 từ, latency 16–17s). Bằng cách yêu cầu "súc tích", ta giảm output tokens → giảm chi phí và tăng tốc độ đọc.
> 2. **"luôn dùng tiếng Việt"**: Qua thử nghiệm, khi gặp thuật ngữ kỹ thuật (như "blockchain", "DLT"), model có xu hướng tự động chuyển sang giải thích tiếng Anh. Chỉ định ngôn ngữ tường minh giúp model giữ nhất quán tiếng Việt. Lưu ý: tiếng Việt tốn nhiều token hơn (1.3–1.9x so với tiếng Anh như thử nghiệm đã chứng minh), nên cần cân nhắc giữa UX và chi phí.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất:** Ngữ cảnh bị cắt xén cứng nhắc (chỉ nhớ 3 lượt = 6 message cuối), làm mất bối cảnh trong hội thoại dài. Thêm vào đó, trợ lý hiện chỉ dùng 1 model cố định — không thể chuyển model linh hoạt theo độ phức tạp của câu hỏi.
> **Đề xuất cải thiện (2 hướng):**
> 1. **Summarized History:** Thay vì xóa hẳn tin nhắn cũ, gọi một lượt API phụ (dùng model rẻ như GPT-4o-mini chỉ $6.30/ngày) để tóm tắt các lượt hội thoại cũ thành 1–2 câu, ghép vào cuối system prompt. Vừa duy trì bối cảnh lâu dài, vừa tiết kiệm input tokens.
> 2. **Model Router:** Tự động chọn model dựa trên độ phức tạp câu hỏi: câu hỏi đơn giản → GPT-4o-mini/Gemini Flash (rẻ, nhanh 2–3s), câu hỏi phức tạp cần lập luận sâu → GPT-4o hoặc Claude Sonnet 4 (chất lượng cao hơn, nhưng latency 8–17s như thử nghiệm đã đo).

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
