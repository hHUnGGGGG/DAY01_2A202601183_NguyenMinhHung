"""
run_experiments.py — Chạy thử nghiệm đa model cho Lab01 exercises.md
Kết quả được in ra console và lưu vào experiment_results.txt
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import template


OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.txt")


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")


def run_all():
    f = open(OUTPUT_FILE, "w", encoding="utf-8")

    # ==================================================================
    # THỬ NGHIỆM 1: Temperature trên nhiều model (Câu 1.1)
    # ==================================================================
    log("\n" + "=" * 70, f)
    log("THỬ NGHIỆM 1: Temperature trên nhiều model (Câu 1.1)", f)
    log("=" * 70, f)
    prompt_11 = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
    log(f"Prompt: {prompt_11}\n", f)

    temp_results = template.test_temperature_across_models(prompt_11)
    current_model = ""
    for r in temp_results:
        if r["model"] != current_model:
            current_model = r["model"]
            log(f"\n--- Model: {current_model} ---", f)
        log(f"  Temp {r['temperature']}: {r['response'][:120]}...", f)
        log(f"    Latency: {r['latency']}s", f)

    # ==================================================================
    # THỬ NGHIỆM 2: So sánh latency & cost 5 model (Câu 1.2, 1.3)
    # ==================================================================
    log("\n" + "=" * 70, f)
    log("THỬ NGHIỆM 2: So sánh latency & cost trên 5 model (Câu 1.2, 1.3)", f)
    log("=" * 70, f)
    prompt_cost = "Giải thích khác biệt giữa machine learning và deep learning trong 3 câu."
    log(f"Prompt: {prompt_cost}\n", f)

    cost_results = template.multi_model_compare(prompt_cost, max_tokens=350)
    log(f"{'Model':<30} {'Latency':>8} {'Tokens':>7} {'Cost (USD)':>12}", f)
    log("-" * 60, f)
    for r in cost_results:
        log(f"{r['model']:<30} {r['latency']:>7.2f}s {r['output_tokens']:>6} ${r['estimated_cost']:>10.6f}", f)

    # Tính tỷ lệ chi phí
    log("\n--- Bảng tỷ lệ chi phí output (so với gpt-4o-mini) ---", f)
    mini_price = template.PRICING_PER_1K_TOKENS["gpt-4o-mini"]["output"]
    for model_name, prices in template.PRICING_PER_1K_TOKENS.items():
        ratio = prices["output"] / mini_price
        log(f"  {model_name:<30} {ratio:>6.1f}x so với mini", f)

    # Workload projection: 10K users, 3 calls/user, 350 tokens/call
    log("\n--- Workload projection: 10K users × 3 calls × 350 tokens output/ngày ---", f)
    daily_tokens = 10000 * 3 * 350  # 10.5M tokens
    for model_name, prices in template.PRICING_PER_1K_TOKENS.items():
        daily_cost = daily_tokens / 1000 * prices["output"]
        monthly_cost = daily_cost * 30
        log(f"  {model_name:<30} ${daily_cost:>8.2f}/ngày  ${monthly_cost:>10.2f}/tháng", f)

    # ==================================================================
    # THỬ NGHIỆM 3: Persona test trên 5 model (Câu 2.1)
    # ==================================================================
    log("\n" + "=" * 70, f)
    log("THỬ NGHIỆM 3: Persona test trên 5 model (Câu 2.1)", f)
    log("=" * 70, f)
    question = "Giải thích blockchain là gì?"
    log(f"Câu hỏi: {question}\n", f)

    persona_results = template.test_persona_across_models(question)
    current_model = ""
    for r in persona_results:
        if r["model"] != current_model:
            current_model = r["model"]
            log(f"\n--- Model: {current_model} ---", f)
        log(f"  [{r['persona_name']}] ({r['word_count']} từ, {r['latency']}s):", f)
        log(f"    {r['response'][:180]}...", f)

    # ==================================================================
    # THỬ NGHIỆM 4: Token count tiếng Việt vs tiếng Anh (Câu 2.2)
    # ==================================================================
    log("\n" + "=" * 70, f)
    log("THỬ NGHIỆM 4: Token count tiếng Việt vs tiếng Anh (Câu 2.2)", f)
    log("=" * 70, f)

    samples = [
        {
            "name": "Mẫu 1 — Lịch sử Hà Nội",
            "vi": "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam, cũng là kinh đô của hầu hết các vương triều Việt trước đây. Do đó, lịch sử Hà Nội gắn liền với sự thăng trầm của lịch sử Việt Nam qua các thời kỳ. Hà Nội là thành phố trực thuộc trung ương có diện tích lớn nhất cả nước, đồng thời cũng là địa phương đứng thứ nhì về dân số với hơn 8 triệu người.",
            "en": "Hanoi is the capital of the Socialist Republic of Vietnam and has been the capital of most previous Vietnamese dynasties. Therefore, Hanoi's history is closely tied to the ups and downs of Vietnamese history through various periods. Hanoi is the largest centrally-governed city in the country by area and the second most populous locality with over 8 million people.",
        },
        {
            "name": "Mẫu 2 — Công nghệ AI",
            "vi": "Trí tuệ nhân tạo là một lĩnh vực khoa học máy tính tập trung vào việc xây dựng các hệ thống có khả năng thực hiện những tác vụ thường đòi hỏi trí thông minh của con người. Các ứng dụng phổ biến bao gồm nhận dạng giọng nói, xử lý ngôn ngữ tự nhiên, thị giác máy tính và xe tự lái.",
            "en": "Artificial intelligence is a field of computer science focused on building systems capable of performing tasks that typically require human intelligence. Common applications include speech recognition, natural language processing, computer vision, and self-driving cars.",
        },
        {
            "name": "Mẫu 3 — Ẩm thực",
            "vi": "Phở là món ăn truyền thống nổi tiếng nhất của Việt Nam, được chế biến từ nước dùng xương hầm trong nhiều giờ cùng với bánh phở và thịt bò hoặc gà. Hương vị đặc trưng đến từ các gia vị như hồi, quế, thảo quả và gừng nướng.",
            "en": "Pho is the most famous traditional Vietnamese dish, made from broth simmered with bones for many hours along with rice noodles and beef or chicken. The distinctive flavor comes from spices such as star anise, cinnamon, cardamom, and roasted ginger.",
        },
    ]

    for sample in samples:
        log(f"\n--- {sample['name']} ---", f)
        result = template.compare_token_languages(sample["vi"], sample["en"])
        log(f"  Tiếng Việt: {result['vi_words']} từ → {result['vi_tokens']} tokens (ước lượng: {result['vi_estimated']})", f)
        log(f"  Tiếng Anh:  {result['en_words']} từ → {result['en_tokens']} tokens (ước lượng: {result['en_estimated']})", f)
        log(f"  Tỷ lệ VI/EN: {result['vi_vs_en_ratio']}x", f)
        log(f"  Sai số ước lượng tiếng Việt: {result['vi_estimate_error_pct']}%", f)

    # ==================================================================
    # THỬ NGHIỆM 5: Streaming vs Non-streaming (Câu 3.1) — chỉ ghi chú
    # ==================================================================
    log("\n" + "=" * 70, f)
    log("THỬ NGHIỆM 5: Streaming vs Non-streaming latency (Câu 3.1)", f)
    log("=" * 70, f)
    prompt_stream = "Viết 1 đoạn 3 câu về lợi ích của AI trong y tế."

    # Non-streaming (đo TTLT = Time-To-Last-Token)
    log(f"Prompt: {prompt_stream}\n", f)
    log("--- Non-streaming (đo tổng thời gian) ---", f)
    for model in template.EXTENDED_MODELS:
        try:
            text, lat, name = template.call_model(prompt_stream, model=model, max_tokens=200)
            log(f"  {name:<30} TTLT={lat:.2f}s  ({len(text.split())} từ)", f)
        except Exception as e:
            log(f"  {model:<30} [ERROR] {e}", f)

    # Streaming (đo TTFT = Time-To-First-Token)
    log("\n--- Streaming (đo TTFT — Time-To-First-Token) ---", f)
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for model in template.EXTENDED_MODELS:
        try:
            start = time.time()
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_stream}],
                stream=True,
                max_tokens=200,
            )
            ttft = None
            full_text = []
            for chunk in stream:
                if ttft is None:
                    ttft = time.time() - start
                delta = chunk.choices[0].delta.content or ""
                full_text.append(delta)
            ttlt = time.time() - start
            log(f"  {model:<30} TTFT={ttft:.2f}s  TTLT={ttlt:.2f}s  (gain={ttlt-ttft:.2f}s)", f)
        except Exception as e:
            log(f"  {model:<30} [ERROR] {e}", f)

    # ==================================================================
    # THỬ NGHIỆM 6: Multi-model creative prompt (bổ sung tổng quan)
    # ==================================================================
    log("\n" + "=" * 70, f)
    log("THỬ NGHIỆM 6: Multi-model creative & reasoning comparison", f)
    log("=" * 70, f)
    prompts_extra = [
        "Viết 1 bài thơ 4 câu về mùa thu Hà Nội.",
        "Tính 17 × 23 + 45 ÷ 9 và giải thích từng bước.",
    ]
    for p in prompts_extra:
        log(f"\nPrompt: {p}", f)
        results = template.multi_model_compare(p, max_tokens=200)
        for r in results:
            log(f"  [{r['model']}] ({r['latency']}s, {r['output_tokens']} tokens):", f)
            log(f"    {r['response'][:150]}...", f)

    log("\n" + "=" * 70, f)
    log("HOÀN TẤT! Kết quả đã lưu tại: " + OUTPUT_FILE, f)
    log("=" * 70, f)
    f.close()


if __name__ == "__main__":
    run_all()
