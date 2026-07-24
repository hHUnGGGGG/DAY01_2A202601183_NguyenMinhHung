"""
K3 — Ngày 1: Khám Phá LLM API (9h00–13h00)
AICB-P1: AI Practical Competency Program, Phase 1

Hướng dẫn:
    1. Làm theo LAB_GUIDE.md — mỗi block có các bước chi tiết và checkpoint.
    2. Điền vào tất cả các chỗ đánh dấu TODO.
    3. KHÔNG đổi chữ ký hàm (tên hàm, tham số).
    4. Import OpenAI BÊN TRONG hàm (xem gợi ý) — nếu import ở đầu file,
       các bài test mock sẽ không hoạt động.
    5. Kiểm tra tiến độ:  pytest tests/test_part1.py -v  (từng phần)
       Chấm điểm tổng:    python grade.py
"""

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "deepseek-chat": {"input": 0.0014, "output": 0.0028},
    "deepseek-reasoner": {"input": 0.0055, "output": 0.0219},
    "gemini-2.5-flash": {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
}

# Danh sách model mở rộng để kiểm thử đa nền tảng qua yescale.io
# Lưu ý: deepseek-chat không khả dụng trên tài khoản starter
EXTENDED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gemini-2.5-flash",
    "claude-sonnet-4-20250514",
]

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: 10h00–10h40)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model OpenAI sử dụng (mặc định: gpt-4o).
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from openai import OpenAI            # import BÊN TRONG hàm
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # đo thời gian bằng time.time() trước và sau lời gọi API
    """
    # TODO: import OpenAI, tạo client, gọi chat.completions.create,
    #       đo start/end time, trả về (response_text, latency)
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    end = time.time()
    return response.choices[0].message.content or "", end - start
    # raise NotImplementedError("Implement call_openai")


# ---------------------------------------------------------------------------
# Task 1.2 — Gọi GPT-4o-mini
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với model gpt-4o-mini — nhanh hơn và rẻ hơn.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        Tái sử dụng call_openai() với model=OPENAI_MINI_MODEL — 1 dòng code.
    """
    # TODO: gọi call_openai với model=OPENAI_MINI_MODEL
    return call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    # raise NotImplementedError("Implement call_openai_mini")


# ---------------------------------------------------------------------------
# Task 1.3 — So sánh GPT-4o vs GPT-4o-mini
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Gọi cả hai model với cùng một prompt và trả về dict so sánh.

    Returns:
        Dict với các key:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (USD ước tính cho phản hồi)

    Gợi ý:
        cost = (len(response.split()) / 0.75) / 1000 \
               * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
        (0.75 từ ≈ 1 token — ước lượng thô; Part 2 sẽ tính chính xác hơn)
    """
    # TODO: gọi call_openai và call_openai_mini, ghép dict kết quả
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    gpt4o_cost_estimate = (
        (len(gpt4o_response.split()) / 0.75)
        / 1000
        * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    )

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }
    # raise NotImplementedError("Implement compare_models")


# ===========================================================================
# PART 2 — SYSTEM PROMPT & TOKEN (Block 2: 10h40–11h20)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 2.1 — Chat với system prompt (persona)
# ---------------------------------------------------------------------------
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi API với MESSAGES gồm 2 phần: system prompt (định hình vai trò/persona
    của model) và user prompt (câu hỏi thật).

    Args:
        system_prompt: Chỉ dẫn vai trò, ví dụ "Bạn là giáo viên tiểu học,
                       giải thích mọi thứ thật đơn giản."
        user_prompt:   Tin nhắn của người dùng.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    """
    # TODO: giống call_openai nhưng messages có thêm phần tử role="system"
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    end = time.time()
    return response.choices[0].message.content or "", end - start
    # raise NotImplementedError("Implement chat_with_system_prompt")


# ---------------------------------------------------------------------------
# Task 2.2 — Đếm token bằng tiktoken
# ---------------------------------------------------------------------------
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """
    Đếm số token của một đoạn text bằng thư viện tiktoken.

    Args:
        text:  Đoạn text cần đếm.
        model: Model dùng để chọn bộ mã hóa (encoding).

    Returns:
        Số token (int).

    Gợi ý:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

        tiktoken cần tải bộ mã hóa từ mạng ở lần chạy đầu. Hãy bọc trong
        try/except — nếu lỗi (offline, model lạ), dùng ước lượng dự phòng:
        max(1, len(text) // 4)   (trung bình 1 token ≈ 4 ký tự)
    """
    # TODO: dùng tiktoken để đếm token, có fallback khi lỗi
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
    # raise NotImplementedError("Implement count_tokens")


# ---------------------------------------------------------------------------
# Task 2.3 — Ước tính chi phí chính xác
# ---------------------------------------------------------------------------
def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """
    Tính chi phí một lượt gọi API dựa trên số token THẬT (đếm bằng
    count_tokens) và bảng giá PRICING_PER_1K_TOKENS — tách riêng chi phí
    input (prompt) và output (response).

    Returns:
        Dict với các key:
            - "input_tokens":  int
            - "output_tokens": int
            - "input_cost":    float  (USD)
            - "output_cost":   float  (USD)
            - "total_cost":    float  (USD)

    Gợi ý:
        pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])
        input_cost = input_tokens / 1000 * pricing["input"]
        (.get với fallback: model không có trong bảng giá — ví dụ model NIM
         miễn phí — thì lấy giá gpt-4o làm tham chiếu học tập)
    """
    # TODO: đếm token prompt/response, tra bảng giá, trả về dict 5 key
    input_tokens = count_tokens(prompt, model=model)
    output_tokens = count_tokens(response, model=model)
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS["gpt-4o"])

    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
    # raise NotImplementedError("Implement estimate_cost")


# ===========================================================================
# PART 3 — STREAMING & ĐỘ BỀN (Block 3: 11h30–12h10)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 3.1 — Chatbot streaming có lịch sử hội thoại
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Chatbot dòng lệnh tương tác dùng streaming.

    Hành vi:
        - Stream token từ OpenAI ngay khi chúng được sinh ra (in từng chunk).
        - Duy trì 3 lượt hội thoại gần nhất trong history.
        - Gõ 'quit' hoặc 'exit' để thoát.

    Gợi ý:
        - Giữ list `history` gồm các dict {"role": ..., "content": ...}.
        - Dùng stream=True trong client.chat.completions.create() và lặp:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - Sau mỗi lượt, thêm phản hồi assistant vào history.
        - Cắt history còn 3 lượt cuối (6 message): history = history[-6:]
    """
    # TODO: vòng lặp while, đọc input, stream phản hồi, duy trì history
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ["quit", "exit"]:
            break

        if not user_input.strip():
            continue

        history.append({"role": "user", "content": user_input})

        response_stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history[-6:],
            stream=True,
        )

        full_response = []
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response.append(delta)
        print()

        assistant_text = "".join(full_response)
        history.append({"role": "assistant", "content": assistant_text})
        history = history[-6:]
    # raise NotImplementedError("Implement streaming_chatbot")


# ---------------------------------------------------------------------------
# Task 3.2 — Retry với exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Gọi fn(). Nếu ném exception, thử lại tối đa max_retries lần với
    exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Callable không tham số.
        max_retries: Số lần thử lại tối đa.
        base_delay:  Delay ban đầu (giây) trước lần thử lại đầu tiên.

    Returns:
        Giá trị trả về của fn() khi thành công.

    Raises:
        Exception cuối cùng của fn() sau khi hết số lần thử.
    """
    # TODO: vòng lặp retry với exponential backoff
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                raise e
            time.sleep(base_delay * (2**attempt))
    # raise NotImplementedError("Implement retry_with_backoff")


# ===========================================================================
# PART 4 — MINI-PROJECT: TRỢ LÝ CLI HOÀN CHỈNH (Block 4: 12h10–12h50)
# ===========================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """
    Trợ lý CLI hoàn chỉnh — ghép mọi thứ bạn đã xây trong Part 1–3.

    Hành vi:
        1. Dùng `persona` làm system prompt cho TOÀN BỘ phiên chat.
        2. Mỗi lượt: đọc tin nhắn qua get_input(); nếu là 'quit'/'exit'
           (không phân biệt hoa thường) → kết thúc phiên.
        3. Gọi API với stream=True, messages = system + history + tin nhắn mới.
           Bọc lời gọi API trong retry_with_backoff để chịu lỗi tạm thời.
        4. In từng chunk khi stream về, ghép lại thành reply hoàn chỉnh.
        5. Cập nhật history (user + assistant), giữ tối đa 3 lượt cuối
           (6 message): history = history[-6:]
        6. Cộng dồn thống kê bằng count_tokens và estimate_cost.
        7. Dừng khi đạt max_turns (nếu được đặt).

    Args:
        persona:   Mô tả vai trò, dùng làm system prompt.
        get_input: Hàm đọc input (mặc định: input). Tham số này giúp
                   test tự động không cần bàn phím thật.
        max_turns: Số lượt tối đa (None = không giới hạn).

    Returns:
        Dict thống kê phiên chat:
            - "num_turns":    int   (số lượt hỏi–đáp đã thực hiện)
            - "total_tokens": int   (tổng token user + assistant)
            - "total_cost":   float (tổng USD ước tính)
            - "history":      list  (history còn lại sau khi cắt, ≤ 6 message)

    Gợi ý khung sườn:
        if get_input is None:
            get_input = input
        history, num_turns, total_tokens, total_cost = [], 0, 0, 0.0
        while True:
            if max_turns is not None and num_turns >= max_turns:
                break
            user_msg = get_input()
            if user_msg.strip().lower() in ("quit", "exit"):
                break
            messages = [{"role": "system", "content": persona}] + history \
                       + [{"role": "user", "content": user_msg}]
            # stream = retry_with_backoff(lambda: client.chat...create(
            #              model=..., messages=messages, stream=True))
            # reply = ghép các chunk...
            ...
        return {"num_turns": num_turns, "total_tokens": total_tokens,
                "total_cost": total_cost, "history": history}
    """
    # TODO: triển khai theo khung sườn trong docstring
    from openai import OpenAI

    if get_input is None:
        get_input = input

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict[str, str]] = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break

        try:
            user_msg = get_input()
        except (EOFError, KeyboardInterrupt):
            break

        if user_msg is None or user_msg.strip().lower() in ("quit", "exit"):
            break

        if not user_msg.strip():
            continue

        messages = (
            [{"role": "system", "content": persona}]
            + history
            + [{"role": "user", "content": user_msg}]
        )

        response_stream = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
        )

        full_response = []
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response.append(delta)
        print()

        reply = "".join(full_response)

        cost_info = estimate_cost(user_msg, reply, model=OPENAI_MODEL)
        num_turns += 1
        total_tokens += cost_info["input_tokens"] + cost_info["output_tokens"]
        total_cost += cost_info["total_cost"]

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }
    # raise NotImplementedError("Implement run_assistant")


# ===========================================================================
# BONUS (không bắt buộc — cho bạn nào xong sớm)
# ===========================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Chạy compare_models cho từng prompt trong list.

    Returns:
        List các dict — mỗi dict là kết quả compare_models kèm thêm
        key "prompt" chứa prompt gốc.
    """
    # TODO (bonus): lặp qua prompts, gọi compare_models, thêm key "prompt"
    results = []
    for prompt in prompts:
        res = compare_models(prompt)
        res["prompt"] = prompt
        results.append(res)
    return results
    # raise NotImplementedError("Implement batch_compare")


def format_comparison_table(results: list[dict]) -> str:
    """
    Định dạng kết quả batch_compare thành bảng text dễ đọc.

    Cột: Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency
    Gợi ý: cắt text dài còn 40 ký tự cho dễ nhìn.
    """
    # TODO (bonus): dựng chuỗi bảng và trả về
    lines = [
        f"{'Prompt':<40} | {'GPT-4o Response':<40} | {'Mini Response':<40} | {'GPT-4o Latency':<15} | {'Mini Latency':<15}",
        "-" * 160,
    ]
    for r in results:
        p = r.get("prompt", "")
        p_str = p[:37] + "..." if len(p) > 40 else p
        g4o = r.get("gpt4o_response", "")
        g4o_str = g4o[:37] + "..." if len(g4o) > 40 else g4o
        mini = r.get("mini_response", "")
        mini_str = mini[:37] + "..." if len(mini) > 40 else mini
        g4o_lat = f"{r.get('gpt4o_latency', 0.0):.2f}s"
        mini_lat = f"{r.get('mini_latency', 0.0):.2f}s"
        lines.append(
            f"{p_str:<40} | {g4o_str:<40} | {mini_str:<40} | {g4o_lat:<15} | {mini_lat:<15}"
        )
    return "\n".join(lines)
    # raise NotImplementedError("Implement format_comparison_table")


# ---------------------------------------------------------------------------
# BONUS MỞ RỘNG — Multi-model testing qua yescale.io
# ---------------------------------------------------------------------------
def call_model(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
    system_prompt: str = None,
) -> tuple[str, float, str]:
    """
    Gọi bất kỳ model nào qua OpenAI-compatible API (yescale.io).

    Args:
        prompt:        Tin nhắn người dùng.
        model:         Tên model (gpt-4o, deepseek-chat, gemini-2.5-flash, ...).
        temperature:   Độ ngẫu nhiên (0.0–2.0).
        max_tokens:    Số token tối đa sinh ra.
        system_prompt: System prompt (nếu có).

    Returns:
        Tuple (response_text, latency_seconds, model_name).
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    end = time.time()
    return response.choices[0].message.content or "", end - start, model


def multi_model_compare(
    prompt: str,
    models: list[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> list[dict]:
    """
    So sánh nhiều model với cùng một prompt.

    Returns:
        List of dict, mỗi dict chứa:
            - "model": tên model
            - "response": nội dung trả lời (cắt 200 ký tự)
            - "full_response": nội dung đầy đủ
            - "latency": thời gian phản hồi (giây)
            - "output_tokens": số token output (ước tính bằng tiktoken)
            - "estimated_cost": chi phí ước tính (USD)
    """
    if models is None:
        models = EXTENDED_MODELS

    results = []
    for model in models:
        try:
            text, latency, name = call_model(
                prompt, model=model, temperature=temperature, max_tokens=max_tokens
            )
            out_tokens = count_tokens(text)
            pricing = PRICING_PER_1K_TOKENS.get(
                model, PRICING_PER_1K_TOKENS["gpt-4o"]
            )
            cost = out_tokens / 1000 * pricing["output"]
            results.append({
                "model": name,
                "response": text[:200] + ("..." if len(text) > 200 else ""),
                "full_response": text,
                "latency": round(latency, 2),
                "output_tokens": out_tokens,
                "estimated_cost": round(cost, 6),
            })
        except Exception as e:
            results.append({
                "model": model,
                "response": f"[ERROR] {e}",
                "full_response": "",
                "latency": 0,
                "output_tokens": 0,
                "estimated_cost": 0,
            })
    return results


def test_temperature_across_models(
    prompt: str,
    temperatures: list[float] = None,
    models: list[str] = None,
    max_tokens: int = 256,
) -> list[dict]:
    """
    Test nhiều mức temperature trên nhiều model.

    Returns:
        List of dict: {model, temperature, response, latency}
    """
    if temperatures is None:
        temperatures = [0.0, 0.5, 1.0, 1.5]
    if models is None:
        models = EXTENDED_MODELS

    results = []
    for model in models:
        for temp in temperatures:
            try:
                text, latency, name = call_model(
                    prompt, model=model, temperature=temp, max_tokens=max_tokens
                )
                results.append({
                    "model": name,
                    "temperature": temp,
                    "response": text[:150] + ("..." if len(text) > 150 else ""),
                    "full_response": text,
                    "latency": round(latency, 2),
                })
            except Exception as e:
                results.append({
                    "model": model,
                    "temperature": temp,
                    "response": f"[ERROR] {e}",
                    "full_response": "",
                    "latency": 0,
                })
    return results


def test_persona_across_models(
    user_prompt: str,
    personas: list[dict] = None,
    models: list[str] = None,
    max_tokens: int = 300,
) -> list[dict]:
    """
    Test nhiều persona (system prompt) trên nhiều model.

    Args:
        user_prompt: Câu hỏi chung.
        personas:    List of dict {"name": ..., "system_prompt": ...}.
        models:      Danh sách model.

    Returns:
        List of dict: {model, persona_name, response, latency, word_count}
    """
    if personas is None:
        personas = [
            {
                "name": "Giáo viên tiểu học",
                "system_prompt": "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi.",
            },
            {
                "name": "Chuyên gia tài chính",
                "system_prompt": "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật.",
            },
        ]
    if models is None:
        models = EXTENDED_MODELS

    results = []
    for model in models:
        for persona in personas:
            try:
                text, latency, name = call_model(
                    user_prompt,
                    model=model,
                    system_prompt=persona["system_prompt"],
                    max_tokens=max_tokens,
                )
                results.append({
                    "model": name,
                    "persona_name": persona["name"],
                    "response": text[:200] + ("..." if len(text) > 200 else ""),
                    "full_response": text,
                    "latency": round(latency, 2),
                    "word_count": len(text.split()),
                })
            except Exception as e:
                results.append({
                    "model": model,
                    "persona_name": persona["name"],
                    "response": f"[ERROR] {e}",
                    "full_response": "",
                    "latency": 0,
                    "word_count": 0,
                })
    return results


def compare_token_languages(
    text_vi: str,
    text_en: str,
) -> dict:
    """
    So sánh số token giữa đoạn tiếng Việt và tiếng Anh cùng nội dung.

    Returns:
        Dict: {
            "vi_text": ..., "en_text": ...,
            "vi_words": int, "en_words": int,
            "vi_tokens": int, "en_tokens": int,
            "vi_estimated": float,  # words / 0.75
            "en_estimated": float,
            "vi_vs_en_ratio": float,  # vi_tokens / en_tokens
            "vi_estimate_error_pct": float,  # % chênh lệch ước lượng vs thực tế
        }
    """
    vi_tokens = count_tokens(text_vi)
    en_tokens = count_tokens(text_en)
    vi_words = len(text_vi.split())
    en_words = len(text_en.split())
    vi_est = vi_words / 0.75
    en_est = en_words / 0.75

    vi_err = abs(vi_tokens - vi_est) / vi_est * 100 if vi_est > 0 else 0

    return {
        "vi_text": text_vi[:100] + "...",
        "en_text": text_en[:100] + "...",
        "vi_words": vi_words,
        "en_words": en_words,
        "vi_tokens": vi_tokens,
        "en_tokens": en_tokens,
        "vi_estimated": round(vi_est, 1),
        "en_estimated": round(en_est, 1),
        "vi_vs_en_ratio": round(vi_tokens / en_tokens, 2) if en_tokens > 0 else 0,
        "vi_estimate_error_pct": round(vi_err, 1),
    }


# ---------------------------------------------------------------------------
# Entry point — demo chạy thật (cần OPENAI_API_KEY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== So sánh model ===")
    result = compare_models(
        "Giải thích khác biệt giữa temperature và top_p trong một câu."
    )
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Trợ lý CLI (gõ 'quit' để thoát) ===")
    stats = run_assistant(
        persona="Bạn là trợ giảng thân thiện của khóa AI, "
                "trả lời ngắn gọn bằng tiếng Việt.",
    )
    print("\n--- Thống kê phiên chat ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
