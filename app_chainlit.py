"""
app_chainlit.py
Giao diện chatbot tuyển sinh HaUI — Chainlit với streaming thực sự.

Cài đặt:
    pip install chainlit

Chạy:
    chainlit run app_chainlit.py --port 8000
"""

import sys
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load .env trước khi import bất kỳ thứ gì
load_dotenv()

# Thêm thư mục gốc project vào sys.path để import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chainlit as cl
from src.pipeline.chatbot import Chatbot
from src.pipeline.router  import IntentType
from src.pipeline.logger  import init_db, log_chat

init_db()

# Lưu PID để admin có thể restart
import os as _os
Path(".chainlit_pid").write_text(str(_os.getpid()))


# ── Intent labels ─────────────────────────────────────────────────────────────

INTENT_LABEL = {
    IntentType.JSON_DIEM_CHUAN      : "Điểm chuẩn",
    IntentType.JSON_HOC_PHI         : "Học phí",
    IntentType.JSON_CHI_TIEU_TO_HOP : "Chỉ tiêu & Tổ hợp",
    IntentType.JSON_QUY_DOI_DIEM    : "Tính điểm",
    IntentType.JSON_DAU_TRUOT       : "Kiểm tra đậu/trượt",
    IntentType.RAG_MO_TA_NGANH      : "Mô tả ngành",
    IntentType.RAG_FAQ              : "FAQ",
    IntentType.RAG_TRUONG_HOC_BONG  : "Trường & Học bổng",
    IntentType.UNKNOWN              : "Tìm kiếm",
}

QUICK_QUESTIONS = [
    "Điểm chuẩn ngành CNTT năm 2024?",
    "Học phí K20 đại trà bao nhiêu?",
    "TSA 80 KV2-NT điểm xét tuyển là bao nhiêu?",
    "Ngành Robot và AI ra làm gì?",
    "Học bổng HaUI có những loại nào?",
    "Hướng dẫn đăng ký xét tuyển 2025?",
]


# ── Global Retriever — load 1 lần, dùng chung ─────────────────────────────────
# Retriever (embedding model + BM25) nặng ~500MB, load 1 lần khi server start.
# Mỗi session tạo Chatbot riêng nhưng dùng chung Retriever → tiết kiệm RAM.

_GLOBAL_RETRIEVER = None

def _get_retriever():
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        from src.retrieval.retriever import Retriever
        _GLOBAL_RETRIEVER = Retriever(use_reranker=False, use_bm25=True)
    return _GLOBAL_RETRIEVER


# ── Chat start ────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    # Mỗi session có Chatbot riêng (history riêng) nhưng dùng chung Retriever
    retriever = _get_retriever()
    chatbot   = Chatbot(retriever=retriever)
    cl.user_session.set("chatbot",    chatbot)
    cl.user_session.set("turn",       0)
    cl.user_session.set("session_id", str(uuid.uuid4())[:8])

    actions = [
        cl.Action(name=f"q{i}", label=q, payload={"q": q})
        for i, q in enumerate(QUICK_QUESTIONS)
    ]

    await cl.Message(
        content=(
            "## Xin chào! Tôi là trợ lý tư vấn tuyển sinh HaUI\n\n"
            "Tôi có thể giúp bạn tra cứu **điểm chuẩn**, **học phí**, "
            "**tính điểm ưu tiên**, **kiểm tra đậu/trượt**, tìm hiểu "
            "**ngành học** và các thông tin **tuyển sinh** khác.\n\n"
            "**Chọn câu hỏi gợi ý hoặc nhập tự do:**"
        ),
        actions=actions,
    ).send()


# ── Quick action callbacks ────────────────────────────────────────────────────

async def _handle_quick(action: cl.Action):
    question = action.payload.get("q", "")
    await cl.Message(content=question, author="Bạn").send()
    await _answer(question)

for _i in range(len(QUICK_QUESTIONS)):
    cl.action_callback(f"q{_i}")(_handle_quick)


# ── Tin nhắn từ user ──────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    await _answer(message.content)


# ── Hàm trả lời với streaming ─────────────────────────────────────────────────

async def _answer(user_input: str):
    """Gọi chatbot với streaming thực sự — token hiện ra ngay từ Ollama."""
    import asyncio
    chatbot   : Chatbot = cl.user_session.get("chatbot")
    turn      : int     = cl.user_session.get("turn", 0)
    session_id: str     = cl.user_session.get("session_id", "unknown")

    msg = cl.Message(content="")
    await msg.send()

    t_start   = time.perf_counter()
    full_text = ""

    try:
        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()
        _DONE = object()  # sentinel

        def _run_stream():
            """Chạy sync generator trong thread, đẩy token vào queue."""
            try:
                for token in chatbot.chat_stream(user_input):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        # Chạy generator trong thread pool, không block event loop
        loop.run_in_executor(None, _run_stream)

        # Đọc token từ queue và stream ra UI ngay lập tức
        while True:
            token = await queue.get()
            if token is _DONE:
                break
            full_text += token
            await msg.stream_token(token)

    except Exception as e:
        err_msg = f"\n\nLỗi: {str(e)}"
        await msg.stream_token(err_msg)
        full_text += err_msg

    latency_ms = int((time.perf_counter() - t_start) * 1000)
    await msg.update()
    cl.user_session.set("turn", turn + 1)

    # Ghi log
    try:
        log_chat(
            session_id   = session_id,
            turn         = turn + 1,
            user_message = user_input,
            bot_answer   = full_text,
            intent       = "STREAM",
            method       = "stream",
            confidence   = 0.0,
            latency_ms   = latency_ms,
            platform     = "web",
        )
    except Exception:
        pass  # log lỗi không nên crash app

    # ── Metadata ─────────────────────────────────────────────────────────
    await cl.Message(
        content = f"`Streaming` · `{latency_ms}ms` · Lượt #{turn + 1}",
        author  = "system",
    ).send()

    # ── Nếu đang chờ làm rõ phương thức → thêm nút bấm ──────────────────
    if "__CLARIFY__" in full_text:
        actions = [
            cl.Action(name="pt_thpt",   label="Thi THPT (PT3)",        payload={"q": "thi THPT"}),
            cl.Action(name="pt_tsa",    label="Đánh giá tư duy TSA",   payload={"q": "đánh giá tư duy TSA"}),
            cl.Action(name="pt_hsa",    label="Đánh giá năng lực HSA", payload={"q": "đánh giá năng lực HSA"}),
            cl.Action(name="pt_hocba",  label="Xét học bạ (PT2)",      payload={"q": "xét học bạ"}),
            cl.Action(name="pt_cancel", label="Bỏ qua",                payload={"q": "thôi"}),
        ]
        await cl.Message(
            content = "Chọn phương thức xét tuyển:",
            actions = actions,
        ).send()
        return


# ── Callbacks nút phương thức xét tuyển ──────────────────────────────────────

async def _handle_pt(action: cl.Action):
    question = action.payload.get("q", "")
    await cl.Message(content=question, author="Bạn").send()
    await _answer(question)

for _name in ["pt_thpt", "pt_tsa", "pt_hsa", "pt_hocba", "pt_cancel"]:
    cl.action_callback(_name)(_handle_pt)


# ── Chat end ──────────────────────────────────────────────────────────────────

@cl.on_chat_end
async def on_chat_end():
    chatbot: Chatbot | None = cl.user_session.get("chatbot")
    if chatbot:
        chatbot.reset()