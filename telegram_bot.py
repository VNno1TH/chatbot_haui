"""
telegram_bot.py
Telegram Bot tư vấn tuyển sinh HaUI.

Cài đặt:
    pip install python-telegram-bot

Cấu hình .env:
    TELEGRAM_BOT_TOKEN=<token từ BotFather>

Chạy:
    python telegram_bot.py
"""

import os
import logging
import re
from dotenv import load_dotenv

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ChatAction

from src.pipeline.chatbot import Chatbot

# ── Cấu hình ─────────────────────────────────────────────────────────────────

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

logging.basicConfig(
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level  = logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Chatbot — mỗi user có instance riêng ─────────────────────────────────────

_user_chatbots: dict[int, Chatbot] = {}

def get_chatbot(chat_id: int) -> Chatbot:
    if chat_id not in _user_chatbots:
        _user_chatbots[chat_id] = Chatbot()
        logger.info(f"Tạo chatbot mới cho user {chat_id}")
    return _user_chatbots[chat_id]


# ── Keyboards ─────────────────────────────────────────────────────────────────

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("📊 Điểm chuẩn CNTT 2024"),  KeyboardButton("💰 Học phí K20")],
        [KeyboardButton("🔢 Tính điểm TSA/HSA"),      KeyboardButton("🎓 Ngành Robot & AI")],
        [KeyboardButton("🏫 Học bổng HaUI"),           KeyboardButton("📋 Hướng dẫn đăng ký")],
    ],
    resize_keyboard   = True,
    one_time_keyboard = False,
)

BUTTON_MAP = {
    "📊 Điểm chuẩn CNTT 2024" : "Điểm chuẩn ngành Công nghệ thông tin năm 2024?",
    "💰 Học phí K20"           : "Học phí K20 đại trà bao nhiêu một tín chỉ?",
    "🔢 Tính điểm TSA/HSA"     : "Cách tính điểm xét tuyển từ TSA và HSA?",
    "🎓 Ngành Robot & AI"      : "Ngành Robot và Trí tuệ nhân tạo HaUI ra làm gì?",
    "🏫 Học bổng HaUI"         : "HaUI có những loại học bổng nào?",
    "📋 Hướng dẫn đăng ký"     : "Hướng dẫn đăng ký xét tuyển vào HaUI năm 2025?",
}

# Inline keyboard khi hỏi lại phương thức xét tuyển
PHUONG_THUC_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("📝 Thi THPT",      callback_data="pt:thi THPT"),
        InlineKeyboardButton("🧮 Tư duy TSA",    callback_data="pt:đánh giá tư duy TSA"),
    ],
    [
        InlineKeyboardButton("🎯 Năng lực HSA",  callback_data="pt:đánh giá năng lực HSA"),
        InlineKeyboardButton("📋 Xét học bạ",    callback_data="pt:xét học bạ"),
    ],
    [
        InlineKeyboardButton("❌ Bỏ qua",        callback_data="pt:thôi"),
    ],
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def md_to_telegram(text: str) -> str:
    """Chuyển **bold** → *bold* cho Telegram Markdown."""
    return re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)


async def process_and_reply(
    update  : Update,
    context : ContextTypes.DEFAULT_TYPE,
    question: str,
):
    """Gọi chatbot, xử lý response và gửi lại user."""
    chat_id = update.effective_chat.id
    bot     = get_chatbot(chat_id)

    # Hiện "đang gõ..."
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        response = bot.chat(question)
        answer   = md_to_telegram(response.answer)
        ctx      = response.context or ""

        # Chatbot đang hỏi lại phương thức → hiện Inline Keyboard
        if "__CLARIFY__" in ctx:
            await update.effective_message.reply_text(
                answer,
                parse_mode   = "Markdown",
                reply_markup = PHUONG_THUC_KEYBOARD,
            )
            return

        # Trả lời bình thường
        await update.effective_message.reply_text(
            answer,
            parse_mode   = "Markdown",
            reply_markup = MAIN_KEYBOARD,
        )

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        await update.effective_message.reply_text(
            "❌ Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.",
            reply_markup = MAIN_KEYBOARD,
        )


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    name = user.first_name if user else "bạn"
    await update.message.reply_text(
        f"👋 Xin chào *{name}*!\n\n"
        f"Tôi là trợ lý tư vấn tuyển sinh *Đại học Công nghiệp Hà Nội (HaUI)*.\n\n"
        f"Tôi có thể giúp bạn:\n"
        f"• 📊 Tra *điểm chuẩn* theo ngành, năm, phương thức\n"
        f"• 💰 Tra *học phí* theo chương trình\n"
        f"• 🔢 *Tính điểm* xét tuyển có ưu tiên khu vực\n"
        f"• ✅ Kiểm tra *đậu/trượt* và so sánh qua các năm\n"
        f"• 🎓 Tìm hiểu *ngành học* và cơ hội việc làm\n"
        f"• ❓ *Hướng dẫn* đăng ký, nhập học, học bổng\n\n"
        f"Hãy đặt câu hỏi hoặc chọn gợi ý bên dưới 👇",
        parse_mode   = "Markdown",
        reply_markup = MAIN_KEYBOARD,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "*Hướng dẫn sử dụng:*\n\n"
        "Gõ câu hỏi bất kỳ về tuyển sinh HaUI.\n\n"
        "*Ví dụ:*\n"
        "• Điểm chuẩn CNTT 2024 bao nhiêu?\n"
        "• Tôi thi TSA 80 điểm KV2-NT được bao nhiêu?\n"
        "• 24 điểm KV1 có đậu CNTT không?\n"
        "• Học phí K20 tiếng Anh bao nhiêu?\n"
        "• So sánh điểm chuẩn CNTT 3 năm?\n\n"
        "*Lệnh:*\n"
        "/start — Bắt đầu lại\n"
        "/reset — Xóa lịch sử hội thoại\n"
        "/help  — Hướng dẫn này",
        parse_mode   = "Markdown",
        reply_markup = MAIN_KEYBOARD,
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in _user_chatbots:
        _user_chatbots[chat_id].reset()
    await update.message.reply_text(
        "✅ Đã xóa lịch sử hội thoại. Bạn có thể bắt đầu cuộc trò chuyện mới!",
        reply_markup = MAIN_KEYBOARD,
    )


# ── Message handler ───────────────────────────────────────────────────────────

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip()
    question   = BUTTON_MAP.get(user_input, user_input)
    await process_and_reply(update, context, question)


# ── Callback handler (Inline Keyboard) ───────────────────────────────────────

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("pt:"):
        return

    question = data[3:]  # "pt:thi THPT" → "thi THPT"

    # Xóa inline keyboard
    await query.edit_message_reply_markup(reply_markup=None)

    # Hiện lựa chọn của user
    await query.message.reply_text(
        f"Bạn chọn: *{question}*",
        parse_mode   = "Markdown",
        reply_markup = MAIN_KEYBOARD,
    )

    await process_and_reply(update, context, question)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not TOKEN:
        raise ValueError(
            "Chưa có TELEGRAM_BOT_TOKEN!\n"
            "Thêm vào .env: TELEGRAM_BOT_TOKEN=<token từ BotFather>"
        )

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("reset",  cmd_reset))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🤖 HaUI Bot đang chạy... Nhấn Ctrl+C để dừng.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()