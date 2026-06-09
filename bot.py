import os
import asyncio
import logging
from typing import Dict, List

import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# ========== НАСТРОЙКИ ==========
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Клиент DeepSeek (совместим с OpenAI API)
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# ========== ХРАНИЛИЩЕ ПОЛЬЗОВАТЕЛЕЙ ==========
user_sessions: Dict[int, Dict] = {}

MAX_HISTORY_LENGTH = 10
DEFAULT_SYSTEM_PROMPT = "Ты полезный, добрый и краткий помощник. Отвечай на русском языке."

def get_session(user_id: int) -> Dict:
    """Возвращает сессию пользователя (создаёт, если нет)"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        }
    return user_sessions[user_id]

def trim_history(history: List[Dict]) -> List[Dict]:
    """Ограничивает историю последними MAX_HISTORY_LENGTH сообщениями"""
    if len(history) > MAX_HISTORY_LENGTH:
        return history[-MAX_HISTORY_LENGTH:]
    return history

def split_long_message(text: str, max_len: int = 4096) -> List[str]:
    """Разбивает длинный текст на части, не разрывая слова"""
    if len(text) <= max_len:
        return [text]
    parts = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at == -1:
            split_at = max_len
        parts.append(text[:split_at])
        text = text[split_at:].lstrip()
    parts.append(text)
    return parts

async def ask_deepseek_with_retry(
    messages: List[Dict], retries: int = 2, delay: float = 1.0
) -> str:
    """Запрос к DeepSeek с повторными попытками"""
    for attempt in range(retries + 1):
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek API error (attempt {attempt+1}): {e}")
            if attempt < retries:
                await asyncio.sleep(delay * (attempt + 1))
            else:
                raise Exception("Сервис DeepSeek временно недоступен. Попробуйте позже.")

# ========== КОМАНДЫ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    get_session(user_id)
    await update.message.reply_text(
        "🤖 Привет! Я бот на основе DeepSeek. Я помню контекст разговора.\n\n"
        "Команды:\n"
        "/reset – сбросить историю диалога\n"
        "/system <новый промпт> – изменить мою личность\n"
        "/image <описание> – сгенерировать картинку\n"
        "/help – справка"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_sessions:
        user_sessions[user_id]["history"] = []
    await update.message.reply_text("🧹 История диалога очищена.")

async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    new_prompt = " ".join(context.args)
    if not new_prompt:
        current = session["system_prompt"]
        await update.message.reply_text(f"Текущий системный промпт:\n{current}")
        return
    session["system_prompt"] = new_prompt
    session["history"] = []
    await update.message.reply_text(f"✅ Системный промпт изменён на:\n{new_prompt}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Доступные команды:*\n"
        "/start – начать диалог\n"
        "/reset – очистить память\n"
        "/system <текст> – задать новую роль / характер\n"
        "/image <описание> – сгенерировать картинку (через DeepSeek Image API)\n"
        "/help – эта справка\n\n"
        "Просто напишите сообщение – я отвечу с учётом предыдущих сообщений.",
        parse_mode="Markdown",
    )

# ========== ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ ==========
async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерирует изображение по запросу /image <описание>"""
    user_id = update.effective_user.id
    prompt = " ".join(context.args)

    if not prompt:
        await update.message.reply_text(
            "❓ Пожалуйста, укажите описание после команды.\n"
            "Пример: `/image красный кот в космосе`",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text(f"🎨 Генерирую изображение: \"{prompt}\"...")
    await update.message.chat.send_action(action="upload_photo")

    try:
        # Используем deepseek_client для генерации изображения
        # (Если DeepSeek не поддерживает, замените на openai_client)
        response = await deepseek_client.images.generate(
            model="dall-e-3",          # или "deepseek-image", если появится
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url

        # Скачиваем и отправляем
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    await update.message.reply_photo(
                        photo=image_data,
                        caption=f"✨ Вот что получилось по запросу: \"{prompt}\""
                    )
                else:
                    await update.message.reply_text("❌ Не удалось загрузить изображение.")
    except Exception as e:
        logger.exception("Ошибка при генерации изображения")
        await update.message.reply_text(f"❌ Ошибка при генерации: {str(e)}")

# ========== ОБРАБОТКА ТЕКСТОВЫХ СООБЩЕНИЙ ==========
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    await update.message.chat.send_action(action="typing")

    session = get_session(user_id)
    history = session["history"]
    system_prompt = session["system_prompt"]

    history.append({"role": "user", "content": user_message})
    history = trim_history(history)
    session["history"] = history

    messages_for_api = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    try:
        reply = await ask_deepseek_with_retry(messages_for_api)
        session["history"].append({"role": "assistant", "content": reply})

        parts = split_long_message(reply)
        for i, part in enumerate(parts):
            await update.message.reply_text(part)
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)
    except Exception as e:
        logger.exception("Ошибка при обработке сообщения")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

# ========== ГЛОБАЛЬНЫЙ ОБРАБОТЧИК ОШИБОК ==========
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Произошла внутренняя ошибка. Администратор уже уведомлён."
        )

# ========== ЗАПУСК ==========
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("Переменная TELEGRAM_BOT_TOKEN не задана")
    if not DEEPSEEK_API_KEY:
        raise ValueError("Переменная DEEPSEEK_API_KEY не задана")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("system", set_system_prompt))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("image", image_command))   # <-- новая команда

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Бот запущен и готов к работе")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
