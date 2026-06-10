import os
import asyncio
import logging
import urllib.parse
from typing import Dict, List

import aiohttp
import httpx
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
BFL_API_KEY = os.getenv("BFL_API_KEY")  # API-ключ Black Forest Labs (FLUX)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Клиент DeepSeek
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    http_client=httpx.AsyncClient(),
)

# ========== ХРАНИЛИЩЕ ПОЛЬЗОВАТЕЛЕЙ ==========
user_sessions: Dict[int, Dict] = {}
MAX_HISTORY_LENGTH = 20
DEFAULT_SYSTEM_PROMPT = "Ты полезный, добрый и краткий помощник. Отвечай на русском языке."

def get_session(user_id: int) -> Dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "system_prompt": DEFAULT_SYSTEM_PROMPT,
        }
    return user_sessions[user_id]

def trim_history(history: List[Dict]) -> List[Dict]:
    if len(history) > MAX_HISTORY_LENGTH:
        return history[-MAX_HISTORY_LENGTH:]
    return history

def split_long_message(text: str, max_len: int = 4096) -> List[str]:
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

async def ask_deepseek_with_retry(messages: List[Dict], retries: int = 2, delay: float = 1.0) -> str:
    for attempt in range(retries + 1):
        try:
            response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=None,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"DeepSeek API error (attempt {attempt+1}): {e}")
            if attempt < retries:
                await asyncio.sleep(delay * (attempt + 1))
            else:
                raise

# ========== КОМАНДЫ ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    get_session(user_id)
    await update.message.reply_text(
        "🤖 Привет! Я бот на основе DeepSeek.\n\n"
        "Команды:\n"
        "/reset – сбросить историю\n"
        "/system <промпт> – сменить личность\n"
        "/image <описание> – сгенерировать картинку (FLUX)\n"
        "/help – справка"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in user_sessions:
        user_sessions[user_id]["history"] = []
    await update.message.reply_text("🧹 История очищена.")

async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_session(user_id)
    new_prompt = " ".join(context.args)
    if not new_prompt:
        await update.message.reply_text(f"Текущий промпт:\n{session['system_prompt']}")
        return
    session["system_prompt"] = new_prompt
    session["history"] = []
    await update.message.reply_text(f"✅ Промпт изменён:\n{new_prompt}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - приветствие\n/reset - сброс истории\n/system <текст> - задать роль\n/image <описание> - сгенерировать картинку (FLUX)"
    )

# ========== ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ (FLUX через Black Forest Labs) ==========
async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text(
            "❓ Напиши описание после /image, например: `/image cat in space`",
            parse_mode="Markdown"
        )
        return

    processing_msg = await update.message.reply_text(f"🎨 Генерирую изображение: \"{prompt}\"...")

    if not BFL_API_KEY:
        await processing_msg.edit_text("❌ API-ключ BFL не настроен. Добавьте переменную BFL_API_KEY в Railway.")
        return

    # Эндпоинт для модели FLUX.2 klein (бюджетная, но качественная)
    # Цена: $0.014 за изображение
    url = "https://api.bfl.ai/v1/flux-2-klein-4b"
    headers = {
        "X-Key": BFL_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "width": 1024,
        "height": 768,
        "steps": 25,
    }

    try:
        async with aiohttp.ClientSession() as session:
            # 1. Отправляем запрос на генерацию
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    await processing_msg.edit_text(f"❌ Ошибка API FLUX: {resp.status}\n{error_text[:200]}")
                    return
                result = await resp.json()
                request_id = result.get("id")
                if not request_id:
                    await processing_msg.edit_text("❌ Не удалось получить ID задания.")
                    return

            # 2. Ожидаем результат (polling)
            status_url = f"https://api.bfl.ai/v1/get_result?id={request_id}"
            max_attempts = 40  # ~40 секунд
            for attempt in range(max_attempts):
                await processing_msg.edit_text(f"🎨 Генерация... (попытка {attempt+1}/{max_attempts})")
                async with session.get(status_url, headers=headers) as status_resp:
                    if status_resp.status == 200:
                        data = await status_resp.json()
                        status = data.get("status")
                        if status == "Ready":
                            image_url = data["result"]["sample"]
                            await processing_msg.delete()
                            await update.message.reply_photo(photo=image_url, caption=f"✨ \"{prompt}\"")
                            return
                        elif status == "Error":
                            error_msg = data.get("error", "Неизвестная ошибка")
                            await processing_msg.edit_text(f"❌ Ошибка FLUX: {error_msg}")
                            return
                    elif status_resp.status == 404:
                        # Ещё не готово, ждём
                        await asyncio.sleep(1)
                        continue
                    else:
                        error_text = await status_resp.text()
                        await processing_msg.edit_text(f"❌ Ошибка при проверке статуса: {status_resp.status}\n{error_text[:200]}")
                        return
                await asyncio.sleep(1)

            # Если не дождались
            await processing_msg.edit_text("❌ Генерация заняла слишком много времени. Попробуйте ещё раз.")

    except Exception as e:
        logger.exception("Ошибка в image_command")
        await processing_msg.edit_text(f"❌ Исключение: {str(e)}")

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

    messages_for_api = [{"role": "system", "content": system_prompt}, *history]

    try:
        reply = await ask_deepseek_with_retry(messages_for_api)
        session["history"].append({"role": "assistant", "content": reply})

        parts = split_long_message(reply)
        for i, part in enumerate(parts):
            await update.message.reply_text(part)
            if i < len(parts) - 1:
                await asyncio.sleep(0.3)
    except Exception as e:
        logger.exception("Ошибка в handle_message")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text("⚠️ Внутренняя ошибка. Администратор уведомлён.")

# ========== ЗАПУСК ==========
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан")
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY не задан")
    if not BFL_API_KEY:
        logger.warning("BFL_API_KEY не задан. Команда /image не будет работать.")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("system", set_system_prompt))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Бот запущен и ожидает сообщения...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
