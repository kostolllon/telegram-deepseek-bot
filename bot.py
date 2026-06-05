import os
import asyncio
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# Токены и ключи из переменных окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# Если используешь ProxyAPI, раскомментируй следующую строку и укажи base_url
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

async def start(update: Update, context):
    await update.message.reply_text("Привет! Я бот на DeepSeek. Просто напиши мне что-нибудь.")

async def handle_message(update: Update, context):
    user_msg = update.message.text
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": user_msg}],
        "stream": False
    }
    try:
        # Если используешь ProxyAPI, замени URL на DEEPSEEK_BASE_URL + "/chat/completions"
        resp = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=data, timeout=30)
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"]
        else:
            reply = f"Ошибка API: {resp.status_code} - {resp.text}"
    except Exception as e:
        reply = f"Ошибка: {e}"
    await update.message.reply_text(reply)

def main():
    # Создаём приложение без автоматического обновления webhook
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запускаем polling (без webhook)
    print("Бот запущен и слушает сообщения...")
    application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
