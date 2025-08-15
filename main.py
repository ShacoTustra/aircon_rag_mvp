import asyncio
import logging
import os
from typing import Dict, List
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv

from llm_functions import (
    llm_dispatcher, 
    llm_extractor, 
    llm_relevance_checker, 
    llm_answer_generator
)
from retrieving_utils import find_model_database, search_similar_models, hybrid_search

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Словарь для хранения истории сообщений пользователей
user_history: Dict[int, List[str]] = {}

def get_user_history(user_id: int, max_messages: int = 3) -> List[str]:
    """Получает последние сообщения пользователя"""
    return '\n'.join(user_history.get(user_id, [])[-max_messages:])

def add_user_message(user_id: int, message: str):
    """Добавляет сообщение пользователя в историю"""
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append(message)
    # Ограничиваем историю 10 сообщениями
    if len(user_history[user_id]) > 10:
        user_history[user_id] = user_history[user_id][-10:]

@dp.message(Command("start"))
async def start_handler(message: Message):
    welcome_text = """
🌟 Добро пожаловать в службу поддержки "Лучшие кондиционеры"! 

Я ваш виртуальный помощник по кондиционерам. Могу помочь вам с:
• Инструкциями по установке
• Настройкой и управлением
• Устранением неполадок
• Техническими характеристиками

Просто напишите название модели вашего кондиционера и ваш вопрос!
    """
    await message.answer(welcome_text)

@dp.message()
async def message_handler(message: Message):
    """Основной обработчик сообщений"""
    user_id = message.from_user.id
    user_message = message.text
    
    # Добавляем сообщение в историю
    add_user_message(user_id, user_message)
    
    try:
        # Этап 1: LLM-диспетчер определяет тип запроса
        recent_messages = get_user_history(user_id, 3)
        dispatcher_response = await llm_dispatcher(recent_messages)
        
        # Если диспетчер может ответить сразу
        if dispatcher_response.get("type") == "direct_answer":
            await message.answer(dispatcher_response.get("answer", "Извините, не могу ответить на ваш вопрос."))
            return
        
        # Если диспетчер просит уточнить модель
        if dispatcher_response.get("type") == "need_clarification":
            await message.answer(dispatcher_response.get("answer", "Пожалуйста, уточните модель кондиционера."))
            return
        
        # Если нужен поиск по мануалам
        if dispatcher_response.get("type") == "search_manual":
            # Уведомляем пользователя о начале поиска
            await message.answer("🔍 Мне понадобится время для ответа. Подождите...")
            extraction_result = await llm_extractor(recent_messages)
            
            if not extraction_result:
                await message.answer("❌ Не удалось определить модель кондиционера и тему запроса.")
                return
            
            model_name = extraction_result.get("model")
            topic = extraction_result.get("topic")
            
            if not model_name or not topic:
                await message.answer("❌ Пожалуйста, укажите модель кондиционера и ваш вопрос более четко.")
                return
            
            # Этап 3: Поиск векторной базы по модели
            vector_db_path = find_model_database(model_name)
            
            if not vector_db_path:
                # Ищем похожие модели
                similar_models = search_similar_models(model_name)
                if similar_models:
                    models_text = "\n".join([f"• {model}" for model in similar_models])
                    await message.answer(f"❓ Модель '{model_name}' не найдена. Возможно, вы имели в виду:\n{models_text}")
                else:
                    await message.answer(f"❌ Кондиционер модели '{model_name}' не найден в нашей базе данных.")
                return
            
            # Этап 4: Извлечение релевантных чанков
            chunks = hybrid_search(vector_db_path, topic)
            
            if not chunks:
                await message.answer("❌ Информацию по вашему запросу найти не удалось.")
                return
            
            # Этап 5: Проверка релевантности найденной информации
            is_relevant = await llm_relevance_checker(user_message, chunks)
            
            if not is_relevant:
                await message.answer("❌ Информацию найти не удалось. Попробуйте переформулировать вопрос.")
                return
            
            # Этап 6: Генерация финального ответа
            final_answer = await llm_answer_generator(user_message, chunks, model_name)
            
            if final_answer:
                await message.answer(f"✅ {final_answer}")
            else:
                await message.answer("❌ Не удалось сформировать ответ. Попробуйте переформулировать вопрос.")
        
        else:
            await message.answer("❌ Извините, не могу обработать ваш запрос.")
    
    except Exception as e:
        logging.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer("❌ Произошла ошибка при обработке запроса. Попробуйте еще раз.")

async def main():
    """Главная функция запуска бота"""
    logging.info("Бот запускается...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())