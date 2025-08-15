import json
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Инициализация OpenAI клиента
LLM_URL = os.getenv("LLM_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_URL:
    client = AsyncOpenAI(base_url=LLM_URL, api_key="dummy")
else:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def llm_dispatcher(user_message: str) -> Dict[str, str]:
    """
    LLM-диспетчер определяет тип запроса пользователя
    """
    prompt = f"""
Ты - виртуальный помощник компании "Лучшие кондиционеры". 

Проанализируй сообщение пользователя и определи один из типов запроса:

1. "direct_answer" - если это простое приветствие, вопрос о тебе, общие вопросы о компании
2. "need_clarification" - если пользователь спрашивает о кондиционере, но не указал конкретную модель
3. "search_manual" - если пользователь задает конкретный вопрос о конкретной модели кондиционера

О компании "Лучшие кондиционеры":
- Ведущий производитель климатической техники
- Более 15 лет на рынке
- Широкий ассортимент бытовых и промышленных кондиционеров
- Сервисная поддержка по всей стране

Сообщение пользователя: "{user_message}"

Ответь в формате JSON:
{{
    "type": "direct_answer|need_clarification|search_manual",
    "answer": "текст ответа пользователю"
}}

Если type = "direct_answer", дай краткий дружелюбный ответ.
Если type = "need_clarification", попроси указать модель кондиционера.
Если type = "search_manual", поставь answer = "search_needed".
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    
    except Exception as e:
        print(f"Ошибка в llm_dispatcher: {e}")
        return {
            "type": "direct_answer",
            "answer": "Извините, произошла ошибка. Попробуйте еще раз."
        }

async def llm_extractor(recent_messages: str) -> Optional[Dict[str, str]]:
    """
    LLM-извлекатель извлекает модель кондиционера и тему из сообщений
    """
    
    prompt = f"""
Проанализируй последние сообщения пользователя и извлеки:
1. Модель кондиционера (точное название)
2. Тему/вопрос, который интересует пользователя

Сообщения:
{recent_messages}

Ответь в формате JSON:
{{
    "model": "точное название модели",
    "topic": "краткое описание темы/вопроса"
}}

Если модель или тему определить невозможно, поставь null.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    
    except Exception as e:
        print(f"Ошибка в llm_extractor: {e}")
        return None

async def llm_relevance_checker(user_question: str, chunks: List[str]) -> bool:
    """
    LLM-ассистент проверяет релевантность найденной информации
    """
    chunks_text = "\n---\n".join(chunks)
    
    prompt = f"""
Проверь, содержится ли в предоставленных фрагментах документации ответ на вопрос пользователя.

Вопрос пользователя: "{user_question}"

Фрагменты документации:
{chunks_text}

Ответь только "да" или "нет".
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip().lower()
        return "да" in result or "yes" in result
    
    except Exception as e:
        print(f"Ошибка в llm_relevance_checker: {e}")
        return False

async def llm_answer_generator(user_question: str, chunks: List[str], model_name: str) -> Optional[str]:
    """
    LLM-ассистент генерирует финальный ответ пользователю
    """
    chunks_text = "\n---\n".join(chunks)
    
    prompt = f"""
Ты - эксперт по кондиционерам компании "Лучшие кондиционеры". 
Ответь на вопрос пользователя, используя предоставленную документацию.

Модель кондиционера: {model_name}
Вопрос пользователя: "{user_question}"

Документация:
{chunks_text}

Требования к ответу:
- Используй только информацию из документации
- Будь конкретным и полезным
- Структурируй ответ для удобства чтения
- Если нужны пошаговые инструкции, пронумеруй их
- Упомини модель кондиционера в начале ответа

Ответ:
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Ошибка в llm_answer_generator: {e}")
        return None