import os
import asyncio
import base64
import re
from typing import List, Tuple
from dotenv import load_dotenv
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
import pickle

load_dotenv()

VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "db")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
LLM_URL = os.getenv("LLM_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if LLM_URL:
    client = AsyncOpenAI(base_url=LLM_URL, api_key="dummy")
else:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Загружаем модель для эмбеддингов
embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

# Создаем директорию для баз данных
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

async def extract_models_from_page(pdf_path: str, page_num: int) -> List[str]:
    """Извлекает названия моделей кондиционеров со страницы PDF"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Конвертируем страницу в изображение
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Увеличиваем разрешение
    img_data = pix.tobytes("png")
    img_base64 = base64.b64encode(img_data).decode()
    
    doc.close()
    
    # Отправляем в VLM
    try:
        response = await client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Найди на этой странице названия моделей кондиционеров. Верни только названия моделей через запятую, без дополнительного текста. Если моделей нет, верни 'НЕТ'."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        if result == "НЕТ" or not result:
            return []
        
        # Парсим названия моделей
        models = []
        # Разделяем по запятым, слэшам и другим разделителям
        parts = re.split(r'[,/;|&\n\r]+', result)
        for part in parts:
            model = part.strip()
            if model and len(model) > 1:  # Исключаем слишком короткие строки
                models.append(model)
        
        return models
        
    except Exception as e:
        print(f"Ошибка при извлечении моделей: {e}")
        return []

async def get_models_from_pdf(pdf_path: str) -> List[str]:
    """Получает названия моделей из первых двух страниц PDF"""
    models = []
    
    # Проверяем первую страницу
    first_page_models = await extract_models_from_page(pdf_path, 0)
    models.extend(first_page_models)
    
    # Если не нашли на первой странице, проверяем вторую
    if not models:
        doc = fitz.open(pdf_path)
        if len(doc) > 1:
            second_page_models = await extract_models_from_page(pdf_path, 1)
            models.extend(second_page_models)
        doc.close()
    
    return models

def has_images_or_tables(page) -> bool:
    """Проверяет, есть ли на странице изображения или таблицы"""
    # Проверяем изображения
    images = page.get_images()
    if images:
        return True
    
    # Простая проверка таблиц по наличию табов или выравненного текста
    text = page.get_text()
    lines = text.split('\n')
    tab_count = sum(1 for line in lines if '\t' in line or len(re.findall(r'\s{3,}', line)) > 2)
    
    return tab_count > 3  # Если много строк с табами/пробелами

async def process_page_with_vlm(pdf_path: str, page_num: int) -> str:
    """Обрабатывает страницу с изображениями/таблицами через VLM"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Конвертируем страницу в изображение
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    img_base64 = base64.b64encode(img_data).decode()
    
    doc.close()
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Опиши текстом все, что видишь на этой странице. Если есть таблицы, преобразуй их в сплошной текст. Если есть изображения с инструкциями, опиши последовательность действий. Будь максимально подробным."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Ошибка при обработке страницы {page_num}: {e}")
        return ""

async def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF, используя VLM для страниц с изображениями/таблицами"""
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        if has_images_or_tables(page):
            # Обрабатываем через VLM
            vlm_text = await process_page_with_vlm(pdf_path, page_num)
            full_text += f"\n\nСтраница {page_num + 1}:\n{vlm_text}\n"
        else:
            # Извлекаем обычный текст
            page_text = page.get_text()
            if page_text.strip():
                full_text += f"\n\nСтраница {page_num + 1}:\n{page_text}\n"
    
    doc.close()
    return full_text

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 150) -> List[str]:
    """Разбивает текст на чанки с перекрытием"""
    chunks = []
    
    # Сначала пробуем разбить по абзацам
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Если абзац слишком длинный, разбиваем его
            if len(paragraph) > chunk_size:
                # Разбиваем длинный абзац по предложениям
                sentences = re.split(r'[.!?]+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += sentence + '. '
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + '. '
                
                if temp_chunk:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = paragraph + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Добавляем перекрытие
    final_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and overlap > 0:
            # Добавляем перекрытие с предыдущим чанком
            prev_chunk = chunks[i-1]
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
            chunk = overlap_text + " " + chunk
        
        final_chunks.append(chunk)
    
    return final_chunks

def create_faiss_index(chunks: List[str], model_names: List[str]):
    """Создает и сохраняет FAISS индекс"""
    if not chunks:
        return
    
    # Создаем эмбеддинги
    embeddings = embedding_model.encode(chunks)
    
    # Создаем FAISS индекс
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Используем inner product для cosine similarity
    
    # Нормализуем эмбеддинги для cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    for model_name in model_names:
    # Сохраняем индекс и метаданные
        safe_model_name = re.sub(r'[^\w\-_\.]', '_', model_name)
        index_path = os.path.join(VECTOR_DB_DIR, f"{safe_model_name}.faiss")
        metadata_path = os.path.join(VECTOR_DB_DIR, f"{safe_model_name}_metadata.pkl")
        
        faiss.write_index(index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': chunks,
                'model_name': model_name,
                'embeddings_model': EMBEDDINGS_MODEL
            }, f)
        
        print(f"Создана база для модели '{model_name}' с {len(chunks)} чанками")

async def process_pdf_file(pdf_path: str):
    """Обрабатывает один PDF файл"""
    print(f"Обрабатываем: {pdf_path}")
    
    # Извлекаем названия моделей
    models = await get_models_from_pdf(pdf_path)
    
    if not models:
        print(f"Не удалось найти модели в {pdf_path}")
        return
    
    print(f"Найдены модели: {models}")
    
    # Извлекаем текст
    text = await extract_text_from_pdf(pdf_path)
    
    if not text.strip():
        print(f"Не удалось извлечь текст из {pdf_path}")
        return
    
    # Создаем чанки
    chunks = create_chunks(text)
    
    create_faiss_index(chunks, models)

async def main():
    """Основная функция"""
    # Ищем все PDF файлы в текущей директории
    pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("PDF файлы не найдены в текущей директории")
        return
    
    print(f"Найдено {len(pdf_files)} PDF файлов")
    
    # Обрабатываем каждый файл
    for pdf_file in pdf_files:
        try:
            await process_pdf_file(pdf_file)
        except Exception as e:
            print(f"Ошибка при обработке {pdf_file}: {e}")
    
    print("Обработка завершена!")

if __name__ == "__main__":
    asyncio.run(main())