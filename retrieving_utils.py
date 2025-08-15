import os
import difflib
from typing import List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from dotenv import load_dotenv

load_dotenv()
# Пути к векторным базам
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR","db")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL","sberbank-ai/sbert_large_mt_nlu_ru")

# Инициализация модели для эмбеддингов
embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)

def find_model_database(model_name: str) -> Optional[str]:
    """
    Ищет векторную базу данных для конкретной модели кондиционера
    """
    if not os.path.exists(VECTOR_DB_DIR):
        return None
    
    # Нормализуем название модели
    normalized_model = model_name.lower().strip()
    
    # Ищем точное совпадение
    for filename in os.listdir(VECTOR_DB_DIR):
        if filename.endswith('.faiss'):
            # Убираем расширение .faiss
            db_model_name = filename[:-6].lower()
            if db_model_name == normalized_model:
                return os.path.join(VECTOR_DB_DIR, filename)
    
    return None

def search_similar_models(model_name: str, max_suggestions: int = 5) -> List[str]:
    """
    Ищет похожие модели кондиционеров по названию
    """
    if not os.path.exists(VECTOR_DB_DIR):
        return []
    
    # Получаем все доступные модели
    available_models = []
    for filename in os.listdir(VECTOR_DB_DIR):
        if filename.endswith('.faiss'):
            model = filename[:-6]  # Убираем расширение .faiss
            available_models.append(model)
    
    if not available_models:
        return []
    
    # Ищем похожие модели используя difflib
    similar_models = difflib.get_close_matches(
        model_name.lower(), 
        [model.lower() for model in available_models], 
        n=max_suggestions, 
        cutoff=0.4
    )
    
    # Возвращаем оригинальные названия моделей
    result = []
    for similar in similar_models:
        for original in available_models:
            if original.lower() == similar:
                result.append(original)
                break
    
    return result

def load_faiss_index_and_chunks(db_path: str) -> tuple:
    """
    Загружает FAISS индекс и соответствующие текстовые чанки
    """
    try:
        # Загружаем FAISS индекс
        index = faiss.read_index(db_path)
        
        # Загружаем соответствующие текстовые чанки
        chunks_path = db_path.replace('.faiss', '_chunks.pkl')
        if os.path.exists(chunks_path):
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
        else:
            # Если файл с чанками не найден, используем заглушки
            chunks = [f"Chunk {i}" for i in range(index.ntotal)]
        
        return index, chunks
    
    except Exception as e:
        print(f"Ошибка при загрузке индекса {db_path}: {e}")
        return None, None

def hybrid_search(db_path: str, query: str, top_k: int = 5) -> List[str]:
    semantic_chunks = extract_chunks_from_vector_db(db_path, query, top_k)
    keyword_chunks = extract_chunks_from_text_file(db_path, query, top_k)
    return semantic_chunks + keyword_chunks

def extract_chunks_from_text_file(db_path: str, query: str, top_k: int = 5) -> List[str]:
    text_file = txt_path = db_path.replace('.faiss', '.txt')
    if not os.path.exists(text_file):
        return []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Разбиваем текст на параграфы/чанки
    chunks = content.split('\n\n')  # Разделение по двойным переносам строк
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Извлекаем ключевые слова из запроса (простая токенизация)
    query_words = set(query.lower().split())
    
    # Подсчитываем релевантность каждого чанка
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0
        
        # Подсчитываем количество совпадений ключевых слов
        for word in query_words:
            if len(word) > 2:  # Игнорируем слишком короткие слова
                score += chunk_lower.count(word)
        
        if score > 0:
            chunk_scores.append((score, i, chunk))
    
    # Сортируем по релевантности и возвращаем топ-к
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    
    result = []
    for score, idx, chunk in chunk_scores[:top_k]:
        result.append(chunk)
    
    return result

def extract_chunks_from_vector_db(db_path: str, query: str, top_k: int = 5) -> List[str]:
    """
    Извлекает релевантные чанки из векторной базы данных
    """
    try:
        # Загружаем индекс и чанки
        index, chunks = load_faiss_index_and_chunks(db_path)
        
        if index is None or chunks is None:
            return []
        
        # Создаем эмбеддинг для запроса
        query_embedding = embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        
        # Выполняем поиск
        scores, indices = index.search(query_embedding, min(top_k, len(chunks)))
        
        # Извлекаем соответствующие чанки
        result_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                result_chunks.append(chunks[idx])
        
        return result_chunks
    
    except Exception as e:
        print(f"Ошибка при извлечении чанков: {e}")
        return []

def get_available_models() -> List[str]:
    """
    Возвращает список всех доступных моделей кондиционеров
    """
    if not os.path.exists(VECTOR_DB_DIR):
        os.makedirs(VECTOR_DB_DIR)
        return []
    
    models = []
    for filename in os.listdir(VECTOR_DB_DIR):
        if filename.endswith('.faiss'):
            model_name = filename[:-6]  # Убираем расширение .faiss
            models.append(model_name)
    
    return sorted(models)
