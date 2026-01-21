# src/core/rag.py
from pathlib import Path
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance

from src.core.logger import logger
from src.core.settings import settings

# Инициализация Qdrant
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

if not client.collection_exists(settings.collection_name):
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
    )

# Эмбеддинги
embeddings = OllamaEmbeddings(model=settings.embedding_model_name)

# LLM
llm = OllamaLLM(model="qwen3:latest", temperature=0.1)

# Векторное хранилище
vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.collection_name,
    embedding=embeddings,
)

def add_documents(documents: list[Document]):
    vector_store.add_documents(documents)
    logger.info(f'Добавлено {len(documents)} документов')

def search(query: str, k: int = 5):
    return vector_store.similarity_search(query, k=k)

def generate_answer(query: str, context: str) -> str:
    """Генерирует ответ с помощью LLM."""
    prompt = f"""
        Ты — эксперт по российскому законодательству.
        Используй только предоставленный контекст для ответа.
        Если в контексте нет информации — скажи "Не знаю".

        Контекст:
        {context}

        Вопрос:
        {query}

        Ответ:
    """
    return llm.invoke(prompt)