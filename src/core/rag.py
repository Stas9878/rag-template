from pathlib import Path
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance

from src.core.logger import logger
from src.core.settings import settings

# Инициализация
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

# Создаём коллекцию при импорте
if not client.collection_exists(settings.collection_name):
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=settings.vector_size, distance=Distance.COSINE),
    )

embeddings = OllamaEmbeddings(model=settings.embedding_model_name)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.collection_name,
    embedding=embeddings,
)

def add_documents(documents: list[Document]):
    """Добавляет документы в векторное хранилище."""
    vector_store.add_documents(documents)
    logger.info(f'Добавлено {len(documents)} документов')

def search(query: str, k: int):
    """Выполняет поиск по запросу."""
    return vector_store.similarity_search(query, k=k)