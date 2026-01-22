import logging
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from src.core.settings import settings

logger = logging.getLogger('__name__')

# Инициализация Qdrant
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

if not client.collection_exists(settings.collection_name):
        # 1. Создаём коллекцию
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=models.VectorParams(size=settings.vector_size, distance=models.Distance.COSINE),
            on_disk_payload=True,
        )

# Эмбеддинги
embeddings = OllamaEmbeddings(model=settings.embedding_model_name)

# LLM
llm = OllamaLLM(model=settings.llm_model_name, temperature=0.1)

# Векторное хранилище
vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.collection_name,
    embedding=embeddings,
)


def add_documents(documents: list[Document]):
    vector_store.add_documents(documents)
    logger.info(f'Добавлено {len(documents)} документов')


def search(query: str, k: int):
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key='metadata.is_current',
                match=models.MatchValue(value='true'),
            )
        ]
    )
    return vector_store.similarity_search(query, k=k, filter=query_filter)


def generate_answer(query: str, context: str) -> str:
    """Генерирует ответ с помощью LLM."""
    prompt = f"""
        Ты — эксперт по российскому законодательству.
        Используй только предоставленный контекст для ответа.
        Не добавляй ничего от себя.
        Если в контексте нет информации — скажи "Не знаю".

        Контекст:
        {context}

        Вопрос:
        {query}
    """
    return llm.invoke(prompt)