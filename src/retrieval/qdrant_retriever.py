from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from src.core.settings import settings


def get_retriever(top_k: int = 3):
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )
    search_kwargs = {
        'filter': Filter(
            must=[FieldCondition(key='is_current', match=MatchValue(value=True))]
        ),
        'k': top_k,
    }
    return vector_store.as_retriever(search_kwargs=search_kwargs)