import hashlib
from datetime import datetime
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.core.settings import settings


def upsert_document(
    text: str,
    filename: str,
    source_type: str = 'manual',
    modified_at: datetime = None,
):
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    if modified_at is None:
        modified_at = datetime.now()

    metadata = {
        'filename': filename,
        'source_type': source_type,
        'content_hash': content_hash,
        'modified_at': modified_at.isoformat(),
        'is_current': True,
    }

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    doc = Document(page_content=text, metadata=metadata)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents([doc])
    print(f'✅ Документ \'{filename}\' добавлен в Qdrant')