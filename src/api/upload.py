import hashlib
import mimetypes
from pathlib import Path
from pypdf import PdfReader
from qdrant_client import QdrantClient
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks

from src.core.logger import logger
from src.core.settings import settings

router = APIRouter()

# Инициализация эмбеддингов
EMBEDDINGS = HuggingFaceEmbeddings(model_name=settings.embedding_model)


def extract_text_from_pdf(file_path: Path) -> str:
    try:
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        return text.strip()
    except Exception as e:
        logger.error(f'Ошибка извлечения текста из PDF: {e}')
        raise HTTPException(status_code=400, detail='Не удалось прочитать PDF')


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=['\n\n', '\n', '. ', ' ', ''],
    )
    return splitter.split_text(text)


async def upsert_document_chunks(
    filename: str,
    text: str,
    file_size: int,
    mime_type: str,
    modified_at: datetime,
):
    content_hash = compute_hash(text)
    chunks = chunk_text(text)

    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    # Создаём коллекцию, если не существует
    if not client.collection_exists(settings.collection_name):
        from qdrant_client.http.models import Distance, VectorParams
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # Шаг 1: Пометить старые версии как неактуальные
    old_points = client.scroll(
        collection_name=settings.collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key='filename', match=MatchValue(value=filename))
            ]
        ),
        limit=10000,  # достаточно для одного файла
        with_payload=True,
    )[0]

    if old_points:
        # Собираем ID всех точек с этим filename
        point_ids = [point.id for point in old_points]
        # Обновляем payload: is_current = False
        for point_id in point_ids:
            client.set_payload(
                collection_name=settings.collection_name,
                payload={'is_current': False},
                points=[point_id],
            )
        logger.info(f'Помечено {len(point_ids)} старых чанков как устаревшие для \'{filename}\'')

    # Шаг 2: Добавить новые чанки
    documents = []
    for ind, chunk in enumerate(chunks):
        metadata = {
            'filename': filename,
            'content_hash': content_hash,
            'file_size_bytes': file_size,
            'mime_type': mime_type,
            'modified_at': modified_at.isoformat(),
            'chunk_index': ind,
            'total_chunks': len(chunks),
            'is_current': True,
        }
        doc = Document(page_content=chunk, metadata=metadata)
        documents.append(doc)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=EMBEDDINGS,
    )
    await vector_store.aadd_documents(documents)
    logger.info(f'Добавлено {len(documents)} чанков для \'{filename}\' в Qdrant')


@router.post('/upload')
async def upload_pdf(
    file: UploadFile,
    background_tasks: BackgroundTasks,
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Поддерживаются только PDF-файлы')

    # Временное сохранение
    temp_path = Path(f'/tmp/{file.filename}')
    try:
        contents = await file.read()
        temp_path.write_bytes(contents)

        # Метаданные
        file_size = len(contents)
        mime_type, _ = mimetypes.guess_type(file.filename) or ('application/pdf', None)
        modified_at = datetime.now(timezone.utc)

        # Извлечение текста
        text = extract_text_from_pdf(temp_path)
        if not text:
            raise HTTPException(status_code=400, detail='PDF не содержит извлекаемого текста')

        # Запуск в фоне (чтобы не блокировать запрос)
        background_tasks.add_task(
            upsert_document_chunks,
            filename=file.filename,
            text=text,
            file_size=file_size,
            mime_type=mime_type,
            modified_at=modified_at,
        )

        return {
            'status': 'accepted',
            'filename': file.filename,
            'message': 'Файл принят, обработка запущена в фоне',
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Ошибка при загрузке PDF: {e}')
        raise HTTPException(status_code=500, detail='Внутренняя ошибка сервера')
    finally:
        if temp_path.exists():
            temp_path.unlink()