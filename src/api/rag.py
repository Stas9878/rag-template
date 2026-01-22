import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from langchain_core.documents import Document
from fastapi import APIRouter, UploadFile, HTTPException, Query

from src.utils.timer import Timer
from src.core.logger import logger
from src.core.retrieve import search, generate_answer, add_documents
from src.utils.job_with_text import extract_text_from_pdf, chunk_text

router = APIRouter()


@router.get('/query')
def query_rag(query: str = Query(..., description='Поисковый запрос')):
    timer = Timer()

    try:
        # Этап 1: Retrieval
        with timer.measure('retrieval_time_sec'):
            docs = search(query, k=10)

        if docs:
            context = '\n\n'.join([doc.page_content for doc in docs])

            # Этап 2: Generation
            with timer.measure('generation_time_sec'):
                answer = generate_answer(query, context)
        else:
            answer = None

        results = [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
            }
            for doc in docs
        ]

        metrics = timer.get_metrics()
        metrics['num_results'] = len(results)

        return {
            'query': query,
            'answer': answer,
            'results': results,
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f'Ошибка в /query: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/upload')
async def upload_pdf(file: UploadFile):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail='Только PDF')

    temp_path = Path(f'/tmp/{file.filename}')
    contents = await file.read()
    temp_path.write_bytes(contents)

    try:
        text = extract_text_from_pdf(temp_path)
        if not text:
            raise HTTPException(status_code=400, detail='PDF пустой')

        file_size = len(contents)
        mime_type = mimetypes.guess_type(file.filename)[0] or 'application/pdf'
        modified_at = datetime.now(timezone.utc)
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

        # Разбиваем на чанки
        chunks = chunk_text(text)
        logger.info(f'Разбито на {len(chunks)} чанков')

        # Подготавливаем документы
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                'filename': file.filename,
                'content_hash': content_hash,
                'file_size_bytes': file_size,
                'mime_type': mime_type,
                'modified_at': modified_at.isoformat(),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'is_current': 'true',
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

        # Добавляем в RAG
        add_documents(documents)

        return {'status': 'ok', 'chunks': len(chunks)}

    finally:
        if temp_path.exists():
            temp_path.unlink()