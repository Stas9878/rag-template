import time
from fastapi import APIRouter, HTTPException

from src.retrieval.qdrant_retriever import get_retriever
from src.core.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)


@router.post('/query')
async def query_rag(query: str):
    start_time = time.time()
    try:
        retriever = get_retriever(top_k=3)
        docs = await retriever.ainvoke(query)

        results = []
        for doc in docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
            })

        duration = time.time() - start_time
        logger.info(f'Query processed in {duration:.2f}s: {query[:50]}...')

        return {
            'query': query,
            'results': results,
            'metrics': {
                'retrieval_time_sec': round(duration, 3),
                'num_results': len(results),
            }
        }

    except Exception as e:
        logger.error(f'Error in /query: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/dashboard')
async def redirect_to_qdrant_dashboard():
    return {'detail': 'Redirect to Qdrant Dashboard', 'url': '/dashboard'}