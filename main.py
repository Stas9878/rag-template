from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.core.logger import logger
from src.api.routes import router as api_router
from src.api.upload import router as upload_router

app = FastAPI(title='RAG Service', version='0.1.0')


app.include_router(api_router, prefix='/api')
app.include_router(upload_router, prefix='/api')


@app.get('/')
async def root():
    return RedirectResponse(url='/docs')


@app.get('/health')
async def health():
    return {'status': 'ok'}