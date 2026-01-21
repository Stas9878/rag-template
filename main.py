from fastapi import FastAPI

from src.api.rag import router as upload_router

app = FastAPI(title='RAG Service', version='0.1.0')


app.include_router(upload_router, prefix='/api')