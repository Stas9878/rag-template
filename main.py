import logging.config
from pathlib import Path
from fastapi import FastAPI

from src.api.rag import router as upload_router

config_path = Path(__file__).parent.parent / 'logging.conf'
if config_path.exists():
    logging.config.fileConfig(config_path, disable_existing_loggers=False)
else:
    # Резервная настройка (если файла нет)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

app = FastAPI(title='RAG Service', version='0.1.0')
app.include_router(upload_router, prefix='/api')