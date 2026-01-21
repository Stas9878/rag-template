from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    collection_name: str
    embedding_model: str


settings = Settings()