from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_host: str
    qdrant_port: int
    collection_name: str
    embedding_model_name: str
    vector_size: int = 768


    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',  # игнорировать лишние переменные в .env
        case_sensitive=False
    )


settings = Settings()