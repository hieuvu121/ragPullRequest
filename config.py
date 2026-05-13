from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    qdrant_url: str
    qdrant_api_key: str = ""
    openai_api_key: str
    github_app_id: str = ""
    github_app_private_key_b64: str = ""
    github_webhook_secret: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"

settings = Settings()
