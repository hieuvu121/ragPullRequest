import base64
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str = ""
    database_url: str
    redis_url: str
    github_app_id: int = 0
    github_app_private_key_b64: str = ""
    github_webhook_secret: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @computed_field
    @property
    def github_private_key_pem(self) -> str:
        if not self.github_app_private_key_b64:
            return ""
        return base64.b64decode(self.github_app_private_key_b64).decode()

    model_config = {"env_file": ".env"}

settings = Settings()