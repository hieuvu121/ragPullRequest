from openai import AsyncOpenAI
from config import settings

EMBEDDING_MODEL="text-embedding-3-small"
BATCH_SIZE=100

class Embedder:
    def __init__(self):
        self.client=AsyncOpenAI(api_key=settings.openai_api_key)
    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            results.extend(item.embedding for item in response.data)
        return results