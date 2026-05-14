import os

from openai import AsyncOpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

class Embedder:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
        )

    #async funct takes chunk as list[str] return vector list
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            results.extend(item.embedding for item in response.data)
        return results