from contextlib import asynccontextmanager
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from config import settings
from api.routes.index import router as index_router
from api.routes.search import router as search_router

QDRANT_COLLECTIONS = ["code_chunks", "adr_docs", "pr_history"]
VECTOR_DIM = 1536

#init connection 1 time for the whole controller
async def init_qdrant(app: FastAPI):
    client=AsyncQdrantClient(url=settings.qdrant_url,api_key=settings.qdrant_api_key or None)
    for name in QDRANT_COLLECTIONS:
        if not await client.collection_exists(name):
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_DIM,distance=Distance.COSINE)
            )
    app.state.qdrant=client

@asynccontextmanager
async def lifespan(app:FastAPI):
    await init_qdrant(app)
    yield

app=FastAPI(lifespan=lifespan)
app.include_router(index_router)
app.include_router(search_router)

@app.get("/healthz")
def health():
    return {"status": "ok"}