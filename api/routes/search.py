from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve, ScoredChunk
from api.dependencies import get_qdrant

router=APIRouter()

class SearchRequest(BaseModel):
    query:str
    limit:int=5

@router.post("/search")
async def post_search(body:SearchRequest,qdrant: AsyncQdrantClient=Depends(get_qdrant)):
    store=QdrantStore()
    chunks=await retrieve(store=store,top_k=body.limit,query=body.query)
    return {
        "results": [
            {
                "file_path": c.payload.get("file_path"),
                "name": c.payload.get("name"),
                "chunk_type": c.payload.get("chunk_type"),
                "start_line": c.payload.get("start_line"),
                "end_line": c.payload.get("end_line"),
                "content": c.payload.get("content"),
                "score": c.score,
            }
            for c in chunks
        ]
    }