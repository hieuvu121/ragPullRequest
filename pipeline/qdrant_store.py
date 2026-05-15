import os
from dataclasses import dataclass
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from dotenv import load_dotenv

load_dotenv()

COLLECTION = "code_chunks"
VECTOR_DIM = 1536

#define search format
@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict

class QdrantStore:
    def __init__(self):
        #create async connection to qdrant
        self.client=AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    async def create_collection(self)->None:
        existing={c.name for c in(await self.client.get_collections())}
        if COLLECTION not in existing:
            await self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
            )

    #point struct is 1 point on the map with many attributes: id, vector, payload
    async def upsert(self,points:list[PointStruct])-> None:
        await self.client.upsert(collection_name=COLLECTION,points=points)

    async def search(self,query_vector:list[float],limit:int=20)->list[SearchHit]:
        hits=await self.client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=limit
        )
#        SearchHit=[]
 #       for h in hits:
  #          hit=SearchHit(id=str(h.id),score=h.score,payload=h.payload or{})
   #         SearchHit.append(hit)
        return [SearchHit(id=str(h.id),score=h.score,payload=h.payload or{})for h in hits]

#need to pass repo and path id
    async def delete_by_filter(self, repo_id:str, file_path:str)->None:
        await self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[
                #same with sql where key=match
                FieldCondition(key="repo_id",match=MatchValue(value=repo_id)),
                FieldCondition(key="file_path",match=MatchValue(value=file_path))
            ])
        )
