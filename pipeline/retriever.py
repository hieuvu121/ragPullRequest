import asyncio
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

from indexer.embedder import EMBEDDING_MODEL
from pipeline.qdrant_store import QdrantStore, SearchHit

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
RERANKING_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None

@dataclass
#will rerank the result so set the score again
class ScoredChunk:
    id:str
    score:float
    payload:dict

def reciprocal_rank_fusion(ranked_lists:list[list[str]],k:int=60 )->list[tuple[str,float]]:
    #dict to store rank and score after each times
    scores:dict[str,float]={}
    for ranked_list in ranked_lists:
        for r,doc_id in enumerate(ranked_list):
            #if first time -> 0.0 else take old val+ new point
            scores[doc_id]=scores.get(doc_id,0.0)+1.0/(k+r+1)
        #return list tuple (doc1,0.2), (doc2,0.3)
        #lambda func take first element(x) to sort
        return sorted(scores.items(),key=lambda x:x[1],reverse=True)

#define lazy load for get rerank
def _get_rerank()-> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker=CrossEncoder(RERANKING_MODEL)
    return _reranker

async def _embed_one(client: AsyncOpenAI, text: str) -> list[float]:
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding

def _strip_diff_markers(diff_hunk:str)->str:
    lines=[]
    for line in diff_hunk.splitlines():
        if line.startswith(("+","-")):
            lines.append(line[1:])
        elif not line.startswith("@@"):
            lines.append(line)
    return "\n".join(lines)

async def retrieve(
        store: QdrantStore,
        query:str,
        top_k:int =5,
        candidate_pool:int=20,
)->list[ScoredChunk]:
    client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    #strip diff marker
    stripped=_strip_diff_markers(query)

    #embed parallel(await run async 2 query and gather after)
    original_vector,stripped_vector=await asyncio.gather(
        _embed_one(client, query),
        _embed_one(client, stripped)
    )

    #search parallel
    original_hits,stripped_hits=await asyncio.gather(
        store.search(original_vector,limit=candidate_pool),
        store.search(stripped_vector,limit=candidate_pool)
    )
    #key is score, when rrf cal score again, score changed so need this dict to look up after rrf
    id_to_hit:dict[str,SearchHit]={}
    for hit in original_hits+stripped_hits:
        id_to_hit[hit.id]=hit

    #rrf
    merged=reciprocal_rank_fusion(
        [[h.id for h in original_hits],
         [h.id for h in stripped_hits]]
    )
    #save payload
    candidates=[
        ScoredChunk(
            id=doc_id,
            score=score,
            payload=id_to_hit[doc_id].payload
        )
        for doc_id,score in merged
        if doc_id in id_to_hit
    ]

    if len(candidates)<=top_k:
        return candidates

    #rerank
    reranker = _get_reranker()
    pairs = [[query, c.payload.get("content", "")] for c in candidates]
    ce_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:top_k]]