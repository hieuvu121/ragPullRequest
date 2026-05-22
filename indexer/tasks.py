import asyncio
from pathlib import Path

from worker import celery_app
from config import settings
from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve
from pipeline.generator import generate_review
from scripts.index_repo import index_directory
from db.session import AsyncSessionLocal
from db.models import IndexedFile, PRReview
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, timezone
import hashlib

@celery_app.task(name="full_index",bind=True,max_retries=3,default_retry_delay=60)
def full_index(self,repo_full_name:str, installation_id:int):
    try:
        #create event loop
        asyncio.run(run_full_index(repo_full_name,installation_id))
    except Exception as e:
        raise self.retry(exc=e,countdown=2**self.request.retries*30)

@celery_app.task(name="incremental_index",bind=True,default_retry_delay=60)
def incremental_index(self, repo_full_name:str, changed_files:list[dict], installation_id:int):
    try:
        asyncio.run(run_incremental_index(repo_full_name,installation_id))
    except Exception as e:
        #time wait for retry increment
        raise self.retry(exc=e,countdown=2**self.request.retries*30)

#check if hash content changed or not
async def get_hash(db,repo_full_name:str,file_path:str)->str|None:
    from sqlalchemy import select
    result=await db.execute(
        select(IndexedFile.content_hash).where(
            IndexedFile.file_path==file_path
        )
    )
    return result.scalar_one_or_none()

def make_store()->QdrantStore:
    return QdrantStore()

async def upsert_indexed_file(db,repo_full_name:str, file_path:str,content_hash:str, chunk_count:int)->None:
    #equivelant to insert statement
    stmt=pg_insert(IndexedFile).values(
        file_path=file_path,
        content_hash=content_hash,
        status="indexed",
        retry_count=0,
        chunk_count=chunk_count,
        indexed_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        #if index elements exist before-> update only
        index_elements=["repo_id","file_path"],
        set_={"content_hash":content_hash, "status":"indexed",
              "chunk_count":chunk_count,"indexed_at":datetime.now(timezone.utc)}
    )

    await db.execute(stmt)
    await db.commit()

#track failed attempts
async def mark_failed(db,repo_full_name:str, file_path:str)->None:
    from sqlalchemy import update
    await db.execute(
        update(IndexedFile)
        .where(IndexedFile.file_path==file_path)
        .values(
            content_hash=None,
            status="failed",
            retry_count=IndexedFile.retry_count+1
        )
    )

    #if failed more than 3 times-> marked as permanent
    await db.execute(
        update(IndexedFile)
        .where(IndexedFile.file_path==file_path,IndexedFile.retry_count>=3)
        .values(status="failed_permanent")
    )

    await db.commit()

#soft delete, run when delete file from qdrant, not hard delete for audit trail,...
async def marked_deleted(db,repo_full_name:str, file_path:str)->None:
    from sqlalchemy import update
    await db.execute(
        update(IndexedFile)
        .where(IndexedFile.file_path==file_path)
        .values(status="deleted", content_hash=None)
    )

    await db.commit()

async def run_full_index(repo_full_name:str, installation_id:int)->None:
    import tempfile,subprocess
    store=make_store()
    await store.create_collection()
    repo_id=repo_full_name.replace("/","-")

    #clone to a temp folder
    with tempfile.TemporaryDirectory() as tmp:
        #subprocess allow run as cmd
        subprocess.run(
            ["git", "clone", "--depth", "1",
             f"https://github.com/{repo_full_name}.git", tmp],
            check=True, capture_output=True,
        )
        await index_directory(Path(tmp),repo_id)

async def run_incremental_index(repo_full_name: str, changed_files: list[dict], installation_id: int) -> None:
    from github.auth import get_installation_token
    from github import Github
    from pipeline.chunker import chunk_file
    from pipeline.embedder import Embedder
    import uuid
    from qdrant_client.models import PointStruct

    store = make_store()
    token = get_installation_token(installation_id)
    g = Github(token)
    repo = g.get_repo(repo_full_name)
    repo_id = repo_full_name.replace("/", "-")
    embedder = Embedder()

    #equal to create db instance and start a session-> async with will auto close session, dont need to close manually
    async with AsyncSessionLocal() as db:
        for f in changed_files:
            path=f["path"]
            if not path.endswith(".py"):
                continue
            if f["status"]=="removed":
                await store.delete_by_filter(repo_id,path)
                await marked_deleted(db,repo_full_name,path)
                continue
            try:
                #query content and compare with hash
                content=repo.get_contents(path).decoded_content.decode("utf-8", errors="ignore")
                new_hash=hashlib.sha256(content.encode()).hexdigest()
                existing=await get_hash(db,repo_full_name,path)

                if existing==new_hash:
                    continue
                #chunk files if new and want to add to dtb
                chunks=chunk_file(path,content)
                if not chunks:
                    continue

                #embed new chunks
                vectors=await embedder.embed([c.content for c in chunks])

                #delete old chunks
                await store.delete_by_filter(repo_id,path)
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=v,
                        payload={"repo_id": repo_id, "file_path": c.file_path,
                                 "name": c.name, "chunk_type": c.chunk_type,
                                 "start_line": c.start_line, "end_line": c.end_line,
                                 "content": c.content, "content_hash": new_hash},
                    )
                    for c, v in zip(chunks, vectors)
                ]
                await store.upsert(points)
                await upsert_indexed_file(db,repo_full_name,path,new_hash,len(chunks))
            except Exception:
                await mark_failed(db,repo_full_name,path)

async def run_review_pr(repo_full_name:str, pr_number:int, diff_text:str,installation_id:int)-> None:
    from scripts.review_pipeline import parse_diff_lines
    import time

    store=make_store()
    diff_lines=parse_diff_lines(diff_text)
    t0=time.monotonic()
    chunks=await retrieve(store=store,query=diff_text,top_k=5)
    comments=await generate_review(diff_text=diff_text,chunks=chunks,diff_lines=diff_lines)
    latency_ms=int((time.monotonic()-t0)*1000)

    #insert pr review to
    async with AsyncSessionLocal() as db:
        stmt = pg_insert(PRReview).values(
            pr_number=pr_number,
            status="posted" if comments else "failed",
            latency_ms=latency_ms,
            raw_output={"comments": [
                {"line": c.line, "path": c.path, "severity": c.severity,
                 "issue": c.issue, "suggestion": c.suggestion, "citation": c.citation}
                for c in comments
            ]},
        )
        await db.execute(stmt)
        await db.commit()