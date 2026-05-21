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

@celery_app.task(name="full_index",bind=True,max_retries=3,default_reetry_delay=60)
def full_index(self,repo_full_name:str, installation_id:int):
    try:
        #create event loop
        asyncio.run(run_full_index(repo_full_name,installation_id))
    except Exception as e:
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
        set={"content_hash":content_hash, "status":"indexed",
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
        .values(status=="deleted",content_hash=None)
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