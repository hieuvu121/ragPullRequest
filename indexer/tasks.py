import asyncio
from pathlib import Path

from worker import celery_app
from config import settings
from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve
from pipeline.generator import generate_review
from scripts.index_repo import index_directory
from db.session import AsyncSessionLocal
from db.models import IndexedFile, PRReview, ReviewFeedback, Repo
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
def incremental_index(self, repo_full_name:str, installation_id:int, changed_files:list[str]=[], removed_files:list[str]=[]):
    try:
        asyncio.run(run_incremental_index(repo_full_name, installation_id, changed_files, removed_files))
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

async def run_incremental_index(repo_full_name: str, installation_id: int, changed_files: list[str], removed_files: list[str]) -> None:
    from gh_app.auth import make_auth
    from github import Github
    from pipeline.chunker import chunk_file
    from pipeline.embedder import Embedder
    import uuid
    from qdrant_client.models import PointStruct

    store = make_store()
    token = make_auth(installation_id).get_installation_token()
    g = Github(token)
    repo = g.get_repo(repo_full_name)
    repo_id = repo_full_name.replace("/", "-")
    embedder = Embedder()

    async with AsyncSessionLocal() as db:
        for path in removed_files:
            if not path.endswith(".py"):
                continue
            await store.delete_by_filter(repo_id, path)
            await marked_deleted(db, repo_full_name, path)

        for path in changed_files:
            if not path.endswith(".py"):
                continue
            try:
                content=repo.get_contents(path).decoded_content.decode("utf-8", errors="ignore")
                new_hash=hashlib.sha256(content.encode()).hexdigest()
                existing=await get_hash(db,repo_full_name,path)

                if existing==new_hash:
                    continue

                chunks=chunk_file(path,content)
                if not chunks:
                    continue

                vectors=await embedder.embed([c.content for c in chunks])
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

@celery_app.task(name="review_pr", bind=True, max_retries=3)
def review_pr(self, repo_full_name: str, pr_number: int, installation_id: int):
    try:
        asyncio.run(_run_review(repo_full_name, pr_number, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


async def _run_review(repo_full_name: str, pr_number: int, installation_id: int) -> None:
    import time
    from sqlalchemy import select
    from gh_app.auth import make_auth
    from gh_app.client import GithubClient
    from scripts.review_pipeline import parse_diff_lines

    start = time.monotonic()
    token = make_auth(installation_id).get_installation_token()
    client = GithubClient(token=token)
    diff = client.get_diff(repo_full_name, pr_number)

    store = make_store()

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Repo).where(Repo.full_name == repo_full_name))
        db_repo = result.scalar_one_or_none()
        if not db_repo:
            return

        pr_review = PRReview(repo_id=db_repo.id, pr_number=pr_number, status="pending")
        db.add(pr_review)
        await db.flush()

        try:
            diff_lines = parse_diff_lines(diff)
            chunks = await retrieve(store=store, query=diff, top_k=5)
            comments = await generate_review(diff_text=diff, chunks=chunks, diff_lines=diff_lines)

            gh_comments = [
                {
                    "path": c.path,
                    "line": c.line,
                    "side": "RIGHT",
                    "body": f"**[{c.severity.upper()}]** {c.issue}\n\n{c.suggestion}\n\n> _{c.citation}_",
                }
                for c in comments
            ]
            summary = f"AI review for PR #{pr_number} — {len(comments)} comment(s) generated."
            github_review_id = client.post_review(repo_full_name, pr_number, summary, gh_comments)

            pr_review.status = "posted"
            pr_review.github_review_id = github_review_id
            pr_review.latency_ms = int((time.monotonic() - start) * 1000)
            pr_review.raw_output = {
                "comments": [
                    {"line": c.line, "path": c.path, "severity": c.severity,
                     "issue": c.issue, "suggestion": c.suggestion, "citation": c.citation}
                    for c in comments
                ]
            }
        except Exception:
            pr_review.status = "failed"
            raise

        await db.commit()


@celery_app.task(name="record_feedback", bind=True, max_retries=3)
def record_feedback(self, comment_id: int, action: str, raw: dict):
    try:
        asyncio.run(_run_feedback(comment_id, action, raw))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

async def _run_feedback(comment_id: int, action: str, raw: dict) -> None:
    from sqlalchemy import select

    score_map = {"resolved": 1.0, "dismissed": -1.0, "created": 0.0}
    value = score_map.get(action, 0.0)
    github_review_id = raw.get("pull_request_review_id")

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PRReview).where(PRReview.github_review_id == github_review_id)
        )
        pr_review = result.scalar_one_or_none()
        if not pr_review:
            return

        db.add(ReviewFeedback(
            review_id=pr_review.id,
            comment_id=comment_id,
            action=action,
            value=value,
        ))
        await db.commit()