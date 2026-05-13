# Phase 2: Infrastructure (FastAPI + Celery + Postgres + Docker) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 1 complete — `pipeline/` module validated, `index_repo.py` and `review_pipeline.py` produce correct results.

**Goal:** Wrap the working Phase 1 RAG pipeline in production infrastructure. No new AI logic — only wiring. `docker compose up` → `POST /index` → Celery worker indexes repo → `POST /search` returns relevant chunks.

**Architecture:** Phase 1 `pipeline/` module is imported directly by Celery tasks. FastAPI routes enqueue tasks and never call pipeline functions directly. SQLAlchemy tracks indexing state. All services run via Docker Compose.

**Tech Stack:** FastAPI, SQLAlchemy 2 async + asyncpg + Alembic, Celery + Redis, pydantic-settings, Docker Compose, pytest + pytest-asyncio + httpx

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add FastAPI, SQLAlchemy, Celery, Alembic, asyncpg |
| `docker-compose.yml` | Create | 5 services: api, worker, redis, qdrant, postgres |
| `Dockerfile` | Create | Build image for api + worker |
| `.env` | Modify | Add `DATABASE_URL`, `REDIS_URL` |
| `config.py` | Create | `Settings` — all env vars via pydantic-settings |
| `db/__init__.py` | Create | Package marker |
| `db/session.py` | Create | Async engine + `AsyncSessionLocal` |
| `db/models.py` | Create | 4 models: `Repo`, `IndexedFile`, `PRReview`, `ReviewFeedback` |
| `alembic.ini` | Create | Via `alembic init` |
| `alembic/env.py` | Modify | Async migration runner |
| `worker.py` | Create | `celery_app` init |
| `indexer/__init__.py` | Create | Package marker |
| `indexer/tasks.py` | Create | `full_index`, `incremental_index`, `review_pr` Celery tasks |
| `api/__init__.py` | Create | Package marker |
| `api/main.py` | Create | FastAPI app + lifespan (Qdrant init + Alembic) |
| `api/dependencies.py` | Create | `get_db`, `get_qdrant` |
| `api/routes/__init__.py` | Create | Package marker |
| `api/routes/index.py` | Create | `POST /index` — enqueues `full_index` |
| `api/routes/search.py` | Create | `POST /search` — calls retriever directly |
| `tests/test_config.py` | Create | Settings loading |
| `tests/test_models.py` | Create | Column + constraint assertions |
| `tests/test_tasks.py` | Create | Task registration |
| `tests/test_routes.py` | Create | POST /index, POST /search |

---

## Task 1: Extend `pyproject.toml` + Docker files

**Files:**
- Modify: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Modify: `.env`

- [ ] **Step 1: Add Phase 2 dependencies to `pyproject.toml`**

Add under `[tool.poetry.dependencies]`:
```toml
fastapi = "^0.111.0"
uvicorn = {extras = ["standard"], version = "^0.29.0"}
celery = {extras = ["redis"], version = "^5.3.6"}
redis = "^5.0.4"
sqlalchemy = {extras = ["asyncio"], version = "^2.0.29"}
asyncpg = "^0.29.0"
alembic = "^1.13.1"
pydantic-settings = "^2.2.1"
```

Add under `[tool.poetry.group.dev.dependencies]`:
```toml
httpx = "^0.27.0"
```

```bash
poetry install
```

- [ ] **Step 2: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install poetry==1.8.2
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root
COPY . .
```

- [ ] **Step 3: Create `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [postgres, redis, qdrant]
    volumes: [.:/app]

  worker:
    build: .
    command: celery -A worker.celery_app worker --loglevel=info --concurrency=4
    env_file: .env
    depends_on: [postgres, redis, qdrant]
    volumes: [.:/app]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  qdrant:
    image: qdrant/qdrant:v1.9.2
    ports: ["6333:6333"]
    volumes: [qdrant_data:/qdrant/storage]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: rag_pr_reviewer
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports: ["5432:5432"]
    volumes: [postgres_data:/var/lib/postgresql/data]

volumes:
  qdrant_data:
  postgres_data:
```

- [ ] **Step 4: Add to `.env`**

```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/rag_pr_reviewer
REDIS_URL=redis://redis:6379/0
```

Keep existing `OPENAI_API_KEY` and `QDRANT_URL=http://qdrant:6333`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Dockerfile docker-compose.yml .env
git commit -m "feat: add FastAPI, Celery, SQLAlchemy deps + Docker Compose"
```

---

## Task 2: `config.py`

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import base64
import importlib
import pytest


def _set_env(monkeypatch):
    pem = b"-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("GITHUB_APP_ID", "123")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_B64", base64.b64encode(pem).decode())
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "secret")


def test_settings_loads_required_fields(monkeypatch):
    _set_env(monkeypatch)
    import config as cfg
    importlib.reload(cfg)
    s = cfg.Settings()
    assert s.openai_api_key == "sk-test"
    assert s.database_url.startswith("postgresql")


def test_github_private_key_decoded(monkeypatch):
    _set_env(monkeypatch)
    import config as cfg
    importlib.reload(cfg)
    s = cfg.Settings()
    assert s.github_private_key.startswith("-----BEGIN RSA PRIVATE KEY-----")


def test_missing_openai_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    import config as cfg
    importlib.reload(cfg)
    with pytest.raises(Exception):
        cfg.Settings()
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write `config.py`**

```python
import base64
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str = ""
    database_url: str
    redis_url: str
    github_app_id: str = ""
    github_app_private_key_b64: str = ""
    github_webhook_secret: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def github_private_key(self) -> str:
        if not self.github_app_private_key_b64:
            return ""
        return base64.b64decode(self.github_app_private_key_b64).decode()

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 4: Run — verify all pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: pydantic Settings — all env vars, base64 private key decode"
```

---

## Task 3: `db/session.py` + `db/models.py` + Alembic

**Files:**
- Create: `db/__init__.py`, `db/session.py`, `db/models.py`
- Create: `alembic.ini`, `alembic/env.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from db.models import Repo, IndexedFile, PRReview, ReviewFeedback


def test_indexed_file_columns():
    cols = {c.name for c in IndexedFile.__table__.columns}
    assert {"file_path", "content_hash", "status", "retry_count",
            "chunk_count", "indexed_at"}.issubset(cols)


def test_indexed_file_unique_constraint():
    types = {type(c).__name__ for c in IndexedFile.__table__.constraints}
    assert "UniqueConstraint" in types


def test_pr_review_columns():
    cols = {c.name for c in PRReview.__table__.columns}
    assert {"pr_number", "status", "langfuse_trace_id", "latency_ms"}.issubset(cols)


def test_review_feedback_columns():
    cols = {c.name for c in ReviewFeedback.__table__.columns}
    assert {"action", "value"}.issubset(cols)
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'db.models'`

- [ ] **Step 3: Write `db/session.py`**

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from config import settings

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
```

- [ ] **Step 4: Write `db/models.py`**

```python
import uuid
from datetime import datetime
from sqlalchemy import Text, Integer, Float, BigInteger, ForeignKey, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    github_repo_id: Mapped[int] = mapped_column(Integer, unique=True)
    full_name: Mapped[str] = mapped_column(Text)
    installation_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class IndexedFile(Base):
    __tablename__ = "indexed_files"
    __table_args__ = (UniqueConstraint("repo_id", "file_path"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"))
    file_path: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)  # NULL = failed
    status: Mapped[str] = mapped_column(Text, default="indexed")  # indexed|failed|failed_permanent|deleted
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class PRReview(Base):
    __tablename__ = "pr_reviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"))
    pr_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, default="pending")  # pending|posted|failed
    langfuse_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ReviewFeedback(Base):
    __tablename__ = "review_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    review_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pr_reviews.id"))
    action: Mapped[str] = mapped_column(Text)   # "+1" or "-1"
    value: Mapped[float] = mapped_column(Float)  # 1.0 or 0.0
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
```

- [ ] **Step 5: Run test — verify pass**

```bash
pytest tests/test_models.py -v
```

Expected: 4 `PASSED`

- [ ] **Step 6: Set up Alembic**

```bash
alembic init alembic
```

In `alembic.ini`, clear the default URL:
```
sqlalchemy.url =
```

Replace `alembic/env.py` entirely:

```python
import asyncio
from logging.config import fileConfig
from sqlalchemy.ext.asyncio import create_async_engine
from alembic import context
from config import settings
from db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online():
    connectable = create_async_engine(settings.database_url)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_offline():
    context.configure(url=settings.database_url, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

- [ ] **Step 7: Generate and apply migration**

```bash
docker compose up postgres -d
alembic revision --autogenerate -m "initial"
alembic upgrade head
```

Expected: `Running upgrade -> <hash>, initial`

- [ ] **Step 8: Commit**

```bash
git add db/ alembic.ini alembic/ tests/test_models.py
git commit -m "feat: SQLAlchemy models + Alembic async migration"
```

---

## Task 4: `worker.py` + `indexer/tasks.py`

**Files:**
- Create: `worker.py`
- Create: `indexer/__init__.py`, `indexer/tasks.py`
- Create: `tests/test_tasks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tasks.py
def test_celery_app_configured():
    from worker import celery_app
    assert celery_app.conf.task_serializer == "json"
    assert "indexer.tasks" in celery_app.conf.include


def test_full_index_task_registered():
    from worker import celery_app
    from indexer import tasks  # noqa: F401 — triggers registration
    assert "full_index" in celery_app.tasks


def test_incremental_index_task_registered():
    from worker import celery_app
    from indexer import tasks  # noqa: F401
    assert "incremental_index" in celery_app.tasks


def test_review_pr_task_registered():
    from worker import celery_app
    from indexer import tasks  # noqa: F401
    assert "review_pr" in celery_app.tasks
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_tasks.py -v
```

Expected: `ModuleNotFoundError: No module named 'worker'`

- [ ] **Step 3: Write `worker.py`**

```python
from celery import Celery
from config import settings

celery_app = Celery(
    "rag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["indexer.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
```

- [ ] **Step 4: Write `indexer/tasks.py`**

The tasks import Phase 1 functions and run them inside `asyncio.run()`. All retry logic lives here — the pipeline functions themselves are unaware of Celery.

```python
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


def _make_store() -> QdrantStore:
    return QdrantStore()


@celery_app.task(name="full_index", bind=True, max_retries=3, default_retry_delay=60)
def full_index(self, repo_full_name: str, installation_id: int):
    try:
        asyncio.run(_run_full_index(repo_full_name, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


@celery_app.task(name="incremental_index", bind=True, max_retries=3, default_retry_delay=60)
def incremental_index(self, repo_full_name: str, changed_files: list[dict], installation_id: int):
    try:
        asyncio.run(_run_incremental_index(repo_full_name, changed_files, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


@celery_app.task(name="review_pr", bind=True, max_retries=3, default_retry_delay=60)
def review_pr(self, repo_full_name: str, pr_number: int, diff_text: str, installation_id: int):
    try:
        asyncio.run(_run_review_pr(repo_full_name, pr_number, diff_text, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


async def _run_full_index(repo_full_name: str, installation_id: int) -> None:
    import tempfile, subprocess
    store = _make_store()
    await store.create_collection()
    repo_id = repo_full_name.replace("/", "-")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1",
             f"https://github.com/{repo_full_name}.git", tmp],
            check=True, capture_output=True,
        )
        await index_directory(Path(tmp), repo_id)


async def _run_incremental_index(
    repo_full_name: str, changed_files: list[dict], installation_id: int
) -> None:
    from github.auth import get_installation_token
    from github import Github
    from pipeline.chunker import chunk_file
    from pipeline.embedder import Embedder
    import uuid
    from qdrant_client.models import PointStruct

    store = _make_store()
    token = get_installation_token(installation_id)
    g = Github(token)
    repo = g.get_repo(repo_full_name)
    repo_id = repo_full_name.replace("/", "-")
    embedder = Embedder()

    async with AsyncSessionLocal() as db:
        for f in changed_files:
            path = f["path"]
            if not path.endswith(".py"):
                continue
            if f["status"] == "removed":
                await store.delete_by_filter(repo_id, path)
                await _mark_deleted(db, repo_full_name, path)
                continue
            try:
                content = repo.get_contents(path).decoded_content.decode("utf-8", errors="ignore")
                new_hash = hashlib.sha256(content.encode()).hexdigest()
                existing = await _get_hash(db, repo_full_name, path)
                if existing == new_hash:
                    continue
                chunks = chunk_file(path, content)
                if not chunks:
                    continue
                vectors = await embedder.embed([c.content for c in chunks])
                await store.delete_by_filter(repo_id, path)
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
                await _upsert_indexed_file(db, repo_full_name, path, new_hash, len(chunks))
            except Exception:
                await _mark_failed(db, repo_full_name, path)


async def _run_review_pr(
    repo_full_name: str, pr_number: int, diff_text: str, installation_id: int
) -> None:
    from scripts.review_pipeline import parse_diff_lines
    import time

    store = _make_store()
    diff_lines = parse_diff_lines(diff_text)
    t0 = time.monotonic()
    chunks = await retrieve(store=store, query=diff_text, top_k=5)
    comments = await generate_review(diff_text=diff_text, chunks=chunks, diff_lines=diff_lines)
    latency_ms = int((time.monotonic() - t0) * 1000)

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


async def _get_hash(db, repo_full_name: str, file_path: str) -> str | None:
    from sqlalchemy import select
    result = await db.execute(
        select(IndexedFile.content_hash).where(
            IndexedFile.file_path == file_path
        )
    )
    return result.scalar_one_or_none()


async def _upsert_indexed_file(db, repo_full_name: str, file_path: str,
                                content_hash: str, chunk_count: int) -> None:
    stmt = pg_insert(IndexedFile).values(
        file_path=file_path,
        content_hash=content_hash,
        status="indexed",
        retry_count=0,
        chunk_count=chunk_count,
        indexed_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        index_elements=["repo_id", "file_path"],
        set_={"content_hash": content_hash, "status": "indexed",
              "chunk_count": chunk_count, "indexed_at": datetime.now(timezone.utc)},
    )
    await db.execute(stmt)
    await db.commit()


async def _mark_failed(db, repo_full_name: str, file_path: str) -> None:
    from sqlalchemy import update
    await db.execute(
        update(IndexedFile)
        .where(IndexedFile.file_path == file_path)
        .values(
            content_hash=None,
            status="failed",
            retry_count=IndexedFile.retry_count + 1,
        )
    )
    # Permanently skip after 3 failures
    await db.execute(
        update(IndexedFile)
        .where(IndexedFile.file_path == file_path, IndexedFile.retry_count >= 3)
        .values(status="failed_permanent")
    )
    await db.commit()


async def _mark_deleted(db, repo_full_name: str, file_path: str) -> None:
    from sqlalchemy import update
    await db.execute(
        update(IndexedFile)
        .where(IndexedFile.file_path == file_path)
        .values(status="deleted", content_hash=None)
    )
    await db.commit()
```

- [ ] **Step 5: Run — verify all pass**

```bash
pytest tests/test_tasks.py -v
```

Expected: 4 `PASSED`

- [ ] **Step 6: Commit**

```bash
git add worker.py indexer/ tests/test_tasks.py
git commit -m "feat: Celery tasks — full_index, incremental_index, review_pr with 3x retry backoff"
```

---

## Task 5: `api/main.py` + `api/dependencies.py` + Routes

**Files:**
- Create: `api/__init__.py`, `api/routes/__init__.py`
- Create: `api/dependencies.py`
- Create: `api/main.py`
- Create: `api/routes/index.py`
- Create: `api/routes/search.py`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_routes.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_post_index_enqueues_full_index():
    with patch("api.routes.index.full_index") as mock_task, \
         patch("api.main.AsyncQdrantClient"), \
         patch("api.main.run_migrations"):
        mock_task.delay.return_value = MagicMock(id="task-1")
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/index", json={"repo_full_name": "owner/repo", "installation_id": 1})
    assert resp.status_code == 200
    assert resp.json()["task_id"] == "task-1"
    mock_task.delay.assert_called_once_with("owner/repo", 1)


@pytest.mark.asyncio
async def test_post_search_returns_chunks():
    mock_hit = MagicMock()
    mock_hit.score = 0.9
    mock_hit.payload = {
        "file_path": "src/auth.py", "name": "verify", "chunk_type": "function",
        "start_line": 1, "end_line": 10, "content": "def verify(): pass"
    }
    with patch("api.routes.search.retrieve", AsyncMock(return_value=[])), \
         patch("api.main.AsyncQdrantClient") as MockQ, \
         patch("api.main.run_migrations"):
        MockQ.return_value.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        MockQ.return_value.create_collection = AsyncMock()
        MockQ.return_value.search = AsyncMock(return_value=[mock_hit])
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/search", json={"query": "token verification", "limit": 5})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_healthz():
    with patch("api.main.AsyncQdrantClient"), patch("api.main.run_migrations"):
        from api.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/healthz")
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_routes.py -v
```

Expected: `ModuleNotFoundError: No module named 'api.main'`

- [ ] **Step 3: Write `api/dependencies.py`**

```python
from typing import AsyncGenerator
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_qdrant(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant
```

- [ ] **Step 4: Write `api/routes/index.py`**

```python
from fastapi import APIRouter
from pydantic import BaseModel
from indexer.tasks import full_index

router = APIRouter()


class IndexRequest(BaseModel):
    repo_full_name: str
    installation_id: int


@router.post("/index")
def post_index(body: IndexRequest):
    task = full_index.delay(body.repo_full_name, body.installation_id)
    return {"task_id": task.id, "status": "queued"}
```

- [ ] **Step 5: Write `api/routes/search.py`**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve, ScoredChunk
from api.dependencies import get_qdrant

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


@router.post("/search")
async def post_search(body: SearchRequest, qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    store = QdrantStore()
    chunks = await retrieve(store=store, query=body.query, top_k=body.limit)
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
```

- [ ] **Step 6: Write `api/main.py`**

```python
from contextlib import asynccontextmanager
from alembic.config import Config as AlembicConfig
from alembic import command as alembic_command
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from config import settings
from api.routes.index import router as index_router
from api.routes.search import router as search_router

QDRANT_COLLECTIONS = ["code_chunks", "adr_docs", "pr_history"]
VECTOR_DIM = 1536


def run_migrations():
    cfg = AlembicConfig("alembic.ini")
    alembic_command.upgrade(cfg, "head")


async def init_qdrant(app: FastAPI):
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    existing = {c.name for c in (await client.get_collections()).collections}
    for name in QDRANT_COLLECTIONS:
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    app.state.qdrant = client


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()
    await init_qdrant(app)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(index_router)
app.include_router(search_router)


@app.get("/healthz")
def health():
    return {"status": "ok"}
```

- [ ] **Step 7: Run — verify all pass**

```bash
pytest tests/test_routes.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 8: Run all tests**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add api/ tests/test_routes.py
git commit -m "feat: FastAPI app — lifespan, POST /index, POST /search, /healthz"
```

---

## Task 6: Acceptance Test

- [ ] **Step 1: Start all services**

```bash
docker compose up --build
```

- [ ] **Step 2: Trigger full index**

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_full_name": "pallets/flask", "installation_id": 0}'
```

Expected: `{"task_id": "...", "status": "queued"}`

Watch worker logs — should see indexing progress.

- [ ] **Step 3: Verify Qdrant has points**

```bash
curl http://localhost:6333/collections/code_chunks
```

Expected: `vectors_count > 0`

- [ ] **Step 4: Verify search returns relevant chunks**

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does Flask handle routing", "limit": 3}'
```

Expected: JSON array with `file_path`, `content`, `score` — content should be relevant Flask routing code.

- [ ] **Step 5: Final commit**

```bash
git commit -m "feat: Phase 2 complete — RAG pipeline wired into FastAPI + Celery + Docker"
```

---

**Phase 2 complete.**

Acceptance criteria met:
- `docker compose up` → all 5 services healthy
- `POST /index` → Celery worker runs `full_index` → Qdrant has points
- `POST /search` returns semantically relevant chunks
- All unit tests pass
