# AI-Powered PR Reviewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GitHub App bot that auto-reviews PRs using RAG — fetching context from indexed code, ADR docs, and PR history — and posts a single atomic inline review with citations.

**Architecture:** Monorepo: FastAPI webhook handler enqueues Celery tasks (never blocks); Celery workers run both the indexing pipeline (tree-sitter → OpenAI embeddings → Qdrant) and the review pipeline (HyDE retrieval → cross-encoder rerank → GPT-4o → GitHub PR review). All infrastructure runs via Docker Compose locally and deploys to Railway.

**Tech Stack:** Python 3.11, FastAPI, Celery + Redis, SQLAlchemy (asyncpg), Qdrant, OpenAI (text-embedding-3-small + GPT-4o), sentence-transformers (cross-encoder), tree-sitter, PyJWT, httpx, unidiff, Langfuse, Docker Compose, Railway.

---

## File Map

```
rag_pr_reviewer/
├── config.py
├── worker.py
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── alembic.ini
├── alembic/
│   └── versions/0001_initial.py
├── api/
│   ├── main.py
│   ├── dependencies.py
│   ├── routes/
│   │   └── webhooks.py
│   └── handlers/
│       ├── indexing.py
│       ├── review.py
│       └── feedback.py
├── db/
│   ├── models.py
│   └── session.py
├── github/
│   ├── auth.py
│   ├── client.py
│   └── webhook.py
├── indexer/
│   ├── chunker.py
│   ├── embedder.py
│   ├── pipeline.py
│   └── tasks.py
├── reviewer/
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── pipeline.py
│   └── tasks.py
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── github/
    │   ├── test_webhook.py
    │   ├── test_auth.py
    │   └── test_client.py
    ├── indexer/
    │   ├── test_chunker.py
    │   ├── test_embedder.py
    │   └── test_pipeline.py
    ├── reviewer/
    │   ├── test_retriever.py
    │   ├── test_reranker.py
    │   ├── test_generator.py
    │   └── test_pipeline.py
    └── api/
        └── test_webhooks.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`, `tests/github/__init__.py`, `tests/indexer/__init__.py`, `tests/reviewer/__init__.py`, `tests/api/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[tool.poetry]
name = "rag-pr-reviewer"
version = "0.1.0"
description = "AI-powered PR reviewer with codebase context"
authors = []
packages = [{include = "."}]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.111.0"
uvicorn = {extras = ["standard"], version = "^0.29.0"}
celery = {extras = ["redis"], version = "^5.3.6"}
redis = "^5.0.4"
sqlalchemy = {extras = ["asyncio"], version = "^2.0.29"}
asyncpg = "^0.29.0"
alembic = "^1.13.1"
pydantic-settings = "^2.2.1"
openai = "^1.25.0"
qdrant-client = {extras = ["async"], version = "^1.9.1"}
PyGithub = "^2.3.0"
unidiff = "^0.7.5"
tree-sitter = "^0.21.3"
tree-sitter-python = "^0.21.0"
sentence-transformers = "^2.7.0"
langfuse = "^2.36.2"
PyJWT = {extras = ["cryptography"], version = "^2.8.0"}
cryptography = "^42.0.5"
httpx = "^0.27.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.1.1"
pytest-asyncio = "^0.23.6"
pytest-mock = "^3.14.0"
httpx = "^0.27.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
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
version: "3.9"

services:
  api:
    build: .
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    env_file: .env
    depends_on: [postgres, redis, qdrant]
    volumes: [.:/app]

  worker:
    build: .
    command: celery -A worker.celery_app worker --loglevel=info --concurrency=4
    env_file: .env
    depends_on: [postgres, redis, qdrant]
    volumes: [.:/app]

  beat:
    build: .
    command: celery -A worker.celery_app beat --loglevel=info
    env_file: .env
    depends_on: [redis]
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

- [ ] **Step 4: Create `.env.example`**

```
GITHUB_APP_ID=
GITHUB_APP_PRIVATE_KEY_B64=
GITHUB_WEBHOOK_SECRET=
OPENAI_API_KEY=
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/rag_pr_reviewer
REDIS_URL=redis://redis:6379/0
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
ADMIN_SECRET=changeme
```

- [ ] **Step 5: Create `tests/conftest.py` and `__init__.py` files**

```python
# tests/conftest.py
import pytest
```

```python
# tests/__init__.py  (and repeat for each subdirectory)
```

Run: `touch tests/__init__.py tests/github/__init__.py tests/indexer/__init__.py tests/reviewer/__init__.py tests/api/__init__.py`

- [ ] **Step 6: Install dependencies**

```bash
poetry install
```

Expected: All packages installed without errors.

- [ ] **Step 7: Commit**

```bash
git init
git add pyproject.toml Dockerfile docker-compose.yml .env.example tests/conftest.py tests/__init__.py tests/github/__init__.py tests/indexer/__init__.py tests/reviewer/__init__.py tests/api/__init__.py
git commit -m "feat: project scaffold with dependencies and docker compose"
```

---

## Task 2: Configuration (`config.py`)

**Files:**
- Create: `config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import base64
import pytest
import os


def test_github_private_key_decoded(monkeypatch):
    raw_pem = b"-----BEGIN RSA PRIVATE KEY-----\nfakekey\n-----END RSA PRIVATE KEY-----"
    b64 = base64.b64encode(raw_pem).decode()

    monkeypatch.setenv("GITHUB_APP_ID", "12345")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_B64", b64)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    # Re-import Settings fresh to pick up env vars
    import importlib
    import config as cfg_module
    importlib.reload(cfg_module)
    from config import Settings
    s = Settings()
    assert s.github_private_key == raw_pem.decode()


def test_missing_required_var_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from pydantic import ValidationError
    from config import Settings
    with pytest.raises(ValidationError):
        Settings()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement `config.py`**

```python
import base64
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    github_app_id: str
    github_app_private_key_b64: str
    github_webhook_secret: str

    openai_api_key: str

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""

    database_url: str
    redis_url: str = "redis://localhost:6379/0"

    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "https://cloud.langfuse.com"

    admin_secret: str = "changeme"

    @property
    def github_private_key(self) -> str:
        return base64.b64decode(self.github_app_private_key_b64).decode("utf-8")


settings = Settings()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_config.py -v
```

Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: pydantic settings with base64 private key decode"
```

---

## Task 3: Database Models (`db/models.py`)

**Files:**
- Create: `db/__init__.py`
- Create: `db/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from db.models import Repo, IndexedFile, PRReview, ReviewFeedback, ReviewStatus, FeedbackEvent
import uuid


def test_repo_model_has_expected_columns():
    r = Repo(github_repo_id=1, full_name="owner/repo", installation_id=99)
    assert r.github_repo_id == 1
    assert r.full_name == "owner/repo"
    assert r.installation_id == 99
    assert r.id is None  # not persisted yet


def test_review_status_values():
    assert ReviewStatus.pending.value == "pending"
    assert ReviewStatus.posted.value == "posted"
    assert ReviewStatus.failed.value == "failed"


def test_feedback_event_values():
    assert FeedbackEvent.dismissed.value == "dismissed"
    assert FeedbackEvent.resolved.value == "resolved"
    assert FeedbackEvent.replied.value == "replied"


def test_pr_review_defaults():
    pr = PRReview(
        repo_id=uuid.uuid4(),
        pr_number=42,
        pr_title="Fix auth",
    )
    assert pr.status == ReviewStatus.pending
    assert pr.github_review_id is None
    assert pr.raw_llm_output is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_models.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Create `db/__init__.py`**

```bash
touch db/__init__.py
```

- [ ] **Step 4: Implement `db/models.py`**

```python
import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import BigInteger, DateTime, Enum, Float, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ReviewStatus(PyEnum):
    pending = "pending"
    posted = "posted"
    failed = "failed"


class FeedbackEvent(PyEnum):
    dismissed = "dismissed"
    resolved = "resolved"
    replied = "replied"


class Repo(Base):
    __tablename__ = "repos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    github_repo_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    installation_id: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    indexed_files: Mapped[list["IndexedFile"]] = relationship(back_populates="repo")
    pr_reviews: Mapped[list["PRReview"]] = relationship(back_populates="repo")


class IndexedFile(Base):
    __tablename__ = "indexed_files"
    __table_args__ = (UniqueConstraint("repo_id", "file_path"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    repo: Mapped["Repo"] = relationship(back_populates="indexed_files")


class PRReview(Base):
    __tablename__ = "pr_reviews"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("repos.id"), nullable=False)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    pr_title: Mapped[str] = mapped_column(Text, nullable=False)
    github_review_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[ReviewStatus] = mapped_column(Enum(ReviewStatus), nullable=False, default=ReviewStatus.pending)
    raw_llm_output: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    langfuse_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    repo: Mapped["Repo"] = relationship(back_populates="pr_reviews")
    feedback: Mapped[list["ReviewFeedback"]] = relationship(back_populates="pr_review")


class ReviewFeedback(Base):
    __tablename__ = "review_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pr_review_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("pr_reviews.id"), nullable=False)
    comment_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event: Mapped[FeedbackEvent] = mapped_column(Enum(FeedbackEvent), nullable=False)
    langfuse_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    pr_review: Mapped["PRReview"] = relationship(back_populates="feedback")
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/test_models.py -v
```

Expected: PASSED

- [ ] **Step 6: Commit**

```bash
git add db/__init__.py db/models.py tests/test_models.py
git commit -m "feat: sqlalchemy models for repos, indexed_files, pr_reviews, review_feedback"
```

---

## Task 4: Database Session + Alembic Migrations

**Files:**
- Create: `db/session.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/versions/0001_initial.py`

- [ ] **Step 1: Create `db/session.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
```

- [ ] **Step 2: Initialize Alembic**

```bash
alembic init alembic
```

- [ ] **Step 3: Replace `alembic/env.py`**

```python
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context
from config import settings
from db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    import asyncio
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Generate initial migration**

```bash
alembic revision --autogenerate -m "initial"
```

Expected: Creates `alembic/versions/<hash>_initial.py` with all four tables.

- [ ] **Step 5: Verify migration looks correct**

Open the generated file and confirm it contains `create_table` calls for `repos`, `indexed_files`, `pr_reviews`, `review_feedback`.

- [ ] **Step 6: Commit**

```bash
git add db/session.py alembic.ini alembic/
git commit -m "feat: async sqlalchemy session and alembic initial migration"
```

---

## Task 5: GitHub Webhook Verification (`github/webhook.py`)

**Files:**
- Create: `github/__init__.py`
- Create: `github/webhook.py`
- Create: `tests/github/test_webhook.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/github/test_webhook.py
import hashlib
import hmac
from github.webhook import verify_signature, parse_event


def _make_signature(payload: bytes, secret: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def test_valid_signature_returns_true():
    payload = b'{"action": "opened"}'
    secret = "mysecret"
    sig = _make_signature(payload, secret)
    assert verify_signature(payload, sig, secret) is True


def test_invalid_signature_returns_false():
    payload = b'{"action": "opened"}'
    assert verify_signature(payload, "sha256=badhash", "mysecret") is False


def test_missing_prefix_returns_false():
    payload = b'{"action": "opened"}'
    assert verify_signature(payload, "badhash", "mysecret") is False


def test_empty_signature_returns_false():
    assert verify_signature(b"payload", "", "secret") is False


def test_parse_event_returns_type_and_action():
    headers = {"x-github-event": "pull_request"}
    body = {"action": "opened"}
    event_type, action = parse_event(headers, body)
    assert event_type == "pull_request"
    assert action == "opened"


def test_parse_event_missing_action():
    headers = {"x-github-event": "push"}
    event_type, action = parse_event(headers, {})
    assert event_type == "push"
    assert action == ""
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/github/test_webhook.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'github.webhook'`

- [ ] **Step 3: Create `github/__init__.py` and implement `github/webhook.py`**

```bash
touch github/__init__.py
```

```python
# github/webhook.py
import hashlib
import hmac


def verify_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, signature_header)


def parse_event(headers: dict, body: dict) -> tuple[str, str]:
    event_type = headers.get("x-github-event", "")
    action = body.get("action", "")
    return event_type, action
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/github/test_webhook.py -v
```

Expected: PASSED (6 tests)

- [ ] **Step 5: Commit**

```bash
git add github/__init__.py github/webhook.py tests/github/test_webhook.py tests/github/__init__.py
git commit -m "feat: github webhook HMAC-SHA256 signature verification"
```

---

## Task 6: GitHub App Auth (`github/auth.py`)

**Files:**
- Create: `github/auth.py`
- Create: `tests/github/test_auth.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/github/test_auth.py
import pytest
import jwt as pyjwt
from unittest.mock import AsyncMock, patch, MagicMock


def test_create_app_jwt_contains_correct_claims(monkeypatch):
    import base64
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    b64 = base64.b64encode(pem).decode()

    monkeypatch.setenv("GITHUB_APP_ID", "99999")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_B64", b64)
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s")
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk2")

    import importlib, config as cfg
    importlib.reload(cfg)
    from config import Settings
    settings = Settings()

    import importlib, github.auth as auth_module
    importlib.reload(auth_module)

    token = auth_module.create_app_jwt()
    pub_key = private_key.public_key()
    claims = pyjwt.decode(token, pub_key, algorithms=["RS256"])
    assert claims["iss"] == "99999"
    assert "iat" in claims
    assert "exp" in claims
    assert claims["exp"] - claims["iat"] < 700


@pytest.mark.asyncio
async def test_get_installation_token_calls_github_api(mocker):
    mock_response = MagicMock()
    mock_response.json.return_value = {"token": "ghs_test_token"}
    mock_response.raise_for_status = MagicMock()

    mock_post = AsyncMock(return_value=mock_response)
    mocker.patch("github.auth.create_app_jwt", return_value="fake.jwt.token")

    import httpx
    mocker.patch.object(httpx.AsyncClient, "post", mock_post)

    from github.auth import get_installation_token
    token = await get_installation_token(12345)
    assert token == "ghs_test_token"
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "12345" in call_args[0][0]
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/github/test_auth.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'github.auth'`

- [ ] **Step 3: Implement `github/auth.py`**

```python
import time
import httpx
import jwt

from config import settings


def create_app_jwt() -> str:
    now = int(time.time())
    payload = {
        "iat": now - 60,
        "exp": now + 540,
        "iss": settings.github_app_id,
    }
    return jwt.encode(payload, settings.github_private_key, algorithm="RS256")


async def get_installation_token(installation_id: int) -> str:
    app_jwt = create_app_jwt()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github.v3+json",
            },
        )
        response.raise_for_status()
        return response.json()["token"]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/github/test_auth.py -v
```

Expected: PASSED

- [ ] **Step 5: Commit**

```bash
git add github/auth.py tests/github/test_auth.py
git commit -m "feat: github app JWT auth and installation token exchange"
```

---

## Task 7: GitHub Client (`github/client.py`)

**Files:**
- Create: `github/client.py`
- Create: `tests/github/test_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/github/test_client.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

SAMPLE_DIFF = """diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,6 +10,10 @@ def login(user):
     if not user:
         return None
+    if user.is_banned:
+        raise PermissionError("banned")
+    token = generate_token(user)
+    return token
     return user
"""


@pytest.mark.asyncio
async def test_fetch_pr_diff_parses_added_lines(mocker):
    mock_response = MagicMock()
    mock_response.text = SAMPLE_DIFF
    mock_response.raise_for_status = MagicMock()

    mocker.patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response))

    from github.client import GitHubClient
    client = GitHubClient("token123", "owner/repo")
    hunks = await client.fetch_pr_diff(42)

    assert len(hunks) > 0
    assert hunks[0].file_path == "src/auth.py"
    assert any(line_no > 0 for line_no, _ in hunks[0].added_lines)


@pytest.mark.asyncio
async def test_post_review_returns_review_id(mocker):
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": 99887766}
    mock_response.raise_for_status = MagicMock()

    mocker.patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=mock_response))

    from github.client import GitHubClient, ReviewComment
    client = GitHubClient("token123", "owner/repo")
    review_id = await client.post_review(
        pr_number=1,
        summary="Looks good overall",
        comments=[ReviewComment(path="src/auth.py", line=13, side="RIGHT", body="Check this")],
    )
    assert review_id == 99887766


@pytest.mark.asyncio
async def test_get_file_content_decodes_base64(mocker):
    import base64
    content = "def foo(): pass\n"
    encoded = base64.b64encode(content.encode()).decode()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"encoding": "base64", "content": encoded}
    mock_response.raise_for_status = MagicMock()

    mocker.patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response))

    from github.client import GitHubClient
    client = GitHubClient("token123", "owner/repo")
    result = await client.get_file_content("src/auth.py")
    assert result == content


@pytest.mark.asyncio
async def test_get_file_content_returns_none_on_404(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 404

    mocker.patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=mock_response))

    from github.client import GitHubClient
    client = GitHubClient("token123", "owner/repo")
    result = await client.get_file_content("nonexistent.py")
    assert result is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/github/test_client.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'github.client'`

- [ ] **Step 3: Implement `github/client.py`**

```python
import base64
from dataclasses import dataclass

import httpx
from unidiff import PatchSet


@dataclass
class DiffHunk:
    file_path: str
    hunk_text: str
    added_lines: list[tuple[int, str]]
    total_changes: int


@dataclass
class ReviewComment:
    path: str
    line: int
    side: str
    body: str


class GitHubClient:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, repo_full_name: str):
        self.token = token
        self.repo_full_name = repo_full_name
        self._headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    async def fetch_pr_diff(self, pr_number: int) -> list[DiffHunk]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/repos/{self.repo_full_name}/pulls/{pr_number}",
                headers={**self._headers, "Accept": "application/vnd.github.v3.diff"},
            )
            response.raise_for_status()
            patch = PatchSet(response.text)

        hunks = []
        for patched_file in patch:
            if patched_file.is_binary_file:
                continue
            for hunk in patched_file:
                added_lines = [
                    (line.target_line_no, line.value)
                    for line in hunk
                    if line.is_added and line.target_line_no is not None
                ]
                if not added_lines:
                    continue
                total_changes = len(added_lines) + sum(1 for line in hunk if line.is_removed)
                hunks.append(DiffHunk(
                    file_path=patched_file.path,
                    hunk_text=str(hunk),
                    added_lines=added_lines,
                    total_changes=total_changes,
                ))
        return hunks

    async def post_review(self, pr_number: int, summary: str, comments: list[ReviewComment]) -> int:
        payload = {
            "body": summary,
            "event": "COMMENT",
            "comments": [
                {"path": c.path, "line": c.line, "side": c.side, "body": c.body}
                for c in comments
            ],
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/repos/{self.repo_full_name}/pulls/{pr_number}/reviews",
                headers=self._headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()["id"]

    async def get_file_content(self, file_path: str, ref: str = "HEAD") -> str | None:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/repos/{self.repo_full_name}/contents/{file_path}",
                headers=self._headers,
                params={"ref": ref},
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return data.get("content", "")

    async def get_repo_tree(self, sha: str = "HEAD") -> list[str]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/repos/{self.repo_full_name}/git/trees/{sha}",
                headers=self._headers,
                params={"recursive": "1"},
            )
            response.raise_for_status()
            tree = response.json().get("tree", [])
            return [item["path"] for item in tree if item["type"] == "blob"]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/github/test_client.py -v
```

Expected: PASSED (4 tests)

- [ ] **Step 5: Commit**

```bash
git add github/client.py tests/github/test_client.py
git commit -m "feat: github client for diff fetching and review posting"
```

---

## Task 8: Tree-sitter Chunker (`indexer/chunker.py`)

**Files:**
- Create: `indexer/__init__.py`
- Create: `indexer/chunker.py`
- Create: `tests/indexer/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/indexer/test_chunker.py
from indexer.chunker import chunk_file, chunk_python_file, CodeChunk

PYTHON_SOURCE = '''
def hello(name: str) -> str:
    """Return greeting."""
    return f"Hello, {name}!"


class Greeter:
    def __init__(self, prefix: str):
        self.prefix = prefix

    def greet(self, name: str) -> str:
        return f"{self.prefix} {name}"
'''


def test_chunk_python_file_extracts_function():
    chunks = chunk_python_file(PYTHON_SOURCE, "greet.py")
    types = [c.chunk_type for c in chunks]
    assert "function" in types


def test_chunk_python_file_extracts_class():
    chunks = chunk_python_file(PYTHON_SOURCE, "greet.py")
    types = [c.chunk_type for c in chunks]
    assert "class" in types


def test_chunks_have_line_numbers():
    chunks = chunk_python_file(PYTHON_SOURCE, "greet.py")
    for chunk in chunks:
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line


def test_chunks_have_content_hash():
    chunks = chunk_python_file(PYTHON_SOURCE, "greet.py")
    for chunk in chunks:
        assert len(chunk.content_hash) == 64  # SHA256 hex


def test_chunk_file_dispatches_python():
    chunks = chunk_file(PYTHON_SOURCE, "src/greet.py")
    assert len(chunks) > 0


def test_small_non_python_file_returns_module_chunk():
    content = "DATABASE_URL=postgres://localhost/db\nDEBUG=true\n"
    chunks = chunk_file(content, "config.env")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"


def test_large_non_python_file_returns_empty():
    content = "x" * 9000
    chunks = chunk_file(content, "bigfile.txt")
    assert chunks == []


def test_empty_python_file_returns_module_chunk():
    chunks = chunk_python_file("", "empty.py")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/indexer/test_chunker.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'indexer'`

- [ ] **Step 3: Create `indexer/__init__.py` and implement `indexer/chunker.py`**

```bash
touch indexer/__init__.py
```

```python
# indexer/chunker.py
import hashlib
from dataclasses import dataclass

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)

CHUNK_NODE_TYPES = {"function_definition", "class_definition"}


@dataclass
class CodeChunk:
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str
    content_hash: str


def _node_type_to_chunk_type(node_type: str) -> str:
    return {"function_definition": "function", "class_definition": "class"}.get(node_type, "module")


def _extract(source: bytes, node: Node, file_path: str) -> list[CodeChunk]:
    if node.type in CHUNK_NODE_TYPES:
        text = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        return [CodeChunk(
            file_path=file_path,
            content=text,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            chunk_type=_node_type_to_chunk_type(node.type),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )]
    chunks = []
    for child in node.children:
        chunks.extend(_extract(source, child, file_path))
    return chunks


def chunk_python_file(content: str, file_path: str) -> list[CodeChunk]:
    source = content.encode("utf-8")
    tree = _parser.parse(source)
    chunks = _extract(source, tree.root_node, file_path)
    if not chunks:
        chunks = [CodeChunk(
            file_path=file_path,
            content=content,
            start_line=1,
            end_line=max(content.count("\n") + 1, 1),
            chunk_type="module",
            content_hash=hashlib.sha256(source).hexdigest(),
        )]
    return chunks


def chunk_file(content: str, file_path: str) -> list[CodeChunk]:
    if file_path.endswith(".py"):
        return chunk_python_file(content, file_path)
    source = content.encode("utf-8")
    if len(source) <= 8192:
        return [CodeChunk(
            file_path=file_path,
            content=content,
            start_line=1,
            end_line=max(content.count("\n") + 1, 1),
            chunk_type="module",
            content_hash=hashlib.sha256(source).hexdigest(),
        )]
    return []
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/indexer/test_chunker.py -v
```

Expected: PASSED (8 tests)

- [ ] **Step 5: Commit**

```bash
git add indexer/__init__.py indexer/chunker.py tests/indexer/__init__.py tests/indexer/test_chunker.py
git commit -m "feat: tree-sitter AST chunker for Python files"
```

---

## Task 9: OpenAI Embedder (`indexer/embedder.py`)

**Files:**
- Create: `indexer/embedder.py`
- Create: `tests/indexer/test_embedder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/indexer/test_embedder.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_embed_texts_returns_correct_count(mocker):
    fake_embedding = [0.1] * 1536
    mock_item = MagicMock()
    mock_item.embedding = fake_embedding
    mock_response = MagicMock()
    mock_response.data = [mock_item, mock_item, mock_item]

    mocker.patch("indexer.embedder._client.embeddings.create", AsyncMock(return_value=mock_response))

    from indexer.embedder import embed_texts
    result = await embed_texts(["text1", "text2", "text3"])
    assert len(result) == 3
    assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_embed_texts_batches_large_input(mocker):
    fake_embedding = [0.0] * 1536
    mock_item = MagicMock()
    mock_item.embedding = fake_embedding

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = MagicMock()
        mock_response.data = [mock_item] * len(kwargs["input"])
        return mock_response

    mocker.patch("indexer.embedder._client.embeddings.create", side_effect=mock_create)

    from indexer.embedder import embed_texts
    texts = [f"text{i}" for i in range(150)]
    result = await embed_texts(texts)
    assert len(result) == 150
    assert call_count == 2  # 100 + 50


@pytest.mark.asyncio
async def test_embed_texts_empty_returns_empty(mocker):
    from indexer.embedder import embed_texts
    result = await embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_text_single(mocker):
    fake_embedding = [0.5] * 1536
    mock_item = MagicMock()
    mock_item.embedding = fake_embedding
    mock_response = MagicMock()
    mock_response.data = [mock_item]

    mocker.patch("indexer.embedder._client.embeddings.create", AsyncMock(return_value=mock_response))

    from indexer.embedder import embed_text
    result = await embed_text("hello world")
    assert len(result) == 1536
    assert result[0] == 0.5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/indexer/test_embedder.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'indexer.embedder'`

- [ ] **Step 3: Implement `indexer/embedder.py`**

```python
from openai import AsyncOpenAI
from config import settings

_client = AsyncOpenAI(api_key=settings.openai_api_key)
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = await _client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


async def embed_text(text: str) -> list[float]:
    return (await embed_texts([text]))[0]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/indexer/test_embedder.py -v
```

Expected: PASSED (4 tests)

- [ ] **Step 5: Commit**

```bash
git add indexer/embedder.py tests/indexer/test_embedder.py
git commit -m "feat: openai text-embedding-3-small wrapper with batching"
```

---

## Task 10: Indexing Pipeline (`indexer/pipeline.py`)

**Files:**
- Create: `indexer/pipeline.py`
- Create: `tests/indexer/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/indexer/test_pipeline.py
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_index_file_skips_unchanged_file(mocker):
    repo_id = uuid.uuid4()
    import hashlib
    content = "def foo(): pass\n"
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    mock_file = MagicMock()
    mock_file.content_hash = content_hash

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_file

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mocker.patch("indexer.pipeline.async_session_factory", return_value=mock_session)
    mocker.patch("indexer.pipeline.get_installation_token", AsyncMock(return_value="tok"))

    mock_gh = AsyncMock()
    mock_gh.get_file_content = AsyncMock(return_value=content)
    mocker.patch("indexer.pipeline.GitHubClient", return_value=mock_gh)

    from indexer.pipeline import index_file
    await index_file(repo_id, "owner/repo", "src/foo.py", 123)

    # embed_texts should not have been called since hash matches
    mock_gh.get_file_content.assert_called_once_with("src/foo.py")


@pytest.mark.asyncio
async def test_ensure_collections_creates_missing(mocker):
    mock_client = AsyncMock()
    mock_collections_response = MagicMock()
    mock_collections_response.collections = []
    mock_client.get_collections = AsyncMock(return_value=mock_collections_response)
    mock_client.create_collection = AsyncMock()

    mocker.patch("indexer.pipeline.get_qdrant_client", return_value=mock_client)

    from indexer.pipeline import ensure_collections
    await ensure_collections()

    assert mock_client.create_collection.call_count == 3  # code_chunks, adr_docs, pr_history
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/indexer/test_pipeline.py -v
```

Expected: FAILED — `ModuleNotFoundError: No module named 'indexer.pipeline'`

- [ ] **Step 3: Implement `indexer/pipeline.py`**

```python
import hashlib
import uuid
from datetime import datetime, timezone

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import settings
from db.models import IndexedFile
from db.session import async_session_factory
from github.auth import get_installation_token
from github.client import GitHubClient
from indexer.chunker import chunk_file
from indexer.embedder import embed_texts

VECTOR_SIZE = 1536
COLLECTIONS = {
    "code_chunks": Distance.COSINE,
    "adr_docs": Distance.COSINE,
    "pr_history": Distance.COSINE,
}


def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


async def ensure_collections() -> None:
    client = get_qdrant_client()
    existing = {c.name for c in (await client.get_collections()).collections}
    for name, distance in COLLECTIONS.items():
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=distance),
            )


async def index_file(
    repo_id: uuid.UUID,
    repo_full_name: str,
    file_path: str,
    installation_id: int,
) -> None:
    installation_token = await get_installation_token(installation_id)
    gh_client = GitHubClient(installation_token, repo_full_name)

    content = await gh_client.get_file_content(file_path)
    if content is None:
        return

    content_hash = hashlib.sha256(content.encode()).hexdigest()

    async with async_session_factory() as session:
        result = await session.execute(
            select(IndexedFile).where(
                IndexedFile.repo_id == repo_id,
                IndexedFile.file_path == file_path,
            )
        )
        existing = result.scalar_one_or_none()
        if existing and existing.content_hash == content_hash:
            return

    chunks = chunk_file(content, file_path)
    if not chunks:
        return

    embeddings = await embed_texts([c.content for c in chunks])

    qdrant = get_qdrant_client()
    await qdrant.delete(
        collection_name="code_chunks",
        points_selector=Filter(must=[
            FieldCondition(key="repo_id", match=MatchValue(value=str(repo_id))),
            FieldCondition(key="file_path", match=MatchValue(value=file_path)),
        ]),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "repo_id": str(repo_id),
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "content_hash": chunk.content_hash,
                "content": chunk.content,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    await qdrant.upsert(collection_name="code_chunks", points=points)

    now = datetime.now(timezone.utc)
    async with async_session_factory() as session:
        stmt = pg_insert(IndexedFile).values(
            repo_id=repo_id,
            file_path=file_path,
            content_hash=content_hash,
            indexed_at=now,
        ).on_conflict_do_update(
            index_elements=["repo_id", "file_path"],
            set_={"content_hash": content_hash, "indexed_at": now},
        )
        await session.execute(stmt)
        await session.commit()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/indexer/test_pipeline.py -v
```

Expected: PASSED (2 tests)

- [ ] **Step 5: Commit**

```bash
git add indexer/pipeline.py tests/indexer/test_pipeline.py
git commit -m "feat: indexing pipeline with Qdrant upsert and hash dedup"
```

---

## Task 11: Celery Worker + Indexer Tasks

**Files:**
- Create: `worker.py`
- Create: `indexer/tasks.py`

- [ ] **Step 1: Create `worker.py`**

```python
from celery import Celery
from config import settings

celery_app = Celery(
    "rag_pr_reviewer",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["indexer.tasks", "reviewer.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={
        "daily-reindex-all": {
            "task": "indexer.tasks.reindex_all_repos",
            "schedule": 86400.0,
        }
    },
)
```

- [ ] **Step 2: Create `indexer/tasks.py`**

```python
import asyncio
import uuid

from celery import shared_task
from sqlalchemy import select

from db.models import Repo
from db.session import async_session_factory
from github.auth import get_installation_token
from github.client import GitHubClient
from indexer.pipeline import index_file
from db.models import IndexedFile
from sqlalchemy.dialects.postgresql import insert as pg_insert

FILE_BATCH_SIZE = 50


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@shared_task(bind=True, max_retries=3)
def incremental_index(self, repo_id: str, changed_files: list[str]):
    try:
        _run(_incremental_index(uuid.UUID(repo_id), changed_files))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@shared_task(bind=True, max_retries=3)
def full_index(self, repo_id: str):
    try:
        _run(_full_index(uuid.UUID(repo_id)))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@shared_task
def reindex_all_repos():
    """Beat-scheduled task: queries all repos and dispatches full_index for each."""
    async def _all():
        async with async_session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(select(Repo))
            repos = result.scalars().all()
        for repo in repos:
            full_index.delay(str(repo.id))
    _run(_all())


async def _get_repo(repo_id: uuid.UUID) -> Repo:
    async with async_session_factory() as session:
        result = await session.execute(select(Repo).where(Repo.id == repo_id))
        return result.scalar_one()


async def _incremental_index(repo_id: uuid.UUID, changed_files: list[str]) -> None:
    repo = await _get_repo(repo_id)
    for file_path in changed_files:
        try:
            await index_file(repo_id, repo.full_name, file_path, repo.installation_id)
        except Exception:
            async with async_session_factory() as session:
                stmt = pg_insert(IndexedFile).values(
                    repo_id=repo_id, file_path=file_path, content_hash=None,
                ).on_conflict_do_update(
                    index_elements=["repo_id", "file_path"],
                    set_={"content_hash": None},
                )
                await session.execute(stmt)
                await session.commit()


async def _full_index(repo_id: uuid.UUID) -> None:
    repo = await _get_repo(repo_id)
    token = await get_installation_token(repo.installation_id)
    client = GitHubClient(token, repo.full_name)
    all_files = await client.get_repo_tree()
    for i in range(0, len(all_files), FILE_BATCH_SIZE):
        for file_path in all_files[i:i + FILE_BATCH_SIZE]:
            try:
                await index_file(repo_id, repo.full_name, file_path, repo.installation_id)
            except Exception:
                pass
```

- [ ] **Step 3: Add `index_pr_history` task to `indexer/tasks.py`**

Append to `indexer/tasks.py`:

```python
@shared_task(bind=True, max_retries=3)
def index_pr_history(self, repo_id: str, pr_number: int):
    """Fetch all review comments from a closed PR and index them into pr_history collection."""
    try:
        _run(_index_pr_history(uuid.UUID(repo_id), pr_number))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


async def _index_pr_history(repo_id: uuid.UUID, pr_number: int) -> None:
    repo = await _get_repo(repo_id)
    token = await get_installation_token(repo.installation_id)
    client = GitHubClient(token, repo.full_name)

    # Fetch review comments via GitHub API
    import httpx
    async with httpx.AsyncClient() as http:
        resp = await http.get(
            f"https://api.github.com/repos/{repo.full_name}/pulls/{pr_number}/reviews",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
        )
        resp.raise_for_status()
        reviews = resp.json()

        comments_resp = await http.get(
            f"https://api.github.com/repos/{repo.full_name}/pulls/{pr_number}/comments",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
        )
        comments_resp.raise_for_status()
        comments = comments_resp.json()

    from indexer.embedder import embed_texts
    from indexer.pipeline import get_qdrant_client
    from qdrant_client.models import PointStruct
    import uuid as _uuid

    texts = [c.get("body", "") for c in comments if c.get("body")]
    if not texts:
        return

    embeddings = await embed_texts(texts)
    qdrant = get_qdrant_client()

    # Determine source: bot (our app) vs human
    import os
    bot_login = os.getenv("GITHUB_BOT_LOGIN", "github-actions[bot]")
    points = []
    for comment, embedding in zip(comments, embeddings):
        source = "bot" if comment.get("user", {}).get("login") == bot_login else "human"
        points.append(PointStruct(
            id=str(_uuid.uuid4()),
            vector=embedding,
            payload={
                "repo_id": str(repo_id),
                "pr_number": pr_number,
                "comment_body": comment.get("body", ""),
                "diff_hunk": comment.get("diff_hunk", ""),
                "source": source,
                "file_path": comment.get("path", ""),
                "line": comment.get("line") or comment.get("original_line"),
            },
        ))

    if points:
        await qdrant.upsert(collection_name="pr_history", points=points)
```

Also add `GITHUB_BOT_LOGIN=` to `.env.example` (the GitHub App's bot username, e.g. `mybot[bot]`).

- [ ] **Step 4: Verify tasks import cleanly**

```bash
python -c "from indexer.tasks import incremental_index, full_index, index_pr_history; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add worker.py indexer/tasks.py
git commit -m "feat: celery worker, indexer tasks including pr_history indexing on PR close"
```

---

## Task 12: Retriever (`reviewer/retriever.py`)

**Files:**
- Create: `reviewer/__init__.py`
- Create: `reviewer/retriever.py`
- Create: `tests/reviewer/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reviewer/test_retriever.py
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_retrieve_returns_deduplicated_chunks(mocker):
    repo_id = uuid.uuid4()

    mocker.patch("reviewer.retriever._hyde_expand", AsyncMock(return_value="hypothetical code"))
    mocker.patch("reviewer.retriever.embed_text", AsyncMock(return_value=[0.1] * 1536))

    def make_hit(content_hash, content, collection):
        hit = MagicMock()
        hit.payload = {
            "content": content,
            "file_path": "src/foo.py",
            "start_line": 1,
            "end_line": 10,
            "content_hash": content_hash,
        }
        hit.score = 0.9
        hit.id = content_hash
        return hit

    code_hit = make_hit("hash1", "def foo(): pass", "code_chunks")
    adr_hit = make_hit("hash2", "ADR-001 content", "adr_docs")
    history_hit = make_hit("hash1", "def foo(): pass", "pr_history")  # duplicate hash

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=[
        [code_hit],   # code_chunks
        [adr_hit],    # adr_docs
        [history_hit], # pr_history
    ])

    mocker.patch("reviewer.retriever.get_qdrant_client", return_value=mock_client)

    from reviewer.retriever import retrieve
    results = await retrieve(repo_id, "diff hunk text")

    assert len(results) == 2  # hash1 deduplicated
    hashes = {r.content_hash for r in results}
    assert "hash1" in hashes
    assert "hash2" in hashes


@pytest.mark.asyncio
async def test_retrieve_labels_source_collection(mocker):
    repo_id = uuid.uuid4()

    mocker.patch("reviewer.retriever._hyde_expand", AsyncMock(return_value="hypo"))
    mocker.patch("reviewer.retriever.embed_text", AsyncMock(return_value=[0.1] * 1536))

    hit = MagicMock()
    hit.payload = {"content": "x", "file_path": "f.py", "start_line": 1, "end_line": 5, "content_hash": "abc"}
    hit.score = 0.8
    hit.id = "abc"

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(side_effect=[[hit], [], []])

    mocker.patch("reviewer.retriever.get_qdrant_client", return_value=mock_client)

    from reviewer.retriever import retrieve
    results = await retrieve(repo_id, "hunk")
    assert results[0].source_collection == "code_chunks"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/reviewer/test_retriever.py -v
```

Expected: FAILED

- [ ] **Step 3: Create `reviewer/__init__.py` and implement `reviewer/retriever.py`**

```bash
touch reviewer/__init__.py
```

```python
# reviewer/retriever.py
import asyncio
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import settings
from indexer.embedder import embed_text
from indexer.pipeline import get_qdrant_client

_openai = AsyncOpenAI(api_key=settings.openai_api_key)


@dataclass
class RetrievedChunk:
    content: str
    source_collection: str
    file_path: str
    start_line: int | None
    end_line: int | None
    content_hash: str
    score: float


async def _hyde_expand(diff_hunk: str) -> str:
    response = await _openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a diff hunk, write a hypothetical code snippet that would be "
                    "most relevant context for reviewing this change. Return only the code, no explanation."
                ),
            },
            {"role": "user", "content": f"Diff hunk:\n{diff_hunk}"},
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content or diff_hunk


async def _search_collection(
    client,
    collection_name: str,
    vector: list[float],
    repo_id: str,
    limit: int,
) -> list[RetrievedChunk]:
    results = await client.search(
        collection_name=collection_name,
        query_vector=vector,
        query_filter=Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]),
        limit=limit,
        with_payload=True,
    )
    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append(RetrievedChunk(
            content=payload.get("content") or payload.get("comment_body", ""),
            source_collection=collection_name,
            file_path=payload.get("file_path", ""),
            start_line=payload.get("start_line"),
            end_line=payload.get("end_line"),
            content_hash=payload.get("content_hash", str(hit.id)),
            score=hit.score,
        ))
    return chunks


async def retrieve(
    repo_id: uuid.UUID,
    diff_hunk: str,
    code_limit: int = 5,
    adr_limit: int = 3,
    history_limit: int = 3,
) -> list[RetrievedChunk]:
    hyde_text = await _hyde_expand(diff_hunk)
    vector = await embed_text(hyde_text)

    client = get_qdrant_client()
    repo_id_str = str(repo_id)

    batches = await asyncio.gather(
        _search_collection(client, "code_chunks", vector, repo_id_str, code_limit),
        _search_collection(client, "adr_docs", vector, repo_id_str, adr_limit),
        _search_collection(client, "pr_history", vector, repo_id_str, history_limit),
    )

    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for batch in batches:
        for chunk in batch:
            if chunk.content_hash not in seen:
                seen.add(chunk.content_hash)
                merged.append(chunk)
    return merged
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/reviewer/test_retriever.py -v
```

Expected: PASSED (2 tests)

- [ ] **Step 5: Commit**

```bash
git add reviewer/__init__.py reviewer/retriever.py tests/reviewer/__init__.py tests/reviewer/test_retriever.py
git commit -m "feat: multi-collection retriever with HyDE query expansion and deduplication"
```

---

## Task 13: Reranker (`reviewer/reranker.py`)

**Files:**
- Create: `reviewer/reranker.py`
- Create: `tests/reviewer/test_reranker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reviewer/test_reranker.py
from reviewer.retriever import RetrievedChunk
from reviewer.reranker import rerank


def _make_chunk(content: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        source_collection="code_chunks",
        file_path="foo.py",
        start_line=1,
        end_line=5,
        content_hash=content[:8],
        score=score,
    )


def test_rerank_returns_top_k(mocker):
    chunks = [_make_chunk(f"content {i}") for i in range(15)]
    mocker.patch(
        "reviewer.reranker._model.predict",
        return_value=list(range(15, 0, -1)),
    )
    result = rerank("query", chunks, top_k=8)
    assert len(result) == 8


def test_rerank_orders_by_score(mocker):
    chunks = [_make_chunk("low relevance"), _make_chunk("high relevance")]
    mocker.patch("reviewer.reranker._model.predict", return_value=[0.2, 0.9])
    result = rerank("find auth code", chunks, top_k=2)
    assert result[0].content == "high relevance"


def test_rerank_empty_input_returns_empty():
    result = rerank("query", [], top_k=8)
    assert result == []


def test_rerank_fewer_chunks_than_top_k(mocker):
    chunks = [_make_chunk("only one")]
    mocker.patch("reviewer.reranker._model.predict", return_value=[0.7])
    result = rerank("query", chunks, top_k=8)
    assert len(result) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/reviewer/test_reranker.py -v
```

Expected: FAILED

- [ ] **Step 3: Implement `reviewer/reranker.py`**

```python
from sentence_transformers import CrossEncoder
from reviewer.retriever import RetrievedChunk

_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
    if not chunks:
        return []
    pairs = [(query, chunk.content) for chunk in chunks]
    scores = _model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/reviewer/test_reranker.py -v
```

Expected: PASSED (4 tests)

- [ ] **Step 5: Commit**

```bash
git add reviewer/reranker.py tests/reviewer/test_reranker.py
git commit -m "feat: cross-encoder reranker using ms-marco-MiniLM-L-6-v2"
```

---

## Task 14: Generator (`reviewer/generator.py`)

**Files:**
- Create: `reviewer/generator.py`
- Create: `tests/reviewer/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reviewer/test_generator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from reviewer.retriever import RetrievedChunk


def _make_chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        source_collection="code_chunks",
        file_path="src/auth.py",
        start_line=1,
        end_line=10,
        content_hash="abc123",
        score=0.9,
    )


@pytest.mark.asyncio
async def test_generate_review_returns_structured_output(mocker):
    mock_output = {
        "summary": "Looks mostly good, one issue found.",
        "comments": [
            {
                "path": "src/auth.py",
                "line": 42,
                "side": "RIGHT",
                "body": "Missing null check here.",
                "severity": "warning",
                "citations": ["code_chunks:src/utils.py:10-25"],
            }
        ],
    }
    mock_message = MagicMock()
    mock_message.content = json.dumps(mock_output)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mocker.patch("reviewer.generator._client.chat.completions.create", AsyncMock(return_value=mock_response))

    from reviewer.generator import generate_review
    result = await generate_review("diff text", [_make_chunk("some context")])
    assert result.summary == "Looks mostly good, one issue found."
    assert len(result.comments) == 1
    assert result.comments[0].path == "src/auth.py"
    assert result.comments[0].line == 42
    assert result.comments[0].severity == "warning"


@pytest.mark.asyncio
async def test_generate_review_retries_on_invalid_json(mocker):
    good_output = json.dumps({"summary": "ok", "comments": []})

    call_count = 0
    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock_message = MagicMock()
        mock_message.content = "not json" if call_count == 1 else good_output
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    mocker.patch("reviewer.generator._client.chat.completions.create", side_effect=mock_create)

    from reviewer.generator import generate_review
    result = await generate_review("diff", [_make_chunk("ctx")])
    assert result.summary == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_review_empty_comments_on_no_issues(mocker):
    mock_output = {"summary": "All good!", "comments": []}
    mock_message = MagicMock()
    mock_message.content = json.dumps(mock_output)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mocker.patch("reviewer.generator._client.chat.completions.create", AsyncMock(return_value=mock_response))

    from reviewer.generator import generate_review
    result = await generate_review("diff", [])
    assert result.comments == []
    assert result.summary == "All good!"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/reviewer/test_generator.py -v
```

Expected: FAILED

- [ ] **Step 3: Implement `reviewer/generator.py`**

```python
import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config import settings
from reviewer.retriever import RetrievedChunk

_client = AsyncOpenAI(api_key=settings.openai_api_key)


@dataclass
class ReviewComment:
    path: str
    line: int
    side: str
    body: str
    severity: str
    citations: list[str] = field(default_factory=list)


@dataclass
class ReviewOutput:
    summary: str
    comments: list[ReviewComment]


SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
Review ONLY based on the retrieved context provided. Do not invent issues without context support.
Return a JSON object:
{
  "summary": "2-3 sentence overall summary",
  "comments": [
    {
      "path": "file path",
      "line": <integer line number in new file>,
      "side": "RIGHT",
      "body": "detailed comment",
      "severity": "error|warning|suggestion",
      "citations": ["collection:file:lines"]
    }
  ]
}
If no issues, return empty comments array with positive summary."""

STRICT_SUFFIX = "\nIMPORTANT: Return ONLY valid JSON. No markdown fences. No extra text."


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = f"{chunk.source_collection}:{chunk.file_path}"
        if chunk.start_line and chunk.end_line:
            ref += f":{chunk.start_line}-{chunk.end_line}"
        parts.append(f"[{i}] {ref}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


async def generate_review(diff_text: str, context_chunks: list[RetrievedChunk]) -> ReviewOutput:
    user_msg = f"## Diff\n\n{diff_text}\n\n## Retrieved Context\n\n{_build_context(context_chunks)}"

    for attempt in range(2):
        system = SYSTEM_PROMPT + (STRICT_SUFFIX if attempt == 1 else "")
        response = await _client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2000,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            comments = [
                ReviewComment(
                    path=c["path"],
                    line=int(c["line"]),
                    side=c.get("side", "RIGHT"),
                    body=c["body"],
                    severity=c.get("severity", "suggestion"),
                    citations=c.get("citations", []),
                )
                for c in data.get("comments", [])
            ]
            return ReviewOutput(summary=data.get("summary", ""), comments=comments)
        except (json.JSONDecodeError, KeyError, TypeError):
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/reviewer/test_generator.py -v
```

Expected: PASSED (3 tests)

- [ ] **Step 5: Commit**

```bash
git add reviewer/generator.py tests/reviewer/test_generator.py
git commit -m "feat: GPT-4o structured review generator with JSON retry"
```

---

## Task 15: Review Pipeline + Task (`reviewer/pipeline.py`, `reviewer/tasks.py`)

**Files:**
- Create: `reviewer/pipeline.py`
- Create: `reviewer/tasks.py`
- Create: `tests/reviewer/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reviewer/test_pipeline.py
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_run_review_posts_review_on_success(mocker):
    repo_id = uuid.uuid4()

    mock_repo = MagicMock()
    mock_repo.full_name = "owner/repo"
    mock_repo.installation_id = 123

    mock_result = MagicMock()
    mock_result.scalar_one = MagicMock(return_value=mock_repo)

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mocker.patch("reviewer.pipeline.async_session_factory", return_value=mock_session)
    mocker.patch("reviewer.pipeline.get_installation_token", AsyncMock(return_value="tok"))

    from github.client import DiffHunk
    mock_hunks = [DiffHunk(file_path="src/auth.py", hunk_text="@@ -1 +1 @@\n+code", added_lines=[(1, "code")], total_changes=1)]
    mock_gh = AsyncMock()
    mock_gh.fetch_pr_diff = AsyncMock(return_value=mock_hunks)
    mock_gh.post_review = AsyncMock(return_value=88776655)
    mocker.patch("reviewer.pipeline.GitHubClient", return_value=mock_gh)

    from reviewer.retriever import RetrievedChunk
    mock_chunk = RetrievedChunk("code", "code_chunks", "src/auth.py", 1, 5, "hash1", 0.9)
    mocker.patch("reviewer.pipeline.retrieve", AsyncMock(return_value=[mock_chunk]))
    mocker.patch("reviewer.pipeline.rerank", return_value=[mock_chunk])

    from reviewer.generator import ReviewOutput, ReviewComment
    mock_output = ReviewOutput(
        summary="Good PR",
        comments=[ReviewComment(path="src/auth.py", line=1, side="RIGHT", body="Nice", severity="suggestion")],
    )
    mocker.patch("reviewer.pipeline.generate_review", AsyncMock(return_value=mock_output))

    mock_trace = MagicMock()
    mock_span = MagicMock()
    mock_span.end = MagicMock()
    mock_trace.span = MagicMock(return_value=mock_span)
    mock_trace.id = "trace-abc"
    mocker.patch("reviewer.pipeline._langfuse.trace", return_value=mock_trace)

    from reviewer.pipeline import run_review
    await run_review(repo_id, 42)

    mock_gh.post_review.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/reviewer/test_pipeline.py -v
```

Expected: FAILED

- [ ] **Step 3: Implement `reviewer/pipeline.py`**

```python
import uuid
from datetime import datetime, timezone

from langfuse import Langfuse
from sqlalchemy import select

from config import settings
from db.models import PRReview, Repo, ReviewStatus
from db.session import async_session_factory
from github.auth import get_installation_token
from github.client import GitHubClient, ReviewComment as GHReviewComment
from reviewer.generator import generate_review
from reviewer.reranker import rerank
from reviewer.retriever import retrieve

_langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

MAX_FILES_FOR_RETRIEVAL = 10


async def run_review(repo_id: uuid.UUID, pr_number: int) -> None:
    async with async_session_factory() as session:
        result = await session.execute(select(Repo).where(Repo.id == repo_id))
        repo = result.scalar_one()

    trace = _langfuse.trace(
        name="review_pr",
        metadata={"repo": repo.full_name, "pr": pr_number},
    )
    pr_review_id = uuid.uuid4()

    async with async_session_factory() as session:
        pr_review = PRReview(
            id=pr_review_id,
            repo_id=repo_id,
            pr_number=pr_number,
            pr_title="",
            status=ReviewStatus.pending,
            langfuse_trace_id=trace.id,
        )
        session.add(pr_review)
        await session.commit()

    installation_token = await get_installation_token(repo.installation_id)
    gh_client = GitHubClient(installation_token, repo.full_name)

    retrieval_span = trace.span(name="retrieval", start_time=datetime.now(timezone.utc))
    diff_hunks = await gh_client.fetch_pr_diff(pr_number)

    top_hunks = sorted(diff_hunks, key=lambda h: h.total_changes, reverse=True)[:MAX_FILES_FOR_RETRIEVAL]

    all_chunks = []
    for hunk in top_hunks:
        chunks = await retrieve(repo_id, hunk.hunk_text)
        all_chunks.extend(rerank(hunk.hunk_text, chunks, top_k=8))

    retrieval_span.end(metadata={"chunks_retrieved": len(all_chunks)})

    full_diff = "\n".join(h.hunk_text for h in diff_hunks)

    gen_span = trace.span(name="generation", start_time=datetime.now(timezone.utc))
    review_output = await generate_review(full_diff, all_chunks)
    gen_span.end()

    gh_comments = []
    for c in review_output.comments:
        body = f"**[{c.severity.upper()}]** {c.body}"
        if c.citations:
            body += f"\n\n*Citations: {', '.join(c.citations)}*"
        gh_comments.append(GHReviewComment(path=c.path, line=c.line, side=c.side, body=body))

    try:
        github_review_id = await gh_client.post_review(pr_number, review_output.summary, gh_comments)
        status = ReviewStatus.posted
    except Exception:
        github_review_id = None
        status = ReviewStatus.failed

    raw_output = {
        "summary": review_output.summary,
        "comments": [
            {"path": c.path, "line": c.line, "body": c.body, "severity": c.severity, "citations": c.citations}
            for c in review_output.comments
        ],
    }

    async with async_session_factory() as session:
        result = await session.execute(select(PRReview).where(PRReview.id == pr_review_id))
        saved = result.scalar_one()
        saved.status = status
        saved.github_review_id = github_review_id
        saved.raw_llm_output = raw_output
        await session.commit()
```

- [ ] **Step 4: Create `reviewer/tasks.py`**

```python
import asyncio
import uuid

from celery import shared_task

from reviewer.pipeline import run_review


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@shared_task(bind=True, max_retries=3)
def review_pr(self, repo_id: str, pr_number: int):
    try:
        _run(run_review(uuid.UUID(repo_id), pr_number))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/reviewer/test_pipeline.py -v
```

Expected: PASSED

- [ ] **Step 6: Commit**

```bash
git add reviewer/pipeline.py reviewer/tasks.py tests/reviewer/test_pipeline.py
git commit -m "feat: review pipeline with Langfuse tracing and Celery task"
```

---

## Task 16: API Handlers

**Files:**
- Create: `api/__init__.py`, `api/routes/__init__.py`, `api/handlers/__init__.py`
- Create: `api/dependencies.py`
- Create: `api/handlers/indexing.py`
- Create: `api/handlers/review.py`
- Create: `api/handlers/feedback.py`

- [ ] **Step 1: Create `__init__.py` files and `api/dependencies.py`**

```bash
touch api/__init__.py api/routes/__init__.py api/handlers/__init__.py
```

```python
# api/dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from db.session import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
```

- [ ] **Step 2: Implement `api/handlers/indexing.py`**

```python
from sqlalchemy import select
from db.models import Repo
from db.session import async_session_factory
from indexer.tasks import incremental_index


async def handle_push(body: dict) -> None:
    github_repo_id = body.get("repository", {}).get("id")
    if not github_repo_id:
        return

    async with async_session_factory() as session:
        result = await session.execute(select(Repo).where(Repo.github_repo_id == github_repo_id))
        repo = result.scalar_one_or_none()
        if not repo:
            return

    changed_files: set[str] = set()
    for commit in body.get("commits", []):
        changed_files.update(commit.get("added", []))
        changed_files.update(commit.get("modified", []))

    if changed_files:
        incremental_index.delay(str(repo.id), list(changed_files))
```

- [ ] **Step 3: Implement `api/handlers/review.py`**

```python
import uuid
from sqlalchemy import select
from db.models import Repo
from db.session import async_session_factory
from reviewer.tasks import review_pr
from indexer.tasks import index_pr_history


async def handle_pr_opened(body: dict) -> None:
    github_repo_id = body.get("repository", {}).get("id")
    pr_number = body.get("pull_request", {}).get("number")
    if not github_repo_id or not pr_number:
        return

    async with async_session_factory() as session:
        result = await session.execute(select(Repo).where(Repo.github_repo_id == github_repo_id))
        repo = result.scalar_one_or_none()
        if not repo:
            return

    review_pr.delay(str(repo.id), pr_number)


async def handle_pr_closed(body: dict) -> None:
    """Index review comments from the closed PR into the pr_history collection."""
    github_repo_id = body.get("repository", {}).get("id")
    pr_number = body.get("pull_request", {}).get("number")
    if not github_repo_id or not pr_number:
        return

    async with async_session_factory() as session:
        result = await session.execute(select(Repo).where(Repo.github_repo_id == github_repo_id))
        repo = result.scalar_one_or_none()
        if not repo:
            return

    index_pr_history.delay(str(repo.id), pr_number)
```

- [ ] **Step 4: Implement `api/handlers/feedback.py`**

```python
from langfuse import Langfuse
from sqlalchemy import select

from config import settings
from db.models import FeedbackEvent, PRReview, ReviewFeedback
from db.session import async_session_factory

_langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

_EVENT_MAP = {
    "dismissed": FeedbackEvent.dismissed,
    "resolved": FeedbackEvent.resolved,
    "created": FeedbackEvent.replied,
}
_SCORE_MAP = {"dismissed": -1.0, "resolved": 1.0, "created": 0.0}


async def handle_review_comment(body: dict) -> None:
    action = body.get("action", "")
    if action not in _EVENT_MAP:
        return

    comment = body.get("comment", {})
    comment_id = comment.get("id")
    pull_request_review_id = comment.get("pull_request_review_id")
    if not comment_id or not pull_request_review_id:
        return

    async with async_session_factory() as session:
        result = await session.execute(
            select(PRReview).where(PRReview.github_review_id == pull_request_review_id)
        )
        pr_review = result.scalar_one_or_none()
        if not pr_review:
            return

        langfuse_score = _SCORE_MAP[action]
        feedback = ReviewFeedback(
            pr_review_id=pr_review.id,
            comment_id=comment_id,
            event=_EVENT_MAP[action],
            langfuse_score=langfuse_score,
        )
        session.add(feedback)
        await session.commit()

        if pr_review.langfuse_trace_id:
            _langfuse.score(
                trace_id=pr_review.langfuse_trace_id,
                name="comment_quality",
                value=langfuse_score,
            )
```

- [ ] **Step 5: Commit**

```bash
git add api/__init__.py api/routes/__init__.py api/handlers/__init__.py api/dependencies.py api/handlers/indexing.py api/handlers/review.py api/handlers/feedback.py
git commit -m "feat: api handlers for push indexing, PR review, and feedback scoring"
```

---

## Task 17: Webhook Route + FastAPI App

**Files:**
- Create: `api/routes/webhooks.py`
- Create: `api/main.py`
- Create: `tests/api/test_webhooks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_webhooks.py
import hashlib
import hmac
import json
import pytest
from httpx import AsyncClient, ASGITransport


def _sign(payload: bytes, secret: str) -> str:
    mac = hmac.new(secret.encode(), payload, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_signature(mocker):
    mocker.patch("config.settings.github_webhook_secret", "real-secret")
    mocker.patch("indexer.pipeline.ensure_collections", new=AsyncMock(return_value=None))
    mocker.patch("db.session.engine.begin")  # prevent real DB connection

    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/webhook/github",
            content=b'{"action":"opened"}',
            headers={
                "x-hub-signature-256": "sha256=badsig",
                "x-github-event": "pull_request",
                "content-type": "application/json",
            },
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_webhook_accepts_valid_push_event(mocker):
    from unittest.mock import AsyncMock
    secret = "test-secret"
    mocker.patch("config.settings.github_webhook_secret", secret)
    mocker.patch("indexer.pipeline.ensure_collections", new=AsyncMock(return_value=None))
    mocker.patch("api.handlers.indexing.handle_push", new=AsyncMock(return_value=None))

    payload = json.dumps({"repository": {"id": 1}, "commits": []}).encode()
    sig = _sign(payload, secret)

    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/webhook/github",
            content=payload,
            headers={
                "x-hub-signature-256": sig,
                "x-github-event": "push",
                "content-type": "application/json",
            },
        )
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(mocker):
    from unittest.mock import AsyncMock
    mocker.patch("indexer.pipeline.ensure_collections", new=AsyncMock(return_value=None))
    from api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/api/test_webhooks.py -v
```

Expected: FAILED

- [ ] **Step 3: Implement `api/routes/webhooks.py`**

```python
from fastapi import APIRouter, Header, HTTPException, Request, status

from api.handlers import feedback, indexing, review
from config import settings
from github.webhook import verify_signature

router = APIRouter()


@router.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED)
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None),
):
    payload = await request.body()

    if not verify_signature(payload, x_hub_signature_256 or "", settings.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    body = await request.json()
    action = body.get("action", "")

    if x_github_event == "push":
        await indexing.handle_push(body)
    elif x_github_event == "pull_request" and action == "opened":
        await review.handle_pr_opened(body)
    elif x_github_event == "pull_request" and action == "closed":
        await review.handle_pr_closed(body)
    elif x_github_event == "pull_request_review_comment":
        await feedback.handle_review_comment(body)

    return {"status": "accepted"}
```

- [ ] **Step 4: Implement `api/main.py`**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, status

from api.routes.webhooks import router as webhook_router
from config import settings
from db.models import Base
from db.session import engine
from indexer.pipeline import ensure_collections
from indexer.tasks import full_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await ensure_collections()
    yield


app = FastAPI(title="RAG PR Reviewer", lifespan=lifespan)
app.include_router(webhook_router)


@app.post("/admin/full-index", status_code=status.HTTP_202_ACCEPTED)
async def trigger_full_index(repo_id: str, x_admin_secret: str = Header(None)):
    if x_admin_secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    full_index.delay(repo_id)
    return {"status": "queued"}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/api/test_webhooks.py -v
```

Expected: PASSED (3 tests)

- [ ] **Step 6: Run full test suite**

```bash
pytest -v
```

Expected: All tests PASSED

- [ ] **Step 7: Commit**

```bash
git add api/routes/webhooks.py api/main.py tests/api/__init__.py tests/api/test_webhooks.py
git commit -m "feat: fastapi webhook endpoint, admin route, and health check"
```

---

## Task 18: End-to-End Smoke Test

**Files:**
- No new files — validates the running system

- [ ] **Step 1: Copy `.env.example` to `.env` and fill in test values**

```bash
cp .env.example .env
```

Edit `.env`: at minimum set `OPENAI_API_KEY`, `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY_B64`, `GITHUB_WEBHOOK_SECRET`. For a smoke test, set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to any placeholder values.

- [ ] **Step 2: Start Docker Compose**

```bash
docker compose up --build -d
```

Expected: All 6 containers start (api, worker, beat, redis, qdrant, postgres).

- [ ] **Step 3: Wait for API to be ready**

```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 4: Verify Qdrant collections were created**

```bash
curl -s http://localhost:6333/collections | python3 -m json.tool
```

Expected: Response contains `code_chunks`, `adr_docs`, `pr_history` in `result.collections`.

- [ ] **Step 5: Register a test repo via psql**

```bash
docker compose exec postgres psql -U postgres -d rag_pr_reviewer -c \
  "INSERT INTO repos (id, github_repo_id, full_name, installation_id) \
   VALUES (gen_random_uuid(), 12345, 'owner/testrepo', 99999);"
```

Expected: `INSERT 1 0`

- [ ] **Step 6: Send a simulated push webhook and verify task enqueues**

```bash
REPO_ID=$(docker compose exec postgres psql -U postgres -d rag_pr_reviewer -Atc \
  "SELECT id FROM repos WHERE github_repo_id=12345;")

SECRET="changeme"  # matches ADMIN_SECRET / GITHUB_WEBHOOK_SECRET in .env
PAYLOAD='{"repository":{"id":12345},"commits":[{"added":[],"modified":["src/main.py"]}]}'
SIG="sha256=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" | awk '{print $2}')"

curl -s -X POST http://localhost:8000/webhook/github \
  -H "Content-Type: application/json" \
  -H "X-Hub-Signature-256: $SIG" \
  -H "X-GitHub-Event: push" \
  -d "$PAYLOAD"
```

Expected: `{"status":"accepted"}`

- [ ] **Step 7: Verify worker logs show task received**

```bash
docker compose logs worker --tail=20
```

Expected: Log line containing `incremental_index` task received.

- [ ] **Step 8: Commit final state**

```bash
git add .
git commit -m "feat: complete RAG PR reviewer implementation"
```

---

## Running All Tests

```bash
pytest -v --tb=short
```

Expected: All tests pass. Coverage across config, models, webhook verification, auth, client, chunker, embedder, indexing pipeline, retriever, reranker, generator, review pipeline, and API endpoints.
