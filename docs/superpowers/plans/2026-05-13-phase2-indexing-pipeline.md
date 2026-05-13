# Phase 2: Indexing Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 1 complete — all 6 Docker services running, Qdrant collections exist, `POST /index` enqueues Celery tasks.

**Goal:** Build the full indexing pipeline — tree-sitter AST chunks Python files, embedder vectorises them, pipeline orchestrates clone → chunk → embed → upsert to Qdrant with incremental re-index logic. `POST /index` triggers the real pipeline.

**Architecture:** `chunker.py` parses Python AST into `Chunk` dataclasses. `indexer/pipeline.py` orchestrates all steps per file. Celery tasks `full_index` and `incremental_index` call the pipeline in sub-batches. `POST /search` is added to verify retrieval works.

**Tech Stack:** tree-sitter 0.21.x + tree-sitter-python, OpenAI text-embedding-3-small, Qdrant AsyncClient, SQLAlchemy async upsert, hashlib SHA256, subprocess git clone

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `indexer/chunker.py` | Create | `Chunk` dataclass + `chunk_file()` via tree-sitter |
| `indexer/pipeline.py` | Create | `index_file()`, `run_full_index()`, `run_incremental_index()` |
| `indexer/tasks.py` | Replace | Real `full_index` + `incremental_index` Celery tasks |
| `api/routes/search.py` | Create | `POST /search` — embed query → Qdrant search |
| `api/main.py` | Modify | Register search router |
| `tests/test_chunker.py` | Create | AST chunking, fallback, skip |
| `tests/test_pipeline.py` | Create | Pipeline orchestration (all services mocked) |
| `tests/test_routes_search.py` | Create | POST /search endpoint |

---

## Task 1: `indexer/chunker.py`

**Files:**
- Create: `indexer/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker.py
import pytest
from indexer.chunker import chunk_file, Chunk

SIMPLE_PYTHON = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    """A simple calculator."""

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b
'''


def test_extracts_module_level_functions():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    fn_names = {c.name for c in chunks if c.chunk_type == "function"}
    assert fn_names == {"add", "subtract"}


def test_extracts_class_header_not_body():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert class_chunks[0].name == "Calculator"
    assert "multiply" not in class_chunks[0].content


def test_extracts_methods():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    method_names = {c.name for c in chunks if c.chunk_type == "method"}
    assert method_names == {"multiply", "divide"}


def test_chunk_has_line_numbers():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_has_file_path():
    chunks = chunk_file("src/math.py", SIMPLE_PYTHON)
    assert all(c.file_path == "src/math.py" for c in chunks)


def test_small_unparseable_file_returns_whole_file_chunk():
    bad_python = "this is not python syntax !!!"
    chunks = chunk_file("bad.py", bad_python)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"
    assert chunks[0].content == bad_python


def test_large_unparseable_file_returns_empty():
    bad_python = "not python " * 1000  # > 8KB
    chunks = chunk_file("big_bad.py", bad_python)
    assert chunks == []


def test_empty_file_returns_empty():
    chunks = chunk_file("empty.py", "")
    assert chunks == []
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'indexer.chunker'`

- [ ] **Step 3: Write `indexer/chunker.py`**

```python
from dataclasses import dataclass
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language(), "python")
FILE_FALLBACK_MAX_BYTES = 8192


@dataclass
class Chunk:
    file_path: str
    name: str
    chunk_type: str  # "function" | "class" | "method" | "module"
    start_line: int
    end_line: int
    content: str


def chunk_file(file_path: str, content: str) -> list[Chunk]:
    if not content.strip():
        return []

    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    tree = parser.parse(content.encode())

    chunks: list[Chunk] = []
    _walk(tree.root_node, content, file_path, chunks, inside_class=False)

    if not chunks:
        if len(content.encode()) <= FILE_FALLBACK_MAX_BYTES:
            chunks.append(Chunk(
                file_path=file_path,
                name="<module>",
                chunk_type="module",
                start_line=1,
                end_line=content.count("\n") + 1,
                content=content,
            ))

    return chunks


def _walk(node, content: str, file_path: str, chunks: list[Chunk], inside_class: bool):
    if node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        name = content[name_node.start_byte:name_node.end_byte] if name_node else "<func>"
        chunks.append(Chunk(
            file_path=file_path,
            name=name,
            chunk_type="method" if inside_class else "function",
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            content=content[node.start_byte:node.end_byte],
        ))
        return

    if node.type == "class_definition":
        name_node = node.child_by_field_name("name")
        name = content[name_node.start_byte:name_node.end_byte] if name_node else "<class>"
        body_node = node.child_by_field_name("body")

        # Class chunk = signature line + optional docstring only
        header_end_byte = body_node.start_byte if body_node else node.end_byte
        class_content = content[node.start_byte:header_end_byte].rstrip()
        header_end_line = (content[:header_end_byte].count("\n") + 1)

        if body_node and body_node.child_count > 0:
            first = body_node.children[0]
            if first.type == "expression_statement" and first.child_count > 0:
                expr = first.children[0]
                if expr.type == "string":
                    docstring = content[expr.start_byte:expr.end_byte]
                    class_content = class_content + "\n    " + docstring
                    header_end_line = first.end_point[0] + 1

        chunks.append(Chunk(
            file_path=file_path,
            name=name,
            chunk_type="class",
            start_line=node.start_point[0] + 1,
            end_line=header_end_line,
            content=class_content,
        ))

        if body_node:
            for child in body_node.children:
                _walk(child, content, file_path, chunks, inside_class=True)
        return

    for child in node.children:
        _walk(child, content, file_path, chunks, inside_class=inside_class)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 8 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add indexer/chunker.py tests/test_chunker.py
git commit -m "feat: tree-sitter AST chunker — functions, class headers, methods, fallback"
```

---

## Task 2: `indexer/pipeline.py`

**Files:**
- Create: `indexer/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from indexer.chunker import Chunk


REPO_ID = uuid.uuid4()
REPO_FULL_NAME = "owner/repo"
INSTALLATION_ID = 42


def _make_chunk(name="func", file_path="src/a.py"):
    return Chunk(
        file_path=file_path,
        name=name,
        chunk_type="function",
        start_line=1,
        end_line=10,
        content=f"def {name}(): pass",
    )


@pytest.mark.asyncio
async def test_index_file_upserts_vectors_to_qdrant():
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()

    mock_qdrant = MagicMock()
    mock_qdrant.delete = AsyncMock()
    mock_qdrant.upsert = AsyncMock()

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 1536])

    chunk = _make_chunk()
    with patch("indexer.pipeline.chunk_file", return_value=[chunk]), \
         patch("indexer.pipeline.Embedder", return_value=mock_embedder):
        from indexer.pipeline import index_file
        await index_file(
            db=mock_db,
            qdrant=mock_qdrant,
            repo_id=REPO_ID,
            repo_full_name=REPO_FULL_NAME,
            file_path="src/a.py",
            content="def func(): pass",
        )

    mock_qdrant.delete.assert_awaited_once()
    mock_qdrant.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_index_file_skips_unchanged_content_hash():
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.delete = AsyncMock()
    mock_qdrant.upsert = AsyncMock()

    content = "def func(): pass"
    import hashlib
    existing_hash = hashlib.sha256(content.encode()).hexdigest()

    with patch("indexer.pipeline.get_content_hash", return_value=existing_hash), \
         patch("indexer.pipeline.get_indexed_file_hash", AsyncMock(return_value=existing_hash)):
        from indexer.pipeline import index_file
        await index_file(
            db=mock_db,
            qdrant=mock_qdrant,
            repo_id=REPO_ID,
            repo_full_name=REPO_FULL_NAME,
            file_path="src/a.py",
            content=content,
        )

    mock_qdrant.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_file_removes_qdrant_vectors():
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_qdrant = MagicMock()
    mock_qdrant.delete = AsyncMock()

    from indexer.pipeline import delete_file
    await delete_file(
        db=mock_db,
        qdrant=mock_qdrant,
        repo_id=REPO_ID,
        repo_full_name=REPO_FULL_NAME,
        file_path="src/deleted.py",
    )

    mock_qdrant.delete.assert_awaited_once()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'indexer.pipeline'`

- [ ] **Step 3: Write `indexer/pipeline.py`**

```python
import hashlib
import uuid
from datetime import datetime, timezone

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sqlalchemy import select, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import IndexedFile
from indexer.chunker import chunk_file, Chunk
from indexer.embedder import Embedder

COLLECTION = "code_chunks"


def get_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


async def get_indexed_file_hash(db: AsyncSession, repo_id: uuid.UUID, file_path: str) -> str | None:
    result = await db.execute(
        select(IndexedFile.content_hash).where(
            IndexedFile.repo_id == repo_id,
            IndexedFile.file_path == file_path,
        )
    )
    row = result.scalar_one_or_none()
    return row


async def index_file(
    db: AsyncSession,
    qdrant: AsyncQdrantClient,
    repo_id: uuid.UUID,
    repo_full_name: str,
    file_path: str,
    content: str,
) -> None:
    new_hash = get_content_hash(content)
    existing_hash = await get_indexed_file_hash(db, repo_id, file_path)
    if existing_hash == new_hash:
        return

    chunks = chunk_file(file_path, content)
    if not chunks:
        return

    embedder = Embedder()
    vectors = await embedder.embed([c.content for c in chunks])

    await qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=[
            FieldCondition(key="repo_id", match=MatchValue(value=str(repo_id))),
            FieldCondition(key="file_path", match=MatchValue(value=file_path)),
        ]),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "repo_id": str(repo_id),
                "repo_full_name": repo_full_name,
                "file_path": chunk.file_path,
                "name": chunk.name,
                "chunk_type": chunk.chunk_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "content_hash": new_hash,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    await qdrant.upsert(collection_name=COLLECTION, points=points)

    stmt = pg_insert(IndexedFile).values(
        repo_id=repo_id,
        file_path=file_path,
        content_hash=new_hash,
        indexed_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        index_elements=["repo_id", "file_path"],
        set_={"content_hash": new_hash, "indexed_at": datetime.now(timezone.utc)},
    )
    await db.execute(stmt)
    await db.commit()


async def delete_file(
    db: AsyncSession,
    qdrant: AsyncQdrantClient,
    repo_id: uuid.UUID,
    repo_full_name: str,
    file_path: str,
) -> None:
    await qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(must=[
            FieldCondition(key="repo_id", match=MatchValue(value=str(repo_id))),
            FieldCondition(key="file_path", match=MatchValue(value=file_path)),
        ]),
    )
    stmt = pg_insert(IndexedFile).values(
        repo_id=repo_id,
        file_path=file_path,
        content_hash=None,
        indexed_at=datetime.now(timezone.utc),
    ).on_conflict_do_update(
        index_elements=["repo_id", "file_path"],
        set_={"content_hash": None, "indexed_at": datetime.now(timezone.utc)},
    )
    await db.execute(stmt)
    await db.commit()
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add indexer/pipeline.py tests/test_pipeline.py
git commit -m "feat: indexer pipeline — chunk, embed, delete_by_filter, upsert, hash-skip"
```

---

## Task 3: Real `indexer/tasks.py`

**Files:**
- Replace: `indexer/tasks.py`
- Create: `tests/test_tasks.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tasks.py
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


REPO_FULL_NAME = "owner/repo"
INSTALLATION_ID = 1


def test_full_index_task_exists():
    from indexer.tasks import full_index
    assert callable(full_index)


def test_incremental_index_task_exists():
    from indexer.tasks import incremental_index
    assert callable(incremental_index)


@patch("indexer.tasks.run_full_index")
@patch("indexer.tasks.AsyncSessionLocal")
@patch("indexer.tasks.AsyncQdrantClient")
def test_full_index_calls_run_full_index(mock_qdrant, mock_session, mock_run):
    mock_run.return_value = None
    from indexer.tasks import full_index
    # apply() runs task synchronously in-process
    full_index.apply(args=[REPO_FULL_NAME, INSTALLATION_ID])
    mock_run.assert_called_once()


@patch("indexer.tasks.run_incremental_index")
@patch("indexer.tasks.AsyncSessionLocal")
@patch("indexer.tasks.AsyncQdrantClient")
def test_incremental_index_calls_run_incremental_index(mock_qdrant, mock_session, mock_run):
    mock_run.return_value = None
    from indexer.tasks import incremental_index
    incremental_index.apply(args=[REPO_FULL_NAME, ["src/a.py"], INSTALLATION_ID])
    mock_run.assert_called_once()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_tasks.py -v
```

Expected: 2 pass (existence checks), 2 fail (the `run_full_index` not found)

- [ ] **Step 3: Add `run_full_index` and `run_incremental_index` to `indexer/pipeline.py`**

Append to `indexer/pipeline.py`:

```python
import asyncio
import subprocess
import tempfile
import os
from pathlib import Path
from github import Github  # PyGitHub
from db.session import AsyncSessionLocal
from config import settings


async def run_full_index(
    qdrant: AsyncQdrantClient,
    repo_full_name: str,
    installation_id: int,
) -> None:
    # Fetch repo tree via GitHub API
    token = _get_installation_token(installation_id)
    g = Github(token)
    repo = g.get_repo(repo_full_name)
    contents = _get_all_py_files(repo)

    async with AsyncSessionLocal() as db:
        repo_row = await _get_or_create_repo(db, repo, installation_id)
        for item in contents:
            file_content = item.decoded_content.decode("utf-8", errors="ignore")
            await index_file(
                db=db,
                qdrant=qdrant,
                repo_id=repo_row.id,
                repo_full_name=repo_full_name,
                file_path=item.path,
                content=file_content,
            )


async def run_incremental_index(
    qdrant: AsyncQdrantClient,
    repo_full_name: str,
    changed_files: list[dict],  # [{"path": str, "status": "added"|"modified"|"removed"}]
    installation_id: int,
) -> None:
    token = _get_installation_token(installation_id)
    g = Github(token)
    repo = g.get_repo(repo_full_name)

    async with AsyncSessionLocal() as db:
        repo_row = await _get_or_create_repo(db, repo, installation_id)
        for f in changed_files:
            path = f["path"]
            if not path.endswith(".py"):
                continue
            if f["status"] == "removed":
                await delete_file(db, qdrant, repo_row.id, repo_full_name, path)
            else:
                try:
                    content_file = repo.get_contents(path)
                    content = content_file.decoded_content.decode("utf-8", errors="ignore")
                    await index_file(db, qdrant, repo_row.id, repo_full_name, path, content)
                except Exception:
                    pass


def _get_all_py_files(repo) -> list:
    results = []
    contents = repo.get_contents("")
    while contents:
        item = contents.pop(0)
        if item.type == "dir":
            contents.extend(repo.get_contents(item.path))
        elif item.path.endswith(".py"):
            results.append(item)
    return results


def _get_installation_token(installation_id: int) -> str:
    # Imported here to avoid circular import; implemented fully in Phase 3
    from github.auth import get_installation_token
    return get_installation_token(installation_id)


async def _get_or_create_repo(db: AsyncSession, repo, installation_id: int):
    from db.models import Repo
    result = await db.execute(
        select(Repo).where(Repo.github_repo_id == repo.id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = Repo(
            github_repo_id=repo.id,
            full_name=repo.full_name,
            installation_id=installation_id,
        )
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return row
```

- [ ] **Step 4: Replace `indexer/tasks.py`**

```python
import asyncio
from qdrant_client import AsyncQdrantClient
from worker import celery_app
from config import settings
from indexer.pipeline import run_full_index, run_incremental_index


def _make_qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


@celery_app.task(name="full_index", bind=True, max_retries=3, default_retry_delay=60)
def full_index(self, repo_full_name: str, installation_id: int):
    try:
        qdrant = _make_qdrant()
        asyncio.run(run_full_index(qdrant, repo_full_name, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


@celery_app.task(name="incremental_index", bind=True, max_retries=3, default_retry_delay=60)
def incremental_index(self, repo_full_name: str, changed_files: list[dict], installation_id: int):
    try:
        qdrant = _make_qdrant()
        asyncio.run(run_incremental_index(qdrant, repo_full_name, changed_files, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_tasks.py tests/test_pipeline.py tests/test_chunker.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add indexer/tasks.py indexer/pipeline.py tests/test_tasks.py
git commit -m "feat: real full_index + incremental_index Celery tasks with retry backoff"
```

---

## Task 4: `api/routes/search.py`

**Files:**
- Create: `api/routes/search.py`
- Modify: `api/main.py` — register search router
- Create: `tests/test_routes_search.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_routes_search.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.mark.asyncio
async def test_search_returns_results():
    mock_hit = MagicMock()
    mock_hit.score = 0.95
    mock_hit.payload = {
        "file_path": "src/auth.py",
        "name": "verify_token",
        "chunk_type": "function",
        "start_line": 10,
        "end_line": 25,
        "content": "def verify_token(token): ...",
    }

    with patch("api.routes.search.Embedder") as MockEmbedder, \
         patch("api.main.AsyncQdrantClient") as MockQdrant, \
         patch("api.main.run_migrations"):
        MockEmbedder.return_value.embed = AsyncMock(return_value=[[0.1] * 1536])
        mock_qdrant_instance = MagicMock()
        mock_qdrant_instance.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        mock_qdrant_instance.create_collection = AsyncMock()
        mock_qdrant_instance.search = AsyncMock(return_value=[mock_hit])
        MockQdrant.return_value = mock_qdrant_instance

        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/search",
                json={"query": "how does token verification work", "limit": 5},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["file_path"] == "src/auth.py"
    assert data["results"][0]["score"] == 0.95
```

- [ ] **Step 2: Run test — verify it fails**

```bash
pytest tests/test_routes_search.py -v
```

Expected: `ModuleNotFoundError: No module named 'api.routes.search'`

- [ ] **Step 3: Write `api/routes/search.py`**

```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from indexer.embedder import Embedder
from api.dependencies import get_qdrant

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchHit(BaseModel):
    file_path: str
    name: str
    chunk_type: str
    start_line: int
    end_line: int
    content: str
    score: float


@router.post("/search")
async def post_search(
    body: SearchRequest,
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    embedder = Embedder()
    vectors = await embedder.embed([body.query])
    hits = await qdrant.search(
        collection_name="code_chunks",
        query_vector=vectors[0],
        limit=body.limit,
    )
    results = [
        SearchHit(
            file_path=h.payload.get("file_path", ""),
            name=h.payload.get("name", ""),
            chunk_type=h.payload.get("chunk_type", ""),
            start_line=h.payload.get("start_line", 0),
            end_line=h.payload.get("end_line", 0),
            content=h.payload.get("content", ""),
            score=h.score,
        )
        for h in hits
    ]
    return {"results": results}
```

- [ ] **Step 4: Register router in `api/main.py`**

Add after the existing `index_router` import and `include_router`:

```python
from api.routes.search import router as search_router
# ...
app.include_router(search_router)
```

- [ ] **Step 5: Run test — verify it passes**

```bash
pytest tests/test_routes_search.py -v
```

Expected: `PASSED`

- [ ] **Step 6: Run all tests**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Integration smoke test**

With Docker Compose running and a real repo indexed via `POST /index`:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication token validation", "limit": 5}'
```

Expected: JSON response with `results` array containing relevant code chunks.

- [ ] **Step 8: Commit**

```bash
git add api/routes/search.py api/main.py tests/test_routes_search.py
git commit -m "feat: POST /search — embed query → Qdrant code_chunks retrieval"
```

---

**Phase 2 complete.**

Acceptance criteria met:
- Index a real Python repo → Qdrant dashboard shows points per function/class/method
- `POST /search` returns semantically relevant chunks for natural language queries
- All unit tests pass
