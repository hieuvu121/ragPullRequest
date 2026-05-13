# Phase 1: RAG Pipeline (Pure Python) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the entire RAG loop as standalone Python scripts — no FastAPI, no Celery, no Docker Compose. Prove the AI logic works before adding infrastructure.

**Architecture:** All logic lives in `pipeline/` as importable modules. Two entry-point scripts in `scripts/` drive the two workflows: indexing a repo and reviewing a diff. A single `docker run` starts Qdrant locally. Everything else is pure Python.

**Tech Stack:** Python 3.11, OpenAI (text-embedding-3-small + GPT-4o + GPT-4o-mini), Qdrant (local via docker run), tree-sitter 0.21.x, sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2), python-dotenv, pytest + pytest-asyncio

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Phase 1 dependencies |
| `.env` | `OPENAI_API_KEY`, `QDRANT_URL` |
| `pipeline/__init__.py` | Package marker |
| `pipeline/chunker.py` | `Chunk` dataclass + `chunk_file()` via tree-sitter |
| `pipeline/embedder.py` | `Embedder.embed()` — OpenAI batched calls |
| `pipeline/qdrant_store.py` | `QdrantStore`: create collection, upsert, search, delete_by_filter |
| `pipeline/retriever.py` | `retrieve()` — HyDE + parallel search + RRF + cross-encoder |
| `pipeline/generator.py` | `generate_review()` — GPT-4o JSON, returns `list[ReviewComment]` |
| `scripts/index_repo.py` | CLI: clone repo → walk `.py` files → chunk → embed → upsert |
| `scripts/review_pipeline.py` | CLI: raw diff string → retrieve → generate → print JSON |
| `tests/__init__.py` | Package marker |
| `tests/test_chunker.py` | AST chunking, fallback, skip, line numbers |
| `tests/test_embedder.py` | Batch logic (mocked OpenAI) |
| `tests/test_retriever.py` | RRF merging (pure logic, no network) |
| `tests/test_generator.py` | JSON parsing, comment filtering (mocked OpenAI) |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env`
- Create: `pipeline/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[tool.poetry]
name = "rag-pr-reviewer"
version = "0.1.0"
description = "AI-powered PR reviewer — Phase 1: pure Python RAG pipeline"
authors = []

[tool.poetry.dependencies]
python = "^3.11"
openai = "^1.30.0"
qdrant-client = {extras = ["async"], version = "^1.9.1"}
tree-sitter = "^0.21.3"
tree-sitter-python = "^0.21.0"
sentence-transformers = "^2.7.0"
httpx = "^0.27.0"
python-dotenv = "^1.0.0"
unidiff = "^0.7.5"

[tool.poetry.group.dev.dependencies]
pytest = "^8.1.1"
pytest-asyncio = "^0.23.6"
pytest-mock = "^3.14.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create `.env`**

```
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
```

- [ ] **Step 3: Create package markers**

```bash
mkdir -p pipeline scripts tests
touch pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 4: Start Qdrant locally**

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

Verify: `curl http://localhost:6333/healthz` → `{"title":"qdrant - vector search engine",...}`

- [ ] **Step 5: Install and commit**

```bash
poetry install
git init
git add .
git commit -m "feat: Phase 1 scaffold — pure Python RAG pipeline"
```

---

## Task 2: `pipeline/chunker.py`

**Files:**
- Create: `pipeline/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker.py
from pipeline.chunker import chunk_file, Chunk

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


def test_extracts_class_header_not_full_body():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert class_chunks[0].name == "Calculator"
    # class chunk must NOT contain the method bodies
    assert "return a * b" not in class_chunks[0].content


def test_class_chunk_includes_docstring():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert "A simple calculator" in class_chunks[0].content


def test_extracts_methods():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    method_names = {c.name for c in chunks if c.chunk_type == "method"}
    assert method_names == {"multiply", "divide"}


def test_chunk_has_accurate_line_numbers():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_carries_file_path():
    chunks = chunk_file("src/math.py", SIMPLE_PYTHON)
    assert all(c.file_path == "src/math.py" for c in chunks)


def test_small_unparseable_returns_whole_file_chunk():
    bad = "this is not valid python syntax @#$"
    chunks = chunk_file("bad.py", bad)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"
    assert chunks[0].content == bad


def test_large_unparseable_returns_empty():
    bad = "not python " * 300  # > 2 KB
    chunks = chunk_file("big.py", bad)
    assert chunks == []


def test_empty_file_returns_empty():
    assert chunk_file("empty.py", "") == []
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.chunker'`

- [ ] **Step 3: Write `pipeline/chunker.py`**

```python
from dataclasses import dataclass
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language(), "python")
FILE_FALLBACK_MAX_BYTES = 2048


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

        header_end_byte = body_node.start_byte if body_node else node.end_byte
        class_content = content[node.start_byte:header_end_byte].rstrip()
        header_end_line = content[:header_end_byte].count("\n") + 1

        # Include docstring if first statement in body is a string
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

- [ ] **Step 4: Run — verify all pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 9 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/chunker.py tests/test_chunker.py
git commit -m "feat: tree-sitter AST chunker — functions, class headers+docstring, methods, fallback"
```

---

## Task 3: `pipeline/embedder.py`

**Files:**
- Create: `pipeline/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedder.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _fake_response(texts):
    r = MagicMock()
    r.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]
    return r


@pytest.mark.asyncio
async def test_embed_returns_one_vector_per_text():
    with patch("pipeline.embedder.AsyncOpenAI") as M:
        M.return_value.embeddings.create = AsyncMock(
            side_effect=lambda model, input: _fake_response(input)
        )
        from pipeline.embedder import Embedder
        result = await Embedder().embed(["a", "b", "c"])
    assert len(result) == 3
    assert len(result[0]) == 1536


@pytest.mark.asyncio
async def test_embed_batches_100_texts_into_two_requests():
    with patch("pipeline.embedder.AsyncOpenAI") as M:
        create = AsyncMock(side_effect=lambda model, input: _fake_response(input))
        M.return_value.embeddings.create = create
        from pipeline.embedder import Embedder
        await Embedder().embed([f"t{i}" for i in range(150)])
    assert create.call_count == 2


@pytest.mark.asyncio
async def test_embed_empty_list_returns_empty():
    with patch("pipeline.embedder.AsyncOpenAI"):
        from pipeline.embedder import Embedder
        result = await Embedder().embed([])
    assert result == []
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_embedder.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.embedder'`

- [ ] **Step 3: Write `pipeline/embedder.py`**

```python
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


class Embedder:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch
            )
            results.extend(item.embedding for item in response.data)
        return results
```

- [ ] **Step 4: Run — verify all pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/embedder.py tests/test_embedder.py
git commit -m "feat: Embedder — OpenAI text-embedding-3-small, 100-text batches"
```

---

## Task 4: `pipeline/qdrant_store.py`

**Files:**
- Create: `pipeline/qdrant_store.py`

No unit tests here — this module is thin glue over the Qdrant SDK. It's validated by the integration smoke test in Task 6.

- [ ] **Step 1: Write `pipeline/qdrant_store.py`**

```python
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


@dataclass
class SearchHit:
    id: str
    score: float
    payload: dict


class QdrantStore:
    def __init__(self):
        self.client = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    async def create_collection(self) -> None:
        existing = {c.name for c in (await self.client.get_collections()).collections}
        if COLLECTION not in existing:
            await self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    async def upsert(self, points: list[PointStruct]) -> None:
        await self.client.upsert(collection_name=COLLECTION, points=points)

    async def search(self, vector: list[float], limit: int = 20) -> list[SearchHit]:
        hits = await self.client.search(
            collection_name=COLLECTION, query_vector=vector, limit=limit
        )
        return [SearchHit(id=str(h.id), score=h.score, payload=h.payload or {}) for h in hits]

    async def delete_by_filter(self, repo_id: str, file_path: str) -> None:
        await self.client.delete(
            collection_name=COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="repo_id", match=MatchValue(value=repo_id)),
                FieldCondition(key="file_path", match=MatchValue(value=file_path)),
            ]),
        )
```

- [ ] **Step 2: Smoke test against live Qdrant**

```bash
python - <<'EOF'
import asyncio
from pipeline.qdrant_store import QdrantStore
from qdrant_client.models import PointStruct

async def main():
    store = QdrantStore()
    await store.create_collection()
    await store.upsert([
        PointStruct(id="test-1", vector=[0.1]*1536, payload={"test": True})
    ])
    hits = await store.search([0.1]*1536, limit=1)
    print("hits:", hits)
    await store.delete_by_filter("test-repo", "test-file.py")
    print("OK")

asyncio.run(main())
EOF
```

Expected: prints `hits: [SearchHit(id='test-1', score=1.0, ...)]` and `OK`

- [ ] **Step 3: Commit**

```bash
git add pipeline/qdrant_store.py
git commit -m "feat: QdrantStore — create collection, upsert, search, delete_by_filter"
```

---

## Task 5: `scripts/index_repo.py`

**Files:**
- Create: `scripts/index_repo.py`

No unit tests — validated by running against a real repo in the acceptance test (Task 8).

- [ ] **Step 1: Write `scripts/index_repo.py`**

```python
"""
Usage:
  python scripts/index_repo.py <git_url_or_local_path> <repo_id>

Example:
  python scripts/index_repo.py https://github.com/pallets/flask my-flask
  python scripts/index_repo.py /path/to/local/repo my-local-repo
"""
import asyncio
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
from qdrant_client.models import PointStruct

from pipeline.chunker import chunk_file
from pipeline.embedder import Embedder
from pipeline.qdrant_store import QdrantStore

SUPPORTED_EXTENSIONS = {".py"}


def _get_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if ".git" not in p.parts]


async def index_directory(root: Path, repo_id: str) -> int:
    store = QdrantStore()
    await store.create_collection()

    embedder = Embedder()
    py_files = _get_py_files(root)
    total_chunks = 0

    for file_path in py_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        relative_path = str(file_path.relative_to(root))
        chunks = chunk_file(relative_path, content)
        if not chunks:
            continue

        vectors = await embedder.embed([c.content for c in chunks])

        await store.delete_by_filter(repo_id, relative_path)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "repo_id": repo_id,
                    "file_path": chunk.file_path,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await store.upsert(points)
        total_chunks += len(points)
        print(f"  indexed {relative_path} → {len(chunks)} chunks")

    return total_chunks


async def main(source: str, repo_id: str) -> None:
    source_path = Path(source)

    if source_path.exists() and source_path.is_dir():
        root = source_path
        total = await index_directory(root, repo_id)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            print(f"Cloning {source}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", source, tmp],
                check=True, capture_output=True,
            )
            total = await index_directory(Path(tmp), repo_id)

    print(f"\nDone. Total chunks indexed: {total}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))
```

- [ ] **Step 2: Commit**

```bash
git add scripts/index_repo.py
git commit -m "feat: index_repo script — clone/walk repo, chunk, embed, upsert to Qdrant"
```

---

## Task 6: `pipeline/retriever.py`

**Files:**
- Create: `pipeline/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retriever.py
import pytest
from pipeline.retriever import reciprocal_rank_fusion, ScoredChunk


def test_rrf_doc_in_two_lists_scores_higher_than_doc_in_one():
    list_a = ["doc1", "doc2"]
    list_b = ["doc1", "doc3"]
    ranked = reciprocal_rank_fusion([list_a, list_b])
    scores = dict(ranked)
    assert scores["doc1"] > scores["doc2"]
    assert scores["doc1"] > scores["doc3"]


def test_rrf_equal_rank_equal_score():
    list_a = ["doc1", "doc2"]
    list_b = ["doc2", "doc1"]
    ranked = reciprocal_rank_fusion([list_a, list_b])
    scores = dict(ranked)
    assert scores["doc1"] == scores["doc2"]


def test_rrf_sorted_descending():
    list_a = ["doc3", "doc1"]
    list_b = ["doc1", "doc2"]
    ranked = reciprocal_rank_fusion([list_a, list_b])
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] == "doc1"


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_scored_chunk_fields():
    c = ScoredChunk(
        id="abc", score=0.9,
        payload={"file_path": "src/a.py", "content": "def foo(): pass",
                 "start_line": 1, "end_line": 5}
    )
    assert c.id == "abc"
    assert c.payload["file_path"] == "src/a.py"
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.retriever'`

- [ ] **Step 3: Write `pipeline/retriever.py`**

```python
import asyncio
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pipeline.qdrant_store import QdrantStore, SearchHit

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


@dataclass
class ScoredChunk:
    id: str
    score: float
    payload: dict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


async def _embed_one(client: AsyncOpenAI, text: str) -> list[float]:
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


async def _hyde_query(client: AsyncOpenAI, diff_hunk: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior engineer. Given a code diff hunk, write a short "
                    "Python function (5-10 lines) that represents existing code most "
                    "relevant to reviewing this change. Output only the code."
                ),
            },
            {"role": "user", "content": diff_hunk},
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


async def retrieve(
    store: QdrantStore,
    query: str,
    top_k: int = 5,
    candidate_pool: int = 20,
) -> list[ScoredChunk]:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Embed original query and HyDE hypothesis in parallel
    hyde_text, original_vector = await asyncio.gather(
        _hyde_query(client, query),
        _embed_one(client, query),
    )
    hyde_vector = await _embed_one(client, hyde_text)

    # Average the two vectors
    merged_vector = [(o + h) / 2 for o, h in zip(original_vector, hyde_vector)]

    # Search with both vectors and merge via RRF
    original_hits, hyde_hits = await asyncio.gather(
        store.search(original_vector, limit=candidate_pool),
        store.search(merged_vector, limit=candidate_pool),
    )

    id_to_hit: dict[str, SearchHit] = {}
    for hit in original_hits + hyde_hits:
        id_to_hit[hit.id] = hit

    original_ids = [h.id for h in original_hits]
    hyde_ids = [h.id for h in hyde_hits]
    merged = reciprocal_rank_fusion([original_ids, hyde_ids])

    candidates = [
        ScoredChunk(id=doc_id, score=score, payload=id_to_hit[doc_id].payload)
        for doc_id, score in merged
        if doc_id in id_to_hit
    ]

    if len(candidates) <= top_k:
        return candidates

    # Cross-encoder rerank
    reranker = _get_reranker()
    pairs = [[query, c.payload.get("content", "")] for c in candidates]
    ce_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:top_k]]
```

- [ ] **Step 4: Run — verify all pass**

```bash
pytest tests/test_retriever.py -v
```

Expected: 5 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/retriever.py tests/test_retriever.py
git commit -m "feat: retriever — HyDE expansion, parallel search, RRF, cross-encoder rerank top-5"
```

---

## Task 7: `pipeline/generator.py`

**Files:**
- Create: `pipeline/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pipeline.generator import generate_review, ReviewComment
from pipeline.retriever import ScoredChunk

CHUNKS = [
    ScoredChunk(
        id="c1", score=0.9,
        payload={"file_path": "src/auth.py", "start_line": 10, "end_line": 20,
                 "content": "def verify(token): return token == SECRET"},
    )
]

VALID_OUTPUT = json.dumps({
    "comments": [
        {
            "line": 5,
            "path": "src/login.py",
            "severity": "error",
            "issue": "No null check on token",
            "suggestion": "Add `if not token: raise ValueError()`",
            "citation": "src/auth.py:10-20",
        }
    ]
})

DIFF_LINES = {"src/login.py": {5, 6, 7}}


@pytest.mark.asyncio
async def test_generate_review_returns_review_comments():
    with patch("pipeline.generator.AsyncOpenAI") as M:
        M.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=VALID_OUTPUT))]
            )
        )
        result = await generate_review(
            diff_text="@@ -1 +1,5 @@\n+def login(token):\n+    pass",
            chunks=CHUNKS,
            diff_lines=DIFF_LINES,
        )
    assert len(result) == 1
    assert isinstance(result[0], ReviewComment)
    assert result[0].path == "src/login.py"
    assert result[0].line == 5
    assert result[0].severity == "error"


@pytest.mark.asyncio
async def test_generate_review_filters_comments_outside_diff():
    out_of_diff = json.dumps({
        "comments": [
            # line 99 is NOT in diff_lines for src/login.py
            {"line": 99, "path": "src/login.py", "severity": "warning",
             "issue": "out", "suggestion": "x", "citation": ""},
            # line 5 IS in diff_lines
            {"line": 5, "path": "src/login.py", "severity": "error",
             "issue": "in", "suggestion": "y", "citation": "src/auth.py:10"},
        ]
    })
    with patch("pipeline.generator.AsyncOpenAI") as M:
        M.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=out_of_diff))]
            )
        )
        result = await generate_review(
            diff_text="diff",
            chunks=CHUNKS,
            diff_lines=DIFF_LINES,
        )
    assert len(result) == 1
    assert result[0].issue == "in"


@pytest.mark.asyncio
async def test_generate_review_returns_empty_on_no_comments():
    empty = json.dumps({"comments": []})
    with patch("pipeline.generator.AsyncOpenAI") as M:
        M.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content=empty))]
            )
        )
        result = await generate_review(diff_text="diff", chunks=CHUNKS, diff_lines={})
    assert result == []
```

- [ ] **Step 2: Run — verify failure**

```bash
pytest tests/test_generator.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline.generator'`

- [ ] **Step 3: Write `pipeline/generator.py`**

```python
import json
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from pipeline.retriever import ScoredChunk
from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
You are given a unified diff and a set of retrieved code context chunks.
Respond ONLY with valid JSON in this exact schema — no markdown, no explanation:
{
  "comments": [
    {
      "line": <integer — must be an added line in the diff>,
      "path": "<file path from the diff>",
      "severity": "<error|warning|suggestion>",
      "issue": "<one sentence: what is wrong>",
      "suggestion": "<one sentence: how to fix it>",
      "citation": "<file_path:start_line-end_line of the context chunk used>"
    }
  ]
}
Rules:
- Only comment on lines that appear as additions (+) in the diff.
- Every comment must be grounded in one of the provided context chunks.
- Do not invent issues not supported by the context."""


@dataclass
class ReviewComment:
    line: int
    path: str
    severity: str
    issue: str
    suggestion: str
    citation: str


async def generate_review(
    diff_text: str,
    chunks: list[ScoredChunk],
    diff_lines: dict[str, set[int]],
) -> list[ReviewComment]:
    context_text = "\n\n".join(
        f"### {c.payload.get('file_path', '')}:"
        f"{c.payload.get('start_line', 0)}-{c.payload.get('end_line', 0)}\n"
        f"{c.payload.get('content', '')}"
        for c in chunks
    )
    user_message = f"## Diff\n{diff_text}\n\n## Context Chunks\n{context_text}"

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model=GENERATION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = json.loads(response.choices[0].message.content)
    comments: list[ReviewComment] = []
    for c in raw.get("comments", []):
        path = c.get("path", "")
        line = c.get("line", 0)
        if line not in diff_lines.get(path, set()):
            continue
        comments.append(ReviewComment(
            line=line,
            path=path,
            severity=c.get("severity", "suggestion"),
            issue=c.get("issue", ""),
            suggestion=c.get("suggestion", ""),
            citation=c.get("citation", ""),
        ))
    return comments
```

- [ ] **Step 4: Run — verify all pass**

```bash
pytest tests/test_generator.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add pipeline/generator.py tests/test_generator.py
git commit -m "feat: GPT-4o generator — JSON mode, filters to diff lines, grounded on context"
```

---

## Task 8: `scripts/review_pipeline.py` + Acceptance Test

**Files:**
- Create: `scripts/review_pipeline.py`

- [ ] **Step 1: Write `scripts/review_pipeline.py`**

```python
"""
Usage:
  python scripts/review_pipeline.py <repo_id> <diff_file>

  <diff_file>  path to a file containing a unified diff, or "-" to read from stdin

Example:
  python scripts/review_pipeline.py my-flask review.diff
  git diff HEAD~1 | python scripts/review_pipeline.py my-flask -
"""
import asyncio
import json
import sys
from io import StringIO
from unidiff import PatchSet

from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve
from pipeline.generator import generate_review


def parse_diff_lines(diff_text: str) -> dict[str, set[int]]:
    """Return {file_path: {added_line_numbers}} from a unified diff."""
    patch = PatchSet(StringIO(diff_text))
    result: dict[str, set[int]] = {}
    for patched_file in patch:
        if patched_file.is_removed_file:
            continue
        added: set[int] = set()
        for hunk in patched_file:
            for line in hunk:
                if line.is_added and line.target_line_no:
                    added.add(line.target_line_no)
        result[patched_file.path] = added
    return result


async def main(repo_id: str, diff_text: str) -> None:
    store = QdrantStore()

    diff_lines = parse_diff_lines(diff_text)
    if not diff_lines:
        print("No added lines found in diff.", file=sys.stderr)
        return

    print(f"Retrieving context for {len(diff_lines)} changed files...", file=sys.stderr)
    chunks = await retrieve(store=store, query=diff_text, top_k=5, candidate_pool=20)
    print(f"  → {len(chunks)} chunks retrieved and reranked", file=sys.stderr)

    print("Generating review...", file=sys.stderr)
    comments = await generate_review(
        diff_text=diff_text,
        chunks=chunks,
        diff_lines=diff_lines,
    )

    output = [
        {
            "line": c.line,
            "path": c.path,
            "severity": c.severity,
            "issue": c.issue,
            "suggestion": c.suggestion,
            "citation": c.citation,
        }
        for c in comments
    ]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    repo_id = sys.argv[1]
    diff_source = sys.argv[2]

    if diff_source == "-":
        diff_text = sys.stdin.read()
    else:
        with open(diff_source) as f:
            diff_text = f.read()

    asyncio.run(main(repo_id, diff_text))
```

- [ ] **Step 2: Commit**

```bash
git add scripts/review_pipeline.py
git commit -m "feat: review_pipeline script — diff → retrieve → generate → print JSON"
```

- [ ] **Step 3: Run all unit tests**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Acceptance test — index a real repo**

```bash
# Index flask (small, well-structured Python repo)
python scripts/index_repo.py https://github.com/pallets/flask flask-repo
```

Expected: prints each file as it's indexed, ends with `Total chunks indexed: N` (should be several hundred).

Check Qdrant: `curl http://localhost:6333/collections/code_chunks` — verify `vectors_count > 0`

- [ ] **Step 5: Acceptance test — run a review**

Create a test diff file `test.diff`:
```diff
diff --git a/src/flask/wrappers.py b/src/flask/wrappers.py
index abc123..def456 100644
--- a/src/flask/wrappers.py
+++ b/src/flask/wrappers.py
@@ -1,5 +1,10 @@
 from flask import Flask
+import os
+
+SECRET = os.getenv("APP_SECRET")
+
 def create_app():
+    if not SECRET:
+        raise RuntimeError("APP_SECRET not set")
     app = Flask(__name__)
     return app
```

```bash
python scripts/review_pipeline.py flask-repo test.diff
```

Expected: JSON array of comments. Each comment should:
- Have `line` matching an added line in the diff
- Have a non-empty `citation` referencing a real Flask file
- Have a relevant `issue` and `suggestion`

- [ ] **Step 6: Iterate until quality is acceptable**

If comments are irrelevant: adjust `SYSTEM_PROMPT` in `generator.py` to be more specific about grounding.

If citations are wrong: check `pipeline/retriever.py` — verify HyDE query is reasonable by printing `hyde_text` temporarily.

If chunking is wrong: add `print(chunk)` in `index_repo.py` to inspect what's being stored.

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "feat: Phase 1 complete — end-to-end RAG pipeline validated on real Python repo"
```

---

**Phase 1 complete.**

Acceptance criteria met:
- `index_repo.py` indexes a real Python repo into Qdrant with function/class/method chunks
- `review_pipeline.py` produces relevant, cited, correctly structured JSON comments for a real diff
- All unit tests pass (`pytest -v`)
