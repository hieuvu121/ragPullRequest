# Phase 3: GitHub Bot Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 2 complete — indexing pipeline works, `POST /search` returns relevant chunks.

**Goal:** Wire GitHub App auth, webhook security, and the full review pipeline: fetch diff → HyDE + RRF retrieval → cross-encoder rerank → GPT-4o generate → post inline GitHub Review. Opening a real PR triggers the bot within 30 seconds.

**Architecture:** `github/` handles all GitHub communication (auth, webhook verification, API calls). `reviewer/` pipeline is pure Python with no GitHub imports — it receives diffs and returns comments. Celery task `review_pr` glues them together. Single webhook endpoint routes all events.

**Tech Stack:** PyJWT (RS256), PyGitHub, unidiff, httpx, sentence-transformers cross-encoder, OpenAI GPT-4o + text-embedding-3-small, ngrok (dev)

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `github/__init__.py` | Create | Package marker |
| `github/auth.py` | Create | JWT mint, installation token, cache |
| `github/webhook.py` | Create | HMAC-SHA256 verify, event parsing |
| `github/client.py` | Create | Fetch PR diff, post GitHub Review |
| `reviewer/__init__.py` | Create | Package marker |
| `reviewer/retriever.py` | Create | HyDE expansion + parallel Qdrant search + RRF |
| `reviewer/reranker.py` | Create | Cross-encoder `ms-marco-MiniLM-L-6-v2`, top-k |
| `reviewer/generator.py` | Create | GPT-4o JSON review generation |
| `reviewer/pipeline.py` | Create | Orchestrate retrieve → rerank → generate → post |
| `reviewer/tasks.py` | Create | `review_pr` Celery task with 3× retry |
| `api/routes/webhooks.py` | Create | `POST /webhook/github` — routes all events |
| `api/handlers/__init__.py` | Create | Package marker |
| `api/handlers/indexing.py` | Create | Handle `push` event |
| `api/handlers/review.py` | Create | Handle `pull_request.opened` event |
| `api/main.py` | Modify | Register webhook router |
| `worker.py` | Modify | Add `reviewer.tasks` to `include` |
| `tests/test_github_auth.py` | Create | JWT structure, token caching |
| `tests/test_github_webhook.py` | Create | HMAC verify, event parsing |
| `tests/test_github_client.py` | Create | Diff parsing, review posting |
| `tests/test_retriever.py` | Create | RRF merging logic |
| `tests/test_reranker.py` | Create | Top-k selection |
| `tests/test_generator.py` | Create | JSON output parsing |
| `tests/test_reviewer_pipeline.py` | Create | Full pipeline (all mocked) |
| `tests/test_webhooks_route.py` | Create | Webhook routing + HMAC |

---

## Task 1: `github/auth.py`

**Files:**
- Create: `github/auth.py`
- Create: `tests/test_github_auth.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_github_auth.py
import time
import pytest
import jwt
from unittest.mock import patch, MagicMock
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization


@pytest.fixture
def rsa_private_key():
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def test_create_jwt_is_valid_rs256(rsa_private_key):
    from github.auth import create_jwt
    token = create_jwt(app_id="123", private_key=rsa_private_key)
    # Decode without verification to inspect claims
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["iss"] == "123"
    assert claims["exp"] > int(time.time())
    assert claims["iat"] <= int(time.time())


def test_get_installation_token_calls_github_api(rsa_private_key):
    mock_response = MagicMock()
    mock_response.json.return_value = {"token": "ghs_test_token", "expires_at": "2099-01-01T00:00:00Z"}
    mock_response.raise_for_status = MagicMock()

    with patch("github.auth.httpx.post", return_value=mock_response), \
         patch("github.auth.settings") as mock_settings:
        mock_settings.github_app_id = "123"
        mock_settings.github_private_key = rsa_private_key

        from github.auth import get_installation_token
        token = get_installation_token(installation_id=99)

    assert token == "ghs_test_token"


def test_get_installation_token_caches_result(rsa_private_key):
    mock_response = MagicMock()
    mock_response.json.return_value = {"token": "ghs_cached", "expires_at": "2099-01-01T00:00:00Z"}
    mock_response.raise_for_status = MagicMock()

    with patch("github.auth.httpx.post", return_value=mock_response), \
         patch("github.auth.settings") as mock_settings, \
         patch("github.auth._token_cache", {}):
        mock_settings.github_app_id = "123"
        mock_settings.github_private_key = rsa_private_key

        from github.auth import get_installation_token
        get_installation_token(installation_id=99)
        get_installation_token(installation_id=99)

    # httpx.post called only once — second call used cache
    assert mock_response.json.call_count == 1
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_github_auth.py -v
```

Expected: `ModuleNotFoundError: No module named 'github.auth'`

- [ ] **Step 3: Write `github/auth.py`**

```python
import time
from datetime import datetime, timezone
import httpx
import jwt
from config import settings

_token_cache: dict[int, dict] = {}  # installation_id → {token, expires_at}


def create_jwt(app_id: str, private_key: str) -> str:
    now = int(time.time())
    payload = {
        "iat": now - 60,   # 60s ago to handle clock skew
        "exp": now + 600,  # 10 minutes
        "iss": app_id,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def get_installation_token(installation_id: int) -> str:
    cached = _token_cache.get(installation_id)
    if cached:
        expires_at = datetime.fromisoformat(cached["expires_at"].replace("Z", "+00:00"))
        if expires_at > datetime.now(timezone.utc).replace(second=0, microsecond=0):
            return cached["token"]

    app_jwt = create_jwt(settings.github_app_id, settings.github_private_key)
    response = httpx.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept": "application/vnd.github+json",
        },
    )
    response.raise_for_status()
    data = response.json()
    _token_cache[installation_id] = data
    return data["token"]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_github_auth.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add github/__init__.py github/auth.py tests/test_github_auth.py
git commit -m "feat: GitHub App auth — RS256 JWT, installation token, cache"
```

---

## Task 2: `github/webhook.py`

**Files:**
- Create: `github/webhook.py`
- Create: `tests/test_github_webhook.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_github_webhook.py
import hashlib
import hmac
import json
import pytest
from github.webhook import verify_signature, parse_event, WebhookEvent


WEBHOOK_SECRET = "test_secret"


def _make_signature(payload: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def test_verify_signature_accepts_valid():
    payload = b'{"action": "opened"}'
    sig = _make_signature(payload, WEBHOOK_SECRET)
    assert verify_signature(payload, sig, WEBHOOK_SECRET) is True


def test_verify_signature_rejects_invalid():
    payload = b'{"action": "opened"}'
    assert verify_signature(payload, "sha256=badhash", WEBHOOK_SECRET) is False


def test_verify_signature_rejects_tampered_payload():
    payload = b'{"action": "opened"}'
    sig = _make_signature(payload, WEBHOOK_SECRET)
    assert verify_signature(b'{"action": "closed"}', sig, WEBHOOK_SECRET) is False


def test_parse_pull_request_opened_event():
    payload = {
        "action": "opened",
        "number": 42,
        "repository": {"full_name": "owner/repo", "id": 123},
        "installation": {"id": 99},
        "pull_request": {"title": "Fix bug"},
    }
    event = parse_event("pull_request", json.dumps(payload).encode())
    assert isinstance(event, WebhookEvent)
    assert event.event_type == "pull_request"
    assert event.action == "opened"
    assert event.pr_number == 42
    assert event.repo_full_name == "owner/repo"
    assert event.installation_id == 99


def test_parse_push_event():
    payload = {
        "ref": "refs/heads/main",
        "repository": {"full_name": "owner/repo", "id": 123},
        "installation": {"id": 99},
        "commits": [
            {"added": [], "modified": ["src/auth.py"], "removed": []},
        ],
    }
    event = parse_event("push", json.dumps(payload).encode())
    assert event.event_type == "push"
    assert event.changed_files == [{"path": "src/auth.py", "status": "modified"}]
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_github_webhook.py -v
```

Expected: `ModuleNotFoundError: No module named 'github.webhook'`

- [ ] **Step 3: Write `github/webhook.py`**

```python
import hashlib
import hmac
import json
from dataclasses import dataclass, field


@dataclass
class WebhookEvent:
    event_type: str
    action: str
    repo_full_name: str
    repo_github_id: int
    installation_id: int
    pr_number: int | None = None
    pr_title: str | None = None
    changed_files: list[dict] = field(default_factory=list)


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def parse_event(event_type: str, payload_bytes: bytes) -> WebhookEvent:
    data = json.loads(payload_bytes)
    repo = data.get("repository", {})
    installation_id = data.get("installation", {}).get("id", 0)

    changed_files: list[dict] = []
    pr_number = None
    pr_title = None
    action = data.get("action", "")

    if event_type == "push":
        for commit in data.get("commits", []):
            for path in commit.get("added", []):
                changed_files.append({"path": path, "status": "added"})
            for path in commit.get("modified", []):
                changed_files.append({"path": path, "status": "modified"})
            for path in commit.get("removed", []):
                changed_files.append({"path": path, "status": "removed"})

    if event_type == "pull_request":
        pr = data.get("pull_request", {})
        pr_number = data.get("number")
        pr_title = pr.get("title")

    return WebhookEvent(
        event_type=event_type,
        action=action,
        repo_full_name=repo.get("full_name", ""),
        repo_github_id=repo.get("id", 0),
        installation_id=installation_id,
        pr_number=pr_number,
        pr_title=pr_title,
        changed_files=changed_files,
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_github_webhook.py -v
```

Expected: 5 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add github/webhook.py tests/test_github_webhook.py
git commit -m "feat: webhook HMAC-SHA256 verification and event parsing"
```

---

## Task 3: `github/client.py`

**Files:**
- Create: `github/client.py`
- Create: `tests/test_github_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_github_client.py
import pytest
from unittest.mock import MagicMock, patch
from github.client import GitHubClient, ParsedDiff, ReviewComment

DIFF_TEXT = """\
diff --git a/src/auth.py b/src/auth.py
index 1234567..abcdefg 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -1,5 +1,8 @@
 import os
+import hashlib
+
 def authenticate(token):
+    if not token:
+        raise ValueError("token required")
     return token == os.environ["SECRET"]
"""


def test_parse_diff_extracts_added_lines():
    client = GitHubClient(token="fake")
    result = client.parse_diff(DIFF_TEXT)
    assert len(result) == 1
    assert result[0].file_path == "src/auth.py"
    added_lines = {line for line in result[0].added_lines}
    assert 2 in added_lines  # import hashlib line
    assert 5 in added_lines  # if not token line


def test_parse_diff_returns_parsed_diff_objects():
    client = GitHubClient(token="fake")
    result = client.parse_diff(DIFF_TEXT)
    assert isinstance(result[0], ParsedDiff)
    assert result[0].hunk_text != ""


def test_post_review_calls_github_api():
    mock_pr = MagicMock()
    mock_pr.create_review = MagicMock(return_value=MagicMock(id=9999))

    mock_repo = MagicMock()
    mock_repo.get_pull = MagicMock(return_value=mock_pr)

    with patch("github.client.Github") as MockGithub:
        MockGithub.return_value.get_repo = MagicMock(return_value=mock_repo)
        client = GitHubClient(token="fake")
        comments = [ReviewComment(path="src/auth.py", line=5, side="RIGHT",
                                  body="Missing null check", severity="warning", citations=[])]
        review_id = client.post_review(
            repo_full_name="owner/repo",
            pr_number=1,
            summary="Looks good with minor issues.",
            comments=comments,
        )

    assert review_id == 9999
    mock_pr.create_review.assert_called_once()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_github_client.py -v
```

Expected: `ModuleNotFoundError: No module named 'github.client'`

- [ ] **Step 3: Write `github/client.py`**

```python
from dataclasses import dataclass, field
from github import Github
from unidiff import PatchSet
import io


@dataclass
class ParsedDiff:
    file_path: str
    hunk_text: str
    added_lines: set[int]


@dataclass
class ReviewComment:
    path: str
    line: int
    side: str  # "RIGHT" for new file, "LEFT" for old
    body: str
    severity: str  # "error" | "warning" | "suggestion"
    citations: list[str] = field(default_factory=list)


class GitHubClient:
    def __init__(self, token: str):
        self._token = token
        self._github = Github(token)

    def get_pr_diff(self, repo_full_name: str, pr_number: int) -> str:
        import httpx
        repo = self._github.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        response = httpx.get(
            pr.diff_url,
            headers={"Authorization": f"token {self._token}"},
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text

    def parse_diff(self, diff_text: str) -> list[ParsedDiff]:
        patch_set = PatchSet(io.StringIO(diff_text))
        result = []
        for patched_file in patch_set:
            if patched_file.is_removed_file:
                continue
            added_lines: set[int] = set()
            hunk_parts: list[str] = []
            for hunk in patched_file:
                hunk_parts.append(str(hunk))
                for line in hunk:
                    if line.is_added and line.target_line_no:
                        added_lines.add(line.target_line_no)
            result.append(ParsedDiff(
                file_path=patched_file.path,
                hunk_text="\n".join(hunk_parts),
                added_lines=added_lines,
            ))
        return result

    def post_review(
        self,
        repo_full_name: str,
        pr_number: int,
        summary: str,
        comments: list[ReviewComment],
    ) -> int:
        repo = self._github.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)

        gh_comments = [
            {
                "path": c.path,
                "line": c.line,
                "side": c.side,
                "body": f"**[{c.severity.upper()}]** {c.body}\n\n*Citations: {', '.join(c.citations)}*"
                        if c.citations else f"**[{c.severity.upper()}]** {c.body}",
            }
            for c in comments
        ]

        review = pr.create_review(
            body=summary,
            event="COMMENT",
            comments=gh_comments,
        )
        return review.id
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_github_client.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add github/client.py tests/test_github_client.py
git commit -m "feat: GitHubClient — parse PR diff (unidiff), post review with inline comments"
```

---

## Task 4: `reviewer/retriever.py`

**Files:**
- Create: `reviewer/retriever.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retriever.py
import pytest
from reviewer.retriever import reciprocal_rank_fusion, ScoredChunk


def test_rrf_scores_higher_for_top_ranked_in_multiple_lists():
    list_a = ["doc1", "doc2", "doc3"]
    list_b = ["doc2", "doc1", "doc4"]
    ranked = reciprocal_rank_fusion([list_a, list_b])
    scores = {doc_id: score for doc_id, score in ranked}

    # doc2 appears at rank 2 in list_a and rank 1 in list_b → higher combined score
    # doc1 appears at rank 1 in list_a and rank 2 in list_b → same combined score as doc2
    assert scores["doc1"] == scores["doc2"]
    assert scores["doc1"] > scores["doc3"]
    assert scores["doc1"] > scores["doc4"]


def test_rrf_doc_in_single_list_gets_lower_score_than_doc_in_two():
    list_a = ["doc1"]
    list_b = ["doc1", "doc2"]
    ranked = reciprocal_rank_fusion([list_a, list_b])
    scores = {doc_id: score for doc_id, score in ranked}
    assert scores["doc1"] > scores["doc2"]


def test_rrf_returns_sorted_descending():
    list_a = ["doc3", "doc1"]
    list_b = ["doc1", "doc2"]
    ranked = reciprocal_rank_fusion([list_a, list_b])
    ids = [doc_id for doc_id, _ in ranked]
    assert ids[0] == "doc1"  # doc1 in both lists, highest score


def test_scored_chunk_dataclass():
    chunk = ScoredChunk(
        id="abc",
        score=0.9,
        payload={"file_path": "src/a.py", "content": "def foo(): pass"},
    )
    assert chunk.id == "abc"
    assert chunk.score == 0.9
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'reviewer.retriever'`

- [ ] **Step 3: Write `reviewer/retriever.py`**

```python
import asyncio
from dataclasses import dataclass
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from config import settings

EMBEDDING_MODEL = "text-embedding-3-small"


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


async def _embed(text: str) -> list[float]:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


async def _hyde_query(diff_hunk: str) -> str:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a senior engineer. Given a code diff hunk, write a short Python code snippet (5-10 lines) that represents the kind of existing code that would be most relevant to review for this change.",
            },
            {"role": "user", "content": diff_hunk},
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


async def retrieve(
    qdrant: AsyncQdrantClient,
    diff_hunk: str,
    limit_per_collection: dict[str, int] | None = None,
) -> list[ScoredChunk]:
    if limit_per_collection is None:
        limit_per_collection = {"code_chunks": 5, "adr_docs": 3, "pr_history": 3}

    hyde_text = await _hyde_query(diff_hunk)
    hyde_vector = await _embed(hyde_text)
    original_vector = await _embed(diff_hunk)
    # Average HyDE and original vectors
    query_vector = [(h + o) / 2 for h, o in zip(hyde_vector, original_vector)]

    async def search_collection(name: str, limit: int) -> list:
        return await qdrant.search(
            collection_name=name, query_vector=query_vector, limit=limit
        )

    results = await asyncio.gather(
        *[
            search_collection(name, limit)
            for name, limit in limit_per_collection.items()
        ]
    )

    # Build ID → payload map and ranked lists for RRF
    id_to_payload: dict[str, dict] = {}
    ranked_lists: list[list[str]] = []
    for hits in results:
        ranked = []
        for hit in hits:
            hit_id = str(hit.id)
            id_to_payload[hit_id] = hit.payload or {}
            ranked.append(hit_id)
        ranked_lists.append(ranked)

    merged = reciprocal_rank_fusion(ranked_lists)
    return [
        ScoredChunk(id=doc_id, score=score, payload=id_to_payload.get(doc_id, {}))
        for doc_id, score in merged
    ]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_retriever.py -v
```

Expected: 4 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add reviewer/__init__.py reviewer/retriever.py tests/test_retriever.py
git commit -m "feat: retriever — HyDE expansion, parallel Qdrant search, RRF merge"
```

---

## Task 5: `reviewer/reranker.py`

**Files:**
- Create: `reviewer/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reranker.py
import pytest
from unittest.mock import patch, MagicMock
from reviewer.reranker import Reranker
from reviewer.retriever import ScoredChunk


def _make_chunks(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(id=f"id{i}", score=0.5, payload={"content": f"code snippet {i}"})
        for i in range(n)
    ]


def test_reranker_returns_top_k():
    with patch("reviewer.reranker.CrossEncoder") as MockCE:
        scores = [0.1, 0.9, 0.3, 0.8, 0.5]
        MockCE.return_value.predict = MagicMock(return_value=scores)
        reranker = Reranker()
        chunks = _make_chunks(5)
        result = reranker.rerank(query="how does auth work", chunks=chunks, top_k=3)
    assert len(result) == 3


def test_reranker_returns_highest_scoring_chunks():
    with patch("reviewer.reranker.CrossEncoder") as MockCE:
        scores = [0.1, 0.9, 0.3, 0.8, 0.5]
        MockCE.return_value.predict = MagicMock(return_value=scores)
        reranker = Reranker()
        chunks = _make_chunks(5)
        result = reranker.rerank(query="how does auth work", chunks=chunks, top_k=2)
    # Top 2 should be id1 (0.9) and id3 (0.8)
    ids = {r.id for r in result}
    assert ids == {"id1", "id3"}


def test_reranker_handles_fewer_chunks_than_top_k():
    with patch("reviewer.reranker.CrossEncoder") as MockCE:
        MockCE.return_value.predict = MagicMock(return_value=[0.5, 0.8])
        reranker = Reranker()
        chunks = _make_chunks(2)
        result = reranker.rerank(query="query", chunks=chunks, top_k=10)
    assert len(result) == 2
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_reranker.py -v
```

Expected: `ModuleNotFoundError: No module named 'reviewer.reranker'`

- [ ] **Step 3: Write `reviewer/reranker.py`**

```python
from sentence_transformers import CrossEncoder
from reviewer.retriever import ScoredChunk

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


class Reranker:
    def __init__(self):
        self.model = _get_model()

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int = 8) -> list[ScoredChunk]:
        if not chunks:
            return []
        pairs = [[query, c.payload.get("content", "")] for c in chunks]
        scores = self.model.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:top_k]]
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_reranker.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add reviewer/reranker.py tests/test_reranker.py
git commit -m "feat: cross-encoder reranker (ms-marco-MiniLM-L-6-v2), lazy model load"
```

---

## Task 6: `reviewer/generator.py`

**Files:**
- Create: `reviewer/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from reviewer.generator import generate_review, ReviewOutput
from reviewer.retriever import ScoredChunk
from github.client import ParsedDiff


DIFF = ParsedDiff(
    file_path="src/auth.py",
    hunk_text="@@ -1,3 +1,5 @@\n def auth(token):\n+    if not token:\n+        raise ValueError()\n     return token",
    added_lines={2, 3},
)

CHUNKS = [
    ScoredChunk(
        id="c1",
        score=0.9,
        payload={"file_path": "src/utils.py", "content": "def validate(x): return bool(x)", "start_line": 1, "end_line": 1},
    )
]

VALID_LLM_OUTPUT = json.dumps({
    "summary": "Minor issues found.",
    "comments": [
        {
            "path": "src/auth.py",
            "line": 2,
            "side": "RIGHT",
            "body": "Consider logging before raising.",
            "severity": "suggestion",
            "citations": ["code_chunks:src/utils.py:1-1"],
        }
    ],
})


@pytest.mark.asyncio
async def test_generate_review_returns_review_output():
    with patch("reviewer.generator.AsyncOpenAI") as MockOpenAI:
        mock_msg = MagicMock()
        mock_msg.content = VALID_LLM_OUTPUT
        MockOpenAI.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=mock_msg)])
        )
        result = await generate_review(diffs=[DIFF], chunks=CHUNKS)

    assert isinstance(result, ReviewOutput)
    assert result.summary == "Minor issues found."
    assert len(result.comments) == 1
    assert result.comments[0].path == "src/auth.py"
    assert result.comments[0].line == 2


@pytest.mark.asyncio
async def test_generate_review_filters_comments_not_in_diff():
    bad_output = json.dumps({
        "summary": "Issues found.",
        "comments": [
            # line 99 is NOT in added_lines {2, 3}
            {"path": "src/auth.py", "line": 99, "side": "RIGHT",
             "body": "bad", "severity": "error", "citations": []},
            # line 2 IS in added_lines
            {"path": "src/auth.py", "line": 2, "side": "RIGHT",
             "body": "good", "severity": "warning", "citations": []},
        ],
    })
    with patch("reviewer.generator.AsyncOpenAI") as MockOpenAI:
        mock_msg = MagicMock()
        mock_msg.content = bad_output
        MockOpenAI.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=mock_msg)])
        )
        result = await generate_review(diffs=[DIFF], chunks=CHUNKS)

    assert len(result.comments) == 1
    assert result.comments[0].body == "good"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_generator.py -v
```

Expected: `ModuleNotFoundError: No module named 'reviewer.generator'`

- [ ] **Step 3: Write `reviewer/generator.py`**

```python
from dataclasses import dataclass, field
import json
from openai import AsyncOpenAI
from github.client import ParsedDiff, ReviewComment
from reviewer.retriever import ScoredChunk
from config import settings

GENERATION_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
You will be given a PR diff and relevant context chunks retrieved from the codebase.
Respond ONLY with valid JSON matching this schema:
{
  "summary": "string — overall review summary",
  "comments": [
    {
      "path": "string — file path",
      "line": integer — line number in new file,
      "side": "RIGHT",
      "body": "string — specific, actionable comment grounded in the context",
      "severity": "error|warning|suggestion",
      "citations": ["collection:file_path:lines", ...]
    }
  ]
}
Only comment on lines that are part of the diff. Ground every comment in the provided context."""


@dataclass
class ReviewOutput:
    summary: str
    comments: list[ReviewComment]


async def generate_review(
    diffs: list[ParsedDiff],
    chunks: list[ScoredChunk],
) -> ReviewOutput:
    diff_text = "\n\n".join(
        f"### {d.file_path}\n{d.hunk_text}" for d in diffs
    )
    context_text = "\n\n".join(
        f"### {c.payload.get('file_path', '')}:{c.payload.get('start_line',0)}-{c.payload.get('end_line',0)}\n{c.payload.get('content', '')}"
        for c in chunks
    )
    user_message = f"## Diff\n{diff_text}\n\n## Context\n{context_text}"

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=GENERATION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = json.loads(response.choices[0].message.content)

    # Build allowed lines set from diffs for filtering
    allowed: dict[str, set[int]] = {d.file_path: d.added_lines for d in diffs}

    comments: list[ReviewComment] = []
    for c in raw.get("comments", []):
        path = c.get("path", "")
        line = c.get("line", 0)
        if line not in allowed.get(path, set()):
            continue
        comments.append(ReviewComment(
            path=path,
            line=line,
            side=c.get("side", "RIGHT"),
            body=c.get("body", ""),
            severity=c.get("severity", "suggestion"),
            citations=c.get("citations", []),
        ))

    return ReviewOutput(summary=raw.get("summary", ""), comments=comments)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_generator.py -v
```

Expected: 2 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add reviewer/generator.py tests/test_generator.py
git commit -m "feat: GPT-4o review generator — JSON output, filters comments to diff lines only"
```

---

## Task 7: `reviewer/pipeline.py` + `reviewer/tasks.py`

**Files:**
- Create: `reviewer/pipeline.py`
- Create: `reviewer/tasks.py`
- Modify: `worker.py` — add `reviewer.tasks` to `include`
- Create: `tests/test_reviewer_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reviewer_pipeline.py
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from github.client import ParsedDiff, ReviewComment
from reviewer.retriever import ScoredChunk
from reviewer.generator import ReviewOutput


REPO_FULL_NAME = "owner/repo"
PR_NUMBER = 7
INSTALLATION_ID = 99

DIFF = ParsedDiff(
    file_path="src/auth.py",
    hunk_text="@@ -1 +1,3 @@\n+def auth(): pass",
    added_lines={1},
)

CHUNKS = [ScoredChunk(id="c1", score=0.9, payload={"content": "x"})]

REVIEW = ReviewOutput(
    summary="Looks good.",
    comments=[
        ReviewComment(path="src/auth.py", line=1, side="RIGHT",
                      body="Consider typing.", severity="suggestion", citations=[])
    ],
)


@pytest.mark.asyncio
async def test_run_review_posts_to_github():
    mock_qdrant = MagicMock()
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()

    with patch("reviewer.pipeline.get_installation_token", return_value="tok"), \
         patch("reviewer.pipeline.GitHubClient") as MockClient, \
         patch("reviewer.pipeline.retrieve", AsyncMock(return_value=CHUNKS)), \
         patch("reviewer.pipeline.Reranker") as MockReranker, \
         patch("reviewer.pipeline.generate_review", AsyncMock(return_value=REVIEW)), \
         patch("reviewer.pipeline.AsyncSessionLocal") as MockSession:
        MockClient.return_value.get_pr_diff.return_value = "diff text"
        MockClient.return_value.parse_diff.return_value = [DIFF]
        MockClient.return_value.post_review.return_value = 12345
        MockReranker.return_value.rerank.return_value = CHUNKS
        MockSession.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

        from reviewer.pipeline import run_review
        await run_review(
            qdrant=mock_qdrant,
            repo_full_name=REPO_FULL_NAME,
            pr_number=PR_NUMBER,
            installation_id=INSTALLATION_ID,
        )

    MockClient.return_value.post_review.assert_called_once()


@pytest.mark.asyncio
async def test_run_review_persists_pr_review_row():
    mock_db = MagicMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_qdrant = MagicMock()

    with patch("reviewer.pipeline.get_installation_token", return_value="tok"), \
         patch("reviewer.pipeline.GitHubClient") as MockClient, \
         patch("reviewer.pipeline.retrieve", AsyncMock(return_value=CHUNKS)), \
         patch("reviewer.pipeline.Reranker") as MockReranker, \
         patch("reviewer.pipeline.generate_review", AsyncMock(return_value=REVIEW)), \
         patch("reviewer.pipeline.AsyncSessionLocal") as MockSession:
        MockClient.return_value.get_pr_diff.return_value = "diff"
        MockClient.return_value.parse_diff.return_value = [DIFF]
        MockClient.return_value.post_review.return_value = 12345
        MockReranker.return_value.rerank.return_value = CHUNKS
        MockSession.return_value.__aenter__ = AsyncMock(return_value=mock_db)
        MockSession.return_value.__aexit__ = AsyncMock(return_value=False)

        from reviewer.pipeline import run_review
        await run_review(
            qdrant=mock_qdrant,
            repo_full_name=REPO_FULL_NAME,
            pr_number=PR_NUMBER,
            installation_id=INSTALLATION_ID,
        )

    mock_db.execute.assert_awaited()
    mock_db.commit.assert_awaited()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_reviewer_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'reviewer.pipeline'`

- [ ] **Step 3: Write `reviewer/pipeline.py`**

```python
from sqlalchemy.dialects.postgresql import insert as pg_insert
from qdrant_client import AsyncQdrantClient
from db.models import PRReview
from db.session import AsyncSessionLocal
from github.auth import get_installation_token
from github.client import GitHubClient
from reviewer.retriever import retrieve
from reviewer.reranker import Reranker
from reviewer.generator import generate_review


async def run_review(
    qdrant: AsyncQdrantClient,
    repo_full_name: str,
    pr_number: int,
    installation_id: int,
) -> None:
    token = get_installation_token(installation_id)
    gh = GitHubClient(token=token)

    diff_text = gh.get_pr_diff(repo_full_name, pr_number)
    diffs = gh.parse_diff(diff_text)
    if not diffs:
        return

    # Retrieve context for up to 10 files by change size
    diffs_for_retrieval = sorted(diffs, key=lambda d: len(d.added_lines), reverse=True)[:10]

    all_chunks = []
    for diff in diffs_for_retrieval:
        chunks = await retrieve(qdrant=qdrant, diff_hunk=diff.hunk_text)
        all_chunks.extend(chunks)

    reranker = Reranker()
    top_chunks = reranker.rerank(
        query="\n".join(d.hunk_text for d in diffs_for_retrieval),
        chunks=all_chunks,
        top_k=8,
    )

    review_output = await generate_review(diffs=diffs, chunks=top_chunks)

    github_review_id = gh.post_review(
        repo_full_name=repo_full_name,
        pr_number=pr_number,
        summary=review_output.summary,
        comments=review_output.comments,
    )

    async with AsyncSessionLocal() as db:
        stmt = pg_insert(PRReview).values(
            repo_id=None,  # resolved in Phase 4 with repo lookup
            pr_number=pr_number,
            github_review_id=github_review_id,
            status="posted",
            raw_llm_output={
                "summary": review_output.summary,
                "comments": [
                    {"path": c.path, "line": c.line, "body": c.body, "severity": c.severity}
                    for c in review_output.comments
                ],
            },
        )
        await db.execute(stmt)
        await db.commit()
```

- [ ] **Step 4: Write `reviewer/tasks.py`**

```python
import asyncio
from qdrant_client import AsyncQdrantClient
from worker import celery_app
from config import settings
from reviewer.pipeline import run_review


def _make_qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


@celery_app.task(name="review_pr", bind=True, max_retries=3, default_retry_delay=60)
def review_pr(self, repo_full_name: str, pr_number: int, installation_id: int):
    try:
        qdrant = _make_qdrant()
        asyncio.run(run_review(qdrant, repo_full_name, pr_number, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)
```

- [ ] **Step 5: Add `reviewer.tasks` to `worker.py` include list**

```python
# worker.py — update include list
celery_app = Celery(
    "rag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["indexer.tasks", "reviewer.tasks"],  # added reviewer.tasks
)
```

- [ ] **Step 6: Run tests — verify they pass**

```bash
pytest tests/test_reviewer_pipeline.py -v
```

Expected: 2 `PASSED`

- [ ] **Step 7: Commit**

```bash
git add reviewer/pipeline.py reviewer/tasks.py worker.py tests/test_reviewer_pipeline.py
git commit -m "feat: reviewer pipeline — retrieve, rerank, generate, post GitHub Review"
```

---

## Task 8: Webhook Route + Event Handlers

**Files:**
- Create: `api/handlers/__init__.py`
- Create: `api/handlers/indexing.py`
- Create: `api/handlers/review.py`
- Create: `api/routes/webhooks.py`
- Modify: `api/main.py` — register webhook router
- Create: `tests/test_webhooks_route.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_webhooks_route.py
import hashlib
import hmac
import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock


WEBHOOK_SECRET = "test_secret"


def _sign(payload: bytes) -> str:
    digest = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.mark.asyncio
async def test_webhook_rejects_invalid_signature():
    with patch("api.main.AsyncQdrantClient"), patch("api.main.run_migrations"):
        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/webhook/github",
                content=b'{"action": "opened"}',
                headers={"X-GitHub-Event": "pull_request", "X-Hub-Signature-256": "sha256=bad"},
            )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_webhook_pull_request_opened_enqueues_review_pr():
    payload = json.dumps({
        "action": "opened",
        "number": 5,
        "repository": {"full_name": "owner/repo", "id": 123},
        "installation": {"id": 99},
        "pull_request": {"title": "My PR"},
    }).encode()

    with patch("api.main.AsyncQdrantClient"), patch("api.main.run_migrations"), \
         patch("api.handlers.review.review_pr") as mock_task, \
         patch("api.routes.webhooks.settings") as mock_settings:
        mock_settings.github_webhook_secret = WEBHOOK_SECRET
        mock_task.delay = MagicMock()
        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/webhook/github",
                content=payload,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": _sign(payload),
                },
            )
    assert resp.status_code == 200
    mock_task.delay.assert_called_once_with("owner/repo", 5, 99)


@pytest.mark.asyncio
async def test_webhook_push_enqueues_incremental_index():
    payload = json.dumps({
        "ref": "refs/heads/main",
        "repository": {"full_name": "owner/repo", "id": 123},
        "installation": {"id": 99},
        "commits": [{"added": [], "modified": ["src/a.py"], "removed": []}],
    }).encode()

    with patch("api.main.AsyncQdrantClient"), patch("api.main.run_migrations"), \
         patch("api.handlers.indexing.incremental_index") as mock_task, \
         patch("api.routes.webhooks.settings") as mock_settings:
        mock_settings.github_webhook_secret = WEBHOOK_SECRET
        mock_task.delay = MagicMock()
        from api.main import app
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/webhook/github",
                content=payload,
                headers={
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": _sign(payload),
                },
            )
    assert resp.status_code == 200
    mock_task.delay.assert_called_once()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_webhooks_route.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write `api/handlers/indexing.py`**

```python
from github.webhook import WebhookEvent
from indexer.tasks import incremental_index


def handle_push(event: WebhookEvent) -> None:
    if not event.changed_files:
        return
    incremental_index.delay(
        event.repo_full_name,
        event.changed_files,
        event.installation_id,
    )
```

- [ ] **Step 4: Write `api/handlers/review.py`**

```python
from github.webhook import WebhookEvent
from reviewer.tasks import review_pr


def handle_pull_request(event: WebhookEvent) -> None:
    if event.action not in ("opened", "reopened"):
        return
    review_pr.delay(
        event.repo_full_name,
        event.pr_number,
        event.installation_id,
    )
```

- [ ] **Step 5: Write `api/routes/webhooks.py`**

```python
from fastapi import APIRouter, Request, HTTPException
from github.webhook import verify_signature, parse_event
from api.handlers.indexing import handle_push
from api.handlers.review import handle_pull_request
from config import settings

router = APIRouter()


@router.post("/webhook/github")
async def github_webhook(request: Request):
    payload = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")
    event_type = request.headers.get("X-GitHub-Event", "")

    if not verify_signature(payload, signature, settings.github_webhook_secret):
        raise HTTPException(status_code=401, detail="Invalid signature")

    event = parse_event(event_type, payload)

    if event_type == "push":
        handle_push(event)
    elif event_type == "pull_request":
        handle_pull_request(event)

    return {"status": "accepted"}
```

- [ ] **Step 6: Register webhook router in `api/main.py`**

Add:
```python
from api.routes.webhooks import router as webhook_router
# ...
app.include_router(webhook_router)
```

- [ ] **Step 7: Run tests — verify they pass**

```bash
pytest tests/test_webhooks_route.py -v
```

Expected: 3 `PASSED`

- [ ] **Step 8: Run all tests**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 9: Set up ngrok and test end-to-end with a real PR**

```bash
# Terminal 1 — start all services
docker compose up --build

# Terminal 2 — expose local port to internet
ngrok http 8000
# Copy the https URL e.g. https://abc123.ngrok.io
```

In GitHub App settings → Webhook URL: `https://abc123.ngrok.io/webhook/github`

Open a real PR on the installed repo. Within 30 seconds the bot should post inline review comments.

- [ ] **Step 10: Commit**

```bash
git add api/handlers/ api/routes/webhooks.py api/main.py tests/test_webhooks_route.py
git commit -m "feat: webhook route — HMAC verify, route pull_request and push to Celery tasks"
```

---

**Phase 3 complete.**

Acceptance criteria met:
- Opening a real PR → bot posts inline review comments with citations within 30 seconds
- Webhook signature verification rejects tampered requests
- All unit tests pass
