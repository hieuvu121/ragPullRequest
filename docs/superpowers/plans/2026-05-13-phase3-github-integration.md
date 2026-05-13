# Phase 3: GitHub Integration (Webhook + Bot) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 2 complete — `docker compose up` starts all services, `POST /index` enqueues a Celery task, `POST /search` returns relevant chunks.

**Goal:** Wire GitHub App webhooks so that pushing code triggers incremental indexing and opening a PR triggers an automatic AI review posted as a GitHub PR review with inline comments.

**Architecture:** A single webhook endpoint (`POST /webhook/github`) verifies HMAC-SHA256, parses the event, and dispatches to the appropriate Celery task. `github/auth.py` manages JWT → installation access token rotation. `github/client.py` wraps PyGitHub for diff fetching and review posting. The Phase 1 `review_pipeline` is called inside the `review_pr` Celery task.

**Tech Stack:** PyGitHub, PyJWT (RS256), unidiff, cryptography, ngrok (dev), pytest + pytest-asyncio + respx

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add PyGitHub, PyJWT, unidiff, cryptography |
| `.env` | Modify | Add `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY_B64`, `GITHUB_WEBHOOK_SECRET` |
| `config.py` | Modify | Add GitHub env vars + base64 private key decode |
| `github/__init__.py` | Create | Package marker |
| `github/auth.py` | Create | JWT mint, installation token fetch + in-memory cache |
| `github/events.py` | Create | HMAC-SHA256 verify, event payload parse |
| `github/client.py` | Create | `GithubClient`: `get_diff()`, `post_review()` |
| `api/handlers/__init__.py` | Create | Package marker |
| `api/handlers/indexing.py` | Create | `handle_push()` → enqueue `incremental_index` |
| `api/handlers/review.py` | Create | `handle_pr_opened()` → enqueue `review_pr` |
| `api/handlers/feedback.py` | Create | `handle_review_comment()` → enqueue `record_feedback` |
| `api/routes/webhooks.py` | Create | `POST /webhook/github` route |
| `api/main.py` | Modify | Register webhook router |
| `indexer/tasks.py` | Modify | Add `incremental_index` + `review_pr` + `record_feedback` tasks |
| `tests/test_github_auth.py` | Create | JWT fields, token caching, refresh before expiry |
| `tests/test_github_events.py` | Create | HMAC verification pass/fail, event routing |
| `tests/test_github_client.py` | Create | `get_diff()` + `post_review()` (mocked GitHub API) |
| `tests/test_webhook_route.py` | Create | Full webhook POST — valid signature, wrong signature, unknown event |

---

## Task 1: Add GitHub Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env`

- [ ] **Step 1: Add Phase 3 dependencies to `pyproject.toml`**

Under `[tool.poetry.dependencies]`:
```toml
PyGithub = "^2.3.0"
PyJWT = {extras = ["crypto"], version = "^2.8.0"}
cryptography = "^42.0.5"
unidiff = "^0.7.5"
```

```bash
poetry install
```

- [ ] **Step 2: Extend `.env`**

```
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY_B64=<base64-encoded PEM here>
GITHUB_WEBHOOK_SECRET=your-webhook-secret
```

To base64-encode your PEM for local dev:
```bash
base64 -i your-private-key.pem | tr -d '\n'
```

- [ ] **Step 3: Update `config.py` to decode the private key**

```python
import base64
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field

class Settings(BaseSettings):
    openai_api_key: str
    qdrant_url: str
    database_url: str
    redis_url: str
    github_app_id: int
    github_app_private_key_b64: str
    github_webhook_secret: str
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @computed_field
    @property
    def github_private_key_pem(self) -> str:
        return base64.b64decode(self.github_app_private_key_b64).decode()

    model_config = {"env_file": ".env"}

settings = Settings()
```

- [ ] **Step 4: Verify settings load**

```bash
python -c "from config import settings; print(settings.github_app_id)"
```

Expected: prints the app ID integer from `.env`.

---

## Task 2: GitHub Auth (`github/auth.py`)

**Files:**
- Create: `github/__init__.py`
- Create: `github/auth.py`
- Create: `tests/test_github_auth.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_github_auth.py`:
```python
import time
import pytest
from unittest.mock import patch, MagicMock
from github.auth import GitHubAuth


@pytest.fixture
def auth(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return GitHubAuth(app_id=12345, private_key_pem=pem, installation_id=99)


def test_jwt_fields(auth):
    import jwt
    token = auth._mint_jwt()
    # decode without verification to inspect claims
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["iss"] == "12345"
    assert claims["exp"] - claims["iat"] <= 600  # max 10 minutes


def test_token_cached(auth):
    fake_token = {"token": "ghs_abc", "expires_at": "2099-01-01T00:00:00Z"}
    with patch.object(auth, "_fetch_installation_token", return_value=fake_token) as mock:
        t1 = auth.get_installation_token()
        t2 = auth.get_installation_token()
    mock.assert_called_once()
    assert t1 == t2 == "ghs_abc"


def test_token_refreshed_before_expiry(auth):
    import datetime
    # Set cached token that expires in 4 minutes (within 5-min buffer)
    soon = (datetime.datetime.utcnow() + datetime.timedelta(minutes=4)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    auth._cached_token = {"token": "old_token", "expires_at": soon}
    new_token = {"token": "new_token", "expires_at": "2099-01-01T00:00:00Z"}
    with patch.object(auth, "_fetch_installation_token", return_value=new_token):
        result = auth.get_installation_token()
    assert result == "new_token"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_github_auth.py -v
```

Expected: `ImportError` — `github.auth` does not exist yet.

- [ ] **Step 3: Create `github/__init__.py`**

```python
```
(empty file)

- [ ] **Step 4: Create `github/auth.py`**

```python
import time
import datetime
import requests
import jwt
from config import settings


class GitHubAuth:
    def __init__(self, app_id: int, private_key_pem: str, installation_id: int):
        self.app_id = app_id
        self.private_key_pem = private_key_pem
        self.installation_id = installation_id
        self._cached_token: dict | None = None

    def _mint_jwt(self) -> str:
        now = int(time.time())
        payload = {"iat": now - 60, "exp": now + 540, "iss": str(self.app_id)}
        return jwt.encode(payload, self.private_key_pem, algorithm="RS256")

    def _fetch_installation_token(self) -> dict:
        token = self._mint_jwt()
        resp = requests.post(
            f"https://api.github.com/app/installations/{self.installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def _token_expires_soon(self) -> bool:
        if not self._cached_token:
            return True
        expires_at = datetime.datetime.strptime(
            self._cached_token["expires_at"], "%Y-%m-%dT%H:%M:%SZ"
        )
        return (expires_at - datetime.datetime.utcnow()).total_seconds() < 300

    def get_installation_token(self) -> str:
        if self._token_expires_soon():
            self._cached_token = self._fetch_installation_token()
        return self._cached_token["token"]


def make_auth(installation_id: int) -> GitHubAuth:
    return GitHubAuth(
        app_id=settings.github_app_id,
        private_key_pem=settings.github_private_key_pem,
        installation_id=installation_id,
    )
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_github_auth.py -v
```

Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add github/__init__.py github/auth.py tests/test_github_auth.py config.py
git commit -m "feat: github app JWT auth with installation token caching"
```

---

## Task 2: Webhook Verification + Event Parsing (`github/events.py`)

**Files:**
- Create: `github/events.py`
- Create: `tests/test_github_events.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_github_events.py`:
```python
import hashlib
import hmac
import json
import pytest
from github.events import verify_signature, parse_event, WebhookEvent

SECRET = "test-secret"


def _sign(body: bytes, secret: str = SECRET) -> str:
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def test_valid_signature():
    body = b'{"action": "opened"}'
    sig = _sign(body)
    assert verify_signature(body, sig, SECRET) is True


def test_invalid_signature():
    body = b'{"action": "opened"}'
    assert verify_signature(body, "sha256=deadbeef", SECRET) is False


def test_missing_prefix():
    body = b'{"action": "opened"}'
    raw_hex = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    assert verify_signature(body, raw_hex, SECRET) is False


def test_parse_push_event():
    payload = {"ref": "refs/heads/main", "commits": [{"added": ["a.py"], "modified": ["b.py"], "removed": []}]}
    event = parse_event("push", payload)
    assert event.event_type == "push"
    assert event.action is None
    assert "a.py" in event.changed_files
    assert "b.py" in event.changed_files


def test_parse_pr_opened():
    payload = {"action": "opened", "number": 42, "pull_request": {"title": "Fix bug"}, "installation": {"id": 99}}
    event = parse_event("pull_request", payload)
    assert event.event_type == "pull_request"
    assert event.action == "opened"
    assert event.pr_number == 42
    assert event.installation_id == 99


def test_parse_unknown_event():
    event = parse_event("ping", {"zen": "Keep it simple"})
    assert event.event_type == "ping"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_github_events.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `github/events.py`**

```python
import hashlib
import hmac
from dataclasses import dataclass, field


@dataclass
class WebhookEvent:
    event_type: str
    action: str | None = None
    pr_number: int | None = None
    installation_id: int | None = None
    repo_full_name: str | None = None
    changed_files: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def verify_signature(body: bytes, signature_header: str, secret: str) -> bool:
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature_header)


def parse_event(event_type: str, payload: dict) -> WebhookEvent:
    event = WebhookEvent(event_type=event_type, raw=payload)
    event.action = payload.get("action")
    event.installation_id = (payload.get("installation") or {}).get("id")
    repo = payload.get("repository") or {}
    event.repo_full_name = repo.get("full_name")

    if event_type == "push":
        files: set[str] = set()
        for commit in payload.get("commits", []):
            files.update(commit.get("added", []))
            files.update(commit.get("modified", []))
        event.changed_files = list(files)

    if event_type == "pull_request":
        event.pr_number = payload.get("number")

    return event
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_github_events.py -v
```

Expected: all 6 pass.

- [ ] **Step 5: Commit**

```bash
git add github/events.py tests/test_github_events.py
git commit -m "feat: webhook HMAC verification and event parsing"
```

---

## Task 3: GitHub Client (`github/client.py`)

**Files:**
- Create: `github/client.py`
- Create: `tests/test_github_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_github_client.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from github.client import GithubClient


@pytest.fixture
def client():
    return GithubClient(token="ghs_fake")


def test_get_diff_returns_string(client):
    mock_pr = MagicMock()
    mock_pr.get_files.return_value = []
    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr

    with patch("github.client.Github") as MockGithub:
        MockGithub.return_value.get_repo.return_value = mock_repo
        diff = client.get_diff("owner/repo", 1)
    assert isinstance(diff, str)


def test_get_diff_contains_file_changes(client):
    mock_file = MagicMock()
    mock_file.filename = "src/auth.py"
    mock_file.patch = "@@ -1,3 +1,5 @@\n-old\n+new"
    mock_pr = MagicMock()
    mock_pr.get_files.return_value = [mock_file]
    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr

    with patch("github.client.Github") as MockGithub:
        MockGithub.return_value.get_repo.return_value = mock_repo
        diff = client.get_diff("owner/repo", 1)
    assert "src/auth.py" in diff
    assert "+new" in diff


def test_post_review_calls_create_review(client):
    mock_pr = MagicMock()
    mock_repo = MagicMock()
    mock_repo.get_pull.return_value = mock_pr

    comments = [
        {"path": "src/auth.py", "line": 10, "side": "RIGHT", "body": "Consider null check"}
    ]

    with patch("github.client.Github") as MockGithub:
        MockGithub.return_value.get_repo.return_value = mock_repo
        review_id = client.post_review("owner/repo", 1, "Summary text", comments)

    mock_pr.create_review.assert_called_once()
    call_kwargs = mock_pr.create_review.call_args[1]
    assert call_kwargs["event"] == "COMMENT"
    assert call_kwargs["body"] == "Summary text"
    assert len(call_kwargs["comments"]) == 1
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_github_client.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Create `github/client.py`**

```python
from github import Github
from github.InputPullRequestReviewComment import InputPullRequestReviewComment


class GithubClient:
    def __init__(self, token: str):
        self.token = token

    def _gh(self):
        return Github(self.token)

    def get_diff(self, repo_full_name: str, pr_number: int) -> str:
        repo = self._gh().get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        lines: list[str] = []
        for f in pr.get_files():
            if f.patch:
                lines.append(f"--- a/{f.filename}\n+++ b/{f.filename}\n{f.patch}")
        return "\n".join(lines)

    def post_review(
        self,
        repo_full_name: str,
        pr_number: int,
        summary: str,
        comments: list[dict],
    ) -> int:
        repo = self._gh().get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)
        gh_comments = [
            InputPullRequestReviewComment(
                path=c["path"],
                position=None,
                body=c["body"],
                line=c["line"],
                side=c.get("side", "RIGHT"),
            )
            for c in comments
        ]
        review = pr.create_review(body=summary, event="COMMENT", comments=gh_comments)
        return review.id
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_github_client.py -v
```

Expected: all 3 pass.

- [ ] **Step 5: Commit**

```bash
git add github/client.py tests/test_github_client.py
git commit -m "feat: github client for diff fetching and review posting"
```

---

## Task 4: Event Handlers (`api/handlers/`)

**Files:**
- Create: `api/handlers/__init__.py`
- Create: `api/handlers/indexing.py`
- Create: `api/handlers/review.py`
- Create: `api/handlers/feedback.py`

- [ ] **Step 1: Create `api/handlers/__init__.py`**

```python
```
(empty)

- [ ] **Step 2: Create `api/handlers/indexing.py`**

```python
from github.events import WebhookEvent
from indexer.tasks import incremental_index


def handle_push(event: WebhookEvent) -> None:
    if not event.changed_files or not event.repo_full_name or not event.installation_id:
        return
    incremental_index.delay(
        repo_full_name=event.repo_full_name,
        installation_id=event.installation_id,
        changed_files=event.changed_files,
    )
```

- [ ] **Step 3: Create `api/handlers/review.py`**

```python
from github.events import WebhookEvent
from indexer.tasks import review_pr


def handle_pr_opened(event: WebhookEvent) -> None:
    if event.action != "opened":
        return
    if not event.pr_number or not event.repo_full_name or not event.installation_id:
        return
    review_pr.delay(
        repo_full_name=event.repo_full_name,
        pr_number=event.pr_number,
        installation_id=event.installation_id,
    )
```

- [ ] **Step 4: Create `api/handlers/feedback.py`**

```python
from github.events import WebhookEvent
from indexer.tasks import record_feedback


def handle_review_comment(event: WebhookEvent) -> None:
    if event.action not in ("dismissed", "resolved", "created"):
        return
    comment_id = (event.raw.get("comment") or {}).get("id")
    if not comment_id:
        return
    record_feedback.delay(
        comment_id=comment_id,
        action=event.action,
        raw=event.raw,
    )
```

- [ ] **Step 5: Commit**

```bash
git add api/handlers/
git commit -m "feat: event handlers dispatch push, pr_opened, review_comment to celery"
```

---

## Task 5: Celery Tasks for GitHub Events (`indexer/tasks.py`)

**Files:**
- Modify: `indexer/tasks.py`

- [ ] **Step 1: Add `incremental_index`, `review_pr`, `record_feedback` tasks**

Append to `indexer/tasks.py`:
```python
import asyncio
from worker import celery_app
from config import settings
from github.auth import make_auth
from github.client import GithubClient
from pipeline.chunker import chunk_file
from pipeline.embedder import Embedder
from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve
from pipeline.generator import generate_review
from db.session import AsyncSessionLocal
from db import models
from sqlalchemy import select, update
import tempfile, os


@celery_app.task(name="incremental_index", bind=True, max_retries=3)
def incremental_index(self, repo_full_name: str, installation_id: int, changed_files: list[str]):
    try:
        asyncio.run(_run_incremental(repo_full_name, installation_id, changed_files))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


async def _run_incremental(repo_full_name: str, installation_id: int, changed_files: list[str]):
    token = make_auth(installation_id).get_installation_token()
    from github import Github
    gh = Github(token)
    repo = gh.get_repo(repo_full_name)

    embedder = Embedder()
    store = QdrantStore(url=settings.qdrant_url)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(models.Repo).where(models.Repo.full_name == repo_full_name)
        )
        db_repo = result.scalar_one_or_none()
        if not db_repo:
            return

        for file_path in changed_files:
            if not file_path.endswith(".py"):
                continue
            try:
                content = repo.get_contents(file_path).decoded_content.decode()
                import hashlib
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                result = await session.execute(
                    select(models.IndexedFile).where(
                        models.IndexedFile.repo_id == db_repo.id,
                        models.IndexedFile.file_path == file_path,
                    )
                )
                existing = result.scalar_one_or_none()
                if existing and existing.content_hash == content_hash:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                    f.write(content)
                    tmp_path = f.name

                try:
                    chunks = chunk_file(tmp_path)
                finally:
                    os.unlink(tmp_path)

                for chunk in chunks:
                    chunk.file_path = file_path

                vectors = embedder.embed([c.content for c in chunks])
                store.delete_by_filter(str(db_repo.id), file_path)
                store.upsert(str(db_repo.id), chunks, vectors)

                if existing:
                    await session.execute(
                        update(models.IndexedFile)
                        .where(models.IndexedFile.id == existing.id)
                        .values(content_hash=content_hash, status="indexed", chunk_count=len(chunks), retry_count=0)
                    )
                else:
                    session.add(models.IndexedFile(
                        repo_id=db_repo.id,
                        file_path=file_path,
                        content_hash=content_hash,
                        status="indexed",
                        chunk_count=len(chunks),
                    ))
            except Exception:
                import logging
                logging.exception("Failed to index %s", file_path)

        await session.commit()


@celery_app.task(name="review_pr", bind=True, max_retries=3)
def review_pr(self, repo_full_name: str, pr_number: int, installation_id: int):
    try:
        asyncio.run(_run_review(repo_full_name, pr_number, installation_id))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


async def _run_review(repo_full_name: str, pr_number: int, installation_id: int):
    import time
    start = time.monotonic()

    token = make_auth(installation_id).get_installation_token()
    client = GithubClient(token=token)
    diff = client.get_diff(repo_full_name, pr_number)

    store = QdrantStore(url=settings.qdrant_url)
    embedder = Embedder()

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(models.Repo).where(models.Repo.full_name == repo_full_name)
        )
        db_repo = result.scalar_one_or_none()
        if not db_repo:
            return

        pr_review = models.PRReview(
            repo_id=db_repo.id,
            pr_number=pr_number,
            status="pending",
        )
        session.add(pr_review)
        await session.flush()

        try:
            chunks = retrieve(diff, store, embedder)
            comments = generate_review(diff, chunks)

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

            import json
            raw_output = [c.__dict__ for c in comments]
            review_id = client.post_review(repo_full_name, pr_number, summary, gh_comments)

            pr_review.status = "posted"
            pr_review.github_review_id = review_id
            pr_review.raw_output = raw_output
            pr_review.latency_ms = int((time.monotonic() - start) * 1000)
        except Exception:
            pr_review.status = "failed"
            raise

        await session.commit()


@celery_app.task(name="record_feedback", bind=True, max_retries=3)
def record_feedback(self, comment_id: int, action: str, raw: dict):
    try:
        asyncio.run(_run_feedback(comment_id, action, raw))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)


async def _run_feedback(comment_id: int, action: str, raw: dict):
    score_map = {"resolved": 1.0, "dismissed": -1.0, "created": 0.0}
    value = score_map.get(action, 0.0)

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(models.PRReview).where(
                models.PRReview.github_review_id == raw.get("pull_request_review_id")
            )
        )
        pr_review = result.scalar_one_or_none()
        if not pr_review:
            return

        session.add(models.ReviewFeedback(
            pr_review_id=pr_review.id,
            comment_id=comment_id,
            action=action,
            value=value,
        ))
        await session.commit()
```

- [ ] **Step 2: Commit**

```bash
git add indexer/tasks.py
git commit -m "feat: celery tasks for incremental_index, review_pr, record_feedback"
```

---

## Task 6: Webhook Route (`api/routes/webhooks.py`)

**Files:**
- Create: `api/routes/webhooks.py`
- Modify: `api/main.py`
- Create: `tests/test_webhook_route.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_webhook_route.py`:
```python
import hashlib
import hmac
import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from api.main import app

SECRET = "test-webhook-secret"


def _sign(body: bytes) -> str:
    sig = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


@pytest.fixture
def push_payload():
    return {
        "ref": "refs/heads/main",
        "commits": [{"added": ["new.py"], "modified": [], "removed": []}],
        "repository": {"full_name": "owner/repo"},
        "installation": {"id": 99},
    }


@pytest.mark.asyncio
async def test_valid_push_returns_202(push_payload):
    body = json.dumps(push_payload).encode()
    with patch("api.routes.webhooks.settings") as mock_settings:
        mock_settings.github_webhook_secret = SECRET
        with patch("api.routes.webhooks.handle_push"):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                resp = await ac.post(
                    "/webhook/github",
                    content=body,
                    headers={"X-GitHub-Event": "push", "X-Hub-Signature-256": _sign(body)},
                )
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_invalid_signature_returns_403(push_payload):
    body = json.dumps(push_payload).encode()
    with patch("api.routes.webhooks.settings") as mock_settings:
        mock_settings.github_webhook_secret = SECRET
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/webhook/github",
                content=body,
                headers={"X-GitHub-Event": "push", "X-Hub-Signature-256": "sha256=badhex"},
            )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_unknown_event_returns_204(push_payload):
    body = json.dumps({"zen": "Keep it simple"}).encode()
    with patch("api.routes.webhooks.settings") as mock_settings:
        mock_settings.github_webhook_secret = SECRET
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/webhook/github",
                content=body,
                headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": _sign(body)},
            )
    assert resp.status_code == 204
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_webhook_route.py -v
```

Expected: `ImportError` or 404.

- [ ] **Step 3: Create `api/routes/webhooks.py`**

```python
import json
from fastapi import APIRouter, Request, Response
from github.events import verify_signature, parse_event
from api.handlers.indexing import handle_push
from api.handlers.review import handle_pr_opened
from api.handlers.feedback import handle_review_comment
from config import settings

router = APIRouter()


@router.post("/webhook/github")
async def github_webhook(request: Request):
    body = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(body, sig, settings.github_webhook_secret):
        return Response(status_code=403)

    event_type = request.headers.get("X-GitHub-Event", "")
    payload = json.loads(body)
    event = parse_event(event_type, payload)

    if event_type == "push":
        handle_push(event)
    elif event_type == "pull_request":
        handle_pr_opened(event)
    elif event_type == "pull_request_review_comment":
        handle_review_comment(event)
    else:
        return Response(status_code=204)

    return Response(status_code=202)
```

- [ ] **Step 4: Register router in `api/main.py`**

Add to the `include_router` section of `api/main.py`:
```python
from api.routes.webhooks import router as webhook_router
app.include_router(webhook_router)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_webhook_route.py -v
```

Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add api/routes/webhooks.py api/main.py tests/test_webhook_route.py
git commit -m "feat: POST /webhook/github with HMAC verification and event dispatch"
```

---

## Task 7: Local Dev with ngrok

**Files:** none (setup only)

- [ ] **Step 1: Start all services**

```bash
docker compose up -d
```

Verify:
```bash
docker compose ps
```

All services (`api`, `worker`, `redis`, `qdrant`, `postgres`) should be `running`.

- [ ] **Step 2: Expose the API via ngrok**

```bash
ngrok http 8000
```

Copy the `Forwarding` HTTPS URL (e.g., `https://abc123.ngrok.io`).

- [ ] **Step 3: Configure the GitHub App webhook**

In your GitHub App settings (https://github.com/settings/apps):
1. Set **Webhook URL** to `https://abc123.ngrok.io/webhook/github`
2. Set **Webhook secret** to the value of `GITHUB_WEBHOOK_SECRET` in your `.env`
3. Subscribe to events: **Push**, **Pull request**, **Pull request review comment**

- [ ] **Step 4: Trigger a test push**

In a repo where the GitHub App is installed:
```bash
echo "# test" >> README.md
git add README.md && git commit -m "test webhook" && git push
```

Check the worker logs:
```bash
docker compose logs worker --tail=20
```

Expected: `Task incremental_index[...] received` followed by indexing log lines.

- [ ] **Step 5: Open a test PR and verify review is posted**

Create a branch, push a `.py` change, open a PR on GitHub. Within 30 seconds the bot should post a review.

```bash
docker compose logs worker --tail=40
```

Expected: `Task review_pr[...] succeeded`.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "chore: ngrok dev setup verified, github app webhook functional"
```

---

## Phase 3 Validation Checklist

Run the full test suite before marking Phase 3 complete:

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass with no skips.

Manual smoke test:
- [ ] Push a commit → worker log shows `incremental_index` received and succeeded
- [ ] Open a PR → GitHub shows an AI review comment within 30s
- [ ] Resolve a comment → worker log shows `record_feedback` succeeded
