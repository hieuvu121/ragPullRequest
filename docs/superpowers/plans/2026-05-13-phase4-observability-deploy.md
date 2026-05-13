# Phase 4: Observability + Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 3 complete — GitHub App webhook wired, push triggers indexing, PR open triggers review, bot posts inline comments.

**Goal:** Add Langfuse LLM tracing and human feedback scoring to every review, harden Docker Compose for production, deploy to Railway, and write a README with setup instructions.

**Architecture:** Langfuse wraps the review pipeline in `pipeline/generator.py` and `pipeline/retriever.py` with a single trace per `review_pr` task. Feedback scores are pushed to Langfuse when `record_feedback` runs. The production Compose file removes source volume mounts and adds `restart: always`. Railway is configured via environment variables; no code changes are needed for the deploy.

**Tech Stack:** Langfuse Python SDK, Railway CLI, Docker Compose (production profile), pytest (Langfuse mocked)

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `langfuse` SDK |
| `.env` | Modify | Add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| `config.py` | Modify | Add Langfuse settings |
| `pipeline/retriever.py` | Modify | Accept optional `trace` arg; log retrieval + reranking spans |
| `pipeline/generator.py` | Modify | Accept optional `trace` arg; log generation span |
| `indexer/tasks.py` | Modify | Create Langfuse trace in `_run_review`; pass to retriever + generator; save `langfuse_trace_id` |
| `db/models.py` | Modify | Add `langfuse_trace_id` to `PRReview`; add `ReviewFeedback.value` float column |
| `alembic/versions/` | Create | Migration adding `langfuse_trace_id` + `value` columns |
| `docker-compose.prod.yml` | Create | Production Compose — no volume mounts, restart: always |
| `README.md` | Create | Problem → architecture → tech stack → setup → deploy |
| `tests/test_tracing.py` | Create | Langfuse trace/span calls (mocked SDK) |
| `tests/test_feedback_score.py` | Create | `record_feedback` pushes score to Langfuse |

---

## Task 1: Add Langfuse Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env`
- Modify: `config.py`

- [ ] **Step 1: Add `langfuse` to `pyproject.toml`**

Under `[tool.poetry.dependencies]`:
```toml
langfuse = "^2.28.0"
```

```bash
poetry install
```

- [ ] **Step 2: Extend `.env`**

```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

- [ ] **Step 3: `config.py` already includes these fields**

Verify that `config.py` has:
```python
langfuse_public_key: str = ""
langfuse_secret_key: str = ""
langfuse_host: str = "https://cloud.langfuse.com"
```

(These were added in Phase 3, Task 1. If missing, add them now.)

- [ ] **Step 4: Verify import**

```bash
python -c "from langfuse import Langfuse; print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml poetry.lock .env
git commit -m "chore: add langfuse dependency"
```

---

## Task 2: Instrument `pipeline/retriever.py` and `pipeline/generator.py`

**Files:**
- Modify: `pipeline/retriever.py`
- Modify: `pipeline/generator.py`
- Create: `tests/test_tracing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tracing.py`:
```python
import pytest
from unittest.mock import MagicMock, patch, call


def _make_mock_trace():
    trace = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    trace.span.return_value = span
    return trace, span


def test_retrieve_logs_retrieval_span():
    trace, span = _make_mock_trace()

    mock_store = MagicMock()
    mock_store.search.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 1536]

    with patch("pipeline.retriever.openai_client") as mock_openai:
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hypothetical snippet"))]
        )
        from pipeline.retriever import retrieve
        retrieve("diff content", mock_store, mock_embedder, trace=trace)

    trace.span.assert_any_call(name="retrieval")


def test_generate_review_logs_generation_span():
    trace, span = _make_mock_trace()

    with patch("pipeline.generator.openai_client") as mock_openai:
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='[]'))]
        )
        from pipeline.generator import generate_review
        generate_review("diff", [], trace=trace)

    trace.span.assert_any_call(name="generation")


def test_retrieve_works_without_trace():
    mock_store = MagicMock()
    mock_store.search.return_value = []
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [[0.1] * 1536]

    with patch("pipeline.retriever.openai_client") as mock_openai:
        mock_openai.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hypothetical"))]
        )
        from pipeline.retriever import retrieve
        result = retrieve("diff", mock_store, mock_embedder, trace=None)

    assert isinstance(result, list)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_tracing.py -v
```

Expected: FAIL — `retrieve` does not accept a `trace` argument yet.

- [ ] **Step 3: Modify `pipeline/retriever.py` to accept and use `trace`**

Add `trace=None` parameter and span logging. Wrap the retrieval + reranking block:

```python
# In retrieve(), after existing imports, add typing import:
from typing import Any

# Change the function signature:
def retrieve(
    diff: str,
    store: "QdrantStore",
    embedder: "Embedder",
    trace: Any = None,
) -> list["Chunk"]:
    # --- HyDE expansion (unchanged) ---
    hyde_snippet = _hyde_expand(diff)
    hyde_vec = embedder.embed([hyde_snippet])[0]
    diff_vec = embedder.embed([diff])[0]
    query_vec = [(h + d) / 2 for h, d in zip(hyde_vec, diff_vec)]

    # --- Retrieval span ---
    if trace:
        span = trace.span(name="retrieval")
        span.update(input={"diff_length": len(diff)})

    results = store.search(query_vec, limit=20)

    if trace:
        span.update(output={"num_candidates": len(results)})
        span.end()

    # --- Reranking span ---
    if trace:
        rerank_span = trace.span(name="reranking")

    reranked = _rerank(diff, results)

    if trace:
        rerank_span.update(output={"num_kept": len(reranked)})
        rerank_span.end()

    return reranked
```

- [ ] **Step 4: Modify `pipeline/generator.py` to accept and use `trace`**

```python
from typing import Any

def generate_review(
    diff: str,
    chunks: list["Chunk"],
    trace: Any = None,
) -> list[ReviewComment]:
    prompt = _build_prompt(diff, chunks)

    if trace:
        gen_span = trace.span(name="generation")
        gen_span.update(input={"prompt_chars": len(prompt)})

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content

    if trace:
        gen_span.update(output={"response_chars": len(raw)})
        gen_span.end()

    return _parse_comments(raw, diff)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_tracing.py -v
```

Expected: all 3 pass.

- [ ] **Step 6: Commit**

```bash
git add pipeline/retriever.py pipeline/generator.py tests/test_tracing.py
git commit -m "feat: langfuse retrieval and generation spans in pipeline"
```

---

## Task 3: Create Langfuse Trace in `review_pr` Task

**Files:**
- Modify: `indexer/tasks.py`
- Modify: `db/models.py`

- [ ] **Step 1: Add `langfuse_trace_id` column to `PRReview`**

In `db/models.py`, add to the `PRReview` class:
```python
langfuse_trace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
```

- [ ] **Step 2: Generate and apply migration**

```bash
alembic revision --autogenerate -m "add_langfuse_trace_id_to_pr_reviews"
alembic upgrade head
```

Verify the migration file in `alembic/versions/` added `langfuse_trace_id` column.

- [ ] **Step 3: Modify `_run_review` in `indexer/tasks.py` to create a trace**

Replace the trace section inside `_run_review`:
```python
from config import settings
from langfuse import Langfuse

async def _run_review(repo_full_name: str, pr_number: int, installation_id: int):
    import time
    start = time.monotonic()

    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    trace = lf.trace(
        name="review_pr",
        metadata={"repo": repo_full_name, "pr_number": pr_number},
    )

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
            langfuse_trace_id=trace.id,
        )
        session.add(pr_review)
        await session.flush()

        try:
            chunks = retrieve(diff, store, embedder, trace=trace)
            comments = generate_review(diff, chunks, trace=trace)

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

            raw_output = [c.__dict__ for c in comments]
            review_id = client.post_review(repo_full_name, pr_number, summary, gh_comments)

            pr_review.status = "posted"
            pr_review.github_review_id = review_id
            pr_review.raw_output = raw_output
            pr_review.latency_ms = int((time.monotonic() - start) * 1000)

            trace.update(
                output={"comments": len(comments), "latency_ms": pr_review.latency_ms}
            )
        except Exception:
            pr_review.status = "failed"
            trace.update(metadata={"error": True})
            raise
        finally:
            lf.flush()

        await session.commit()
```

- [ ] **Step 4: Commit**

```bash
git add db/models.py alembic/versions/ indexer/tasks.py
git commit -m "feat: langfuse trace per review_pr with trace_id persisted to postgres"
```

---

## Task 4: Feedback Scores to Langfuse

**Files:**
- Modify: `indexer/tasks.py` (`_run_feedback`)
- Create: `tests/test_feedback_score.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feedback_score.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_feedback_pushes_score_to_langfuse():
    fake_review = MagicMock()
    fake_review.langfuse_trace_id = "trace-abc"
    fake_review.id = "review-uuid"

    with patch("indexer.tasks.AsyncSessionLocal") as mock_session_cls, \
         patch("indexer.tasks.Langfuse") as MockLangfuse:

        mock_lf = MagicMock()
        MockLangfuse.return_value = mock_lf

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=fake_review)))
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session_cls.return_value = mock_session

        from indexer.tasks import _run_feedback
        await _run_feedback(
            comment_id=42,
            action="resolved",
            raw={"pull_request_review_id": 999},
        )

    mock_lf.score.assert_called_once_with(
        trace_id="trace-abc",
        name="comment_quality",
        value=1.0,
    )
    mock_lf.flush.assert_called_once()


@pytest.mark.asyncio
async def test_dismissed_sends_negative_score():
    fake_review = MagicMock()
    fake_review.langfuse_trace_id = "trace-xyz"
    fake_review.id = "review-uuid-2"

    with patch("indexer.tasks.AsyncSessionLocal") as mock_session_cls, \
         patch("indexer.tasks.Langfuse") as MockLangfuse:

        mock_lf = MagicMock()
        MockLangfuse.return_value = mock_lf

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=fake_review)))
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session_cls.return_value = mock_session

        from indexer.tasks import _run_feedback
        await _run_feedback(
            comment_id=43,
            action="dismissed",
            raw={"pull_request_review_id": 888},
        )

    mock_lf.score.assert_called_once_with(
        trace_id="trace-xyz",
        name="comment_quality",
        value=-1.0,
    )
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_feedback_score.py -v
```

Expected: FAIL — `_run_feedback` doesn't call Langfuse yet.

- [ ] **Step 3: Update `_run_feedback` in `indexer/tasks.py`**

```python
async def _run_feedback(comment_id: int, action: str, raw: dict):
    score_map = {"resolved": 1.0, "dismissed": -1.0, "created": 0.0}
    value = score_map.get(action, 0.0)

    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )

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

    if pr_review.langfuse_trace_id:
        lf.score(
            trace_id=pr_review.langfuse_trace_id,
            name="comment_quality",
            value=value,
        )
        lf.flush()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_feedback_score.py -v
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add indexer/tasks.py tests/test_feedback_score.py
git commit -m "feat: feedback scores pushed to langfuse on resolved/dismissed events"
```

---

## Task 5: Production Docker Compose

**Files:**
- Create: `docker-compose.prod.yml`

- [ ] **Step 1: Create `docker-compose.prod.yml`**

```yaml
services:
  api:
    image: rag-pr-reviewer:latest
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000
    restart: always
    env_file: .env.prod
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
      - qdrant

  worker:
    image: rag-pr-reviewer:latest
    command: celery -A worker.celery_app worker --loglevel=info --concurrency=4
    restart: always
    env_file: .env.prod
    depends_on:
      - redis
      - postgres
      - qdrant

  beat:
    image: rag-pr-reviewer:latest
    command: celery -A worker.celery_app beat --loglevel=info
    restart: always
    env_file: .env.prod
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    restart: always
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:16-alpine
    restart: always
    env_file: .env.prod
    volumes:
      - pg_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:v1.9.2
    restart: always
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  redis_data:
  pg_data:
  qdrant_data:
```

Note: no source volume mounts. The built image is used directly.

- [ ] **Step 2: Create `.env.prod` template**

Create `.env.prod.example` (committed; actual `.env.prod` stays out of git):
```
GITHUB_APP_ID=
GITHUB_APP_PRIVATE_KEY_B64=
GITHUB_WEBHOOK_SECRET=
OPENAI_API_KEY=
QDRANT_URL=http://qdrant:6333
DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/rag_reviewer
REDIS_URL=redis://redis:6379/0
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=rag_reviewer
```

- [ ] **Step 3: Add `.env.prod` to `.gitignore`**

```bash
echo ".env.prod" >> .gitignore
```

- [ ] **Step 4: Smoke-test production compose locally**

```bash
docker build -t rag-pr-reviewer:latest .
cp .env .env.prod   # fill in real values
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

Expected: all services `running`.

```bash
curl -s http://localhost:8000/health
```

Expected: `{"status": "ok"}`.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.prod.yml .env.prod.example .gitignore
git commit -m "chore: production docker compose without source mounts, restart always"
```

---

## Task 6: Deploy to Railway

No code changes are required — Railway reads `docker-compose.prod.yml` via Nixpacks or uses the `Dockerfile` directly.

- [ ] **Step 1: Install Railway CLI**

```bash
npm install -g @railway/cli
railway login
```

- [ ] **Step 2: Create a new Railway project**

```bash
railway new
```

Select **Empty project**. Note the project ID.

- [ ] **Step 3: Add managed services in Railway dashboard**

In the Railway dashboard for your project:
1. Click **New** → **Database** → **PostgreSQL** — Railway provisions it, sets `DATABASE_URL`
2. Click **New** → **Database** → **Redis** — sets `REDIS_URL`
3. Click **New** → **Empty service** → name it `qdrant`
   - Source: Docker image `qdrant/qdrant:v1.9.2`
   - Add persistent volume at `/qdrant/storage`

- [ ] **Step 4: Deploy `api` service**

```bash
railway service create api
railway up --service api
```

Set the start command in Railway dashboard:
```
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Set all environment variables from `.env.prod.example` in the Railway dashboard (Settings → Variables).

- [ ] **Step 5: Deploy `worker` service**

```bash
railway service create worker
railway up --service worker
```

Start command:
```
celery -A worker.celery_app worker --loglevel=info --concurrency=4
```

- [ ] **Step 6: Update GitHub App webhook URL**

In GitHub App settings, set Webhook URL to the Railway `api` service public URL:
```
https://<your-railway-api-url>/webhook/github
```

- [ ] **Step 7: Run migrations on Railway**

```bash
railway run --service api alembic upgrade head
```

- [ ] **Step 8: Verify deployment**

Open a PR in a connected repo. Check Railway logs:
```bash
railway logs --service worker
```

Expected: `Task review_pr[...] succeeded` within 30 seconds.

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "chore: railway deploy instructions verified"
```

---

## Task 7: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# RAG PR Reviewer

An AI-powered GitHub bot that automatically reviews Pull Requests using retrieval-augmented generation. When a PR is opened, the bot retrieves relevant context from your codebase, architectural decision records, and past review history, then posts a structured inline review with citations.

---

## Problem

Senior engineers spend hours reviewing PRs but can only keep so much of the codebase in their head. This bot acts as a tireless reviewer that has read every file, every ADR, and every past PR comment — and cites its sources.

---

## Architecture

```
GitHub Webhook → FastAPI → Celery Worker
                               │
                    ┌──────────┴──────────┐
                    │   RAG Pipeline       │
                    │  ┌───────────────┐  │
                    │  │ HyDE Expand   │  │
                    │  │ Embed Query   │  │
                    │  │ Qdrant Search │  │
                    │  │ RRF Merge     │  │
                    │  │ Cross-Encode  │  │
                    │  │ GPT-4o Review │  │
                    │  └───────────────┘  │
                    └─────────────────────┘
                               │
                    GitHub PR Review (inline)
```

---

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embedding | `text-embedding-3-small` | Best cost/quality ratio at 1536 dims |
| Vector DB | Qdrant | Filter-by-payload required for per-repo isolation |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Lifts precision without extra API calls |
| LLM | GPT-4o (JSON mode) | Structured output with citations |
| HyDE | GPT-4o-mini | Cheap query expansion before embedding |
| Queue | Celery + Redis | Webhook returns 202 immediately; review is async |
| Tracing | Langfuse | Per-review traces + human feedback scores |

---

## Setup (Local Dev)

### 1. Prerequisites

- Docker + Docker Compose
- Python 3.11 + Poetry
- A GitHub App with permissions: Contents (read), Pull requests (read/write), Metadata (read)

### 2. Clone and install

```bash
git clone <repo>
cd rag-pr-reviewer
poetry install
```

### 3. Configure `.env`

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY_B64, GITHUB_WEBHOOK_SECRET
```

### 4. Start services

```bash
docker compose up -d
```

### 5. Run migrations

```bash
poetry run alembic upgrade head
```

### 6. Expose webhook via ngrok

```bash
ngrok http 8000
# Set the ngrok URL as the GitHub App webhook URL
```

### 7. Trigger a review

Open a PR in a repo where your GitHub App is installed. The bot will post a review within 30 seconds.

---

## Deploy to Railway

See [Task 6 in the Phase 4 plan](docs/superpowers/plans/2026-05-13-phase4-observability-deploy.md#task-6-deploy-to-railway) for step-by-step Railway deployment instructions.

---

## Metrics (after 30 days of reviews)

| Metric | Value |
|--------|-------|
| Avg review latency | < 20s |
| Comments resolved (positive signal) | tracked via Langfuse |
| Comments dismissed (negative signal) | tracked via Langfuse |

---

## Technical Decisions

- **One webhook endpoint for all events** — simplifies routing; event type is in the header, not the URL
- **HyDE + averaged vectors** — retrieving against a hypothetical snippet is more accurate than embedding the diff directly
- **RRF over score normalization** — rank fusion is robust to different score scales across collections
- **`asyncio.run()` inside Celery tasks** — keeps the pipeline async-native while Celery workers remain sync
- **base64-encoded PEM in env var** — avoids multiline secrets in Railway/Heroku env var fields
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with architecture, tech stack rationale, and setup instructions"
```

---

## Phase 4 Validation Checklist

Run the full test suite:
```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass.

Manual validation:
- [ ] Open a PR → Langfuse dashboard shows a trace with `retrieval`, `reranking`, `generation` spans
- [ ] Resolve a bot comment → Langfuse trace shows `comment_quality` score of `1.0`
- [ ] Dismiss a bot comment → Langfuse trace shows `comment_quality` score of `-1.0`
- [ ] Railway `api` and `worker` services both show healthy status
- [ ] `docker compose -f docker-compose.prod.yml up` starts cleanly from a fresh pull (no source mounts)
