# AI-Powered PR Reviewer with Codebase Context — Design Spec

**Date:** 2026-05-12
**Updated:** 2026-05-13
**Status:** Approved

---

## Overview

A GitHub bot that automatically reviews Pull Requests using RAG (Retrieval-Augmented Generation). When a PR is opened, the bot retrieves relevant context from three indexed sources — codebase chunks, ADR docs, and past PR review history — then posts a single atomic inline review with citations, mimicking a senior engineer who remembers everything about the codebase.

**Scope:** Single GitHub repository per installation (v1). Multi-repo support is out of scope.

---

## Architecture

Single Python monorepo: one FastAPI app + Celery workers sharing the same codebase. All AI logic lives in `pipeline/` as pure Python — validated independently before any infrastructure. Celery tasks in `indexer/tasks.py` import `pipeline/` functions directly via `asyncio.run()`.

```
rag-pr-reviewer/
├── pipeline/                        # Phase 1 — pure Python RAG logic
│   ├── __init__.py
│   ├── chunker.py                   # Chunk dataclass + chunk_file() via tree-sitter
│   ├── embedder.py                  # Embedder.embed() — OpenAI batched calls
│   ├── qdrant_store.py              # QdrantStore: create, upsert, search, delete_by_filter
│   ├── retriever.py                 # retrieve() — strip diff markers + RRF + cross-encoder rerank
│   └── generator.py                 # generate_review() — GPT-4o JSON → list[ReviewComment]
├── scripts/                         # Phase 1 — standalone CLI entry points
│   ├── index_repo.py                # CLI: walk .py files → chunk → embed → upsert
│   └── review_pipeline.py           # CLI: diff string → retrieve → generate → print JSON
├── api/                             # Phase 2 — FastAPI app
│   ├── __init__.py
│   ├── main.py                      # App init, lifespan (migrations + Qdrant collection init)
│   ├── dependencies.py              # get_db, get_qdrant
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── index.py                 # POST /index — enqueues full_index
│   │   ├── search.py                # POST /search — calls retriever directly
│   │   └── webhooks.py              # POST /webhook/github — HMAC verify, routes events
│   └── handlers/
│       ├── __init__.py
│       ├── indexing.py              # handle_push() → enqueue incremental_index
│       ├── review.py                # handle_pr_opened() → enqueue review_pr
│       └── feedback.py              # handle_review_comment() → enqueue record_feedback
├── github/                          # Phase 3 — GitHub App integration
│   ├── __init__.py
│   ├── auth.py                      # JWT RS256 mint + installation token cache
│   ├── events.py                    # HMAC-SHA256 verify + WebhookEvent parsing
│   └── client.py                    # GithubClient: get_diff(), post_review()
├── indexer/                         # Phase 2/3 — Celery tasks
│   ├── __init__.py
│   └── tasks.py                     # full_index, incremental_index, review_pr, record_feedback
├── db/                              # Phase 2 — Postgres
│   ├── __init__.py
│   ├── models.py                    # Repo, IndexedFile, PRReview, ReviewFeedback
│   └── session.py                   # Async engine, AsyncSessionLocal
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   ├── test_generator.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_tasks.py
│   ├── test_routes.py
│   ├── test_github_auth.py
│   ├── test_github_events.py
│   ├── test_github_client.py
│   ├── test_webhook_route.py
│   ├── test_tracing.py
│   └── test_feedback_score.py
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
├── config.py                        # pydantic-settings: all env vars, base64 key decode
├── worker.py                        # Celery app init
├── pyproject.toml
├── alembic.ini
├── Dockerfile
├── docker-compose.yml               # dev: api, worker, beat, redis, qdrant, postgres
├── docker-compose.prod.yml          # prod: no source mounts, restart: always
├── .env
└── README.md
```

**Module ownership rules:**
- `pipeline/` contains all AI logic — no FastAPI, no Celery, no Postgres imports
- `api/` never calls `pipeline/` or `indexer/tasks.py` directly — enqueues Celery tasks only
- `indexer/tasks.py` is the only place that calls `pipeline/` functions (via `asyncio.run()`)
- `github/` is the only module that talks to the GitHub API
- `db/` is the only module that writes to Postgres
- `scripts/` are standalone CLI tools for local dev and Phase 1 validation only

---

## Data Model

### PostgreSQL (SQLAlchemy + asyncpg)

**`repos`**
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| github_repo_id | INT UNIQUE | GitHub's numeric repo ID |
| full_name | TEXT | `"owner/repo"` |
| installation_id | INT | GitHub App installation ID |
| created_at | TIMESTAMP | |

**`indexed_files`**
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| repo_id | UUID FK → repos | |
| file_path | TEXT | Relative path in repo |
| content_hash | TEXT | SHA256 of file content; NULL = needs retry |
| status | TEXT | `indexed`, `failed`, `failed_permanent`, `deleted` |
| retry_count | INT | Incremented on each failure; `failed_permanent` at ≥ 3 |
| chunk_count | INT | Number of chunks produced on last successful index |
| indexed_at | TIMESTAMP | |
| UNIQUE | (repo_id, file_path) | |

**`pr_reviews`**
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| repo_id | UUID FK → repos | |
| pr_number | INT | |
| github_review_id | BIGINT | ID returned by GitHub after posting |
| status | TEXT | `pending`, `posted`, `failed` |
| raw_output | JSONB | Full list of `ReviewComment` dicts from GPT-4o |
| latency_ms | INT | Wall-clock ms from task start to GitHub post |
| langfuse_trace_id | TEXT | |
| created_at | TIMESTAMP | |

**`review_feedback`**
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| pr_review_id | UUID FK → pr_reviews | |
| comment_id | BIGINT | GitHub review comment ID |
| action | TEXT | `dismissed`, `resolved`, `created` |
| value | FLOAT | -1.0 (dismissed), 0.0 (created), +1.0 (resolved) |
| timestamp | TIMESTAMP | |

### Qdrant Collections

All collections use `text-embedding-3-small` (1536 dims), cosine similarity.

**`code_chunks`** _(implemented in Phase 1)_

Payload: `repo_id`, `file_path`, `start_line`, `end_line`, `chunk_type` (`function`/`class`/`module`), `content_hash`

Strategy: deleted by `{repo_id, file_path}` filter before re-indexing a changed file.

**`adr_docs`** _(out of scope v1)_

Payload: `repo_id`, `file_path`, `section_title`, `doc_type`

Strategy: manually triggered index; ADRs are rarely updated.

**`pr_history`** _(out of scope v1)_

Payload: `repo_id`, `pr_number`, `comment_body`, `diff_hunk`, `source` (`human`/`bot`), `file_path`, `line`

Strategy: append-only on each closed PR; never deleted.

---

## Indexing Pipeline

**Triggers:**
- `push` webhook → `incremental_index(repo_id, changed_files)` Celery task
- Manual admin call → `full_index(repo_id)` Celery task (first run)

**Incremental index flow (per changed file):**

1. Fetch file content via GitHub API
2. `SHA256(content)` → compare against `indexed_files.content_hash`
3. If hash unchanged → skip
4. `pipeline/chunker.py`: tree-sitter parse → extract functions and classes as individual chunks
   - Fallback: single whole-file chunk if unparseable and file ≤ 2 KB; skip if larger
   - Each chunk captures `start_line`/`end_line` for inline comment positioning
5. `embedder.py`: batch embed chunks (up to 100 per OpenAI request)
6. Qdrant: `delete_by_filter({repo_id, file_path})`
7. Qdrant: upsert new vectors with full payload
8. Postgres: upsert `indexed_files(file_path, content_hash, indexed_at)`

**Full index flow:**

Same per-file logic, but fetches the full repo tree. Processed in batches of 50 files per Celery task to avoid memory pressure.

**Error handling:**
- Embedding or Qdrant upsert failure → `indexed_files.status = failed`, `content_hash = NULL`, increment `retry_count` → retried on next push
- At `retry_count >= 3` → `status = failed_permanent`, no further retries
- Celery task retries 3× with exponential backoff (`30s * 2^attempt`) on transient errors

---

## Review Pipeline

**Trigger:** `pull_request.opened` webhook → `review_pr(repo_full_name, pr_number, installation_id)` Celery task

**Flow:**

1. **Fetch diff** — `github/client.py` calls PyGitHub `pr.get_files()`, reconstructs a unified diff string with accurate line numbers

2. **Retrieval** (`pipeline/retriever.py`):
   - **Diff marker stripping**: `+`/`-` prefix characters and `@@` hunk headers are removed to produce clean code; both the raw diff and stripped version are embedded in parallel and searched separately
   - Query `code_chunks` collection (top-20 candidates)
   - **RRF merge**: `score(doc) = Σ 1/(60 + rank_i)` across ranked lists
   - **Cross-encoder rerank**: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores top-20 → keep top-5

3. **Generation** (`pipeline/generator.py`) — GPT-4o with `response_format: json_object`:
   ```json
   [
     {
       "line": 42,
       "path": "src/auth.py",
       "severity": "error|warning|suggestion",
       "issue": "Describe the problem...",
       "suggestion": "Describe the fix...",
       "citation": "pipeline/chunker.py:10-25"
     }
   ]
   ```
   Comments are filtered to lines present in the diff before posting.

4. **Post review** — `github/client.py` submits a single GitHub PR review with all inline comments and a summary body

5. **Persist** — insert `pr_reviews` row with `status=posted`, `raw_output`, `latency_ms`, `langfuse_trace_id`

6. **Trace close** — Langfuse trace closes with `retrieval`, `reranking`, `generation`, `post_comment` spans

**Error handling:**
- GitHub posting failure → `pr_reviews.status = failed`; raw output preserved for manual retry
- Celery task retries 3× with exponential backoff (`30s * 2^attempt`) on any transient error

---

## Feedback Loop

**Trigger:** `pull_request_review_comment` GitHub webhook events arrive at `POST /webhook/github` (same endpoint as all other events), routed internally to `api/handlers/feedback.py` after HMAC-SHA256 verification.

**Flow:**
1. HMAC-SHA256 verification (shared with all webhook events)
2. Look up `pr_reviews` by `github_review_id`
3. Insert `review_feedback` row (`event`, `comment_id`, `langfuse_score`)
4. Push score to Langfuse via `client.score(trace_id, name="comment_quality", value=score)`

**Score mapping:**
- `resolved` → `+1.0`
- `dismissed` → `-1.0`
- `replied` → `0.0` (neutral signal, logged for analysis)

---

## GitHub App Integration

**Auth flow:**
1. At startup, `config.py` base64-decodes `GITHUB_APP_PRIVATE_KEY_B64` → RSA private key
2. Per request, `github/auth.py` mints a JWT (10-min expiry) signed with the private key
3. Exchange JWT for an installation access token (1-hour expiry) via `POST /app/installations/{id}/access_tokens`
4. All GitHub API calls use the installation token

**Webhook security:** `github/events.py` validates `X-Hub-Signature-256` header against `GITHUB_WEBHOOK_SECRET` using `hmac.compare_digest` before any processing. All events share a single `POST /webhook/github` endpoint.

**Events handled:**
| Event | Handler | Celery task |
|-------|---------|-------------|
| `push` | `api/handlers/indexing.py` | `incremental_index(repo_full_name, installation_id, changed_files)` |
| `pull_request.opened` | `api/handlers/review.py` | `review_pr(repo_full_name, pr_number, installation_id)` |
| `pull_request_review_comment.dismissed` / `.created` | `api/handlers/feedback.py` | `record_feedback(comment_id, action, raw)` |

---

## Observability

**Langfuse — one trace per `review_pr` task:**

| Span | Recorded data |
|---|---|
| `retrieval` | diff length, number of candidates returned |
| `reranking` | number of results kept after reranking |
| `generation` | prompt character count, response character count |
| `post_comment` | number of comments posted, latency |

Feedback events (`record_feedback` task) push a `comment_quality` score to the originating trace via `lf.score(trace_id, name="comment_quality", value=±1.0)`.

**Postgres** stores raw `review_feedback` rows for offline analysis (e.g., which severities or file types get dismissed most often).

---

## Deployment

### Local Dev (Docker Compose)

`docker-compose.yml` — source volume mounts, auto-reload:

```yaml
services:
  api:      # FastAPI (uvicorn, port 8000, auto-reload)
  worker:   # Celery worker (concurrency=4)
  beat:     # Celery beat (scheduled daily re-index)
  redis:    # Redis 7 (broker + result backend)
  qdrant:   # Qdrant (port 6333, named volume)
  postgres: # Postgres 16 (named volume)
```

GitHub webhook delivery in dev via `ngrok http 8000`.

### Production (Railway)

`docker-compose.prod.yml` — no source volume mounts, `restart: always`, reads from `.env.prod`.

- `api` and `worker` as separate Railway services from the same Docker image, different `CMD`
- Railway managed Postgres and Redis plugins
- Qdrant as a Railway service with persistent volume
- All secrets injected as Railway environment variables

### Environment Variables

```
GITHUB_APP_ID
GITHUB_APP_PRIVATE_KEY_B64     # base64-encoded PEM
GITHUB_WEBHOOK_SECRET
OPENAI_API_KEY
QDRANT_URL
QDRANT_API_KEY
DATABASE_URL
REDIS_URL
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LANGFUSE_HOST
```

---

## Key Constraints

- API never blocks — all heavy work dispatched as Celery tasks
- Incremental indexing: SHA256 per file, unchanged files skipped
- Qdrant delete-by-filter before re-indexing changed files
- LLM output grounded only on retrieved context (no hallucinated references)
- GitHub App private key never stored in plaintext — base64 env var, decoded at startup
- Webhook endpoints protected by HMAC-SHA256 only (no additional auth)

---

## Out of Scope (v1)

- Multi-repo or multi-tenant support
- Support for languages other than Python in tree-sitter chunking (extensible but not implemented)
- PR auto-approve or merge triggering
- Slack/email notifications
- Admin dashboard UI
