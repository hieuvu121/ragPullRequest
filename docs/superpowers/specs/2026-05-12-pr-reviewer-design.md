# AI-Powered PR Reviewer with Codebase Context — Design Spec

**Date:** 2026-05-12
**Status:** Approved

---

## Overview

A GitHub bot that automatically reviews Pull Requests using RAG (Retrieval-Augmented Generation). When a PR is opened, the bot retrieves relevant context from three indexed sources — codebase chunks, ADR docs, and past PR review history — then posts a single atomic inline review with citations, mimicking a senior engineer who remembers everything about the codebase.

**Scope:** Single GitHub repository per installation (v1). Multi-repo support is out of scope.

---

## Architecture

Single Python monorepo, Option A: one FastAPI app + Celery workers sharing the same codebase. No microservices split at launch; internal module boundaries are clean enough to extract later.

```
rag_pr_reviewer/
├── api/
│   ├── main.py              # FastAPI app, lifespan, router registration
│   ├── routes/
│   │   └── webhooks.py      # POST /webhook/github — HMAC-SHA256 verification, routes all events
│   ├── handlers/
│   │   ├── indexing.py      # Handles push events → enqueue incremental_index
│   │   ├── review.py        # Handles pull_request.opened → enqueue review_pr
│   │   └── feedback.py      # Handles pull_request_review_comment events → feedback loop
│   └── dependencies.py      # Shared FastAPI deps (DB session, Qdrant client)
├── indexer/
│   ├── pipeline.py          # Orchestrates clone → chunk → embed → upsert
│   ├── chunker.py           # tree-sitter AST chunking (Python; extensible)
│   ├── embedder.py          # OpenAI text-embedding-3-small wrapper
│   └── tasks.py             # Celery tasks: full_index, incremental_index
├── reviewer/
│   ├── pipeline.py          # Orchestrates retrieval → rerank → generate → post
│   ├── retriever.py         # Multi-collection Qdrant search + HyDE expansion
│   ├── reranker.py          # cross-encoder/ms-marco reranking
│   ├── generator.py         # GPT-4o structured JSON review generation
│   └── tasks.py             # Celery task: review_pr
├── github/
│   ├── auth.py              # JWT + installation token, base64 private key decode
│   ├── client.py            # PyGitHub wrapper (fetch diff, post review)
│   └── webhook.py           # HMAC-SHA256 verification, event parsing
├── db/
│   ├── models.py            # SQLAlchemy: repos, indexed_files, pr_reviews, review_feedback
│   └── session.py           # Async engine, session factory
├── worker.py                # Celery app init, queue config
├── config.py                # Pydantic Settings (env vars, base64 key decode)
└── docker-compose.yml       # api, worker, beat, redis, qdrant, postgres
```

**Module ownership rules:**
- `api/` never calls `indexer/` or `reviewer/` directly — it enqueues Celery tasks only
- `reviewer/` accesses GitHub only through `github/client.py`
- `db/` is the only module that writes to Postgres

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
| indexed_at | TIMESTAMP | |
| UNIQUE | (repo_id, file_path) | |

**`pr_reviews`**
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| repo_id | UUID FK → repos | |
| pr_number | INT | |
| pr_title | TEXT | |
| github_review_id | BIGINT | ID returned by GitHub after posting |
| status | ENUM | `pending`, `posted`, `failed` |
| raw_llm_output | JSONB | Full structured JSON from GPT-4o |
| langfuse_trace_id | TEXT | |
| created_at | TIMESTAMP | |

**`review_feedback`**
| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| pr_review_id | UUID FK → pr_reviews | |
| comment_id | BIGINT | GitHub review comment ID |
| event | ENUM | `dismissed`, `resolved`, `replied` |
| langfuse_score | FLOAT | -1.0 (dismissed) to +1.0 (resolved) |
| recorded_at | TIMESTAMP | |

### Qdrant Collections

All collections use `text-embedding-3-small` (1536 dims), cosine similarity.

**`code_chunks`**

Payload: `repo_id`, `file_path`, `start_line`, `end_line`, `chunk_type` (`function`/`class`/`module`), `content_hash`

Strategy: deleted by `{repo_id, file_path}` filter before re-indexing a changed file.

**`adr_docs`**

Payload: `repo_id`, `file_path`, `section_title`, `doc_type`

Strategy: manually triggered index; ADRs are rarely updated.

**`pr_history`**

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
4. `chunker.py`: tree-sitter parse → extract functions and classes as individual chunks
   - Fallback: single whole-file chunk if unparseable and file ≤ 8 KB; skip if larger
   - Each chunk captures `start_line`/`end_line` for inline comment positioning
5. `embedder.py`: batch embed chunks (up to 100 per OpenAI request)
6. Qdrant: `delete_by_filter({repo_id, file_path})`
7. Qdrant: upsert new vectors with full payload
8. Postgres: upsert `indexed_files(file_path, content_hash, indexed_at)`

**Full index flow:**

Same per-file logic, but fetches the full repo tree. Processed in batches of 50 files per Celery task to avoid memory pressure.

**Error handling:**
- Embedding or Qdrant upsert failure → log file with `content_hash = NULL` → retried on next push
- Celery task retries 3× with exponential backoff on transient errors

---

## Review Pipeline

**Trigger:** `pull_request.opened` webhook → `review_pr(repo_id, pr_number)` Celery task

**Flow:**

1. **Fetch diff** — `github/client.py` fetches PR diff, parsed with `unidiff` → list of `(file_path, hunk, added_lines with line numbers)`

2. **Multi-source retrieval** — for each changed file (up to 10 files by line-change count; the full diff is still fetched, only retrieval is capped to control embedding cost):
   - HyDE expansion: GPT-4o generates a hypothetical relevant code snippet from the diff hunk, used as the embedding query
   - Embed the HyDE query with `text-embedding-3-small`
   - Query all 3 Qdrant collections **in parallel**: `code_chunks` (limit=5), `adr_docs` (limit=3), `pr_history` (limit=3)
   - Merge + deduplicate results by `content_hash`

3. **Reranking** — `cross-encoder/ms-marco` scores all candidates against the original diff hunk text → keep top 8 per file

4. **Generation** — build prompt with diff + retrieved context chunks (with citations: `file_path`, lines, source collection)
   - GPT-4o with `response_format: json_object` returns:
   ```json
   {
     "summary": "Overall review summary...",
     "comments": [
       {
         "path": "src/auth.py",
         "line": 42,
         "side": "RIGHT",
         "body": "Comment body with reasoning...",
         "severity": "error|warning|suggestion",
         "citations": ["code_chunks:src/utils.py:10-25", "adr_docs:ADR-003"]
       }
     ]
   }
   ```

5. **Post review** — `github/client.py` submits a single GitHub PR review via `POST /repos/{owner}/{repo}/pulls/{pr}/reviews` containing all inline comments and the summary body

6. **Persist** — insert `pr_reviews` row with `status=posted`, `raw_llm_output`, `langfuse_trace_id`

7. **Trace close** — Langfuse trace closes with retrieval span and LLM span

**Error handling:**
- GitHub posting failure → `pr_reviews.status = failed`; raw LLM output preserved for manual retry
- Malformed JSON from GPT-4o → retry generation once with stricter system prompt; if still malformed, fail task and log

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

**Webhook security:** `github/webhook.py` validates `X-Hub-Signature-256` header against `GITHUB_WEBHOOK_SECRET` using `hmac.compare_digest` before any processing.

**Events handled:**
- `push` → enqueue `incremental_index`
- `pull_request.opened` → enqueue `review_pr`
- `pull_request_review_comment.dismissed` / `.created` → feedback endpoint

---

## Observability

**Langfuse — one trace per PR review:**

| Span | Recorded data |
|---|---|
| `retrieval` | query, collection, top-k results, reranker scores, latency |
| `generation` | prompt tokens, completion tokens, model, latency, structured output |

Feedback events push a `comment_quality` score to the originating trace, enabling filtering of traces by review quality in the Langfuse dashboard.

**Postgres** stores raw `review_feedback` rows for offline analysis (e.g., which file types or severity levels get dismissed most often).

---

## Deployment

### Local Dev (Docker Compose)

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

- `api` and `worker` as separate Railway services from the same Docker image, different `CMD`
- Railway managed Postgres and Redis plugins
- Qdrant as a Railway service with persistent volume (or Qdrant Cloud)
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
