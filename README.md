# RAG PR Reviewer

An AI-powered GitHub bot that automatically reviews Pull Requests using Retrieval-Augmented Generation. When a PR is opened, the bot fetches the diff, retrieves relevant code context from a vector store, and posts inline review comments grounded in your actual codebase.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry
- A GitHub App
- An OpenAI API key
- ngrok (for local webhook delivery)

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/rag-pr-reviewer.git
cd rag-pr-reviewer
```

### 2. Install Dependencies

```bash
pip install poetry
poetry install
```

### 3. Create a GitHub App

1. Go to **GitHub → Settings → Developer settings → GitHub Apps → New GitHub App**
2. Set the webhook URL to `https://<your-domain>/webhook/github`
3. Set a webhook secret and note it
4. Grant permissions: **Pull requests** (Read & Write), **Contents** (Read), **Metadata** (Read)
5. Subscribe to events: `Push`, `Pull request`, `Pull request review comment`
6. Generate a private key (PEM) and note the **App ID** and **Installation ID**
7. Base64-encode the private key:
   ```bash
   base64 -i your-private-key.pem | tr -d '\n'
   ```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# GitHub App
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY_B64=<base64-encoded PEM from step 3>
GITHUB_WEBHOOK_SECRET=<your webhook secret>

# OpenAI
OPENAI_API_KEY=sk-...

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

# Database
DATABASE_URL=postgresql+asyncpg://rag:rag@postgres:5432/rag

# Redis
REDIS_URL=redis://redis:6379/0
```

### 5. Start Services

```bash
docker-compose up --build
```

This starts:
- **api** — FastAPI on port 8000
- **worker** — Celery worker (concurrency 4)
- **beat** — Celery beat scheduler
- **redis** — Redis 7 on port 6379
- **postgres** — PostgreSQL 16 on port 5433
- **qdrant** — Qdrant vector store on port 6333

### 6. Run Database Migrations

```bash
docker-compose exec api alembic upgrade head
```

### 7. Expose Locally for Webhooks

```bash
ngrok http 8000
```

Copy the ngrok HTTPS URL and update your GitHub App's webhook URL to `https://<ngrok-id>.ngrok.io/webhook/github`.

### 8. Trigger Initial Index

Once the GitHub App is installed on a repository, trigger a full index:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_full_name": "owner/repo", "installation_id": 12345678}'
```

The bot will then automatically review all new PRs opened on that repository.

---

## Running Tests

```bash
poetry run pytest
```

---

## Project Structure

```
├── pipeline/          # Core RAG logic (chunker, embedder, retriever, generator)
├── api/               # FastAPI app, routes, and webhook handlers
├── gh_app/            # GitHub App auth, client, and event parsing
├── indexer/           # Celery tasks
├── db/                # SQLAlchemy models and async session
├── scripts/           # CLI tools for local testing
├── alembic/           # Database migrations
├── tests/             # Unit tests
├── config.py          # Pydantic-settings config
├── worker.py          # Celery app init
├── Dockerfile
└── docker-compose.yml
```
