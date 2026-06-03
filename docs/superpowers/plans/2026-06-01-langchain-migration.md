# LangChain Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** Phase 3 complete — full RAG pipeline working end-to-end.

**Goal:** Replace custom pipeline code with LangChain equivalents **only where performance stays the same or improves**. Components where custom code is measurably better are explicitly kept.

---

## What Changes vs What Stays

| Component | Decision | Reason |
|-----------|----------|--------|
| `pipeline/generator.py` — LLM call | **Replace** | `ChatOpenAI` + `JsonOutputParser` is equivalent — same model, same JSON mode, cleaner chain |
| `pipeline/chunker.py` | **Keep** | AST-aware splitting produces whole functions/classes. LangChain's `RecursiveCharacterTextSplitter` splits by character count and cuts through function bodies — retrieval quality drops |
| `pipeline/retriever.py` | **Keep** | Dual-embedding (raw diff + stripped diff) merged with custom RRF is more accurate than `EnsembleRetriever` with a single retriever |
| `pipeline/qdrant_store.py` | **Keep** | Changing the Qdrant payload format to LangChain's schema would break the custom retriever. Since the retriever stays custom, storage format stays as-is |
| `scripts/index_repo.py` | **Keep** | Tied to the custom chunker and Qdrant store payload format |
| `scripts/review_pipeline.py` (`parse_diff_lines`) | **Keep** | No LangChain equivalent — domain-specific to PR diffs |

**Net result:** Only `pipeline/generator.py` changes.

---

## File Map

| File | Change |
|------|--------|
| `pyproject.toml` | Add `langchain-openai` |
| `pipeline/generator.py` | Replace `AsyncOpenAI` call with `ChatOpenAI` + `JsonOutputParser` chain |
| `tests/test_generator.py` | Update mocks to patch `ChatOpenAI` instead of `AsyncOpenAI` |

---

## Task 1: Add LangChain Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `langchain-openai` to `pyproject.toml`**

Under `[tool.poetry.dependencies]`:
```toml
langchain-openai = "^0.1.0"
```

```bash
poetry install
```

- [ ] **Step 2: Verify import**

```bash
python -c "from langchain_openai import ChatOpenAI; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add langchain-openai dependency"
```

---

## Task 2: Replace LLM Call in `pipeline/generator.py`

**Files:**
- Modify: `pipeline/generator.py`

**What changes:** The `AsyncOpenAI` client call + manual `json.loads()` is replaced by a `ChatOpenAI | JsonOutputParser` chain. The `ReviewComment` dataclass, `diff_lines` filter, and `SYSTEM_PROMPT` are all unchanged — only the API call mechanism changes.

- [ ] **Step 1: Write failing tests**

Create `tests/test_generator.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_generate_review_returns_comments_for_valid_lines():
    diff_lines = {"foo.py": {5, 10}}
    raw_output = {
        "comments": [
            {"line": 5, "path": "foo.py", "severity": "warning",
             "issue": "Missing input validation", "suggestion": "Add a guard clause",
             "citation": "foo.py:1-10"},
            {"line": 99, "path": "foo.py", "severity": "error",
             "issue": "Out of diff", "suggestion": "Fix it",
             "citation": "foo.py:90-100"},  # line 99 not in diff_lines → filtered
        ]
    }

    with patch("pipeline.generator.ChatOpenAI") as MockLLM, \
         patch("pipeline.generator.JsonOutputParser") as MockParser:

        mock_chain = AsyncMock(return_value=raw_output)
        MockLLM.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

        from pipeline.generator import generate_review
        comments = await generate_review(
            diff_text="diff content",
            chunks=[],
            diff_lines=diff_lines,
        )

    assert len(comments) == 1
    assert comments[0].line == 5
    assert comments[0].path == "foo.py"


@pytest.mark.asyncio
async def test_generate_review_returns_empty_when_no_valid_lines():
    diff_lines = {"foo.py": {3}}
    raw_output = {"comments": [
        {"line": 99, "path": "foo.py", "severity": "warning",
         "issue": "Issue", "suggestion": "Fix", "citation": "foo.py:1-5"},
    ]}

    with patch("pipeline.generator.ChatOpenAI") as MockLLM, \
         patch("pipeline.generator.JsonOutputParser") as MockParser:

        mock_chain = AsyncMock(return_value=raw_output)
        MockLLM.return_value.__or__ = MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))

        from pipeline.generator import generate_review
        comments = await generate_review(
            diff_text="diff content",
            chunks=[],
            diff_lines=diff_lines,
        )

    assert comments == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_generator.py -v
```

Expected: FAIL — `generate_review` still uses `AsyncOpenAI`, not `ChatOpenAI`.

- [ ] **Step 3: Rewrite `pipeline/generator.py`**

Replace the file content:
```python
import os
from dataclasses import dataclass
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    trace: Any = None,
) -> list[ReviewComment]:
    context_text = "\n\n".join(
        f"### {c.payload.get('file_path', '')}:"
        f"{c.payload.get('start_line', 0)}-{c.payload.get('end_line', 0)}\n"
        f"{c.payload.get('content', '')}"
        for c in chunks
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "## Diff\n{diff}\n\n## Context Chunks\n{context}"),
    ])

    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        api_key=os.getenv("OPENAI_API_KEY"),
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    chain = prompt | llm | JsonOutputParser()

    if trace:
        gen_span = trace.span(name="generation")
        gen_span.update(input={"prompt_chars": len(diff_text) + len(context_text)})

    raw = await chain.ainvoke({"diff": diff_text, "context": context_text})

    if trace:
        gen_span.update(output={"raw_comment_count": len(raw.get("comments", []))})
        gen_span.end()

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

Key differences from the old code:
- `AsyncOpenAI().chat.completions.create()` → `ChatOpenAI | JsonOutputParser` chain
- `json.loads(response.choices[0].message.content)` → handled automatically by `JsonOutputParser`
- `trace` arg preserved for Langfuse (Phase 4 compatible)
- `ReviewComment` dataclass and `diff_lines` filter are **identical**

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_generator.py -v
```

Expected: both pass.

- [ ] **Step 5: Restart worker and run a manual end-to-end test**

```bash
docker compose restart worker
```

Open a PR in a connected repo. Verify the bot still posts inline comments.

- [ ] **Step 6: Commit**

```bash
git add pipeline/generator.py tests/test_generator.py
git commit -m "refactor: replace AsyncOpenAI call with LangChain ChatOpenAI chain in generator"
```

---

## Validation Checklist

- [ ] `pytest tests/test_generator.py -v` — all pass
- [ ] PR opened → bot posts inline comments (same quality as before)
- [ ] No regression in `review_pr` worker logs
- [ ] Phase 4 Langfuse tracing still works — `trace.span(name="generation")` still called
