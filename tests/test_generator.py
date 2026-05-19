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