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