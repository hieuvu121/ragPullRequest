from typing import AsyncGenerator
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from db.session import AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


def get_qdrant(request: Request) -> AsyncQdrantClient:
    return request.app.state.qdrant