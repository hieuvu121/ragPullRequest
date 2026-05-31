import asyncio
from gh_app.events import WebhookEvent
from db.session import AsyncSessionLocal
from db.models import Repo
from sqlalchemy import select
import uuid


def handle_installation(event: WebhookEvent) -> None:
    if event.action not in("created","added") or not event.installation_id or not event.repositories:
        return
    asyncio.create_task(_insert_repos(event.installation_id, event.repositories))


async def _insert_repos(installation_id: int, repositories: list[dict]) -> None:
    async with AsyncSessionLocal() as db:
        for repo in repositories:
            github_repo_id = repo.get("id")
            full_name = repo.get("full_name")
            if not github_repo_id or not full_name:
                continue
            existing = await db.execute(
                select(Repo).where(Repo.github_repo_id == github_repo_id)
            )
            if existing.scalar_one_or_none():
                continue
            db.add(Repo(
                id=uuid.uuid4(),
                github_repo_id=github_repo_id,
                full_name=full_name,
                installation_id=installation_id,
            ))
        await db.commit()
