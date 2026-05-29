from gh_app.events import WebhookEvent
from indexer.tasks import incremental_index


def handle_push(event: WebhookEvent) -> None:
    if not event.changed_files or not event.repo_full_name or not event.installation_id:
        return
    incremental_index.delay(
        repo_full_name=event.repo_full_name,
        installation_id=event.installation_id,
        changed_files=event.changed_files,
    )