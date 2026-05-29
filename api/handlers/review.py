from indexer.tasks import review_pr
from gh_app.events import WebhookEvent

def handle_pr_opened(event: WebhookEvent) -> None:
    if event.action not in ("opened", "reopened"):
        return
    if not event.pr_number or not event.repo_full_name or not event.installation_id:
        return
    review_pr.delay(
        repo_full_name=event.repo_full_name,
        pr_number=event.pr_number,
        installation_id=event.installation_id,
    )