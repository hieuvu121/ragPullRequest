from gh_app.events import WebhookEvent
from indexer.tasks import record_feedback


def handle_review_comment(event: WebhookEvent) -> None:
    if event.action not in ("dismissed", "resolved", "created"):
        return
    comment_id = (event.raw.get("comment") or {}).get("id")
    if not comment_id:
        return
    record_feedback.delay(
        comment_id=comment_id,
        action=event.action,
        raw=event.raw,
    )