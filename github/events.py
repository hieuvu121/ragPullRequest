import hashlib
import hmac
from dataclasses import dataclass, field

#github send request with signature, using shared key set in env and github
@dataclass
class WebhookEvent:
    event_type: str
    action: str | None = None
    pr_number: int | None = None
    installation_id: int | None = None
    repo_full_name: str | None = None
    changed_files: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

#verify if producer is github webhook
def verify_signature(body: bytes, signature_header: str, secret: str) -> bool:
    if not signature_header.startswith("sha256="):
        return False
    expected = (hmac.new(secret.encode(),#webhook secret from key
                        body,#raw request
                        hashlib.sha256).
                hexdigest())
    return hmac.compare_digest(f"sha256={expected}", signature_header)#compare


def parse_event(event_type: str, payload: dict) -> WebhookEvent:
    event = WebhookEvent(event_type=event_type, raw=payload)
    event.action = payload.get("action")
    event.installation_id = (payload.get("installation") or {}).get("id")
    repo = payload.get("repository") or {}
    event.repo_full_name = repo.get("full_name")

    if event_type == "push":
        files: set[str] = set()
        for commit in payload.get("commits", []):
            files.update(commit.get("added", []))
            files.update(commit.get("modified", []))
        event.changed_files = list(files)

    if event_type == "pull_request":
        event.pr_number = payload.get("number")

    return event