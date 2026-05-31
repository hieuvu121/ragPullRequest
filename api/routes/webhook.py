import json
from fastapi import APIRouter, Request, Response
from gh_app.events import verify_signature, parse_event
from api.handlers.indexing import handle_push
from api.handlers.review import handle_pr_opened
from api.handlers.feedback import handle_review_comment
from config import settings

router=APIRouter()

@router.post("/webhook/github")
async def github_webhook(request:Request):
    body=await request.body()
    sig=request.headers.get("X-Hub-Signature-256", "")
    #check author by webhook sec key, body, sig
    if not verify_signature(body,sig,settings.github_webhook_secret):
        return Response(status_code=403)
    event_type=request.headers.get("X-GitHub-Event", "")
    payload=json.loads(body)
    event=parse_event(event_type,payload)

    if event_type == "push":
        handle_push(event)
    elif event_type == "pull_request":
        handle_pr_opened(event)
    elif event_type == "pull_request_review_comment":
        handle_review_comment(event)
    else:
        return Response(status_code=204)

    return Response(status_code=202)