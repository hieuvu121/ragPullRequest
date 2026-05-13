from fastapi import APIRouter
from pydantic import BaseModel
from indexer.tasks import full_index

router=APIRouter()

class IndexRequest(BaseModel):
    installation_id:int

@router.post("/index")
def post_index(body:IndexRequest):
    task=full_index.delay(body.repo_full_name,body.installation_id)
    return{"task_id":task.id,"status":"queued"}