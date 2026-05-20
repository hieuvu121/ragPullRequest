from fastapi import APIRouter
from pydantic import BaseModel
form indexer.tasks import full_index

router=APIRouter()

class IndexerRequest(BaseModel):
    repo_full_name:str
    installation_id:int

@router.post("/index")
def post_index(body: IndexerRequest):
    task=full_index.delay(body.repo_full_name,body.installation_id)
    return {"task_id":task.id,"status":"queued"}

