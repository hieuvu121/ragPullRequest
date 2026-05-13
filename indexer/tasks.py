from worker import celery_app

@celery_app.task(name="full_index")
def full_index(repo_full_name):
    print(f"[full_index] repo={repo_full_name}")
    return {"status": "queued", "repo": repo_full_name}

@celery_app.task(name="incremental_index")
def incremental_index(repo_full_name: str, changed_files: list[str], installation_id: int):
    print(f"[incremental_index] repo={repo_full_name} files={changed_files}")
    return {"status": "queued", "repo": repo_full_name}