import asyncio
import sys
import subprocess
import tempfile
import uuid
from pathlib import Path
from qdrant_client.models import PointStruct

from pipeline.chunker import chunk_file
from pipeline.embedder import Embedder
from pipeline.qdrant_store import QdrantStore

SUPPORTED_EXTENSIONS = {".py"}

#get all py files
def _get_py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if ".git" not in p.parts]

UPSERT_BATCH = 500

async def index_directory(root: Path, repo_id: str) -> int:
    store = QdrantStore()
    await store.create_collection()
    embedder = Embedder()

    # collect all chunks from all files first
    all_chunks = []
    for file_path in _get_py_files(root):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative_path = str(file_path.relative_to(root))
        chunks = chunk_file(relative_path, content)
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    # 1 delete for the whole repo
    await store.delete_by_repo(repo_id)

    # 1 batched embed call (Embedder batches internally at 100)
    vectors = await embedder.embed([c.content for c in all_chunks])

    # build all points
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "repo_id": repo_id,
                "file_path": chunk.file_path,
                "name": chunk.name,
                "chunk_type": chunk.chunk_type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
            },
        )
        for chunk, vector in zip(all_chunks, vectors)
    ]

    # upsert in batches of 500
    for i in range(0, len(points), UPSERT_BATCH):
        await store.upsert(points[i : i + UPSERT_BATCH])

    print(f"  indexed {len(points)} chunks from {len(_get_py_files(root))} files")
    return len(points)


async def main(source: str, repo_id: str) -> None:
    source_path = Path(source)

    if source_path.exists() and source_path.is_dir():
        root = source_path
        total = await index_directory(root, repo_id)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            print(f"Cloning {source}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", source, tmp],
                check=True, capture_output=True,
            )
            total = await index_directory(Path(tmp), repo_id)

    print(f"\nDone. Total chunks indexed: {total}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    asyncio.run(main(sys.argv[1], sys.argv[2]))