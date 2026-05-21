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


async def index_directory(root: Path, repo_id: str) -> int:
    store = QdrantStore()
    await store.create_collection()

    embedder = Embedder()
    py_files = _get_py_files(root)
    total_chunks = 0

    for file_path in py_files:
        try:
            #read content
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        #chunk files
        relative_path = str(file_path.relative_to(root))
        chunks = chunk_file(relative_path, content)
        if not chunks:
            continue

        #embedding
        vectors = await embedder.embed([c.content for c in chunks])

        #delete old vectors
        await store.delete_by_filter(repo_id, relative_path)

        #upsert to qdrant
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
            for chunk, vector in zip(chunks, vectors)
        ]
        await store.upsert(points)
        total_chunks += len(points)
        print(f"  indexed {relative_path} → {len(chunks)} chunks")

    return total_chunks


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