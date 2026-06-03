#this to merge all component services to an actual pipeline
import asyncio
import json
import sys
from io import StringIO
from unidiff import PatchSet

from pipeline.qdrant_store import QdrantStore
from pipeline.retriever import retrieve
from pipeline.generator import generate_review

def parse_diff_lines(diff_text:str)->dict[str,set[int]]:
    #create dict of diff lines passing into generator
    patch=PatchSet(StringIO(diff_text))
    result:dict[str,set[int]]={}

    for p in patch:
        if p.is_removed_file:
            continue
        #set = list but no duplicate+ not in order
        added:set[int]=set()
        #each hunk is a block of changes
        for hunk in p:
            for line in hunk:
                if (line.is_added or line.is_context) and line.target_line_no:
                    added.add(line.target_line_no)
        result[p.path]=added
    return result

async def main(repo_id:str, diff_text:str)-> None:
    store=QdrantStore()

    #1 parse diff
    #diff text raw string-> parse to only + lines
    diff_lines=parse_diff_lines(diff_text)
    if not diff_lines:
        print("No added lines found",file=sys.stderr)
        return
    #2 retrieve context
    print(f"Retrieving context for {len(diff_lines)} changed files...", file=sys.stderr)
    chunks=await retrieve(store=store, query=diff_text,top_k=5,candidate_pool=20)
    print(f"  → {len(chunks)} chunks retrieved and reranked", file=sys.stderr)

    #3 call ai
    print("Generating review...", file=sys.stderr)
    comments = await generate_review(
        diff_text=diff_text,
        chunks=chunks,
        diff_lines=diff_lines,
    )

    #4 format and print(for dev)
    output = [
        {
            "line": c.line,
            "path": c.path,
            "severity": c.severity,
            "issue": c.issue,
            "suggestion": c.suggestion,
            "citation": c.citation,
        }
        for c in comments
    ]
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    repo_id = sys.argv[1]
    diff_source = sys.argv[2]

    if diff_source == "-":
        diff_text = sys.stdin.read()
    else:
        with open(diff_source) as f:
            diff_text = f.read()

    asyncio.run(main(repo_id, diff_text))