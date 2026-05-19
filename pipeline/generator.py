import json
import os
from dataclasses import dataclass
from openai import AsyncOpenAI
from pipeline.retriever import ScoredChunk
from dotenv import load_dotenv

load_dotenv()

GENERATION_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a senior software engineer performing a code review.
You are given a unified diff and a set of retrieved code context chunks.
Respond ONLY with valid JSON in this exact schema — no markdown, no explanation:
{
  "comments": [
    {
      "line": <integer — must be an added line in the diff>,
      "path": "<file path from the diff>",
      "severity": "<error|warning|suggestion>",
      "issue": "<one sentence: what is wrong>",
      "suggestion": "<one sentence: how to fix it>",
      "citation": "<file_path:start_line-end_line of the context chunk used>"
    }
  ]
}
Rules:
- Only comment on lines that appear as additions (+) in the diff.
- Every comment must be grounded in one of the provided context chunks.
- Do not invent issues not supported by the context."""

@dataclass
#define comment for each + lines
class ReviewComment:
    line:str
    path:str
    severity:str
    issue:str
    suggestion:str
    citation:str

async def generate_review(
        diff_text:str,
        chunks:list[ScoredChunk],
        diff_lines:dict[str,set[int]]
)->list[ReviewComment]:
    #build context include diff and related code chunks from qdrant
    context_text="\n\n".join(
        f"### {c.payload.get('file_path', '')}:"
        f"{c.payload.get('start_line', 0)}-{c.payload.get('end_line', 0)}\n"
        f"{c.payload.get('content', '')}"
        for c in chunks
    )
    user_message = f"## Diff\n{diff_text}\n\n## Context Chunks\n{context_text}"

    #define model generation
    client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response=await client.chat.completions.create(
        model=GENERATION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    #create res generator list and return
    raw=json.loads(response.choices[0].message.content)
    comments:list[ReviewComment]=[]
    for c in raw.get("comments",[]):
        path=c.get("path","")
        line=c.get("line",0)
        #second validation if line in diff dict
        if line not in diff_lines.get(path,set()):
            continue
        comments.append(ReviewComment(
            line=line,
            path=path,
            severity=c.get("severity","suggestion"),
            issue=c.get("issue",""),
            suggestion=c.get("suggestion",""),
            citation=c.get("citation","")
        ))
    return comments