from dataclasses import dataclass
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

#set lang config for ast
PY_LANGUAGE=Language(tspython.language(),"python")
FILE_FALLBACK_MAX_BYTES=2048

#store metadata of chunk files
#langchain can be used but not enough metadata+not use ast chunk
@dataclass
class Chunk:
    file_path: str
    name: str
    chunk_type: str  # "function" | "class" | "method" | "module"
    start_line: int
    end_line: int
    content: str

def chunk_file(file_path: str, content: str)->list[Chunk]:
    if not content.strip():
        return[]

    #parse to tree
    parser=Parser()
    parser.set_language(PY_LANGUAGE)
    tree=parser.parse(content.encode())

    #walk through tree and add to chunk
    chunks:list[Chunk]=[]_walk(tree.root_node, content, file_path, chunks, inside_class=False)

    #when cannot retrieve any chunks-> create 1 chunk with that file content
    if not chunks:
        if len(content.encode())<=FILE_FALLBACK_MAX_BYTES:
            chunks.append(Chunk(
                file_path=file_path,
                name="<module>",
                chunk_type="module",
                start_line=1,
                end_line=content.count("\n") + 1,
                content=content,
            ))
    return chunks;