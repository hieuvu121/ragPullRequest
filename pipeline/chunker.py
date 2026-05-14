from dataclasses import dataclass
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

#set lang config for ast
PY_LANGUAGE=Language(tspython.language())
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
    parser=Parser(PY_LANGUAGE)
    tree=parser.parse(content.encode())

    #walk through tree and add to chunk
    chunks: list[Chunk] = []
    _walk(tree.root_node, content, file_path, chunks, inside_class=False)

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

def _walk(node,content: str,file_path:str,chunks: list[Chunk],inside_class:bool):
    #for helper function stand alone
    if node.type=="function_definition":
        #name node retrieve as byte, have start and end byte
        name_node=node.child_by_field_name("name")
        #actual name string
        name=content[name_node.start_byte:name_node.end_byte] if name_node else "<func>"
        chunks.append(Chunk(
            file_path=file_path,
            name=name,
            chunk_type="method" if inside_class else "function",
            start_line=node.start_point[0]+1,
            end_line=node.end_point[0]+1,
            content=content[node.start_byte:node.end_byte]
        ))
        return
    # for a class with multiples function, create class chunk-> recursive for all funct
    if node.type=="class_definition":
        name_node=node.child_by_field_name("name")
        name=content[name_node.start_byte:name_node.end_byte]if name_node else "<class>"
        #look inside body of class for after recursive
        body_node=node.child_by_field_name("body")

        #take class name, take from start byte to the end of header
        header_end_byte=body_node.start_byte if body_node else node.end_byte
        class_content=content[node.start_byte:header_end_byte].rstrip()
        header_end_line=content[:header_end_byte].count("\n")+1

        # Include docstring if first statement in body is a string
        if body_node and body_node.child_count > 0:
            first = body_node.children[0]
            if first.type == "expression_statement" and first.child_count > 0:
                expr = first.children[0]
                if expr.type == "string":
                    docstring = content[expr.start_byte:expr.end_byte]
                    class_content = class_content + "\n    " + docstring
                    header_end_line = first.end_point[0] + 1

        chunks.append(Chunk(
            file_path=file_path,
            name=name,
            chunk_type="class",
            #ast start from index 1, start point store as tuple(row,col)
            start_line=node.start_point[0]+1,
            end_line=header_end_line,
            content=class_content
        ))

        if body_node:
            for child in body_node.children:
                _walk(child,content,file_path,chunks,inside_class=True)
            return

    #recursive, if not function or class-> go through
    for child in node.children:
        _walk(child,content,file_path,chunks,inside_class=inside_class)