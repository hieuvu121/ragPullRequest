from pipeline.chunker import chunk_file, Chunk

SIMPLE_PYTHON = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    """A simple calculator."""

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        return a / b
'''


def test_extracts_module_level_functions():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    fn_names = {c.name for c in chunks if c.chunk_type == "function"}
    assert fn_names == {"add", "subtract"}


def test_extracts_class_header_not_full_body():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert len(class_chunks) == 1
    assert class_chunks[0].name == "Calculator"
    # class chunk must NOT contain the method bodies
    assert "return a * b" not in class_chunks[0].content


def test_class_chunk_includes_docstring():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    class_chunks = [c for c in chunks if c.chunk_type == "class"]
    assert "A simple calculator" in class_chunks[0].content


def test_extracts_methods():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    method_names = {c.name for c in chunks if c.chunk_type == "method"}
    assert method_names == {"multiply", "divide"}


def test_chunk_has_accurate_line_numbers():
    chunks = chunk_file("math.py", SIMPLE_PYTHON)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_chunk_carries_file_path():
    chunks = chunk_file("src/math.py", SIMPLE_PYTHON)
    assert all(c.file_path == "src/math.py" for c in chunks)


def test_small_unparseable_returns_whole_file_chunk():
    bad = "this is not valid python syntax @#$"
    chunks = chunk_file("bad.py", bad)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "module"
    assert chunks[0].content == bad


def test_large_unparseable_returns_empty():
    bad = "not python " * 300  # > 2 KB
    chunks = chunk_file("big.py", bad)
    assert chunks == []


def test_empty_file_returns_empty():
    assert chunk_file("empty.py", "") == []