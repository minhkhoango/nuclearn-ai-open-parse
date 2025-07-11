import re
from pathlib import Path

import openparse

# Get the directory of the current test file (/.../src/tests)
TESTS_ROOT = Path(__file__).parent

# Go up one level to get the root of the source tree (/.../src)
SRC_ROOT = TESTS_ROOT.parent

# Define the data directory relative to the source root
DATA_ROOT = SRC_ROOT / "evals" / "data"


def test_parse_doc() -> None:
    """Tests basic PDF document parsing."""
    basic_doc_path = DATA_ROOT / "full-pdfs" / "mock-1-page-lease.pdf"
    parser = openparse.DocumentParser()
    parsed_basic_doc = parser.parse(basic_doc_path)
    assert parsed_basic_doc.nodes, "Parsing should produce at least one node."
    assert parsed_basic_doc.nodes[0].text.startswith(
        "**MOCK LEASE AGREEMENT**"
    ), "The first node's text does not match the expected start."


def get_cols(html_string: str) -> str | None:
    """Extracts table header content from an HTML string."""
    pattern = r"<thead>(.*?)</thead>"
    match = re.search(pattern, html_string, re.DOTALL)
    if match:
        return match.group(1)
    return None


def test_parse_tables_with_table_transformers() -> None:
    """Tests table parsing using the 'table-transformers' algorithm."""
    doc_with_tables_path = (
        DATA_ROOT / "tables" / "naic-numerical-list-of-companies-page-94.pdf"
    )

    parser = openparse.DocumentParser(
        table_args={"parsing_algorithm": "table-transformers"}
    )
    parsed_doc = parser.parse(doc_with_tables_path)
    assert (
        parsed_doc.nodes
    ), "Parsing should produce at least one node for the table document."

    found_text = get_cols(parsed_doc.nodes[0].text)

    assert found_text is not None, "Could not find the table header (<thead>)."
    assert "GROUP NAME" in found_text
    assert "GROUP" in found_text
    assert "CO NO" in found_text
    assert "STMT" in found_text
    assert "STATUS" in found_text
    assert "ST" in found_text
    assert "COMPANY NAME" in found_text


def test_parse_tables_with_pymupdf() -> None:
    """Tests table parsing using the 'pymupdf' algorithm."""
    doc_with_tables_path = DATA_ROOT / "tables" / "meta-2022-10k-page-69.pdf"

    parser = openparse.DocumentParser(table_args={"parsing_algorithm": "pymupdf"})

    parsed_doc = parser.parse(doc_with_tables_path)

    print("\n--- DEBUG OUTPUT ---")
    print(f"Number of nodes found: {len(parsed_doc.nodes)}")
    print("Content of the final node:")
    print(parsed_doc.nodes[-1].text)
    print("--- END DEBUG ---")

    assert parsed_doc.nodes, "Parsing should produce at least one node."
    assert parsed_doc.nodes[-1].text


def test_to_llama_index_nodes() -> None:
    """Tests the conversion of parsed documents to LlamaIndex nodes."""
    basic_doc_path = DATA_ROOT / "full-pdfs" / "mock-1-page-lease.pdf"
    parser = openparse.DocumentParser()
    parsed_basic_doc = parser.parse(basic_doc_path)

    nodes = parsed_basic_doc.to_llama_index_nodes()
    assert nodes, "Conversion should produce at least one LlamaIndex node."
