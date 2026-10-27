"""Unit and integration tests for ePub ingestion and chunking."""

from pathlib import Path
import pytest

from cadence.ingestion.epub_parser import parse_epub, Chapter
from cadence.ingestion.chunker import chunk_text, chunk_chapters, Chunk
from cadence.main import ingestion_node, State


@pytest.fixture
def raw_epub_path() -> Path:
    """Fixture to find the raw ePub in data/raw."""
    raw_dir = Path("data/raw")
    epubs = list(raw_dir.glob("*.epub"))
    if not epubs:
        pytest.skip("No .epub files found in data/raw/ to test with.")
    return epubs[0]


def test_parse_ddia_epub(raw_epub_path: Path):
    """Verify that the ePub parser cleanly extracts chapters from the book."""
    chapters = parse_epub(raw_epub_path)

    assert len(chapters) >= 10, f"Expected at least 10 chapters, got {len(chapters)}"
    
    # Verify properties of extracted chapters
    for ch in chapters:
        assert isinstance(ch, Chapter)
        assert ch.title and len(ch.title) > 0
        assert ch.content and len(ch.content) > 0
        assert ch.char_count == len(ch.content)

    # Verify that major topics/chapters are recognized
    all_titles = " ".join(ch.title for ch in chapters)
    assert any(keyword in all_titles for keyword in ["Data Models", "Storage", "Partitioning", "Replication", "Foundations"])


def test_chunk_text_basic():
    """Verify that chunk_text splits paragraphs and respects bounds."""
    sample_text = (
        "This is paragraph one about databases and consistency models.\n\n"
        "This is paragraph two about leaderless replication and dynamo-style systems.\n\n"
        "This is paragraph three about distributed transactions and two-phase commit."
    )
    chunks = chunk_text(sample_text, chapter_title="Chapter 1", chapter_index=0, max_chars=100, overlap=20)
    
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chapter_title == "Chapter 1"
        assert len(c.text) > 0
        assert c.char_count == len(c.text)


def test_chunk_ddia_chapters(raw_epub_path: Path):
    """Verify that chapters chunk into overlapping segments ready for RAG."""
    chapters = parse_epub(raw_epub_path)
    chunks = chunk_chapters(chapters, max_chars=1500, overlap=200)

    assert len(chunks) > 100, f"Expected > 100 chunks for DDIA, got {len(chunks)}"

    # Check contiguous indexing
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert c.chapter_title is not None
        assert len(c.text) > 0


@pytest.mark.asyncio
async def test_ingestion_node_execution(raw_epub_path: Path):
    """Test the LangGraph ingestion_node with the existing ePub file."""
    state = {
        "messages": [],
        "intent": "book_search",
        "raw_query": "Designing Data-Intensive Applications",
        "cleaned_title": "Designing Data-Intensive Applications",
        "author": "Martin Kleppmann",
        "book_results": [],
        "selected_book": {"title": "Designing Data-Intensive Applications"},
        "download_approved": True,
        "epub_path": str(raw_epub_path),
        "chapters": [],
        "chunks": [],
        "processing_status": "download_complete",
        "error": None,
    }

    result = await ingestion_node(state)

    assert "chapters" in result
    assert "chunks" in result
    assert len(result["chapters"]) >= 10
    assert len(result["chunks"]) > 100
    assert "ingestion_complete" in result["processing_status"]
    assert result.get("error") is None
