"""Unit and integration tests for Phase 4: Persistent memory, BookLibrary, and conversation context."""

from types import SimpleNamespace
from pathlib import Path
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from cadence.memory.library import BookLibrary, LibraryEntry
from cadence.memory.context import (
    should_summarize,
    split_messages_for_summary,
    summarize_conversation,
)
from cadence.main import persistent_app


@pytest.fixture
def temp_library_file(tmp_path):
    """Temporary file path for testing BookLibrary isolation."""
    return tmp_path / "test_library.json"


def test_book_library_crud(temp_library_file: Path):
    """Verify that BookLibrary handles registration, search, and retrieval."""
    lib = BookLibrary(storage_path=temp_library_file)

    # Initial state
    assert len(lib.list_books()) == 0

    # Register book
    entry = lib.register_book(
        title="Designing Data-Intensive Applications",
        authors=["Martin Kleppmann"],
        epub_path="data/raw/ddia.epub",
        collection_name="ddia_collection",
        chapter_count=12,
        chunk_count=904,
    )

    assert entry.id == "ddia_collection"
    assert len(lib.list_books()) == 1

    # Retrieve by ID and title
    found_by_id = lib.get_book("ddia_collection")
    assert found_by_id is not None
    assert found_by_id.title == "Designing Data-Intensive Applications"

    found_by_title = lib.get_book("Designing Data-Intensive Applications")
    assert found_by_title is not None
    assert found_by_title.chunk_count == 904

    # Search
    results = lib.search_books("Kleppmann")
    assert len(results) == 1
    assert results[0].authors == ["Martin Kleppmann"]

    results_empty = lib.search_books("Nonexistent Author")
    assert len(results_empty) == 0


def test_context_split_and_threshold():
    """Verify dialog turn threshold checking and message splitting."""
    # Under threshold
    short_history = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="Find a book"),
    ]
    assert not should_summarize(short_history, threshold=6)

    # Exceeding threshold
    long_history = []
    for i in range(7):
        long_history.append(HumanMessage(content=f"Question {i}"))
        long_history.append(AIMessage(content=f"Answer {i}"))

    assert should_summarize(long_history, threshold=10)

    older, recent = split_messages_for_summary(long_history, keep_recent=4)
    assert len(recent) == 4
    assert len(older) == len(long_history) - 4
    assert recent[-1].content == "Answer 6"


def test_summarize_conversation(monkeypatch):
    """Verify summary construction without making a live Gemini request."""
    class FakeClient:
        def models(self):
            raise AssertionError("This method should not be called")

    class FakeModels:
        @staticmethod
        def generate_content(**_kwargs):
            return SimpleNamespace(
                text="Alex downloaded Designing Data-Intensive Applications and asked about replication."
            )

    class FakeGenAIClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr("cadence.memory.context.genai.Client", FakeGenAIClient)
    history = [
        HumanMessage(content="Can you find Designing Data-Intensive Applications by Martin Kleppmann?"),
        AIMessage(content="Yes, I found the book and downloaded it."),
        HumanMessage(content="Tell me what Chapter 5 discusses."),
        AIMessage(content="Chapter 5 discusses single-leader, multi-leader, and leaderless replication."),
    ]

    summary = summarize_conversation(history)
    assert len(summary) > 20
    assert any(term in summary.lower() for term in ["designing", "data", "replication", "kleppmann", "chapter"])


@pytest.mark.asyncio
async def test_sqlite_persistence_across_invocations(monkeypatch, tmp_path):
    """Verify that thread state is persisted to SQLite across calls."""
    import cadence.main as main_module

    monkeypatch.setattr(
        main_module,
        "intent_llm",
        SimpleNamespace(invoke=lambda _messages: SimpleNamespace(intent="conversation")),
    )
    monkeypatch.setattr(
        main_module,
        "llm",
        SimpleNamespace(invoke=lambda _messages: AIMessage(content="Hello, Alex.")),
    )

    thread_id = "test-persistent-thread-offline"
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1
    async with persistent_app(tmp_path / "cadence_memory.db") as app:
        result1 = await app.ainvoke(
            {"messages": [HumanMessage(content="Hello, my name is Alex.")]},
            config=config,
        )
        assert result1["processing_status"] == "conversation_complete"

        # Verify the completed turn was written to SQLite.
        saved_state = await app.aget_state(config)
        messages = saved_state.values.get("messages", [])
        assert len(messages) >= 2


@pytest.mark.asyncio
async def test_library_listing_via_graph(monkeypatch, tmp_path):
    """Verify that asking 'What books do I have in my library?' routes and returns response."""
    import cadence.main as main_module

    monkeypatch.setattr(
        main_module,
        "intent_llm",
        SimpleNamespace(invoke=lambda _messages: SimpleNamespace(intent="book_chat")),
    )
    config = {"configurable": {"thread_id": "test-library-query-thread"}}
    async with persistent_app(tmp_path / "cadence_memory.db") as app:
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="What books do I have in my library?")]},
            config=config,
        )

    assert result.get("intent") in ["book_chat", "conversation"]
    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    assert len(ai_messages) >= 1
    assert "library" in ai_messages[-1].content.lower()
