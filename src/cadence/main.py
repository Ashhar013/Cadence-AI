import asyncio
import time
from contextlib import asynccontextmanager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, interrupt

from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import AsyncIterator, Optional, Annotated
from typing_extensions import TypedDict, Literal

from cadence.discovery.catalog import get_book_details
from cadence.tools.book_downloader import download_file
from cadence.ingestion.epub_parser import parse_epub
from cadence.ingestion.chunker import chunk_chapters
from cadence.memory.vector_store import get_chroma_client, get_or_create_collection, list_collections
from cadence.memory.embedder import aupsert_chunks
from cadence.memory.library import get_library
from cadence.memory.context import should_summarize, split_messages_for_summary, summarize_conversation
from cadence.agents.rag_agent import aanswer_question

load_dotenv()


# ─── State ───────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Intent
    intent: str
    raw_query: str
    # Discovery
    cleaned_title: str
    author: Optional[str]
    book_results: list[dict]
    # Selection
    selected_book: Optional[dict]
    download_approved: bool
    # Download / Ingestion
    epub_path: Optional[str]
    chapters: list[dict]
    chunks: list[dict]
    collection_name: Optional[str]
    # Memory
    conversation_summary: str
    summarized_message_count: int
    # Observability
    processing_status: str
    error: Optional[str]


# ─── Structured Output Schemas ───────────────────────────────────────────────

class BookSearchQuery(BaseModel):
    cleaned_title: str = Field(
        description="The clean, isolated title of the book extracted from the user query. Remove conversational fluff or descriptions."
    )
    author: Optional[str] = Field(
        default=None,
        description="The name of the author of the book if mentioned in query. Otherwise null."
    )

class IntentClassification(BaseModel):
    intent: Literal['book_search', 'book_chat', 'conversation'] = Field(
        description="Classify the user's message as 'book_search' (find/download book), 'book_chat' (ask question or discuss book content), or 'conversation' (greetings, general chat)."
    )


# ─── LLM Setup ──────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract the actual book title and author from a book-search "
        "request. Remove conversational filler. Do not invent an author."
    ),
    (
        "human", "{raw_query}"
    ),
])

intent_llm = llm.with_structured_output(IntentClassification)
structured_llm = query_prompt | llm.with_structured_output(BookSearchQuery)


# ─── Node: chat_node (Intent Classification Only) ───────────────────────────

def chat_node(state: State) -> dict:
    """Classify user intent. Does NOT generate a response — just routes."""
    latest_user_message = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        None,
    )

    if latest_user_message is None or not str(latest_user_message.content).strip():
        return {
            "intent": "conversation",
            "raw_query": "",
            "processing_status": "awaiting_input",
        }

    raw_query = str(latest_user_message.content).strip()

    start = time.perf_counter()
    classification = intent_llm.invoke([
        SystemMessage(
            content=(
                "Classify the latest user message. "
                "'book_search' means the user wants to find, search for, or download a book. "
                "'book_chat' means the user is asking a question about a book, its concepts, chapters, or contents. "
                "'conversation' means greetings, general chit-chat, or any other query. "
                "Classify the latest message, not an earlier request."
            )
        ),
        latest_user_message,
    ])
    print(f"[PERF] Intent classification: {round(time.perf_counter() - start, 2)}s")

    return {
        "intent": classification.intent,
        "raw_query": raw_query,
        "processing_status": f"intent_classified:{classification.intent}",
    }


def route_intent(state: State) -> str:
    """Route based on classified intent."""
    intent = state.get("intent")
    if intent == "book_search":
        return "query_node"
    elif intent == "book_chat":
        return "rag_node"
    return "general_chat_node"


# ─── Node: general_chat_node (Freeform Conversation) ────────────────────────

def general_chat_node(state: State) -> dict:
    """Handle general conversation — not book search."""
    # If the user provided empty or blank input, give a friendly prompt without calling LLM
    if not state.get("raw_query", "").strip():
        return {
            "messages": [AIMessage(content="Hello! How can I assist you with finding books today?")],
            "processing_status": "conversation_complete",
        }

    messages = state["messages"]
    summary = state.get("conversation_summary", "")
    summarized_message_count = state.get("summarized_message_count", 0)

    # Keep recent turns verbatim and compress only older messages that have not
    # already been included in the running summary.
    if should_summarize(messages):
        older_messages, _ = split_messages_for_summary(messages)
        if len(older_messages) > summarized_message_count:
            summary = summarize_conversation(
                older_messages[summarized_message_count:],
                existing_summary=summary,
            )
            summarized_message_count = len(older_messages)

    prompt_messages = messages
    if summary:
        prompt_messages = [
            SystemMessage(content=f"Conversation summary from earlier turns:\n{summary}"),
            *messages,
        ]

    start = time.perf_counter()
    response = llm.invoke(prompt_messages)
    print(f"[PERF] Chat response: {round(time.perf_counter() - start, 2)}s")

    return {
        "messages": [response],
        "processing_status": "conversation_complete",
        "conversation_summary": summary,
        "summarized_message_count": summarized_message_count,
    }


# ─── Node: query_node (Title/Author Extraction) ─────────────────────────────

def query_node(state: State) -> dict:
    """Extract cleaned book title and author from raw query using Gemini."""
    extraction = structured_llm.invoke({"raw_query": state["raw_query"]})

    return {
        "cleaned_title": extraction.cleaned_title,
        "author": extraction.author,
        "book_results": [],
        "selected_book": None,
        "download_approved": False,
        "epub_path": None,
        "chapters": [],
        "chunks": [],
        "collection_name": None,
        "conversation_summary": state.get("conversation_summary", ""),
        "summarized_message_count": state.get("summarized_message_count", 0),
        "error": None,
        "processing_status": f"query_extracted: {extraction.cleaned_title}",
    }


# ─── Node: book_details_node (Search) ───────────────────────────────────────

async def book_details_node(state: State) -> dict:
    """Search for books using the Open Library API."""
    title = state["cleaned_title"]
    author = state.get("author")
    query = f"{title} {author}".strip() if author else title

    try:
        candidates = await get_book_details(query)
        print(f"[INFO] Open Library search results received: {len(candidates)}")
        results = [
            {
                "title": c.title,
                "authors": c.authors,
                "meta": f"{', '.join(c.authors) if c.authors else 'Unknown'} ({c.meta.first_publish_year or 'N/A'}) • {c.meta.total_editions_cataloged} editions",
                "link": c.meta.link,
            }
            for c in candidates
        ]
        return {
            "book_results": results,
            "processing_status": f"search_complete: {len(results)} results",
        }
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return {
            "book_results": [],
            "error": f"Search failed: {e}",
            "processing_status": "search_failed",
        }


# ─── Node: book_selection_node (HITL Interrupt #1 — Pick a Book) ────────────

def book_selection_node(state: State) -> dict:
    """Present search results and let user pick one. Single interrupt."""
    books = state.get("book_results", [])

    if not books:
        return {
            "selected_book": None,
            "error": "No matching books were found. Try another title or author.",
            "processing_status": "no_results",
        }

    options = []
    for index, book in enumerate(books, start=1):
        options.append({
            "index": index,
            "title": book.get("title", "Unknown title"),
            "metadata": book.get("meta", "Metadata unavailable"),
        })

    # ── Interrupt #1: book selection ──
    selection = interrupt({
        "type": "book_selection",
        "message": "Choose a result by number, or enter 'c' to cancel.",
        "options": options,
    })

    selection = str(selection).strip().lower()

    if selection == "c":
        return {
            "selected_book": None,
            "error": "Book selection cancelled.",
            "processing_status": "selection_cancelled",
        }

    if not selection.isdigit():
        return {
            "selected_book": None,
            "error": "Invalid selection. Please start the search again and choose a valid number.",
            "processing_status": "selection_invalid",
        }

    index = int(selection) - 1
    if not 0 <= index < len(books):
        return {
            "selected_book": None,
            "error": "Selection out of range. Please start the search again.",
            "processing_status": "selection_out_of_range",
        }

    selected_book = books[index]
    return {
        "selected_book": selected_book,
        "error": None,
        "processing_status": f"selected: {selected_book.get('title', 'Unknown')}",
    }


def route_after_selection(state: State) -> str:
    """Route after book selection: if a book was picked, go to approval. Otherwise, show error."""
    if state.get("selected_book") is not None and state.get("error") is None:
        return "approval_node"
    return "error_result_node"


# ─── Node: approval_node (HITL Interrupt #2 — Approve Download) ─────────────

def approval_node(state: State) -> dict:
    """Ask user to confirm download of the selected book. Single interrupt."""
    selected_book = state.get("selected_book", {})

    approval = interrupt({
        "type": "download_approval",
        "message": (
            f"Selected: {selected_book.get('title', 'Unknown title')}. "
            "Approve download? Enter 'yes' or 'no'."
        ),
    })

    approved = str(approval).strip().lower() in {"yes", "y", "approve", "approved"}

    return {
        "download_approved": approved,
        "processing_status": "download_approved" if approved else "download_rejected",
    }


def route_after_approval(state: State) -> str:
    """Route after approval: download if approved, else show cancellation."""
    if state.get("download_approved"):
        return "download_node"
    return "error_result_node"


# ─── Node: download_node ────────────────────────────────────────────────────

async def download_node(state: State) -> dict:
    """Download the selected book's ePub file via Playwright."""
    selected_book = state.get("selected_book")
    if not selected_book:
        return {
            "error": "No book was selected.",
            "processing_status": "download_failed",
        }

    link = selected_book.get("link", "")
    title = selected_book.get("title", "")
    authors = selected_book.get("authors", [])
    author = authors[0] if isinstance(authors, list) and authors else ""

    try:
        epub_path = await download_file(link=link, title=title, author=author)
        return {
            "epub_path": str(epub_path),
            "processing_status": "download_complete",
        }
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return {
            "error": f"Download failed: {e}",
            "processing_status": "download_failed",
        }


# ─── Node: ingestion_node ────────────────────────────────────────────────────

async def ingestion_node(state: State) -> dict:
    """Parse downloaded ePub and split into overlapping chunks for RAG."""
    epub_path_str = state.get("epub_path")
    if not epub_path_str:
        return {
            "error": "No ePub file found to ingest.",
            "processing_status": "ingestion_failed",
        }

    epub_path = Path(epub_path_str)
    if not epub_path.exists():
        return {
            "error": f"ePub file does not exist: {epub_path}",
            "processing_status": "ingestion_failed",
        }

    try:
        chapters = parse_epub(epub_path)
        chunks = chunk_chapters(chapters)
        print(f"[INFO] Ingestion successful: {len(chapters)} chapters, {len(chunks)} chunks.")
        return {
            "chapters": [c.to_dict() for c in chapters],
            "chunks": [c.to_dict() for c in chunks],
            "processing_status": f"ingestion_complete:{len(chapters)}_chapters:{len(chunks)}_chunks",
        }
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        return {
            "error": f"ePub ingestion failed: {e}",
            "processing_status": "ingestion_failed",
        }


def route_after_download(state: State) -> str:
    """Route after download: proceed to ingestion if epub_path exists, else show error."""
    if state.get("epub_path") and not state.get("error"):
        return "ingestion_node"
    return "error_result_node"


# ─── Node: embedding_node (Index Chunks into ChromaDB) ───────────────────────

async def embedding_node(state: State) -> dict:
    """Embed parsed chunks into ChromaDB collection for RAG search."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {
            "error": "No chunks found to embed.",
            "processing_status": "embedding_failed",
        }

    title = state.get("selected_book", {}).get("title", "book")
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name=title)

    try:
        count = await aupsert_chunks(collection, chunks=chunks, book_title=title)
        print(f"[INFO] Embedded {count} chunks into Chroma collection '{collection.name}'")

        # Register book in the user's permanent local library
        lib = get_library()
        lib.register_book(
            title=title,
            authors=state.get("selected_book", {}).get("authors", []),
            epub_path=state.get("epub_path", ""),
            collection_name=collection.name,
            chapter_count=len(state.get("chapters", [])),
            chunk_count=count,
        )

        return {
            "collection_name": collection.name,
            "processing_status": f"embedding_complete:{count}_chunks",
        }
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return {
            "error": f"Failed to embed book chunks: {e}",
            "processing_status": "embedding_failed",
        }


def route_after_ingestion(state: State) -> str:
    """Route after ingestion: proceed to embedding if chunks exist, else show error."""
    if not state.get("error") and state.get("chunks"):
        return "embedding_node"
    return "error_result_node"


def route_after_embedding(state: State) -> str:
    """Route after embedding: proceed to result_node if no error, else show error."""
    if not state.get("error"):
        return "result_node"
    return "error_result_node"


# ─── Node: rag_node (Answer Questions from Book Context) ────────────────────

async def rag_node(state: State) -> dict:
    """Answer questions about books using grounded RAG retrieval or library queries."""
    raw_query = state.get("raw_query", "")
    lib = get_library()
    q_lower = raw_query.lower()

    # Handle library queries: "What books do I have?", "List my books", etc.
    if any(phrase in q_lower for phrase in ["what books", "list my books", "show my books", "my library", "all books"]):
        books = lib.list_books()
        if not books:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Your library is currently empty! "
                            "You can search for and download books anytime (e.g., 'Find Designing Data-Intensive Applications')."
                        )
                    )
                ],
                "processing_status": "library_empty",
            }
        book_lines = [
            f"• **{b.title}** by {', '.join(b.authors) if b.authors else 'Unknown'} "
            f"({b.chapter_count} chapters, {b.chunk_count} chunks indexed)"
            for b in books
        ]
        message_text = "Here are the books currently in your library:\n" + "\n".join(book_lines)
        return {
            "messages": [AIMessage(content=message_text)],
            "processing_status": "library_listed",
        }

    client = get_chroma_client()
    collections = list_collections(client)

    if not collections:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I don't have any books indexed in memory yet. "
                        "Please search for and download a book first (e.g. 'Find Designing Data-Intensive Applications')."
                    )
                )
            ],
            "processing_status": "rag_no_books",
        }

    target_col_name = state.get("collection_name")
    if not target_col_name or target_col_name not in collections:
        target_col_name = collections[0]

    collection = client.get_collection(target_col_name)

    try:
        answer = await aanswer_question(question=raw_query, collection=collection)
        return {
            "messages": [AIMessage(content=answer)],
            "processing_status": "rag_answered",
        }
    except Exception as e:
        print(f"[ERROR] RAG query failed: {e}")
        return {
            "messages": [AIMessage(content=f"Sorry, I encountered an error searching the book: {e}")],
            "error": str(e),
            "processing_status": "rag_failed",
        }


# ─── Node: result_node (Success) ────────────────────────────────────────────

def result_node(state: State) -> dict:
    """Format and return the final success message."""
    epub_path = state.get("epub_path")
    selected_title = state.get("selected_book", {}).get("title", "the book")
    num_chapters = len(state.get("chapters", []))
    num_chunks = len(state.get("chunks", []))
    col_name = state.get("collection_name", "")

    if epub_path:
        message = (
            f"Successfully downloaded and indexed '{selected_title}'!\n"
            f"• Saved to: {epub_path}\n"
            f"• Extracted: {num_chapters} chapters and {num_chunks} chunks\n"
            f"• Indexed in Chroma collection: '{col_name}'\n\n"
            "You can now ask me questions about this book!"
        )
    else:
        message = f"Download and indexing completed for '{selected_title}'."

    return {
        "messages": [AIMessage(content=message)],
        "processing_status": "complete",
    }


# ─── Node: error_result_node (Error / Cancellation) ─────────────────────────

def error_result_node(state: State) -> dict:
    """Format and return error or cancellation messages."""
    error = state.get("error") or "No download was performed."

    return {
        "messages": [AIMessage(content=error)],
        "processing_status": "complete_with_error",
    }


# ─── Graph Assembly ─────────────────────────────────────────────────────────

graph = StateGraph(State)

# Register nodes
graph.add_node("chat_node", chat_node)
graph.add_node("general_chat_node", general_chat_node)
graph.add_node("query_node", query_node)
graph.add_node("book_details_node", book_details_node)
graph.add_node("book_selection_node", book_selection_node)
graph.add_node("approval_node", approval_node)
graph.add_node("download_node", download_node)
graph.add_node("ingestion_node", ingestion_node)
graph.add_node("embedding_node", embedding_node)
graph.add_node("rag_node", rag_node)
graph.add_node("result_node", result_node)
graph.add_node("error_result_node", error_result_node)

# Wire edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", route_intent)
graph.add_edge("general_chat_node", END)
graph.add_edge("rag_node", END)
graph.add_edge("query_node", "book_details_node")
graph.add_edge("book_details_node", "book_selection_node")
graph.add_conditional_edges("book_selection_node", route_after_selection)
graph.add_conditional_edges("approval_node", route_after_approval)
graph.add_conditional_edges("download_node", route_after_download)
graph.add_conditional_edges("ingestion_node", route_after_ingestion)
graph.add_conditional_edges("embedding_node", route_after_embedding)
graph.add_edge("result_node", END)
graph.add_edge("error_result_node", END)

def build_app(checkpointer=None):
    """Compile the graph with an optional LangGraph checkpointer."""
    return graph.compile(checkpointer=checkpointer)


# Kept for callers that only need an in-memory, one-shot graph invocation.
# The CLI uses `persistent_app`, which is required for safe async SQLite access.
app = build_app()


@asynccontextmanager
async def persistent_app(
    db_path: Path = Path("data/processed/cadence_memory.db"),
) -> AsyncIterator:
    """Yield an async graph backed by a persistent SQLite checkpointer."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        yield build_app(checkpointer=checkpointer)


# ─── CLI Entry Point ────────────────────────────────────────────────────────

async def main() -> None:
    config = {"configurable": {"thread_id": "cadence-cli"}}

    print("Welcome to Cadence AI. Type 'exit' to quit.")

    async with persistent_app() as persistent_graph:
        while True:
            user_text = input("\nYou: ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            result = await persistent_graph.ainvoke(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
            )

            # LangGraph interrupts pause execution and return __interrupt__.
            while result.get("__interrupt__"):
                interrupt_value = result["__interrupt__"][0].value
                print("\n", interrupt_value)

                if interrupt_value.get("type") == "book_selection":
                    answer = input("Selection: ").strip()
                elif interrupt_value.get("type") == "download_approval":
                    answer = input("Approve (yes/no): ").strip()
                else:
                    answer = input("Your response: ").strip()

                result = await persistent_graph.ainvoke(Command(resume=answer), config=config)

            # Print the newest assistant message produced by this turn.
            for message in reversed(result.get("messages", [])):
                if isinstance(message, AIMessage):
                    content = message.content

                    if isinstance(content, list):
                        content = "".join(
                            item["text"]
                            for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )

                    print(f"\nCadence: {content}")
                    break


if __name__ == "__main__":
    asyncio.run(main())
