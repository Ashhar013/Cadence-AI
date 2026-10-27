"""ChromaDB vector store setup and collection management."""

import re
from pathlib import Path
from typing import List, Optional
import chromadb
from chromadb.api.models.Collection import Collection

from cadence.utils.config import CHROMA_PERSIST_DIR


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize an arbitrary book title into a valid ChromaDB collection name.
    
    Chroma naming rules:
    - 3-63 characters
    - Only alphanumeric, underscores, hyphens
    - Must start and end with an alphanumeric character
    """
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
    clean = re.sub(r"_+", "_", clean).strip("_-")
    if not clean:
        return "book_collection"
    if len(clean) < 3:
        clean = f"book_{clean}".rstrip("_-")
    clean = clean[:63].rstrip("_-")
    return clean or "book_collection"


def get_chroma_client(persist_dir: Optional[str] = None) -> chromadb.PersistentClient:
    """Initialize or connect to a local ChromaDB persistent instance."""
    storage_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(storage_dir))


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    distance_metric: str = "cosine",
) -> Collection:
    """
    Retrieve or create a ChromaDB collection for a book.
    
    Args:
        client: The ChromaDB persistent client.
        collection_name: The name or title for the collection (will be sanitized).
        distance_metric: Distance metric for HNSW index ('cosine', 'l2', 'ip').
    """
    safe_name = sanitize_collection_name(collection_name)
    return client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": distance_metric},
    )


def list_collections(client: chromadb.PersistentClient) -> List[str]:
    """Return all collection names currently stored in ChromaDB."""
    return [col.name for col in client.list_collections()]
