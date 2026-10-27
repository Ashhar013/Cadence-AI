"""Persistent local book registry tracking downloaded, parsed, and indexed books."""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from cadence.utils.config import DATA_PROCESSED_DIR
from cadence.memory.vector_store import sanitize_collection_name


@dataclass
class LibraryEntry:
    id: str
    title: str
    authors: List[str]
    epub_path: str
    collection_name: str
    chapter_count: int
    chunk_count: int
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LibraryEntry":
        return cls(**data)


class BookLibrary:
    """Persistent catalog registry for user's books."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or (Path(DATA_PROCESSED_DIR) / "library.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, LibraryEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._entries = {k: LibraryEntry.from_dict(v) for k, v in data.items()}
            except Exception as e:
                print(f"[WARN] Failed to load book library from {self.storage_path}: {e}")
                self._entries = {}
        else:
            self._entries = {}

    def _save(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self._entries.items()}, f, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to save book library to {self.storage_path}: {e}")

    def register_book(
        self,
        title: str,
        authors: List[str],
        epub_path: str,
        collection_name: Optional[str] = None,
        chapter_count: int = 0,
        chunk_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LibraryEntry:
        """Register a new book or update an existing entry."""
        col_name = collection_name or sanitize_collection_name(title)
        book_id = col_name

        entry = LibraryEntry(
            id=book_id,
            title=title.strip(),
            authors=authors or [],
            epub_path=str(epub_path),
            collection_name=col_name,
            chapter_count=chapter_count,
            chunk_count=chunk_count,
            metadata=metadata or {},
        )
        self._entries[book_id] = entry
        self._save()
        return entry

    def get_book(self, book_id_or_title: str) -> Optional[LibraryEntry]:
        """Find a book by exact ID or title."""
        key = sanitize_collection_name(book_id_or_title)
        if key in self._entries:
            return self._entries[key]

        target = book_id_or_title.strip().lower()
        for entry in self._entries.values():
            if entry.title.lower() == target:
                return entry
        return None

    def list_books(self) -> List[LibraryEntry]:
        """Return all books in the library ordered by addition date."""
        return list(self._entries.values())

    def search_books(self, query: str) -> List[LibraryEntry]:
        """Search library by title, author, or keyword."""
        q = query.strip().lower()
        results = []
        for entry in self._entries.values():
            in_title = q in entry.title.lower()
            in_authors = any(q in a.lower() for a in entry.authors)
            if in_title or in_authors:
                results.append(entry)
        return results

    def remove_book(self, book_id: str) -> bool:
        """Remove a book entry from the registry."""
        key = sanitize_collection_name(book_id)
        if key in self._entries:
            del self._entries[key]
            self._save()
            return True
        return False


_global_library: Optional[BookLibrary] = None


def get_library(storage_path: Optional[Path] = None) -> BookLibrary:
    """Singleton getter for the BookLibrary."""
    global _global_library
    if _global_library is None or storage_path is not None:
        _global_library = BookLibrary(storage_path=storage_path)
    return _global_library
