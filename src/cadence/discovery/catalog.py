import httpx
from typing import List
import asyncio
from cadence.discovery.models import BookCandidate, BookMetadata

async def search_books(query: str) -> List[BookCandidate]:
    """Quick title-only search against Open Library."""
    url = "https://openlibrary.org/search.json"
    params = {"title": query, "limit": 5}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

async def get_book_details(search_query: str) -> List[BookCandidate]:
    """Fetches a book's title, author, metadata, and web link from Open Library."""
    url = "https://openlibrary.org/search.json"

    # 1. Define the fields you want to minimize payload size and improve speed
    fields = "key,title,author_name,first_publish_year,edition_count,subject"
    params = {"q": search_query, "fields": fields, "limit": 5}

    # 2. Add an identified User-Agent for better rate-limiting stability
    headers = {"User-Agent": "CadenceAI/0.1.0 (cadence-book-assistant)"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()

        # Check if any books matched the query
        if not data.get("docs"):
            return []

        candidates = []

        # 3. Iterate over ALL returned docs (up to limit=5), not just the first
        for book in data["docs"]:
            title = book.get("title", "Unknown Title")

            # Authors are returned as a list — keep them as a list
            authors_list = book.get("author_name", [])

            # Construct the web link using the unique work path (e.g., /works/OL27479W)
            web_link = f"https://openlibrary.org{book.get('key', '')}"

            # Consolidate desired metadata into a clean dictionary
            metadata = BookMetadata(
                first_publish_year=book.get("first_publish_year", None),
                total_editions_cataloged=book.get("edition_count", 0),
                top_subjects=book.get("subject", [])[:5],
                link=web_link
            )

            candidate = BookCandidate(
                title=title,
                authors=authors_list,
                meta=metadata
            )
            candidates.append(candidate)

        return candidates

    except httpx.HTTPStatusError as e:
        print(f"[ERROR] HTTP error during Open Library API call: {e}")
        return []
    except httpx.RequestError as e:
        print(f"[ERROR] Network error during Open Library API call: {e}")
        return []

