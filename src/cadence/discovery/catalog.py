import httpx
from typing import List, Union
import asyncio
from functools import partial
from cadence.discovery.models import BookCandidate, BookMetadata

async def search_books(query: str) -> Union[List[BookCandidate], str]:
    url = "https://openlibrary.org/search.json"
    params = {"title": query}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        return response.json()

async def get_book_details(search_query: str) -> List[BookCandidate]:
    """Fetches a book's title, author, metadata, and web link from Open Library."""
    url = "https://openlibrary.org/search.json"

    # 1. Define the fields you want to minimize payload size and improve speed
    fields = "key,title,author_name,first_publish_year,edition_count,subject"
    params = {"q": search_query, "fields": fields, "limit": 5}

    # 2. Add an identified User-Agent for better rate-limiting stability
    headers = {"User-Agent": "BookFinderApp/1.0 (your-email@example.com)"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()  # Check for HTTP request errors

            data = response.json()

        candidates = []

        # Check if any books matched the query
        if not data.get("docs"):
            return "No books found matching that query."

        book = data["docs"][0]

        # 3. Clean and parse the payload
        title = book.get("title", "Unknown Title")

        # Authors are returned as a list, join them with commas
        authors_list = book.get("author_name", [])
        #authors = ", ".join(authors_list) if authors_list else "Unknown Author"

        # Construct the web link using the unique work path (e.g., /works/OL27479W)
        web_link = f"https://openlibrary.org/search.json{book.get('key', '')}"

        # Consolidate desired metadata into a clean dictionary
        metadata = BookMetadata(
            first_publish_year=book.get("first_publish_year", None),
            total_editions_cataloged=book.get("edition_count", 0),
            top_subjects=book.get("subject", [])[:5],
            link=web_link
        )

        candidate = BookCandidate(
            title=title,
            authors= ", ".join(authors_list) if authors_list else [],
            meta=metadata
        )
        candidates.append(candidate)

        return candidates

    except httpx.HTTPStatusError as e:
        return f"An HTTP error occurred during the API call: {e}"
    except httpx.RequestError as e:
        return f"A network error occurred during the API call: {e}"

