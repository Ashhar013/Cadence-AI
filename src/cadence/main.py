import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

from cadence.discovery.models import BookCandidate
from  cadence.discovery.catalog import get_book_details

load_dotenv()

llm = genai.Client()

class BookSearchQuery(BaseModel):
    cleaned_title: str = Field(
        description="The clean, isolated title of the book extracted from the user query. Remove conversational fluff or descriptions."
    )
    author: Optional[str] = Field(
        default=None,
        description="The name of the author of the book if mentioned in query. Otherwise null."
    )

async def generate_book_search_query(raw_query: str) -> BookSearchQuery:
    system_instruction = (
        "You are an expert library search assistant. Analyze the user's raw search phrase, "
        "extract the actual book title and author name, and isolate them from conversational "
        "fluff or background descriptions."
    )

    response = await llm.aio.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Extract the book title and author from this query: '{raw_query}'",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=BookSearchQuery
        )            
    )

    return BookSearchQuery.model_validate_json(response.text)


def display_book_info(book_candidate: BookCandidate):
    print(f"\nTitle:   {book_candidate.title}")
    print(f"Author:  {book_candidate.authors}")
    print(f"Link:    {book_candidate.meta.link}")
    print("Metadata:")
    print(f"   • First Published:  {book_candidate.meta.first_publish_year}")
    print(
        f"   • Total Editions:   {book_candidate.meta.total_editions_cataloged}"
    )
    print(f"   • Key Subjects:     {', '.join(book_candidate.meta.top_subjects)}")

async def main():
    print('''
========================================
        GRAPHCAST
   Your Personal Book Assistant
========================================
            ''')
    while True:

        input_query = await asyncio.to_thread(input, "Enter a book title to search: ")

        if input_query.lower() == 'exit':
            print("Exiting the application. Goodbye!")
            break

        if not input_query.strip():
            continue

        try:
            print("\nSearching for books...")

            search_criteria = await generate_book_search_query(input_query)

            search_description = f"Searching for books with title '{search_criteria.cleaned_title}'"
            if search_criteria.author:
                search_message = f" by '{search_criteria.author}'"
            print(search_description + (search_message if search_criteria.author else ""))

            result = await get_book_details(search_criteria.cleaned_title)

            if isinstance(result, list) and len(result) > 0:
                print(f"\nFound {len(result)} book matches.")
                for index, book in enumerate(result, start=1):
                    print(f"[{index}] {book.title} -by {', '.join(book.authors)}")

                while True:
                    selection = await asyncio.to_thread(
                        input, "\nEnter the number of the book to view details (or 'c' to cancel): "
                    )

                    selection = selection.strip().lower()

                    if selection == 'c':
                        break

                    if selection.isdigit():
                        choice_idx = int(selection) - 1
                        if 0 <= choice_idx < len(result):
                            selected_book = result[choice_idx]
                            display_book_info(selected_book)
                            break

                    print("Invalid selection. Please enter a valid number or 'c' to cancel.")

            elif isinstance(result, BookCandidate):
                display_book_info(result)

            else:
                print("\nNo books found matching that query.")

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
