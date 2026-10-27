import asyncio
from cadence.ingestion.scraper import ShadowLibraryScrapper
from playwright.async_api import async_playwright
from typing import List, Dict

async def get_books_details(query: str) -> List[Dict[str, str]]:
    """Core scraping logic that will be called by our LangGraph tool."""
    scrapper = ShadowLibraryScrapper(download_dir="data/raw", headless=False)

    print(f"Searching for book : '{query}'")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox"
            ]
        )
        context = await browser.new_context(
            accept_downloads=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Search for books
        results = await scrapper.search_books(query, page)
        await browser.close()
        return results


if __name__ == "__main__":
    asyncio.run(get_books_details("test"))



