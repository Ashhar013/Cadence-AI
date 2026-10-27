import asyncio
from playwright.async_api import async_playwright
from cadence.ingestion.scraper import ShadowLibraryScrapper

from typing import Optional
from pathlib import Path

async def download_file(link: str = "", title: Optional[str] = None, author: Optional[str] = None) -> Path:
    """Runs the download step using either a direct shadow library link or title/author search."""
    scrapper = ShadowLibraryScrapper(download_dir="data/raw", headless=False)

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

        try:
            if link and "annas-archive" in link:
                epub_path = await scrapper.download_epub(link, page)
            else:
                search_query = f"{title} {author}".strip() if (title or author) else link
                print(f"[*] Searching shadow library for ePub: '{search_query}'")
                results = await scrapper.search_books(search_query, page)
                if not results:
                    raise RuntimeError(f"No ePub download results found for: {search_query}")
                epub_path = await scrapper.download_epub(results[0]["link"], page)

            return epub_path
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(download_file(title="Designing Data-Intensive Applications"))