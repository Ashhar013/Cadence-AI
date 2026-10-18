import asyncio
from multiprocessing import context
from src.cadence.ingestion.scraper import ShadowLibraryScrapper
from playwright.async_api import async_playwright

async def run_scraper_test():
    scrapper = ShadowLibraryScrapper(download_dir="data/raw", headless=False)
    query = "Designing Data-Intensive Applications"

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
        
        print("\nSearch Results:")
        for idx, item in enumerate(results, start=1):
            print(f"{idx}. {item['title']} - {item['meta']}")

        # Download the first ePub found
        if results:
            first_book_link = results[0]["link"]
            epub_path = await scrapper.download_epub(first_book_link, page)
            print(f"\nDownloaded ePub to: {epub_path}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_scraper_test())