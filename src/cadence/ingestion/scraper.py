import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional
from playwright.async_api import async_playwright, BrowserContext, Page, Error as PlaywrightError

class ShadowLibraryScrapper:
    """
    Playwright-based scrapper to retrieve ePub files from the Shadow Library website.
    It uses HITL (Human-in-the-loop) to handle CAPTCHA challenges that may arise during the scraping process.
    """
    def __init__(self, download_dir: str = "data/raw", headless: bool = False):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless
        self.browser_context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def search_books(self, query: str, page: Page) -> List[Dict[str, str]]:
        """
        Search for books and extract metadata for top ePubs.
        """
        search_url = (
        "https://annas-archive.gl/search"
        f"?index=&page=1&sort=&ext=epub&display=&q={query}&check=1"
    )
        
        # 1. Navigate to search URL
        await page.goto(search_url, wait_until="domcontentloaded")

        # Give the page a moment to settle/redirect
        await asyncio.sleep(3)

        # 2. Check if DDoS-Guard or Cloudflare challenge is present
        content = await page.content()
        if any(term in content.lower() for term in ["ddos-guard", "challenge", "cf-turnstile", "just a moment"]):
            print("\n[!] DDoS-Guard / CAPTCHA detected.")
            print("[!] Solve the challenge in the open browser window...")

            # Poll until the user solves it and search results appear (up to 120 seconds)
            solved = False
            for _ in range(60):  # 60 iterations * 2s = 120s timeout
                await asyncio.sleep(2)
                try:
                    cards = await page.query_selector_all("a[href^='/md5/']")
                    if len(cards) > 0:
                        solved = True
                        print("[✓] Challenge cleared and search results loaded!")
                        break

                except PlaywrightError as e:
                        # Catch 'Execution context was destroyed' when the page reloads after solving CAPTCHA
                        if "context was destroyed" in str(e).lower() or "navigation" in str(e).lower():
                            print("[*] Page reloaded/redirected after verification. Retrying check...")
                            await asyncio.sleep(2)
                            continue
                        raise e

            if not solved:
                raise TimeoutError("CAPTCHA manual solving timed out after 120s.")

        else:
        # If no challenge, wait up to 10s for the result list to render
            await page.wait_for_selector("a[href^='/md5/']", timeout=10000)       

        results = []
        print("[DEBUG] URL after CAPTCHA:", page.url)
        print("[DEBUG] Title after CAPTCHA:", await page.title())
        cards = await page.query_selector_all("a[href^='/md5/']")
        print("[DEBUG] Cards found:", len(cards))

        for card in cards:
            full_card_text = await card.inner_text()
            href = await card.get_attribute("href")

            print(f"href: {href}")
            print(f"contains epub: {'epub' in full_card_text.lower()}")
            print(f"text preview: {full_card_text[:150]!r}")

            title_el = await card.query_selector("h3")
            title = await title_el.inner_text() if title_el else full_card_text.split('\n')[0]  # Fallback to first line of card text if title not found

            full_link = f"https://annas-archive.gl{href}" if href.startswith("/") else href
            results.append({
                "title": title.strip(),
                "meta": full_card_text.replace('\n', ' ').strip()[:100],
                "link": full_link
            })

            if len(results) >= 10:  # Limit to top 5 results
                break

        return results

    async def download_epub(self, book_link: str, page: Page) -> Path:
        """
        Download the ePub file from the given book link.
        """
        print(f"\n[*] Navigating to book page: {book_link}")
        await page.goto(book_link, wait_until="domcontentloaded")
        await asyncio.sleep(2)  # Allow time for the page to load

        # Locate and click the initial slow/fast partner download option
        partner_download_link = await page.query_selector("a[href*='/slow_download/'], a:has-text('Slow Partner Server')")

        if partner_download_link:
            print("[*] Clicking on the partner download link...")
            await partner_download_link.click()
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(3)  # Allow time for the download page to load

        # Check if a secondary challenge page loaded
        content = await page.content()
        if any(term in content.lower() for term in ["ddos-guard", "challenge", "cf-turnstile", "just a moment"]):
            print("\n[!] DDoS-Guard / CAPTCHA detected on download page.")
            print("[!] Solve the challenge in the open browser window...")

            # Poll until the user solves it and download button appears (up to 120 seconds)
            solved = False
            for _ in range(60):  # 60 iterations * 2s = 120s timeout
                await asyncio.sleep(2)
                try:
                    download_button = await page.query_selector("a:has-text('Download now'), a[href*='download'], a[href$='.epub']")
                    if download_button:
                        solved = True
                        print("[✓] Challenge cleared and download button available!")
                        break

                except PlaywrightError as e:
                    if "context was destroyed" in str(e).lower() or "navigation" in str(e).lower():
                        print("[*] Page reloaded/redirected after verification. Retrying check...")
                        await asyncio.sleep(2)
                        continue
                    raise e

            if not solved:
                raise TimeoutError("CAPTCHA manual solving timed out after 120s.")

        #Final candidate download links
        download_links = page.locator("a:has-text('Download now'), a:has-text('Slow Partner Download'), a[href$='.epub'], a[href*='download']")

        count = await download_links.count()
        if count == 0:
            raise RuntimeError("Could not locate any direct download links on the page.")

        print(f"[*] Found {count} potential download link(s).")
        
        # Wait for the download button to appear
        final_download_button = None

        for i in range(count):
            candidate = download_links.nth(i)

            if await candidate.is_visible():
                text = (await candidate.inner_text()).strip()
                href = await candidate.get_attribute("href")

                print(f"[*] Visible candidate: text={text!r}, href={href!r}")

                final_download_button = candidate
                break

        if final_download_button is None:
            await page.screenshot(path="data/raw/download_debug.png",full_page=True)

            raise RuntimeError(
            "No visible download link found. "
            "Inspect data/raw/download_debug.png and the page DOM."
            )

        print(f"[*] Clicking on the download button: {await final_download_button.inner_text()}")

        async with page.expect_download(timeout=120000) as download_info:
            await final_download_button.click()

        download = await download_info.value
        # Save the downloaded file to the specified directory
        filename = download.suggested_filename or "download.epub"
        epub_path = self.download_dir / filename
        await download.save_as(epub_path)
        print(f"[✓] Successfully Downloaded ePub to: {epub_path}")

        return epub_path

