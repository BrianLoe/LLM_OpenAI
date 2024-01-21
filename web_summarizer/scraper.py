from playwright.async_api import async_playwright
from html_cleaner import HTMLCleaner
import pprint
import asyncio

async def ascrape_playwright(url, tags = ["h1", "h2", "h3", "h4", "span"]) -> str:
    """
    An asynchronous Python function that uses Playwright to scrape
    content from a given URL, extracting specified HTML tags and removing unwanted tags and unnecessary
    lines.
    """
    print("Started scraping...")
    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            html_cleaner = HTMLCleaner(page_source)
            results = html_cleaner.clean_html_content(tags)
            print("Content scraped")
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results

if __name__ == "__main__":
    url = "https://www.patagonia.ca/shop/new-arrivals"

    async def scrape_playwright():
        results = await ascrape_playwright(url)
        print(results)

    pprint.pprint(asyncio.run(scrape_playwright()))