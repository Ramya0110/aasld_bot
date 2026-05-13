from scraper_utils import WebScraper
import json

def test_header_extraction():
    url = "https://aasldv2022dev.aasld.org/"
    scraper = WebScraper(url)
    print(f"Scraping {url}...")
    data = scraper.get_page_content(url)
    
    if data:
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(data.get("html_debug", ""))
        
        print("\n--- Header Links Found ---")
        for link in data.get("header_links", []):
            print(f"[{link['context']}] {link['text']} -> {link['url']}")
            
        print(f"\nTotal header links: {len(data.get('header_links', []))}")
        print(f"Total outgoing links: {len(data.get('outgoing_links', []))}")
    else:
        print("Failed to scrape data.")

if __name__ == "__main__":
    test_header_extraction()
