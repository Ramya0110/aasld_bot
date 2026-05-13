from scraper_utils import WebScraper
import json

def verify():
    url = "https://aasldv2022dev.aasld.org/events-and-webinars"
    print(f"Testing scraper on: {url}")
    scraper = WebScraper("https://aasldv2022dev.aasld.org/")
    res = scraper.get_page_content(url)
    
    if res:
        print(f"Title: {res['title']}")
        print(f"Length of text: {len(res['text'])}")
        found_resmetirom = "Resmetirom" in res['text']
        print(f"Found 'Resmetirom': {found_resmetirom}")
        
        # Look for 2026 events as well
        found_2026 = "2026" in res['text']
        print(f"Found '2026': {found_2026}")
        
        if found_resmetirom or found_2026:
            print("SUCCESS: Dynamic content captured.")
        else:
            print("FAILURE: Dynamic content NOT captured. Content might still be empty or error-prone.")
            print("Snippet:", res['text'][:1000])
    else:
        print("FAILURE: Scraper returned None.")

if __name__ == "__main__":
    verify()
