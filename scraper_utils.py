import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib3
import time

# Suppress insecure request warnings for dev environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WebScraper:
    def __init__(self, base_url, verify_ssl=False):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.verify_ssl = verify_ssl
        self.visited_urls = set()
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_page_content(self, url, page=None):
        """Fetches and extracts title and main content from a URL. Optionally uses an existing Playwright page."""
        from playwright.sync_api import sync_playwright
        
        own_browser = False
        if page is None:
            own_browser = True
            p = sync_playwright().start()
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(ignore_https_errors=True, user_agent=self.headers["User-Agent"])
            page = context.new_page()
        
        try:
            print(f"Playwright: Navigating to {url}")
            # Use 'load' which is generally faster than 'networkidle' and almost as good for our needs
            page.goto(url, wait_until="load", timeout=45000)
            
            # Additional small wait to ensure dynamic lists are rendered
            page.wait_for_timeout(3000)
            
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            title = page.title() or "Untitled"

            # Extract Full Body Content
            main_content = soup.body
            
            # Extract Header Links specifically (do this before replacing a tags with text)
            header_links = self.extract_header_links(soup, url)
            
            if main_content:
                # Remove unwanted technical elements
                for tag in main_content.find_all(["script", "style", "noscript", "iframe", "svg"]):
                    tag.decompose()
                
                # Extract all outgoing links
                outgoing_links = []
                for a_tag in main_content.find_all("a", href=True):
                    link_text = a_tag.get_text(strip=True)
                    absolute_link = urljoin(url, a_tag["href"])
                    parent = a_tag.find_parent()
                    context_text = parent.get_text(strip=True) if parent else ""
                    
                    outgoing_links.append({
                        "text": link_text, 
                        "url": absolute_link,
                        "context": context_text
                    })
                    
                    if link_text:
                        a_tag.replace_with(f"[{link_text}]({absolute_link})")
                
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = "" # Ensure text is defined even if main_content is None
            
            # Combine header links text into the main text for context
            if header_links:
                header_text = "\n\n--- SITE NAVIGATION MENU ---\n"
                for link in header_links:
                    header_text += f"{link['context']} > {link['text']}: {link['url']}\n"
                text += header_text
            
            snippet = text[:500] + "..." if len(text) > 500 else text
            
            # Ensure header links are included in outgoing_links for crawling
            header_urls = {link["url"] for link in header_links}
            existing_urls = {link["url"] for link in outgoing_links}
            
            for h_link in header_links:
                if h_link["url"] not in existing_urls:
                    outgoing_links.append(h_link)
            
            return {
                "url": url, 
                "title": title, 
                "text": text, 
                "snippet": snippet,
                "outgoing_links": outgoing_links,
                "header_links": header_links,
                "html_debug": html_content
            }
        except Exception as e:
            print(f"Error scraping {url} with Playwright: {e}")
            return None
        finally:
            if own_browser:
                browser.close()
                p.stop()
        
        return None

    def extract_header_links(self, soup, current_url):
        """Extracts navigation links from the tophat and main dropdown menus."""
        header_links = []
        seen_urls = set()
        
        def add_link(a_tag, context_name):
            link_text = a_tag.get_text(strip=True)
            href = a_tag.get("href", "").strip()
            
            if not link_text or not href or href.startswith("#"):
                return
                
            absolute_link = urljoin(current_url, href)
            # Use a tuple of (text, url) to allow same URL with different text, 
            # or just URL to be strict. Let's do (text, url) to capture all distinct info.
            if (link_text, absolute_link) not in seen_urls:
                header_links.append({
                    "text": link_text,
                    "url": absolute_link,
                    "context": f"Header > {context_name}"
                })
                seen_urls.add((link_text, absolute_link))

        # 1. Tophat Dropdown Links
        tophat_menus = soup.find_all("div", class_="tophat__menu")
        for menu in tophat_menus:
            menu_id = menu.get("id")
            toggle = soup.find("a", href=f"#{menu_id}") if menu_id else None
            context_name = toggle.get_text(strip=True) if toggle else "Tophat Menu"
            
            for a_tag in menu.find_all("a", href=True):
                add_link(a_tag, context_name)

        # 2. Main Navigation Dropdown Links
        dropdown_menus = soup.find_all("div", class_="dropdown-menu")
        for menu in dropdown_menus:
            menu_id = menu.get("id")
            toggle = soup.find("a", href=f"#{menu_id}") if menu_id else None
            context_name = toggle.get_text(strip=True) if toggle else "Navigation"
            
            for a_tag in menu.find_all("a", href=True):
                add_link(a_tag, context_name)
                
        return header_links

    def crawl(self, start_url=None, max_pages=50):
        """Crawl the site recursively starting from start_url using a persistent browser."""
        from playwright.sync_api import sync_playwright
        
        if not start_url:
            start_url = self.base_url
        
        queue = [start_url]
        results = []
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(ignore_https_errors=True, user_agent=self.headers["User-Agent"])
                page = context.new_page()
                
                while queue:
                    if max_pages > 0 and len(self.visited_urls) >= max_pages:
                        break
                    url = queue.pop(0)
                    if url in self.visited_urls:
                        continue
                    
                    self.visited_urls.add(url)
                    print(f"Scraping: {url}")
                    
                    content = self.get_page_content(url, page=page)
                    if content:
                        results.append(content)
                        
                        for link_info in content.get("outgoing_links", []):
                            link = link_info["url"]
                            link = link.split("#")[0]
                            if urlparse(link).netloc == self.domain and link not in self.visited_urls:
                                if not any(ext in link.lower() for ext in [".pdf", ".jpg", ".png", ".zip", ".docx"]):
                                    queue.append(link)
                    
                    time.sleep(1) # Small gap to be polite
                
                browser.close()
        except Exception as e:
            print(f"Crawl failed: {e}")
            
        return results

if __name__ == "__main__":
    # Test multi-page crawling
    scraper = WebScraper("https://aasldv2022dev.aasld.org/")
    # max_pages=3 for a quick test
    data_list = scraper.crawl(max_pages=3)
    
    for i, data in enumerate(data_list):
        print(f"\n--- Page {i+1} ---")
        print(f"Title: {data['title']}")
        print(f"URL: {data['url']}")
        print(f"Text Snippet: {data['text'][:200]}...")
        print(f"Links found: {len(data['outgoing_links'])}")
