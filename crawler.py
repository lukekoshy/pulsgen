"""
Web Crawler Module
Handles recursive crawling of help documentation websites.
Features:
- Same-domain constraint
- Duplicate URL detection
- Pagination handling
- Help center hierarchy following
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
from typing import Set, List, Dict, Optional, Callable
import logging
from dataclasses import dataclass, field
from collections import deque
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Represents a single crawled page"""
    url: str
    title: str
    html_content: str
    text_content: str
    depth: int
    parent_url: Optional[str] = None
    breadcrumbs: List[str] = field(default_factory=list)
    headings: Dict[str, List[str]] = field(default_factory=dict)


class WebCrawler:
    """
    Intelligent web crawler for help documentation sites.
    Handles recursion, pagination, and hierarchy extraction.
    """
    
    # Patterns to exclude (navigation, login, etc.)
    EXCLUDE_PATTERNS = [
        r'/login', r'/signup', r'/register', r'/auth',
        r'/cart', r'/checkout', r'/account',
        r'/search', r'/tag/', r'/category/',
        r'#', r'javascript:', r'mailto:', r'tel:',
        r'/cdn-cgi/', r'/wp-admin/', r'/wp-login',
        r'/feed', r'/rss', r'/atom',
        r'\.(pdf|zip|exe|dmg|pkg|msi|doc|docx|xls|xlsx|ppt|pptx)$',
        r'\.(jpg|jpeg|png|gif|svg|webp|ico|mp4|mp3|wav|avi)$',
    ]
    
    # Patterns that indicate help/documentation pages
    HELP_PATTERNS = [
        r'/help', r'/docs', r'/documentation', r'/support',
        r'/guide', r'/tutorial', r'/article', r'/kb',
        r'/knowledge', r'/faq', r'/how-to', r'/getting-started',
        r'/learn', r'/resources', r'/manual', r'/reference',
    ]
    
    def __init__(
        self,
        max_pages: int = 100,
        max_depth: int = 5,
        delay: float = 0.5,
        timeout: int = 10,
        user_agent: str = "ModuleExtractor/1.0 (Documentation Crawler)"
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.timeout = timeout
        self.user_agent = user_agent
        
        self.visited_urls: Set[str] = set()
        self.results: List[CrawlResult] = []
        self.base_domain: str = ""
        self.base_path: str = ""
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
        # Callback for progress updates
        self.progress_callback: Optional[Callable[[str, int, int], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int], None]):
        """Set callback for progress updates: callback(url, current, total)"""
        self.progress_callback = callback
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes"""
        url, _ = urldefrag(url)
        url = url.rstrip('/')
        return url
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain"""
        try:
            parsed = urlparse(url)
            return parsed.netloc == self.base_domain or parsed.netloc == ""
        except:
            return False
    
    def _is_within_base_path(self, url: str) -> bool:
        """Check if URL is within the base path (for scoped crawling)"""
        try:
            parsed = urlparse(url)
            return parsed.path.startswith(self.base_path) or self.base_path == ""
        except:
            return False
    
    def _should_exclude(self, url: str) -> bool:
        """Check if URL matches exclusion patterns"""
        url_lower = url.lower()
        for pattern in self.EXCLUDE_PATTERNS:
            if re.search(pattern, url_lower):
                return True
        return False
    
    def _is_help_page(self, url: str) -> bool:
        """Check if URL looks like a help/documentation page"""
        url_lower = url.lower()
        for pattern in self.HELP_PATTERNS:
            if re.search(pattern, url_lower):
                return True
        return True  # Default to True for general crawling
    
    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all valid links from the page"""
        links = []
        
        # Remove navigation elements before extracting links
        for nav in soup.find_all(['nav', 'header', 'footer']):
            nav.decompose()
        
        # Also remove common navigation class patterns
        for elem in soup.find_all(class_=re.compile(
            r'(nav|menu|sidebar|footer|header|cookie|popup|modal|banner)',
            re.IGNORECASE
        )):
            elem.decompose()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty or javascript links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue
            
            # Convert relative to absolute URL
            full_url = urljoin(current_url, href)
            normalized_url = self._normalize_url(full_url)
            
            # Apply filters
            if (self._is_same_domain(normalized_url) and
                self._is_within_base_path(normalized_url) and
                not self._should_exclude(normalized_url) and
                normalized_url not in self.visited_urls):
                links.append(normalized_url)
        
        return list(set(links))  # Remove duplicates
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract headings organized by level"""
        headings = {'h1': [], 'h2': [], 'h3': [], 'h4': []}
        
        for level in headings.keys():
            for heading in soup.find_all(level):
                text = heading.get_text(strip=True)
                if text and len(text) > 2:
                    headings[level].append(text)
        
        return headings
    
    def _extract_breadcrumbs(self, soup: BeautifulSoup) -> List[str]:
        """Extract breadcrumb navigation if present"""
        breadcrumbs = []
        
        # Common breadcrumb patterns
        breadcrumb_selectors = [
            ('nav', {'class': re.compile(r'breadcrumb', re.I)}),
            ('ol', {'class': re.compile(r'breadcrumb', re.I)}),
            ('ul', {'class': re.compile(r'breadcrumb', re.I)}),
            ('div', {'class': re.compile(r'breadcrumb', re.I)}),
            (None, {'aria-label': 'breadcrumb'}),
            (None, {'aria-label': 'Breadcrumb'}),
        ]
        
        for tag, attrs in breadcrumb_selectors:
            crumb_elem = soup.find(tag, attrs) if tag else soup.find(attrs=attrs)
            if crumb_elem:
                for item in crumb_elem.find_all(['li', 'a', 'span']):
                    text = item.get_text(strip=True)
                    if text and text not in ['>', '/', '›', '»', '→']:
                        breadcrumbs.append(text)
                break
        
        return breadcrumbs
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a single page"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                logger.debug(f"Skipping non-HTML content: {url}")
                return None
            
            return BeautifulSoup(response.content, 'lxml')
        
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        # Try to find the main title
        title_sources = [
            soup.find('h1'),
            soup.find('title'),
            soup.find('meta', {'property': 'og:title'}),
        ]
        
        for source in title_sources:
            if source:
                if source.name == 'meta':
                    return source.get('content', '').strip()
                return source.get_text(strip=True)
        
        return "Untitled"
    
    def crawl(self, start_urls: List[str]) -> List[CrawlResult]:
        """
        Start crawling from the given URLs.
        Returns list of CrawlResult objects.
        """
        self.visited_urls.clear()
        self.results.clear()
        
        # Queue: (url, depth, parent_url)
        queue = deque()
        
        for url in start_urls:
            url = self._normalize_url(url)
            parsed = urlparse(url)
            self.base_domain = parsed.netloc
            self.base_path = '/'.join(parsed.path.split('/')[:-1]) if '/' in parsed.path else ""
            queue.append((url, 0, None))
        
        pages_crawled = 0
        
        while queue and pages_crawled < self.max_pages:
            url, depth, parent_url = queue.popleft()
            
            if url in self.visited_urls or depth > self.max_depth:
                continue
            
            self.visited_urls.add(url)
            
            # Progress callback
            if self.progress_callback:
                self.progress_callback(url, pages_crawled + 1, min(len(queue) + pages_crawled + 1, self.max_pages))
            
            logger.info(f"Crawling ({pages_crawled + 1}/{self.max_pages}): {url}")
            
            soup = self._fetch_page(url)
            if not soup:
                continue
            
            # Extract content
            title = self._extract_title(soup)
            headings = self._extract_headings(soup)
            breadcrumbs = self._extract_breadcrumbs(soup)
            
            # Store raw HTML for later processing
            html_content = str(soup)
            
            # Basic text extraction (will be refined by parser)
            text_content = soup.get_text(separator='\n', strip=True)
            
            result = CrawlResult(
                url=url,
                title=title,
                html_content=html_content,
                text_content=text_content,
                depth=depth,
                parent_url=parent_url,
                breadcrumbs=breadcrumbs,
                headings=headings
            )
            self.results.append(result)
            pages_crawled += 1
            
            # Extract and queue new links
            new_links = self._extract_links(soup, url)
            for link in new_links:
                if link not in self.visited_urls:
                    queue.append((link, depth + 1, url))
            
            # Respect rate limiting
            time.sleep(self.delay)
        
        logger.info(f"Crawling complete. Total pages: {len(self.results)}")
        return self.results
    
    def get_site_structure(self) -> Dict:
        """Build a hierarchical structure of the crawled site"""
        structure = {}
        
        for result in self.results:
            parsed = urlparse(result.url)
            path_parts = [p for p in parsed.path.split('/') if p]
            
            current = structure
            for part in path_parts:
                if part not in current:
                    current[part] = {'_urls': [], '_children': {}}
                current[part]['_urls'].append({
                    'url': result.url,
                    'title': result.title
                })
                current = current[part]['_children']
        
        return structure


def crawl_urls(urls: List[str], max_pages: int = 100, max_depth: int = 5,
               progress_callback: Optional[Callable] = None) -> List[CrawlResult]:
    """
    Convenience function to crawl multiple URLs.
    """
    crawler = WebCrawler(max_pages=max_pages, max_depth=max_depth)
    if progress_callback:
        crawler.set_progress_callback(progress_callback)
    return crawler.crawl(urls)


if __name__ == "__main__":
    # Test crawling
    test_urls = ["https://help.example.com"]
    results = crawl_urls(test_urls, max_pages=5, max_depth=2)
    for r in results:
        print(f"Title: {r.title}, URL: {r.url}, Depth: {r.depth}")
