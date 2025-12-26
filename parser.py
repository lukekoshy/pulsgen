"""
Content Parser Module
Extracts clean, relevant text from HTML content.
Features:
- Removes boilerplate (headers, footers, navigation)
- Extracts article content
- Preserves heading hierarchy
- Handles various HTML structures
"""

from bs4 import BeautifulSoup, NavigableString
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass, field
import logging

# Try to import trafilatura for better extraction
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedContent:
    """Represents parsed and cleaned content from a page"""
    url: str
    title: str
    main_content: str
    sections: List[Dict[str, str]] = field(default_factory=list)
    headings_hierarchy: List[Dict] = field(default_factory=list)
    bullet_points: List[str] = field(default_factory=list)
    breadcrumbs: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ContentParser:
    """
    Intelligent content parser for extracting clean text from HTML.
    Removes boilerplate and preserves meaningful structure.
    """
    
    # Elements to remove entirely
    REMOVE_TAGS = [
        'script', 'style', 'noscript', 'iframe', 'svg', 'canvas',
        'form', 'input', 'button', 'select', 'textarea',
        'header', 'footer', 'aside', 'nav',
    ]
    
    # Class patterns indicating boilerplate
    BOILERPLATE_CLASSES = [
        r'nav', r'menu', r'sidebar', r'footer', r'header',
        r'cookie', r'popup', r'modal', r'banner', r'advertisement',
        r'social', r'share', r'comment', r'related', r'recommended',
        r'newsletter', r'subscribe', r'signup', r'login',
        r'breadcrumb', r'pagination', r'widget', r'toolbar',
    ]
    
    # ID patterns indicating boilerplate
    BOILERPLATE_IDS = [
        r'nav', r'menu', r'sidebar', r'footer', r'header',
        r'cookie', r'popup', r'modal', r'banner',
    ]
    
    # Patterns indicating main content
    CONTENT_INDICATORS = [
        r'content', r'article', r'main', r'post', r'entry',
        r'text', r'body', r'documentation', r'doc-content',
        r'help-content', r'kb-article', r'support-article',
    ]
    
    def __init__(self, use_trafilatura: bool = True):
        self.use_trafilatura = use_trafilatura and TRAFILATURA_AVAILABLE
    
    def _remove_boilerplate_elements(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove elements that are likely boilerplate"""
        # Remove script, style, etc.
        for tag in self.REMOVE_TAGS:
            for elem in soup.find_all(tag):
                elem.decompose()
        
        # Remove elements with boilerplate classes
        boilerplate_pattern = re.compile(
            '|'.join(self.BOILERPLATE_CLASSES),
            re.IGNORECASE
        )
        
        for elem in soup.find_all(class_=boilerplate_pattern):
            elem.decompose()
        
        # Remove elements with boilerplate IDs
        id_pattern = re.compile(
            '|'.join(self.BOILERPLATE_IDS),
            re.IGNORECASE
        )
        
        for elem in soup.find_all(id=id_pattern):
            elem.decompose()
        
        # Remove hidden elements
        for elem in soup.find_all(style=re.compile(r'display\s*:\s*none', re.I)):
            elem.decompose()
        
        for elem in soup.find_all(attrs={'hidden': True}):
            elem.decompose()
        
        for elem in soup.find_all(attrs={'aria-hidden': 'true'}):
            elem.decompose()
        
        return soup
    
    def _find_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """Find the main content container"""
        
        # Try semantic HTML5 elements first
        main_elem = soup.find('main')
        if main_elem:
            return main_elem
        
        article_elem = soup.find('article')
        if article_elem:
            return article_elem
        
        # Try common content class patterns
        content_pattern = re.compile(
            '|'.join(self.CONTENT_INDICATORS),
            re.IGNORECASE
        )
        
        for elem in soup.find_all(['div', 'section'], class_=content_pattern):
            # Verify it has substantial content
            text = elem.get_text(strip=True)
            if len(text) > 200:
                return elem
        
        # Try content ID patterns
        for elem in soup.find_all(['div', 'section'], id=content_pattern):
            text = elem.get_text(strip=True)
            if len(text) > 200:
                return elem
        
        # Fall back to body
        return soup.find('body') or soup
    
    def _extract_sections(self, content: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract content sections based on headings"""
        sections = []
        current_section = {'heading': '', 'level': 0, 'content': []}
        
        for elem in content.children:
            if isinstance(elem, NavigableString):
                text = str(elem).strip()
                if text and current_section['heading']:
                    current_section['content'].append(text)
                continue
            
            if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Save previous section if it has content
                if current_section['heading'] and current_section['content']:
                    sections.append({
                        'heading': current_section['heading'],
                        'level': current_section['level'],
                        'content': ' '.join(current_section['content'])
                    })
                
                # Start new section
                level = int(elem.name[1])
                current_section = {
                    'heading': elem.get_text(strip=True),
                    'level': level,
                    'content': []
                }
            elif elem.name in ['p', 'div', 'span', 'li', 'ul', 'ol', 'pre', 'code', 'blockquote']:
                text = elem.get_text(separator=' ', strip=True)
                if text:
                    current_section['content'].append(text)
        
        # Don't forget the last section
        if current_section['heading'] and current_section['content']:
            sections.append({
                'heading': current_section['heading'],
                'level': current_section['level'],
                'content': ' '.join(current_section['content'])
            })
        
        return sections
    
    def _extract_bullet_points(self, content: BeautifulSoup) -> List[str]:
        """Extract bullet points and list items"""
        bullets = []
        
        for li in content.find_all('li'):
            text = li.get_text(strip=True)
            if text and len(text) > 5:
                bullets.append(text)
        
        return bullets
    
    def _extract_headings_hierarchy(self, content: BeautifulSoup) -> List[Dict]:
        """Extract headings with their hierarchy"""
        headings = []
        
        for elem in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = elem.get_text(strip=True)
            if text:
                level = int(elem.name[1])
                headings.append({
                    'text': text,
                    'level': level,
                    'tag': elem.name
                })
        
        return headings
    
    def _extract_with_trafilatura(self, html: str) -> Tuple[str, Dict]:
        """Use trafilatura for content extraction"""
        if not self.use_trafilatura:
            return "", {}
        
        try:
            # Extract main text
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=True
            )
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(html)
            meta_dict = {}
            if metadata:
                meta_dict = {
                    'title': metadata.title,
                    'author': metadata.author,
                    'date': metadata.date,
                    'description': metadata.description,
                }
            
            return text or "", meta_dict
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
            return "", {}
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove common boilerplate phrases
        boilerplate_phrases = [
            r'cookie[s]? (policy|notice|consent)',
            r'privacy policy',
            r'terms (of|and) (service|use)',
            r'all rights reserved',
            r'copyright \d{4}',
            r'subscribe to our newsletter',
            r'follow us on',
            r'share (this|on)',
        ]
        
        for phrase in boilerplate_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def parse(self, html_content: str, url: str = "", title: str = "",
              breadcrumbs: List[str] = None) -> ParsedContent:
        """
        Parse HTML content and extract clean, structured text.
        """
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title if not provided
        if not title:
            title_elem = soup.find('h1') or soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "Untitled"
        
        # Try trafilatura first for main content
        traf_content, traf_metadata = self._extract_with_trafilatura(html_content)
        
        # Clean the soup
        soup = self._remove_boilerplate_elements(soup)
        
        # Find main content area
        main_content = self._find_main_content(soup)
        
        # Extract structured content
        sections = self._extract_sections(main_content)
        headings = self._extract_headings_hierarchy(main_content)
        bullets = self._extract_bullet_points(main_content)
        
        # Get main text content
        if traf_content:
            main_text = self._clean_text(traf_content)
        else:
            main_text = self._clean_text(main_content.get_text(separator='\n', strip=True))
        
        # Combine metadata
        metadata = traf_metadata or {}
        
        return ParsedContent(
            url=url,
            title=title,
            main_content=main_text,
            sections=sections,
            headings_hierarchy=headings,
            bullet_points=bullets,
            breadcrumbs=breadcrumbs or [],
            metadata=metadata
        )


def parse_crawl_results(crawl_results) -> List[ParsedContent]:
    """
    Parse a list of CrawlResult objects from the crawler.
    """
    parser = ContentParser()
    parsed_results = []
    
    for result in crawl_results:
        parsed = parser.parse(
            html_content=result.html_content,
            url=result.url,
            title=result.title,
            breadcrumbs=result.breadcrumbs
        )
        parsed_results.append(parsed)
    
    return parsed_results


if __name__ == "__main__":
    # Test parsing
    test_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <nav>Navigation menu</nav>
        <main>
            <h1>Main Title</h1>
            <p>This is the main content paragraph.</p>
            <h2>Section 1</h2>
            <p>Section 1 content here.</p>
            <ul>
                <li>Point 1</li>
                <li>Point 2</li>
            </ul>
            <h2>Section 2</h2>
            <p>Section 2 content here.</p>
        </main>
        <footer>Footer content</footer>
    </body>
    </html>
    """
    
    parser = ContentParser()
    result = parser.parse(test_html, url="https://example.com/test")
    
    print(f"Title: {result.title}")
    print(f"Main content length: {len(result.main_content)}")
    print(f"Sections: {len(result.sections)}")
    print(f"Headings: {result.headings_hierarchy}")
    print(f"Bullets: {result.bullet_points}")
