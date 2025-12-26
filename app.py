"""
Module Extraction AI Agent - Streamlit Application
A powerful tool to extract modules and submodules from help documentation websites.
"""

import streamlit as st
import json
import time
import os
from datetime import datetime
from typing import List, Dict
import validators

# Import local modules
from crawler import WebCrawler, CrawlResult
from parser import ContentParser, ParsedContent, parse_crawl_results
from nlp_agent import NLPAgent, extract_modules_from_content

# Import CrewAI agents (optional)
try:
    from crew_agents import CrewAINLPAgent, extract_modules_with_crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Module Extractor AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #22d3ee;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.8)); }
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Log container */
    .log-container {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .log-entry {
        padding: 0.25rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .log-time {
        color: #6366f1;
        margin-right: 0.5rem;
    }
    
    .log-message {
        color: #e2e8f0;
    }
    
    /* Module card */
    .module-card {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
    }
    
    .module-card:hover {
        background: linear-gradient(145deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border-color: rgba(139, 92, 246, 0.5);
    }
    
    .module-title {
        color: #a5b4fc;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .module-description {
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }
    
    .submodule-item {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    
    .submodule-name {
        color: #22d3ee;
        font-weight: 500;
    }
    
    .submodule-desc {
        color: #94a3b8;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a855f7) !important;
    }
    
    /* Input fields */
    .stTextArea textarea, .stTextInput input {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(30, 27, 75, 0.95)) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #a5b4fc !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6) !important;
        border-radius: 12px !important;
    }
    
    /* JSON display */
    .json-display {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem;
        font-family: 'Fira Code', 'Consolas', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease forwards;
    }
    
    /* Spinner override */
    .stSpinner > div {
        border-color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = []
    if 'parsed_content' not in st.session_state:
        st.session_state.parsed_content = []
    if 'modules' not in st.session_state:
        st.session_state.modules = []
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'extraction_complete' not in st.session_state:
        st.session_state.extraction_complete = False


def add_log(message: str, level: str = "info"):
    """Add a log entry with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        'time': timestamp,
        'message': message,
        'level': level
    })


def display_logs():
    """Display streaming logs"""
    if st.session_state.logs:
        log_html = '<div class="log-container">'
        for log in st.session_state.logs[-50:]:  # Show last 50 logs
            level_color = {
                'info': '#6366f1',
                'success': '#10b981',
                'warning': '#f59e0b',
                'error': '#ef4444'
            }.get(log['level'], '#6366f1')
            
            log_html += f'''
                <div class="log-entry">
                    <span class="log-time">[{log['time']}]</span>
                    <span style="color: {level_color};">●</span>
                    <span class="log-message">{log['message']}</span>
                </div>
            '''
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)


def validate_urls(urls_text: str) -> List[str]:
    """Validate and extract URLs from text input"""
    valid_urls = []
    lines = urls_text.strip().split('\n')
    
    for line in lines:
        url = line.strip()
        if url and validators.url(url):
            valid_urls.append(url)
    
    return valid_urls


def crawl_websites(urls: List[str], max_pages: int, max_depth: int, progress_bar, status_text):
    """Crawl websites and return results"""
    crawler = WebCrawler(
        max_pages=max_pages,
        max_depth=max_depth,
        delay=0.5
    )
    
    def progress_callback(url, current, total):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Crawling: {url[:60]}...")
        add_log(f"Crawled page {current}/{total}: {url[:50]}...")
    
    crawler.set_progress_callback(progress_callback)
    
    add_log(f"Starting crawl of {len(urls)} URL(s)", "info")
    results = crawler.crawl(urls)
    add_log(f"Crawling complete! Found {len(results)} pages", "success")
    
    return results


def parse_content(crawl_results: List[CrawlResult], progress_bar, status_text):
    """Parse crawled content"""
    add_log("Parsing page content...", "info")
    
    parser = ContentParser()
    parsed_results = []
    
    for i, result in enumerate(crawl_results):
        progress = (i + 1) / len(crawl_results)
        progress_bar.progress(progress)
        status_text.text(f"Parsing: {result.title[:50]}...")
        
        parsed = parser.parse(
            html_content=result.html_content,
            url=result.url,
            title=result.title,
            breadcrumbs=result.breadcrumbs
        )
        parsed_results.append(parsed)
        add_log(f"Parsed: {result.title[:40]}")
    
    add_log(f"Parsed {len(parsed_results)} pages", "success")
    return parsed_results


def extract_modules(parsed_content: List[ParsedContent], openai_key: str, progress_bar, status_text, use_crewai: bool = False):
    """Extract modules from parsed content using either CrewAI or traditional NLP"""
    add_log("Starting module extraction...", "info")
    
    def progress_callback(message, progress):
        progress_bar.progress(progress)
        status_text.text(message)
        add_log(message)
    
    if use_crewai and CREWAI_AVAILABLE:
        add_log("Using CrewAI multi-agent system", "info")
        agent = CrewAINLPAgent(openai_api_key=openai_key if openai_key else None)
        modules = agent.extract_modules(parsed_content, progress_callback)
        json_result = agent.modules_to_json(modules)
    else:
        add_log("Using traditional NLP processing", "info")
        agent = NLPAgent(openai_api_key=openai_key if openai_key else None)
        modules = agent.extract_modules(parsed_content, progress_callback)
        json_result = agent.modules_to_json(modules)
    
    add_log(f"Extracted {len(json_result)} modules", "success")
    return json_result


def display_modules(modules: List[Dict]):
    """Display extracted modules in a beautiful format"""
    for module in modules:
        st.markdown(f"""
        <div class="module-card">
            <div class="module-title">📦 {module.get('module', 'Unknown Module')}</div>
            <div class="module-description">{module.get('Description', 'No description available')}</div>
        """, unsafe_allow_html=True)
        
        submodules = module.get('Submodules', {})
        if submodules:
            st.markdown('<div style="margin-top: 0.5rem;">', unsafe_allow_html=True)
            for name, desc in submodules.items():
                st.markdown(f"""
                <div class="submodule-item">
                    <span class="submodule-name">▸ {name}</span>
                    <span class="submodule-desc"> — {desc[:100]}{'...' if len(desc) > 100 else ''}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def save_json_output(modules: List[Dict], filename: str = None):
    """Save modules to JSON file"""
    if not filename:
        filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_dir = os.path.join(os.path.dirname(__file__), 'samples')
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(modules, f, indent=2, ensure_ascii=False)
    
    return filepath


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Module Extractor AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract structured modules from help documentation using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        st.markdown("---")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="For better module extraction. Falls back to heuristics if not provided."
        )
        
        st.markdown("---")
        
        # Crawling settings
        st.markdown("### 🕷️ Crawler Settings")
        max_pages = st.slider("Max Pages", 5, 200, 50, 5)
        max_depth = st.slider("Max Depth", 1, 10, 4)
        
        st.markdown("---")
        
        # Agent Settings
        st.markdown("### 🤖 Agent Settings")
        if CREWAI_AVAILABLE:
            use_crewai = st.toggle(
                "Use CrewAI Agents",
                value=True,
                help="Enable CrewAI multi-agent system for enhanced module extraction"
            )
            if use_crewai:
                st.info("🚀 Using CrewAI agents: Content Analyst, Clustering Specialist, Module Expert, QA Specialist")
        else:
            use_crewai = False
            st.warning("⚠️ CrewAI not installed. Install with: pip install crewai")
        
        st.markdown("---")
        
        # Example URLs
        st.markdown("### 📚 Example Sites")
        st.markdown("""
        Try these help documentation sites:
        - `https://help.neo.com`
        - `https://developer.wordpress.org`
        - `https://help.zluri.com`
        - `https://www.chargebee.com/docs`
        """)
        
        st.markdown("---")
        
        # Stats
        if st.session_state.crawl_results:
            st.markdown("### 📊 Statistics")
            col1, col2 = st.columns(2)
            col1.metric("Pages Crawled", len(st.session_state.crawl_results))
            col2.metric("Modules Found", len(st.session_state.modules))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🌐 Enter Documentation URLs")
        
        urls_input = st.text_area(
            "Enter one URL per line:",
            height=150,
            placeholder="https://help.example.com/docs\nhttps://docs.another-site.com",
            help="Enter the starting URLs for help documentation sites"
        )
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            extract_btn = st.button("🚀 Extract Modules", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.button("🗑️ Clear All", use_container_width=True)
        
        with col_btn3:
            if st.session_state.modules:
                json_str = json.dumps(st.session_state.modules, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    file_name=f"modules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📝 Activity Log")
        display_logs()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear button handler
    if clear_btn:
        st.session_state.crawl_results = []
        st.session_state.parsed_content = []
        st.session_state.modules = []
        st.session_state.logs = []
        st.session_state.extraction_complete = False
        st.rerun()
    
    # Extraction process
    if extract_btn:
        urls = validate_urls(urls_input)
        
        if not urls:
            st.error("❌ Please enter at least one valid URL")
            add_log("No valid URLs provided", "error")
        else:
            st.session_state.processing = True
            st.session_state.logs = []
            add_log(f"Starting extraction for {len(urls)} URL(s)", "info")
            
            # Progress container
            progress_container = st.container()
            
            with progress_container:
                st.markdown("---")
                
                # Phase 1: Crawling
                st.markdown("### 🕷️ Phase 1: Crawling Websites")
                crawl_progress = st.progress(0)
                crawl_status = st.empty()
                
                try:
                    st.session_state.crawl_results = crawl_websites(
                        urls, max_pages, max_depth, crawl_progress, crawl_status
                    )
                    crawl_status.markdown("✅ Crawling complete!")
                except Exception as e:
                    st.error(f"Crawling failed: {str(e)}")
                    add_log(f"Crawling error: {str(e)}", "error")
                    st.session_state.processing = False
                    st.stop()
                
                if not st.session_state.crawl_results:
                    st.warning("⚠️ No pages were crawled. Check the URLs and try again.")
                    st.session_state.processing = False
                    st.stop()
                
                # Phase 2: Parsing
                st.markdown("### 📄 Phase 2: Parsing Content")
                parse_progress = st.progress(0)
                parse_status = st.empty()
                
                try:
                    st.session_state.parsed_content = parse_content(
                        st.session_state.crawl_results, parse_progress, parse_status
                    )
                    parse_status.markdown("✅ Parsing complete!")
                except Exception as e:
                    st.error(f"Parsing failed: {str(e)}")
                    add_log(f"Parsing error: {str(e)}", "error")
                    st.session_state.processing = False
                    st.stop()
                
                # Phase 3: Module Extraction
                st.markdown("### 🧠 Phase 3: Extracting Modules")
                extract_progress = st.progress(0)
                extract_status = st.empty()
                
                try:
                    st.session_state.modules = extract_modules(
                        st.session_state.parsed_content, 
                        openai_key, 
                        extract_progress, 
                        extract_status,
                        use_crewai=use_crewai
                    )
                    extract_status.markdown("✅ Extraction complete!")
                except Exception as e:
                    st.error(f"Extraction failed: {str(e)}")
                    add_log(f"Extraction error: {str(e)}", "error")
                    st.session_state.processing = False
                    st.stop()
                
                st.session_state.processing = False
                st.session_state.extraction_complete = True
                add_log("All phases complete!", "success")
    
    # Display results
    if st.session_state.extraction_complete and st.session_state.modules:
        st.markdown("---")
        st.markdown("## 📦 Extracted Modules")
        
        # Tabs for different views
        tab1, tab2 = st.tabs(["📊 Visual View", "📝 JSON Output"])
        
        with tab1:
            display_modules(st.session_state.modules)
        
        with tab2:
            st.markdown('<div class="json-display">', unsafe_allow_html=True)
            st.json(st.session_state.modules)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown("---")
        st.markdown("### 📈 Extraction Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_submodules = sum(len(m.get('Submodules', {})) for m in st.session_state.modules)
        
        col1.metric("🌐 Pages Crawled", len(st.session_state.crawl_results))
        col2.metric("📄 Pages Parsed", len(st.session_state.parsed_content))
        col3.metric("📦 Modules Found", len(st.session_state.modules))
        col4.metric("🔧 Submodules Found", total_submodules)


if __name__ == "__main__":
    main()
