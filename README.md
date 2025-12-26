# 🔍 Module Extractor AI

<div align="center">

![Module Extractor](https://img.shields.io/badge/Module%20Extractor-AI%20Powered-6366f1?style=for-the-badge&logo=robot&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Extract structured modules and submodules from help documentation websites using AI**

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Architecture](#-architecture) •
[How It Works](#-how-it-works) •
[Limitations](#-limitations)

</div>

---

## 📋 Problem Statement

Organizations often need to understand the structure and features of software products by analyzing their help documentation. Manually reviewing hundreds of documentation pages to extract a structured overview is:

- **Time-consuming**: Large documentation sites can have hundreds or thousands of pages
- **Error-prone**: Manual extraction leads to inconsistencies and missed content
- **Non-standardized**: Different team members may interpret hierarchy differently

### Solution

**Module Extractor AI** automates this process by:
1. Intelligently crawling help documentation websites
2. Extracting and cleaning meaningful content
3. Using AI to infer module/submodule hierarchies
4. Outputting a standardized JSON structure

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🕷️ **Smart Crawling** | Recursive crawling with same-domain constraints, duplicate detection, and help-page prioritization |
| 📄 **Content Cleaning** | Removes boilerplate (headers, footers, navigation), extracts only relevant article content |
| 🧠 **AI-Powered Extraction** | Uses embeddings + clustering + LLM reasoning for accurate module inference |
| 🚫 **No Hallucination** | Descriptions are strictly based on scraped content, never fabricated |
| 📊 **Beautiful UI** | Modern dark-theme interface with real-time progress and streaming logs |
| 📥 **JSON Export** | Outputs in the exact required format for easy integration |
| ⚡ **Caching** | Embedding cache for improved performance on repeated operations |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODULE EXTRACTOR AI                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   STREAMLIT  │     │    URLs      │     │   CONFIG     │                 │
│  │      UI      │◄────│   INPUT      │     │   SIDEBAR    │                 │
│  │   (app.py)   │     │              │     │              │                 │
│  └──────┬───────┘     └──────────────┘     └──────────────┘                 │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PHASE 1: WEB CRAWLING                         │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  crawler.py                                                   │   │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │    │
│  │  │  │   URL       │  │  Same-      │  │   Breadcrumb/       │  │   │    │
│  │  │  │   Queue     │──│  Domain     │──│   Heading           │  │   │    │
│  │  │  │   (BFS)     │  │  Filter     │  │   Extraction        │  │   │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PHASE 2: CONTENT PARSING                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  parser.py                                                    │   │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │    │
│  │  │  │  Boilerplate│  │  Content    │  │   Section/          │  │   │    │
│  │  │  │  Removal    │──│  Extraction │──│   Hierarchy         │  │   │    │
│  │  │  │             │  │ (trafilatura│  │   Builder           │  │   │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     PHASE 3: MODULE EXTRACTION                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  nlp_agent.py                                                 │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │    │
│  │  │  │   Text      │  │  Embedding  │  │   Semantic          │  │   │    │
│  │  │  │   Chunking  │──│  (sentence- │──│   Clustering        │  │   │    │
│  │  │  │             │  │  transformers│  │   (Agglomerative)  │  │   │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │    │
│  │  │                                           │                   │   │    │
│  │  │                                           ▼                   │   │    │
│  │  │                          ┌─────────────────────────────────┐ │   │    │
│  │  │                          │   LLM Inference (OpenAI)        │ │   │    │
│  │  │                          │   OR Heuristic Fallback         │ │   │    │
│  │  │                          │   ┌───────────────────────────┐ │ │   │    │
│  │  │                          │   │ • Extract module name     │ │ │   │    │
│  │  │                          │   │ • Generate description    │ │ │   │    │
│  │  │                          │   │ • Identify submodules     │ │ │   │    │
│  │  │                          │   │ • Merge duplicates        │ │ │   │    │
│  │  │                          │   └───────────────────────────┘ │ │   │    │
│  │  │                          └─────────────────────────────────┘ │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                           JSON OUTPUT                                │    │
│  │  [                                                                   │    │
│  │    {                                                                 │    │
│  │      "module": "Account Settings",                                   │    │
│  │      "Description": "...",                                           │    │
│  │      "Submodules": {                                                 │    │
│  │        "Change Username": "...",                                     │    │
│  │        "Security Settings": "..."                                    │    │
│  │      }                                                               │    │
│  │    }                                                                 │    │
│  │  ]                                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
module_extractor/
│
├── app.py                  # Streamlit UI - Main application interface
├── crawler.py              # Web crawling logic - Recursive page fetching
├── parser.py               # HTML text extraction - Content cleaning
├── nlp_agent.py            # Module + submodule inference - Traditional NLP
├── crew_agents.py          # CrewAI multi-agent system - AI agents
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
│
└── samples/               # Sample output files
    ├── output_neo.json
    ├── output_wordpress.json
    ├── output_zluri.json
    └── output_chargebee.json
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Optional: OpenAI API key for enhanced module extraction

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd module_extractor
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key (optional but recommended):**
   ```bash
   # Windows
   set OPENAI_API_KEY=your-api-key-here
   
   # macOS/Linux
   export OPENAI_API_KEY=your-api-key-here
   ```

   Or enter it directly in the Streamlit sidebar when running the app.

---

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Interface

1. **Enter URLs**: Input one or more help documentation URLs (one per line)
2. **Configure Settings**: Adjust max pages and crawl depth in the sidebar
3. **Add API Key** (optional): Enter OpenAI API key for better extraction
4. **Click Extract**: Start the extraction process
5. **Monitor Progress**: Watch real-time logs and progress bars
6. **View Results**: See extracted modules in visual or JSON format
7. **Download**: Export results as JSON file

### Example URLs to Test

- `https://help.neo.com`
- `https://developer.wordpress.org/documentation`
- `https://help.zluri.com`
- `https://www.chargebee.com/docs`

---

## 🧠 How Module Detection Works

### 1. Content Chunking
The system breaks down parsed content into meaningful chunks based on:
- **Heading sections** (H1 → H2 → H3 hierarchy)
- **Paragraph boundaries** for content without clear headings
- **Breadcrumb context** for understanding page position in hierarchy

### 2. Semantic Embedding
Each chunk is converted to a vector embedding using `sentence-transformers`:
- Model: `all-MiniLM-L6-v2` (lightweight, fast)
- Captures semantic meaning of text
- Cached for performance

### 3. Clustering
Related chunks are grouped using **Agglomerative Clustering**:
- Cosine similarity metric
- Average linkage
- Automatic cluster count estimation

### 4. Module Extraction
For each cluster, the system extracts:

**With OpenAI (Recommended):**
- LLM analyzes the cluster context
- Strict prompt ensures no hallucination
- Generates module name, description, and submodules

**Fallback (No API Key):**
- Uses heuristic analysis
- Module name from common headings/breadcrumbs
- Submodules from lower-level headings
- Descriptions from first sentences

### 5. Deduplication
Similar modules are merged based on:
- Name similarity (substring matching)
- Word overlap ratio
- Submodule consolidation

---

## 🤖 CrewAI Multi-Agent System

The application now supports **CrewAI** for enhanced module extraction using specialized AI agents working together.

### Agent Roles

| Agent | Role | Responsibility |
|-------|------|----------------|
| 📊 **Content Analyst** | Documentation Content Analyst | Chunks and prepares documentation for processing |
| 🔗 **Clustering Specialist** | Semantic Clustering Specialist | Groups related content using embeddings and similarity |
| 📦 **Module Expert** | Module Extraction Expert | Extracts structured module information |
| ✅ **QA Specialist** | Quality Assurance Specialist | Reviews, merges, and refines the final output |

### Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Content        │     │  Clustering     │     │  Module         │     │  QA             │
│  Analyst        │────▶│  Specialist     │────▶│  Expert         │────▶│  Specialist     │
│                 │     │                 │     │                 │     │                 │
│  Chunks content │     │  Groups similar │     │  Extracts       │     │  Merges &       │
│  into segments  │     │  content        │     │  modules        │     │  refines        │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Enable CrewAI

Toggle "Use CrewAI Agents" in the sidebar to enable the multi-agent system.

---

## 📊 Output Format

```json
[
  {
    "module": "Account Settings",
    "Description": "Manage your account preferences, security options, and profile information.",
    "Submodules": {
      "Change Username": "Update your display name and username from the profile settings page.",
      "Security Settings": "Configure two-factor authentication and password requirements.",
      "Deactivate Account": "Permanently close your account and delete associated data."
    }
  }
]
```

---

## ⚙️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| Max Pages | 50 | Maximum number of pages to crawl |
| Max Depth | 4 | Maximum link-following depth from start URL |
| Crawl Delay | 0.5s | Delay between requests to respect rate limits |
| Similarity Threshold | 0.65 | Threshold for clustering (lower = more clusters) |

---

## ⚠️ Limitations

### Crawling Limitations
- **JavaScript-rendered content**: Sites using heavy JavaScript (SPAs) may not be fully accessible
- **Login-protected pages**: Cannot access content behind authentication
- **Rate limiting**: Some sites may block rapid crawling; adjust delay if needed
- **Robots.txt**: The crawler does not currently check robots.txt (manual review recommended)

### Content Extraction Limitations
- **Complex layouts**: Unusual HTML structures may not be parsed correctly
- **Images/Videos**: Visual content is not analyzed; only text is extracted
- **PDFs**: Linked PDF documents are not processed
- **Non-English content**: Optimized for English documentation

### AI Limitations
- **Without OpenAI API**: Falls back to heuristic extraction (less accurate)
- **Token limits**: Very large pages may be truncated
- **Ambiguous content**: May struggle with poorly structured documentation
- **Domain-specific terminology**: May not correctly identify specialized terms

### Performance Limitations
- **Large sites**: Sites with 1000+ pages may take significant time
- **Memory usage**: Embedding models require significant RAM (~1GB)
- **Network dependency**: Requires stable internet connection

---

## 🧪 Testing

### Tested Websites

| Website | Status | Notes |
|---------|--------|-------|
| Neo Help | ✅ Tested | Good structure, clean extraction |
| WordPress Docs | ✅ Tested | Large site, use lower max pages |
| Zluri Help | ✅ Tested | Well-organized help center |
| Chargebee Docs | ✅ Tested | Technical documentation |

### Running Tests

```bash
# Test crawler
python crawler.py

# Test parser
python parser.py

# Test NLP agent
python nlp_agent.py
```

---

## 🔮 Future Improvements

1. **JavaScript Rendering**: Integrate Playwright/Selenium for SPA support
2. **PDF Processing**: Extract content from linked PDF documents
3. **Multi-language Support**: Add support for non-English documentation
4. **Incremental Crawling**: Resume interrupted crawls, detect updates
5. **Parallel Crawling**: Speed up extraction with async/concurrent requests
6. **Custom Extraction Rules**: Allow users to define site-specific patterns
7. **API Endpoint**: Expose extraction as a REST API
8. **Scheduled Extraction**: Automatic periodic extraction with diff reporting
9. **Export Formats**: Add CSV, Excel, and Markdown export options
10. **Visualization**: Interactive module hierarchy visualization

---

## 🛠️ Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| **UI** | Streamlit | Web interface |
| **Crawling** | requests, BeautifulSoup | Fetching and parsing HTML |
| **Content Extraction** | trafilatura | Clean text extraction |
| **Embeddings** | sentence-transformers | Semantic text representation |
| **Clustering** | scikit-learn | Grouping related content |
| **LLM** | OpenAI API | Intelligent module inference |
| **Validation** | validators | URL validation |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Limitations](#-limitations) section
2. Review existing issues
3. Create a new issue with detailed description

---

<div align="center">

**Built with ❤️ using Streamlit and AI**

</div>
