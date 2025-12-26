"""
Lightweight Multi-Agent Module Extraction System
Implements specialized agents for module extraction without requiring external AI frameworks.
Compatible with all Python versions.
"""

import json
import os
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ContentChunk:
    """A chunk of content with metadata"""
    text: str
    source_url: str
    heading: str
    level: int
    breadcrumbs: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


@dataclass
class ExtractedModule:
    """Represents an extracted module"""
    name: str
    description: str
    submodules: Dict[str, str] = field(default_factory=dict)
    source_urls: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result from an agent's task execution"""
    success: bool
    data: Any
    message: str
    agent_name: str


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal
        self.verbose = True
    
    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        """Execute the agent's main task"""
        pass
    
    def log(self, message: str):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            logger.info(f"[{self.name}] {message}")


# ============================================================================
# Specialized Agents
# ============================================================================

class ContentAnalystAgent(BaseAgent):
    """Agent responsible for chunking and preparing documentation content"""
    
    def __init__(self):
        super().__init__(
            name="Content Analyst",
            role="Documentation Content Analyst",
            goal="Analyze and chunk documentation content into meaningful segments"
        )
    
    def execute(self, parsed_contents: List, **kwargs) -> AgentResult:
        """Chunk parsed content into meaningful segments"""
        self.log("Starting content analysis...")
        chunks = []
        
        for content in parsed_contents:
            # Process sections
            if hasattr(content, 'sections') and content.sections:
                for section in content.sections:
                    chunk = ContentChunk(
                        text=f"{section['heading']}\n{section['content']}",
                        source_url=content.url,
                        heading=section['heading'],
                        level=section['level'],
                        breadcrumbs=getattr(content, 'breadcrumbs', [])
                    )
                    chunks.append(chunk)
            
            # Process main content if no sections
            if (not hasattr(content, 'sections') or not content.sections):
                if hasattr(content, 'main_content') and content.main_content:
                    paragraphs = content.main_content.split('\n\n')
                    for i, para in enumerate(paragraphs):
                        if len(para.strip()) > 50:
                            chunk = ContentChunk(
                                text=para.strip(),
                                source_url=content.url,
                                heading=content.title if i == 0 else "",
                                level=1,
                                breadcrumbs=getattr(content, 'breadcrumbs', [])
                            )
                            chunks.append(chunk)
        
        self.log(f"Created {len(chunks)} content chunks")
        return AgentResult(
            success=True,
            data=chunks,
            message=f"Successfully chunked {len(parsed_contents)} pages into {len(chunks)} segments",
            agent_name=self.name
        )


class ClusteringSpecialistAgent(BaseAgent):
    """Agent responsible for semantic clustering of content"""
    
    def __init__(self):
        super().__init__(
            name="Clustering Specialist",
            role="Semantic Clustering Specialist",
            goal="Group related content using semantic similarity"
        )
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.log("Loaded embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                self.log(f"Failed to load embedding model: {e}")
    
    def execute(self, chunks: List[ContentChunk], **kwargs) -> AgentResult:
        """Cluster chunks by semantic similarity"""
        self.log("Starting semantic clustering...")
        
        if not chunks:
            return AgentResult(
                success=True,
                data={0: chunks},
                message="No chunks to cluster",
                agent_name=self.name
            )
        
        # Try semantic clustering first
        if SKLEARN_AVAILABLE and self.embedding_model:
            clusters = self._semantic_cluster(chunks)
        else:
            clusters = self._structure_cluster(chunks)
        
        self.log(f"Created {len(clusters)} clusters")
        return AgentResult(
            success=True,
            data=clusters,
            message=f"Successfully clustered {len(chunks)} chunks into {len(clusters)} groups",
            agent_name=self.name
        )
    
    def _semantic_cluster(self, chunks: List[ContentChunk]) -> Dict[int, List[ContentChunk]]:
        """Cluster using semantic embeddings"""
        texts = [chunk.text[:512] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        if len(embeddings) < 2:
            return {0: chunks}
        
        n_clusters = min(max(2, len(chunks) // 5), 15)
        n_clusters = min(n_clusters, len(chunks))
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings)
            
            clustered = {}
            for chunk, label in zip(chunks, labels):
                if label not in clustered:
                    clustered[label] = []
                clustered[label].append(chunk)
            
            return clustered
        except Exception as e:
            self.log(f"Clustering failed: {e}, falling back to structure-based")
            return self._structure_cluster(chunks)
    
    def _structure_cluster(self, chunks: List[ContentChunk]) -> Dict[int, List[ContentChunk]]:
        """Fallback clustering based on document structure"""
        clustered = {}
        
        for chunk in chunks:
            if chunk.breadcrumbs and len(chunk.breadcrumbs) > 1:
                key = chunk.breadcrumbs[1]
            elif chunk.heading:
                key = chunk.heading.split(':')[0].split('-')[0].strip()
            else:
                key = "General"
            
            cluster_id = hash(key) % 100
            if cluster_id not in clustered:
                clustered[cluster_id] = []
            clustered[cluster_id].append(chunk)
        
        return clustered


class ModuleExpertAgent(BaseAgent):
    """Agent responsible for extracting module information"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        super().__init__(
            name="Module Expert",
            role="Module Extraction Expert",
            goal="Extract structured module information from clustered content"
        )
        self.openai_client = None
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                self.log("OpenAI client initialized")
            except Exception as e:
                self.log(f"Failed to initialize OpenAI: {e}")
    
    def execute(self, clusters: Dict[int, List[ContentChunk]], **kwargs) -> AgentResult:
        """Extract modules from each cluster"""
        self.log("Starting module extraction...")
        modules = []
        
        for cluster_id, cluster_chunks in clusters.items():
            if len(cluster_chunks) >= 1:
                if self.openai_client:
                    module = self._extract_with_llm(cluster_chunks)
                else:
                    module = self._extract_heuristic(cluster_chunks)
                
                if module and module.name:
                    module.name = self._clean_name(module.name)
                    modules.append(module)
        
        self.log(f"Extracted {len(modules)} modules")
        return AgentResult(
            success=True,
            data=modules,
            message=f"Successfully extracted {len(modules)} modules from {len(clusters)} clusters",
            agent_name=self.name
        )
    
    def _extract_with_llm(self, chunks: List[ContentChunk]) -> Optional[ExtractedModule]:
        """Use LLM for extraction"""
        context_parts = []
        source_urls = set()
        
        for chunk in chunks[:10]:
            context_parts.append(f"Heading: {chunk.heading}\nContent: {chunk.text[:500]}")
            source_urls.add(chunk.source_url)
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Analyze the following documentation excerpts and extract the module structure.

IMPORTANT: Only use information explicitly present in the text. Do not hallucinate.

Documentation:
{context}

Respond in JSON format:
{{"module_name": "string", "description": "string", "submodules": {{"name": "description"}}}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract structured information only from provided text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                return ExtractedModule(
                    name=result.get('module_name', 'Unknown'),
                    description=result.get('description', ''),
                    submodules=result.get('submodules', {}),
                    source_urls=list(source_urls)
                )
        except Exception as e:
            self.log(f"LLM extraction failed: {e}")
        
        return self._extract_heuristic(chunks)
    
    def _extract_heuristic(self, chunks: List[ContentChunk]) -> Optional[ExtractedModule]:
        """Heuristic-based extraction"""
        if not chunks:
            return None
        
        headings = {}
        source_urls = set()
        all_content = []
        
        for chunk in chunks:
            source_urls.add(chunk.source_url)
            all_content.append(chunk.text)
            
            if chunk.heading:
                headings[chunk.heading.strip()] = headings.get(chunk.heading.strip(), 0) + 1
            
            if chunk.breadcrumbs and len(chunk.breadcrumbs) > 1:
                headings[chunk.breadcrumbs[1]] = headings.get(chunk.breadcrumbs[1], 0) + 3
        
        if not headings:
            return None
        
        module_name = max(headings.keys(), key=lambda x: headings[x])
        
        # Extract submodules
        submodules = {}
        for chunk in chunks:
            if chunk.heading and chunk.heading != module_name:
                sentences = chunk.text.split('.')
                desc = sentences[0].strip() if sentences else ""
                if desc and len(desc) > 10:
                    if chunk.heading not in submodules:
                        submodules[chunk.heading] = desc[:200]
        
        # Generate description
        combined = ' '.join(all_content)
        sentences = combined.split('.')
        description = '. '.join(sentences[:2]).strip() if sentences else ""
        
        return ExtractedModule(
            name=module_name,
            description=description[:300] if description else f"Documentation for {module_name}",
            submodules=submodules,
            source_urls=list(source_urls)
        )
    
    def _clean_name(self, name: str) -> str:
        """Clean up module name"""
        import re
        name = re.sub(r'^(how to |getting started with |about |introduction to )', '', name, flags=re.I)
        name = re.sub(r'( guide| documentation| docs| help| - .+)$', '', name, flags=re.I)
        if name:
            name = name[0].upper() + name[1:]
        return name.strip()


class QASpecialistAgent(BaseAgent):
    """Agent responsible for quality assurance and merging"""
    
    def __init__(self):
        super().__init__(
            name="QA Specialist",
            role="Quality Assurance Specialist",
            goal="Review, clean, and merge similar modules"
        )
    
    def execute(self, modules: List[ExtractedModule], **kwargs) -> AgentResult:
        """Merge similar modules and perform quality checks"""
        self.log("Starting quality assurance...")
        
        if len(modules) <= 1:
            return AgentResult(
                success=True,
                data=modules,
                message="No merging needed",
                agent_name=self.name
            )
        
        merged = []
        used = set()
        
        for i, mod1 in enumerate(modules):
            if i in used:
                continue
            
            similar_indices = [i]
            for j, mod2 in enumerate(modules[i+1:], start=i+1):
                if j in used:
                    continue
                
                if self._are_similar(mod1.name, mod2.name):
                    similar_indices.append(j)
                    used.add(j)
            
            used.add(i)
            
            if len(similar_indices) > 1:
                merged_module = self._merge_modules([modules[idx] for idx in similar_indices])
                merged.append(merged_module)
            else:
                merged.append(mod1)
        
        self.log(f"Merged {len(modules)} modules into {len(merged)}")
        return AgentResult(
            success=True,
            data=merged,
            message=f"Quality check complete. {len(modules)} → {len(merged)} modules",
            agent_name=self.name
        )
    
    def _are_similar(self, name1: str, name2: str) -> bool:
        """Check if two module names are similar"""
        n1, n2 = name1.lower(), name2.lower()
        if n1 in n2 or n2 in n1:
            return True
        
        words1 = set(n1.split())
        words2 = set(n2.split())
        if not words1 or not words2:
            return False
        
        intersection = words1 & words2
        union = words1 | words2
        overlap = len(intersection) / len(union)
        
        return overlap > 0.5
    
    def _merge_modules(self, modules: List[ExtractedModule]) -> ExtractedModule:
        """Merge multiple modules into one"""
        names = [m.name for m in modules]
        merged_name = max(names, key=len)
        
        descriptions = [m.description for m in modules if m.description]
        merged_description = descriptions[0] if descriptions else ""
        
        merged_submodules = {}
        for mod in modules:
            for name, desc in mod.submodules.items():
                if name not in merged_submodules:
                    merged_submodules[name] = desc
        
        merged_urls = list(set(url for mod in modules for url in mod.source_urls))
        
        return ExtractedModule(
            name=merged_name,
            description=merged_description,
            submodules=merged_submodules,
            source_urls=merged_urls
        )


# ============================================================================
# Multi-Agent Crew
# ============================================================================

class ModuleExtractionCrew:
    """Orchestrates multiple agents for module extraction"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.agents = {
            'analyst': ContentAnalystAgent(),
            'clusterer': ClusteringSpecialistAgent(),
            'extractor': ModuleExpertAgent(openai_api_key),
            'qa': QASpecialistAgent()
        }
        self.results = []
    
    def run(self, parsed_contents: List, progress_callback: Optional[Callable] = None) -> List[ExtractedModule]:
        """Execute the full extraction pipeline"""
        logger.info("🚀 Starting multi-agent extraction pipeline")
        
        # Phase 1: Content Analysis
        if progress_callback:
            progress_callback("Content Analyst: Chunking content...", 0.1)
        result1 = self.agents['analyst'].execute(parsed_contents)
        self.results.append(result1)
        
        if not result1.success:
            return []
        
        # Phase 2: Clustering
        if progress_callback:
            progress_callback("Clustering Specialist: Grouping content...", 0.35)
        result2 = self.agents['clusterer'].execute(result1.data)
        self.results.append(result2)
        
        if not result2.success:
            return []
        
        # Phase 3: Module Extraction
        if progress_callback:
            progress_callback("Module Expert: Extracting modules...", 0.6)
        result3 = self.agents['extractor'].execute(result2.data)
        self.results.append(result3)
        
        if not result3.success:
            return []
        
        # Phase 4: Quality Assurance
        if progress_callback:
            progress_callback("QA Specialist: Merging & cleaning...", 0.85)
        result4 = self.agents['qa'].execute(result3.data)
        self.results.append(result4)
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        logger.info(f"✅ Extraction complete: {len(result4.data)} modules")
        return result4.data


# ============================================================================
# CrewAI-compatible NLP Agent
# ============================================================================

class CrewAINLPAgent:
    """
    Multi-agent NLP Agent for module extraction.
    Uses lightweight custom agents (no external AI framework required).
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        self.crew = ModuleExtractionCrew(openai_api_key)
    
    def extract_modules(self, parsed_contents: List, progress_callback=None) -> List[ExtractedModule]:
        """Extract modules using multi-agent crew"""
        return self.crew.run(parsed_contents, progress_callback)
    
    def modules_to_json(self, modules: List[ExtractedModule]) -> List[Dict]:
        """Convert modules to JSON format"""
        return [
            {
                "module": m.name,
                "Description": m.description,
                "Submodules": m.submodules
            }
            for m in modules
        ]


def extract_modules_with_crew(
    parsed_contents: List,
    openai_api_key: Optional[str] = None,
    progress_callback=None
) -> List[Dict]:
    """
    Extract modules from parsed content using multi-agent system.
    """
    agent = CrewAINLPAgent(openai_api_key=openai_api_key)
    modules = agent.extract_modules(parsed_contents, progress_callback)
    return agent.modules_to_json(modules)


# Compatibility flag
CREWAI_AVAILABLE = True  # Our lightweight implementation is always available


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class MockContent:
        url: str = "https://example.com/docs"
        title: str = "Test Documentation"
        main_content: str = "This is test documentation."
        sections: List = None
        breadcrumbs: List = None
        
        def __post_init__(self):
            self.sections = self.sections or []
            self.breadcrumbs = self.breadcrumbs or []
    
    # Test
    mock = MockContent(
        sections=[
            {'heading': 'Account Settings', 'level': 1, 'content': 'Manage your account here.'},
            {'heading': 'Change Password', 'level': 2, 'content': 'Update your password.'},
            {'heading': 'Profile', 'level': 2, 'content': 'Edit profile information.'},
        ],
        breadcrumbs=['Home', 'Settings', 'Account']
    )
    
    print("Testing Multi-Agent Module Extraction...")
    print("-" * 50)
    
    agent = CrewAINLPAgent()
    modules = agent.extract_modules([mock])
    
    print("\nExtracted Modules:")
    print(json.dumps(agent.modules_to_json(modules), indent=2))
