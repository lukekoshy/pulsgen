"""
NLP Agent Module
Handles module/submodule inference using embeddings, clustering, and LLM reasoning.
Features:
- Text chunking and embedding
- Semantic clustering
- Module/submodule extraction
- Description generation (no hallucination)
"""

import json
import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import hashlib

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
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
class Module:
    """Represents an extracted module"""
    name: str
    description: str
    submodules: Dict[str, str] = field(default_factory=dict)
    source_urls: List[str] = field(default_factory=list)


class NLPAgent:
    """
    NLP Agent for extracting modules and submodules from documentation.
    Uses embeddings for clustering and LLM for inference.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: Optional[str] = None,
        use_embeddings: bool = True,
        similarity_threshold: float = 0.65
    ):
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings and SKLEARN_AVAILABLE
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and use_embeddings:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        # Initialize OpenAI client
        self.openai_client = None
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    def _chunk_content(self, parsed_contents: List) -> List[ContentChunk]:
        """Split parsed content into meaningful chunks"""
        chunks = []
        
        for content in parsed_contents:
            # Create chunks from sections
            if content.sections:
                for section in content.sections:
                    chunk = ContentChunk(
                        text=f"{section['heading']}\n{section['content']}",
                        source_url=content.url,
                        heading=section['heading'],
                        level=section['level'],
                        breadcrumbs=content.breadcrumbs
                    )
                    chunks.append(chunk)
            
            # If no sections, create a single chunk from main content
            if not content.sections and content.main_content:
                # Split by paragraphs if content is long
                paragraphs = content.main_content.split('\n\n')
                for i, para in enumerate(paragraphs):
                    if len(para.strip()) > 50:
                        chunk = ContentChunk(
                            text=para.strip(),
                            source_url=content.url,
                            heading=content.title if i == 0 else "",
                            level=1,
                            breadcrumbs=content.breadcrumbs
                        )
                        chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text (with caching)"""
        if not self.embedding_model:
            return None
        
        # Create cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            embedding = self.embedding_model.encode(text).tolist()
            self._embedding_cache[cache_key] = embedding
            return embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def _embed_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """Add embeddings to chunks"""
        if not self.embedding_model:
            return chunks
        
        texts = [chunk.text[:512] for chunk in chunks]  # Limit text length
        
        try:
            embeddings = self.embedding_model.encode(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
        
        return chunks
    
    def _cluster_chunks(self, chunks: List[ContentChunk], n_clusters: int = None) -> Dict[int, List[ContentChunk]]:
        """Cluster chunks by semantic similarity"""
        if not SKLEARN_AVAILABLE or not chunks:
            # Fallback: group by heading level and breadcrumbs
            return self._cluster_by_structure(chunks)
        
        # Get embeddings
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        
        if len(embeddings) < 2:
            return {0: chunks}
        
        embeddings_array = np.array(embeddings)
        
        # Determine number of clusters
        if n_clusters is None:
            # Use a reasonable default based on content
            n_clusters = min(max(2, len(chunks) // 5), 15)
        
        n_clusters = min(n_clusters, len(chunks))
        
        try:
            # Use Agglomerative Clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_array)
            
            # Group chunks by cluster
            clustered = defaultdict(list)
            for chunk, label in zip(chunks, labels):
                clustered[label].append(chunk)
            
            return dict(clustered)
        
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return self._cluster_by_structure(chunks)
    
    def _cluster_by_structure(self, chunks: List[ContentChunk]) -> Dict[int, List[ContentChunk]]:
        """Fallback clustering based on document structure"""
        clustered = defaultdict(list)
        
        # Group by first breadcrumb or heading
        for chunk in chunks:
            if chunk.breadcrumbs and len(chunk.breadcrumbs) > 1:
                key = chunk.breadcrumbs[1]  # Second breadcrumb is often the module
            elif chunk.heading:
                key = chunk.heading.split(':')[0].split('-')[0].strip()
            else:
                key = "General"
            
            # Hash the key to get a cluster number
            cluster_id = hash(key) % 100
            clustered[cluster_id].append(chunk)
        
        return dict(clustered)
    
    def _extract_module_with_llm(self, cluster_chunks: List[ContentChunk]) -> Optional[Module]:
        """Use LLM to extract module information from a cluster"""
        if not self.openai_client:
            return self._extract_module_heuristic(cluster_chunks)
        
        # Prepare context from chunks
        context_parts = []
        source_urls = set()
        
        for chunk in cluster_chunks[:10]:  # Limit to avoid token overflow
            context_parts.append(f"Heading: {chunk.heading}\nContent: {chunk.text[:500]}")
            source_urls.add(chunk.source_url)
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Analyze the following documentation excerpts and extract the module structure.

IMPORTANT RULES:
1. Only use information explicitly present in the text
2. Do not make up or infer features not mentioned
3. Descriptions must be based on actual content
4. If unsure, use general descriptions from the text

Documentation excerpts:
{context}

Based ONLY on the above content, provide:
1. A concise module name (the main topic/feature)
2. A brief description of what this module covers (1-2 sentences from the docs)
3. List of submodules (specific features/tasks) with their descriptions

Respond in this exact JSON format:
{{
    "module_name": "string",
    "description": "string",
    "submodules": {{
        "submodule_name": "description from docs",
        ...
    }}
}}

Only include submodules that are clearly mentioned in the content."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a documentation analyzer. Extract structured information only from the provided text. Never hallucinate or add information not present in the source."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                
                return Module(
                    name=result.get('module_name', 'Unknown Module'),
                    description=result.get('description', ''),
                    submodules=result.get('submodules', {}),
                    source_urls=list(source_urls)
                )
        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
        
        return self._extract_module_heuristic(cluster_chunks)
    
    def _extract_module_heuristic(self, cluster_chunks: List[ContentChunk]) -> Optional[Module]:
        """Heuristic-based module extraction (fallback)"""
        if not cluster_chunks:
            return None
        
        # Find the most common high-level heading
        headings = defaultdict(int)
        source_urls = set()
        all_content = []
        
        for chunk in cluster_chunks:
            source_urls.add(chunk.source_url)
            all_content.append(chunk.text)
            
            if chunk.heading:
                # Clean heading
                clean_heading = chunk.heading.strip()
                headings[clean_heading] += 1
            
            # Use breadcrumbs for module name
            if chunk.breadcrumbs and len(chunk.breadcrumbs) > 1:
                headings[chunk.breadcrumbs[1]] += 3  # Weight breadcrumbs higher
        
        if not headings:
            return None
        
        # Get most common heading as module name
        module_name = max(headings.keys(), key=lambda x: headings[x])
        
        # Extract submodules from lower-level headings
        submodules = {}
        for chunk in cluster_chunks:
            if chunk.heading and chunk.heading != module_name:
                # Use first sentence of content as description
                content_sentences = chunk.text.split('.')
                description = content_sentences[0].strip() if content_sentences else ""
                
                if description and len(description) > 10:
                    submodule_name = chunk.heading
                    if submodule_name not in submodules:
                        submodules[submodule_name] = description[:200]
        
        # Generate module description from content
        combined_content = ' '.join(all_content)
        sentences = combined_content.split('.')
        description = '. '.join(sentences[:2]).strip() if sentences else ""
        
        return Module(
            name=module_name,
            description=description[:300] if description else f"Documentation for {module_name}",
            submodules=submodules,
            source_urls=list(source_urls)
        )
    
    def _merge_similar_modules(self, modules: List[Module]) -> List[Module]:
        """Merge modules with similar names or content"""
        if len(modules) <= 1:
            return modules
        
        merged = []
        used = set()
        
        for i, mod1 in enumerate(modules):
            if i in used:
                continue
            
            # Find similar modules
            similar_indices = [i]
            for j, mod2 in enumerate(modules[i+1:], start=i+1):
                if j in used:
                    continue
                
                # Check name similarity
                name1 = mod1.name.lower()
                name2 = mod2.name.lower()
                
                if (name1 in name2 or name2 in name1 or
                    self._word_overlap(name1, name2) > 0.5):
                    similar_indices.append(j)
                    used.add(j)
            
            used.add(i)
            
            # Merge similar modules
            if len(similar_indices) > 1:
                merged_module = self._merge_modules([modules[idx] for idx in similar_indices])
                merged.append(merged_module)
            else:
                merged.append(mod1)
        
        return merged
    
    def _word_overlap(self, s1: str, s2: str) -> float:
        """Calculate word overlap ratio"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _merge_modules(self, modules: List[Module]) -> Module:
        """Merge multiple modules into one"""
        # Use the longest name
        names = [m.name for m in modules]
        merged_name = max(names, key=len)
        
        # Combine descriptions
        descriptions = [m.description for m in modules if m.description]
        merged_description = descriptions[0] if descriptions else ""
        
        # Combine submodules
        merged_submodules = {}
        for mod in modules:
            for name, desc in mod.submodules.items():
                if name not in merged_submodules:
                    merged_submodules[name] = desc
        
        # Combine URLs
        merged_urls = list(set(url for mod in modules for url in mod.source_urls))
        
        return Module(
            name=merged_name,
            description=merged_description,
            submodules=merged_submodules,
            source_urls=merged_urls
        )
    
    def _clean_module_name(self, name: str) -> str:
        """Clean up module name"""
        # Remove common prefixes/suffixes
        name = re.sub(r'^(how to |getting started with |about |introduction to )', '', name, flags=re.I)
        name = re.sub(r'( guide| documentation| docs| help| - .+)$', '', name, flags=re.I)
        
        # Capitalize properly
        if name:
            name = name[0].upper() + name[1:]
        
        return name.strip()
    
    def extract_modules(self, parsed_contents: List, progress_callback=None) -> List[Module]:
        """
        Main method to extract modules from parsed content.
        Returns list of Module objects.
        """
        logger.info(f"Processing {len(parsed_contents)} pages")
        
        # Step 1: Chunk content
        if progress_callback:
            progress_callback("Chunking content...", 0.1)
        chunks = self._chunk_content(parsed_contents)
        logger.info(f"Created {len(chunks)} content chunks")
        
        if not chunks:
            return []
        
        # Step 2: Embed chunks (if available)
        if progress_callback:
            progress_callback("Creating embeddings...", 0.3)
        if self.use_embeddings and self.embedding_model:
            chunks = self._embed_chunks(chunks)
        
        # Step 3: Cluster chunks
        if progress_callback:
            progress_callback("Clustering content...", 0.5)
        clusters = self._cluster_chunks(chunks)
        logger.info(f"Created {len(clusters)} clusters")
        
        # Step 4: Extract modules from clusters
        if progress_callback:
            progress_callback("Extracting modules...", 0.7)
        modules = []
        for cluster_id, cluster_chunks in clusters.items():
            if len(cluster_chunks) >= 1:  # At least one chunk
                module = self._extract_module_with_llm(cluster_chunks)
                if module and module.name:
                    module.name = self._clean_module_name(module.name)
                    modules.append(module)
        
        # Step 5: Merge similar modules
        if progress_callback:
            progress_callback("Merging duplicates...", 0.9)
        modules = self._merge_similar_modules(modules)
        
        logger.info(f"Extracted {len(modules)} modules")
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        return modules
    
    def modules_to_json(self, modules: List[Module]) -> List[Dict]:
        """Convert modules to the required JSON format"""
        result = []
        
        for module in modules:
            result.append({
                "module": module.name,
                "Description": module.description,
                "Submodules": module.submodules
            })
        
        return result


def extract_modules_from_content(parsed_contents: List, 
                                 openai_api_key: Optional[str] = None,
                                 progress_callback=None) -> List[Dict]:
    """
    Convenience function to extract modules from parsed content.
    Returns JSON-serializable list.
    """
    agent = NLPAgent(openai_api_key=openai_api_key)
    modules = agent.extract_modules(parsed_contents, progress_callback)
    return agent.modules_to_json(modules)


if __name__ == "__main__":
    # Test with mock data
    from dataclasses import dataclass
    
    @dataclass
    class MockContent:
        url: str = "https://example.com/docs"
        title: str = "Test Documentation"
        main_content: str = "This is a test documentation page about user settings."
        sections: List = None
        headings_hierarchy: List = None
        bullet_points: List = None
        breadcrumbs: List = None
        metadata: Dict = None
        
        def __post_init__(self):
            self.sections = self.sections or []
            self.headings_hierarchy = self.headings_hierarchy or []
            self.bullet_points = self.bullet_points or []
            self.breadcrumbs = self.breadcrumbs or []
            self.metadata = self.metadata or {}
    
    mock_content = MockContent(
        sections=[
            {'heading': 'Account Settings', 'level': 1, 'content': 'Manage your account settings here.'},
            {'heading': 'Change Password', 'level': 2, 'content': 'Update your password for security.'},
            {'heading': 'Profile Settings', 'level': 2, 'content': 'Edit your profile information.'},
        ],
        breadcrumbs=['Home', 'Settings', 'Account']
    )
    
    agent = NLPAgent(use_embeddings=False)  # Test without embeddings
    modules = agent.extract_modules([mock_content])
    
    print(json.dumps(agent.modules_to_json(modules), indent=2))
