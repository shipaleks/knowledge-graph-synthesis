# Backend Structure

This document details the backend architecture of the Knowledge Graph Synthesis System, focusing on the organization of modules, data flow, and key components.

## 1. Overall Structure

The backend follows a modular, layered architecture that separates concerns and enables flexibility. The system is organized into these main layers:

```
┌─────────────────────────────────────────────────────────────┐
│                       Application Layer                      │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Processing Modules                  │
├─────────┬─────────┬─────────┬─────────┬─────────┬───────────┤
│  Text   │Knowledge│  Graph  │Recursive│  Graph  │ Meta-Graph │
│Processing│Extraction│Creation │Reasoning│ Analysis│ Creation   │
├─────────┼─────────┼─────────┼─────────┼─────────┼───────────┤
│ Theory  │ Results │         │         │         │           │
│Formation│Generation│         │         │         │           │
└─────────┴─────────┴─────────┴─────────┴─────────┴───────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                     Infrastructure Layer                     │
├─────────────┬───────────────────────────┬──────────────────┤
│  LLM        │          Storage          │ Utility Services │
│Provider API │                           │                  │
└─────────────┴───────────────────────────┴──────────────────┘
```

## 2. Directory Structure

The project follows this directory structure:

```
src/
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── app_config.py        # Application configuration
│   └── llm_config.py        # LLM provider configuration
│
├── text_processing/         # Text processing module
│   ├── __init__.py
│   ├── loader.py            # Text loading functions
│   ├── normalizer.py        # Text normalization
│   ├── segmenter.py         # Text segmentation
│   └── summarizer.py        # Contextual summarization
│
├── knowledge_extraction/    # Knowledge extraction module
│   ├── __init__.py
│   ├── entity_extractor.py  # Entity extraction
│   ├── relation_extractor.py # Relationship extraction
│   ├── coreference_resolver.py # Coreference resolution
│   └── verifier.py          # Knowledge verification
│
├── graph_management/        # Graph management module
│   ├── __init__.py
│   ├── creator.py           # Graph creation
│   ├── storage.py           # Graph persistence
│   ├── query.py             # Graph querying
│   └── visualizer.py        # Graph visualization
│
├── reasoning/               # Reasoning module
│   ├── __init__.py
│   ├── question_generator.py # Question generation
│   ├── llm_reasoner.py      # LLM reasoning
│   └── knowledge_integrator.py # Knowledge integration
│
├── analysis/                # Graph analysis module
│   ├── __init__.py
│   ├── centrality.py        # Centrality analysis
│   ├── community.py         # Community detection
│   ├── path_analysis.py     # Path analysis
│   └── pattern_detection.py # Pattern detection
│
├── meta_graph/              # Meta-graph module
│   ├── __init__.py
│   ├── abstractor.py        # Concept abstraction
│   ├── meta_relation.py     # Meta-relationship identification
│   └── linker.py            # Graph to meta-graph linking
│
├── theory_formation/        # Theory formation module
│   ├── __init__.py
│   ├── pattern_recognizer.py # Pattern recognition
│   ├── theory_formulator.py  # Theory formulation
│   ├── hypothesis_generator.py # Hypothesis generation
│   └── hypothesis_tester.py  # Hypothesis testing
│
├── results/                 # Results generation module
│   ├── __init__.py
│   ├── format_selector.py   # Output format selection
│   ├── document_generator.py # Document generation
│   ├── visualization_creator.py # Visualization creation
│   └── self_assessor.py     # Self-assessment
│
├── llm/                     # LLM provider integration
│   ├── __init__.py
│   ├── provider.py          # Abstract provider interface
│   ├── claude_provider.py   # Claude integration
│   ├── gpt_provider.py      # GPT integration
│   ├── gemini_provider.py   # Gemini integration
│   ├── deepseek_provider.py # DeepSeek integration
│   └── ollama_provider.py   # Ollama integration
│
├── storage/                 # Storage services
│   ├── __init__.py
│   ├── graph_repository.py  # Graph storage
│   ├── document_repository.py # Document storage
│   └── cache.py             # Caching service
│
├── utils/                   # Utility services
│   ├── __init__.py
│   ├── logger.py            # Logging utility
│   ├── result.py            # Result wrapper
│   ├── language.py          # Language detection and handling
│   ├── text_utils.py        # Text manipulation utilities
│   └── timer.py             # Performance timing
│
├── frontend/                # Frontend integration
│   └── api.py               # API for frontend
│
└── main.py                  # Application entry point
```

## 3. Key Modules

### 3.1 Text Processing Module

Purpose: Transform raw text into structured segments with context.

Components:
- **Loader**: Handles text loading from various sources
- **Normalizer**: Standardizes text encoding and formatting
- **Segmenter**: Divides text into hierarchical segments
- **Summarizer**: Creates contextual summaries for segments

Key Interfaces:
```python
def process_text(text: str, language: str = None) -> List[TextSegment]:
    """Process raw text into structured segments."""
    # ...
```

### 3.2 Knowledge Extraction Module

Purpose: Extract entities and relationships from text segments.

Components:
- **EntityExtractor**: Identifies and categorizes entities
- **RelationExtractor**: Identifies relationships between entities
- **CoreferenceResolver**: Resolves entity references
- **Verifier**: Validates extracted knowledge

Key Interfaces:
```python
def extract_knowledge(segments: List[TextSegment], domain_type: str) -> KnowledgeGraph:
    """Extract structured knowledge from text segments."""
    # ...
```

### 3.3 Graph Management Module

Purpose: Create, store, and manage knowledge graphs.

Components:
- **Creator**: Constructs graph from extracted knowledge
- **Storage**: Persists graph to storage
- **Query**: Retrieves information from graph
- **Visualizer**: Creates visual representations

Key Interfaces:
```python
def create_graph(entities: List[Entity], relations: List[Relation]) -> Graph:
    """Create a graph from entities and relations."""
    # ...

def store_graph(graph: Graph, path: str) -> bool:
    """Store graph to specified location."""
    # ...

def load_graph(path: str) -> Graph:
    """Load graph from specified location."""
    # ...
```

### 3.4 Reasoning Module

Purpose: Expand knowledge graph through iterative reasoning.

Components:
- **QuestionGenerator**: Creates research questions
- **LLMReasoner**: Generates reasoning about questions
- **KnowledgeIntegrator**: Integrates new knowledge into graph

Key Interfaces:
```python
def expand_graph(graph: Graph, iterations: int = 5) -> Graph:
    """Expand graph through recursive reasoning."""
    # ...
```

### 3.5 Analysis Module

Purpose: Analyze graph structure to identify patterns and insights.

Components:
- **Centrality**: Calculates node centrality metrics
- **Community**: Detects communities in graph
- **PathAnalysis**: Analyzes significant paths
- **PatternDetection**: Identifies recurring patterns

Key Interfaces:
```python
def analyze_graph(graph: Graph) -> GraphAnalysis:
    """Perform comprehensive analysis of graph structure."""
    # ...
```

### 3.6 Meta-Graph Module

Purpose: Create higher-level abstractions of knowledge.

Components:
- **Abstractor**: Creates meta-concepts from clusters
- **MetaRelation**: Identifies relationships between meta-concepts
- **Linker**: Links meta-graph with original graph

Key Interfaces:
```python
def create_meta_graph(graph: Graph, analysis: GraphAnalysis) -> MetaGraph:
    """Create meta-graph from original graph."""
    # ...
```

### 3.7 Theory Formation Module

Purpose: Formulate theories and hypotheses from patterns.

Components:
- **PatternRecognizer**: Recognizes significant patterns
- **TheoryFormulator**: Creates theories explaining patterns
- **HypothesisGenerator**: Generates testable hypotheses
- **HypothesisTester**: Tests hypotheses against graph

Key Interfaces:
```python
def generate_theories(graph: Graph, meta_graph: MetaGraph, analysis: GraphAnalysis) -> List[Theory]:
    """Generate theories from graph analysis."""
    # ...
```

### 3.8 Results Generation Module

Purpose: Generate final output in appropriate format.

Components:
- **FormatSelector**: Determines optimal output format
- **DocumentGenerator**: Creates textual documents
- **VisualizationCreator**: Creates visualizations
- **SelfAssessor**: Provides critical assessment

Key Interfaces:
```python
def generate_results(
    graph: Graph, 
    meta_graph: MetaGraph, 
    theories: List[Theory], 
    domain_type: str,
    output_format: str = "auto"
) -> Results:
    """Generate comprehensive results package."""
    # ...
```

### 3.9 LLM Integration Module

Purpose: Provide unified interface to various LLM providers.

Components:
- **Provider**: Abstract provider interface
- **ClaudeProvider**, **GPTProvider**, etc.: Specific provider implementations

Key Interfaces:
```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate_text(self, prompt: str, options: Dict = None) -> Result[str]:
        """Generate text from prompt."""
        pass
```

## 4. Data Models

### 4.1 Core Data Models

```python
@dataclass
class TextSegment:
    """Represents a segment of text with context."""
    id: str
    text: str
    position: Tuple[int, int]  # Start and end positions
    parent_id: Optional[str]
    summary: Optional[str]
    
@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: str
    attributes: Dict[str, Any]
    source_segments: List[str]  # IDs of source segments
    confidence: float
    
@dataclass
class Relation:
    """Represents a relationship between entities."""
    id: str
    source_id: str
    target_id: str
    type: str
    attributes: Dict[str, Any]
    source_segments: List[str]  # IDs of source segments
    confidence: float

@dataclass
class MetaConcept:
    """Represents a meta-concept abstracted from entities."""
    id: str
    name: str
    definition: str
    entities: List[str]  # IDs of included entities
    attributes: Dict[str, Any]
    
@dataclass
class Theory:
    """Represents a theory derived from graph analysis."""
    id: str
    name: str
    description: str
    postulates: List[str]
    evidence: List[Dict]  # Supporting evidence
    hypotheses: List[Dict]  # Generated hypotheses
```

### 4.2 Result Wrapper

For consistent error handling:

```python
@dataclass
class Result[T]:
    """Wrapper for function results with error handling."""
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    details: Optional[Dict] = None
```

## 5. Configuration Management

### 5.1 Application Configuration

```python
class AppConfig:
    """Application configuration."""
    def __init__(self, config_path: str = None):
        self.language = "en"  # Default language
        self.max_iterations = 10  # Default max iterations for recursive reasoning
        self.output_dir = "./output"  # Default output directory
        
        if config_path:
            self._load_from_file(config_path)
    
    def _load_from_file(self, path: str):
        # Load configuration from file
        pass
```

### 5.2 LLM Provider Configuration

```python
class LLMConfig:
    """LLM provider configuration."""
    def __init__(self):
        self.provider = "claude"  # Default provider
        self.model = "claude-3-sonnet-20240229"  # Default model
        self.api_key = None
        self.max_tokens = 4000
        self.temperature = 0.3
        
    def from_env(self):
        """Load configuration from environment variables."""
        self.provider = os.environ.get("LLM_PROVIDER", self.provider)
        self.model = os.environ.get("LLM_MODEL", self.model)
        self.api_key = os.environ.get(f"{self.provider.upper()}_API_KEY")
        return self
```

## 6. Error Handling

### 6.1 Exception Hierarchy

```python
class AppError(Exception):
    """Base class for application errors."""
    pass

class ConfigError(AppError):
    """Configuration error."""
    pass

class TextProcessingError(AppError):
    """Text processing error."""
    pass

class KnowledgeExtractionError(AppError):
    """Knowledge extraction error."""
    pass

class GraphError(AppError):
    """Graph processing error."""
    pass

class LLMError(AppError):
    """LLM provider error."""
    pass
```

### 6.2 Error Handling Strategy

```python
def safe_execute[T](func: Callable[..., T], *args, **kwargs) -> Result[T]:
    """Execute function safely and return Result."""
    try:
        result = func(*args, **kwargs)
        return Result(success=True, value=result)
    except AppError as e:
        logger.error(f"Application error: {str(e)}")
        return Result(success=False, error=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return Result(success=False, error="An unexpected error occurred", details={"exception": str(e)})
```

## 7. Multilingual Support

### 7.1 Language Detection

```python
def detect_language(text: str) -> str:
    """Detect language of text (returns 'en' or 'ru')."""
    try:
        return langdetect.detect(text)
    except:
        return "en"  # Default to English
```

### 7.2 Language-Specific Prompts

```python
PROMPTS = {
    "en": {
        "entity_extraction": "Extract entities from the following text...",
        # Other English prompts
    },
    "ru": {
        "entity_extraction": "Извлеките сущности из следующего текста...",
        # Other Russian prompts
    }
}

def get_prompt(key: str, language: str = "en") -> str:
    """Get prompt for specified key and language."""
    if language not in ["en", "ru"]:
        language = "en"  # Default to English
    
    return PROMPTS[language].get(key, PROMPTS["en"].get(key, ""))
```

## 8. LLM Provider Integration

### 8.1 Provider Factory

```python
def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create LLM provider based on configuration."""
    if config.provider == "claude":
        return ClaudeProvider(api_key=config.api_key, model=config.model)
    elif config.provider == "gpt":
        return GPTProvider(api_key=config.api_key, model=config.model)
    elif config.provider == "gemini":
        return GeminiProvider(api_key=config.api_key, model=config.model)
    elif config.provider == "deepseek":
        return DeepSeekProvider(api_key=config.api_key, model=config.model)
    elif config.provider == "ollama":
        return OllamaProvider(model=config.model)
    else:
        raise ConfigError(f"Unknown LLM provider: {config.provider}")
```

### 8.2 Provider Implementation Example

```python
class ClaudeProvider(LLMProvider):
    """Claude LLM provider."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
    
    async def generate_text(self, prompt: str, options: Dict = None) -> Result[str]:
        """Generate text using Claude."""
        try:
            options = options or {}
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=options.get("max_tokens", 4000),
                temperature=options.get("temperature", 0.3),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return Result(success=True, value=response.content[0].text)
        except anthropic.APIError as e:
            return Result(success=False, error=f"Claude API error: {str(e)}")
        except Exception as e:
            return Result(success=False, error=f"Unexpected error: {str(e)}")
```

## 9. Performance Optimization

### 9.1 Caching

```python
class Cache:
    """Simple in-memory cache."""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove random item if cache is full
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
```

### 9.2 Async Processing

```python
async def process_batch(items: List[Any], processor: Callable[[Any], Awaitable[Any]], batch_size: int = 5) -> List[Any]:
    """Process items in batches asynchronously."""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)
    return results
```

## 10. Dependency Management

Dependencies are managed using Poetry or requirements.txt for simplicity:

```
# Key dependencies
python-dotenv==1.0.0
networkx==3.2.1
spacy==3.7.2
nltk==3.8.1
langdetect==1.0.9
pyvis==0.3.2
anthropic==0.18.0
openai==1.12.0
google-generativeai==0.3.2
ollama==0.1.5
```

Additional development dependencies:
```
# Development dependencies
pytest==7.4.3
black==23.11.0
flake8==6.1.0
mypy==1.7.1
```

The backend structure is designed to be modular, maintainable, and beginner-friendly while still providing a robust foundation for the Knowledge Graph Synthesis System.