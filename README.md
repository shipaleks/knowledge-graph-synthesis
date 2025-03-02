# Knowledge Graph Synthesis System

A system for transforming unstructured text into structured knowledge through automated graph creation, expansion, and analysis.

## Overview

The Knowledge Graph Synthesis System processes text input to extract entities and relationships, builds a knowledge graph, expands it through recursive reasoning, analyzes its structure, creates abstractions, and generates theories and insights. The system supports both Russian and English languages and works with multiple Large Language Model providers.


## Features

- **Text Processing**: Hierarchical segmentation with contextual summarization
- **Knowledge Extraction**: Entity and relationship extraction with coreference resolution
- **Graph Management**: Creation, storage, and visualization of knowledge graphs
- **Recursive Reasoning**: Autonomous expansion of knowledge graphs through questioning and reasoning
- **Graph Analysis**: Calculation of structural metrics, community detection, and pattern identification
- **Meta-Graph Creation**: Abstraction of concepts into higher-level representations
- **Theory Formation**: Generation of theories and hypotheses with testing
- **Results Generation**: Production of documents, visualizations, and insights

## Supported LLM Providers

- Claude (Anthropic)
- GPT (OpenAI)
- Gemini (Google)
- DeepSeek
- Ollama (local models)

## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/knowledge-graph-synthesis.git
   cd knowledge-graph-synthesis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on the example:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Usage

### Command Line Interface

Process a text file and generate insights:

```bash
python src/main.py process --input text.txt --output results/ --language en --provider claude
```

### Streamlit Interface

Start the interactive Streamlit interface:

```bash
python -m streamlit run src/frontend/app.py
```

### Configuration

Configure LLM providers in your `.env` file:

```
CLAUDE_API_KEY=your_api_key
GPT_API_KEY=your_api_key
GEMINI_API_KEY=your_api_key
DEEPSEEK_API_KEY=your_api_key
# For Ollama, no API key is needed
```

## Development

### Project Structure

```
src/
├── config/                  # Configuration management
├── text_processing/         # Text processing module
├── knowledge_extraction/    # Knowledge extraction module
├── graph_management/        # Graph management module
├── reasoning/               # Reasoning module
├── analysis/                # Graph analysis module
├── meta_graph/              # Meta-graph module
├── theory_formation/        # Theory formation module
├── results/                 # Results generation module
├── llm/                     # LLM provider integration
├── storage/                 # Storage services
├── utils/                   # Utility services
├── frontend/                # Frontend integration
└── main.py                  # Application entry point
```

### Running Tests

```bash
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Product Requirements Document](docs/PRD.md)
- [Application Flow](docs/AppFlow.md)
- [Technology Stack](docs/TechStack.md)
- [Frontend Guidelines](docs/FrontendGuidelines.md)
- [Backend Structure](docs/BackendStructure.md)
- [Implementation Plan](docs/Implementation-Plan.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by research in knowledge graph construction and reasoning
- Thanks to the developers of all the libraries and tools used in this project