"""
Test Knowledge Extraction Pipeline Module

This module tests the functionality of the knowledge extraction pipeline.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.knowledge.pipeline import KnowledgeExtractionPipeline, extract_knowledge

# Configure logger
logger = get_logger(__name__)

# Sample texts for testing
SAMPLE_TEXT_EN = """
Knowledge graphs represent a powerful way to organize and reason about information. 
They consist of entities (nodes) and relationships (edges) that connect these entities.
The process of building a knowledge graph involves several key steps:

1. Entity Extraction: Identifying important concepts, objects, or individuals mentioned in the text.
2. Relationship Extraction: Determining how these entities are connected to each other.
3. Entity Resolution: Ensuring that different mentions of the same entity are properly linked.
4. Knowledge Integration: Combining the extracted information with existing knowledge bases.

Various techniques can be used for knowledge extraction, ranging from rule-based approaches to deep learning models like BERT and GPT. These models have shown remarkable capabilities in understanding context and extracting structured information from unstructured text.

When building knowledge graphs, it's important to consider issues like data quality, scalability, and domain specificity. For example, a knowledge graph for medical research would have different requirements than one for financial analysis.

Some common challenges in knowledge graph construction include:
- Handling ambiguity in natural language
- Dealing with incomplete or contradictory information
- Scaling to large volumes of data
- Maintaining the graph over time as new information becomes available

Despite these challenges, knowledge graphs have proven valuable in many applications, from search engines and recommendation systems to scientific research and data integration.
"""

SAMPLE_TEXT_RU = """
Графы знаний представляют собой мощный способ организации и рассуждения об информации.
Они состоят из сущностей (узлов) и отношений (рёбер), которые соединяют эти сущности.
Процесс построения графа знаний включает несколько ключевых этапов:

1. Извлечение сущностей: Выявление важных концепций, объектов или индивидуумов, упомянутых в тексте.
2. Извлечение отношений: Определение того, как эти сущности связаны друг с другом.
3. Разрешение сущностей: Обеспечение правильной связи различных упоминаний одной и той же сущности.
4. Интеграция знаний: Объединение извлеченной информации с существующими базами знаний.

Для извлечения знаний могут использоваться различные методы, от подходов на основе правил до моделей глубокого обучения, таких как BERT и GPT. Эти модели продемонстрировали замечательные способности в понимании контекста и извлечении структурированной информации из неструктурированного текста.

При построении графов знаний важно учитывать такие вопросы, как качество данных, масштабируемость и специфика предметной области. Например, граф знаний для медицинских исследований будет иметь другие требования, чем граф для финансового анализа.

Некоторые распространенные проблемы при построении графов знаний включают:
- Обработка неоднозначности в естественном языке
- Работа с неполной или противоречивой информацией
- Масштабирование для больших объемов данных
- Поддержание графа с течением времени по мере поступления новой информации

Несмотря на эти проблемы, графы знаний доказали свою ценность во многих приложениях, от поисковых систем и рекомендательных систем до научных исследований и интеграции данных.
"""


def test_pipeline_with_text(text: str, language: str = "en") -> None:
    """
    Test the knowledge extraction pipeline with a sample text.
    
    Args:
        text: Text to process
        language: Language code
    """
    logger.info(f"Testing knowledge extraction pipeline with {language} text")
    
    # Create pipeline
    pipeline = KnowledgeExtractionPipeline()
    
    # Process text
    result = pipeline.process(text, language=language, output_prefix=f"test_{language}")
    
    if not result.success:
        logger.error(f"Pipeline processing failed: {result.error}")
        return
    
    # Get results
    extraction_results = result.value
    
    # Log results summary
    entity_count = extraction_results["metadata"]["entity_count"]
    relationship_count = extraction_results["metadata"]["relationship_count"]
    issue_count = extraction_results["metadata"]["issue_count"]
    is_valid = extraction_results["metadata"]["is_valid"]
    
    logger.info(f"Knowledge extraction completed successfully:")
    logger.info(f"  - Entities: {entity_count}")
    logger.info(f"  - Relationships: {relationship_count}")
    logger.info(f"  - Verification issues: {issue_count}")
    logger.info(f"  - Knowledge graph valid: {is_valid}")
    
    # Log timing information
    timing = extraction_results["metadata"]["timing"]
    logger.info(f"Timing information:")
    for step, time_taken in timing.items():
        logger.info(f"  - {step}: {time_taken:.2f} seconds")


def test_pipeline_with_file(file_path: str, language: Optional[str] = None) -> None:
    """
    Test the knowledge extraction pipeline with a file.
    
    Args:
        file_path: Path to the file to process
        language: Language code (auto-detected if None)
    """
    logger.info(f"Testing knowledge extraction pipeline with file: {file_path}")
    
    # Create pipeline
    pipeline = KnowledgeExtractionPipeline()
    
    # Process file
    result = pipeline.process(Path(file_path), language=language, output_prefix="test_file")
    
    if not result.success:
        logger.error(f"Pipeline processing failed: {result.error}")
        return
    
    # Get results
    extraction_results = result.value
    
    # Log results summary
    entity_count = extraction_results["metadata"]["entity_count"]
    relationship_count = extraction_results["metadata"]["relationship_count"]
    issue_count = extraction_results["metadata"]["issue_count"]
    is_valid = extraction_results["metadata"]["is_valid"]
    
    logger.info(f"Knowledge extraction completed successfully:")
    logger.info(f"  - Entities: {entity_count}")
    logger.info(f"  - Relationships: {relationship_count}")
    logger.info(f"  - Verification issues: {issue_count}")
    logger.info(f"  - Knowledge graph valid: {is_valid}")


def test_convenience_function(text: str, language: str = "en") -> None:
    """
    Test the extract_knowledge convenience function.
    
    Args:
        text: Text to process
        language: Language code
    """
    logger.info(f"Testing extract_knowledge convenience function with {language} text")
    
    # Process text using convenience function
    result = extract_knowledge(text, language=language, output_prefix=f"test_convenience_{language}")
    
    if not result.success:
        logger.error(f"Knowledge extraction failed: {result.error}")
        return
    
    # Get results
    extraction_results = result.value
    
    # Log results summary
    entity_count = extraction_results["metadata"]["entity_count"]
    relationship_count = extraction_results["metadata"]["relationship_count"]
    
    logger.info(f"Knowledge extraction completed successfully:")
    logger.info(f"  - Entities: {entity_count}")
    logger.info(f"  - Relationships: {relationship_count}")


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description="Test knowledge extraction pipeline")
    parser.add_argument("--test", choices=["en", "ru", "file", "convenience", "all"], 
                      help="Test to run", default="en")
    parser.add_argument("--file", help="File to process for file test", default=None)
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Run tests
    if args.test == "en" or args.test == "all":
        test_pipeline_with_text(SAMPLE_TEXT_EN, "en")
    
    if args.test == "ru" or args.test == "all":
        test_pipeline_with_text(SAMPLE_TEXT_RU, "ru")
    
    if args.test == "file" or args.test == "all":
        if args.file:
            test_pipeline_with_file(args.file)
        else:
            logger.error("File path must be provided for file test")
    
    if args.test == "convenience" or args.test == "all":
        test_convenience_function(SAMPLE_TEXT_EN, "en")


if __name__ == "__main__":
    main() 