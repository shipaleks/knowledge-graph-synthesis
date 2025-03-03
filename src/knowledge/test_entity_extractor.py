#!/usr/bin/env python3
"""
Test script for entity extraction functionality.

This script demonstrates how to use the entity extraction module
to extract entities from text and segments.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.result import Result
from src.utils.logger import setup_logging, get_logger
from src.config.app_config import AppConfig
from src.text_processing.segment import Segment
from src.text_processing.text_segmenter import segment_text
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.entity_extractor import extract_entities, EntityExtractor

# Set up logging
logger = get_logger(__name__)

# Sample text for testing
SAMPLE_TEXT_EN = """
Knowledge Graph Construction

Knowledge graphs represent structured information about entities and their relationships. The process of building a knowledge graph involves several key steps:

First, entity extraction identifies important concepts, objects, or individuals mentioned in the text. For example, in a document about artificial intelligence, entities might include "machine learning", "neural networks", and "Turing test".

Second, relationship extraction determines how these entities are connected. These connections might be hierarchical (e.g., "neural networks" are a "subset of" machine learning techniques) or causal (e.g., "overfitting" leads to "poor generalization").

Third, entity resolution or coreference resolution ensures that different mentions of the same entity are properly linked. For instance, recognizing that "AI", "artificial intelligence", and "machine intelligence" all refer to the same concept.

Finally, knowledge integration combines the extracted information with existing knowledge bases, enhancing the overall graph structure and filling potential gaps.

These steps can be performed using various techniques, from rule-based approaches to advanced deep learning models like BERT or GPT. The choice of technique depends on factors such as domain specificity, available training data, and required accuracy.
"""

SAMPLE_TEXT_RU = """
Построение графов знаний

Графы знаний представляют структурированную информацию о сущностях и их отношениях. Процесс построения графа знаний включает несколько ключевых этапов:

Во-первых, извлечение сущностей идентифицирует важные концепции, объекты или лица, упомянутые в тексте. Например, в документе об искусственном интеллекте сущностями могут быть "машинное обучение", "нейронные сети" и "тест Тьюринга".

Во-вторых, извлечение отношений определяет, как эти сущности связаны между собой. Эти связи могут быть иерархическими (например, "нейронные сети" являются "подмножеством" методов машинного обучения) или причинно-следственными (например, "переобучение" приводит к "плохой обобщающей способности").

В-третьих, разрешение сущностей или кореференции обеспечивает правильную связь различных упоминаний одной и той же сущности. Например, распознавание того, что "ИИ", "искусственный интеллект" и "машинный интеллект" относятся к одному и тому же понятию.

Наконец, интеграция знаний объединяет извлеченную информацию с существующими базами знаний, улучшая общую структуру графа и заполняя потенциальные пробелы.

Эти шаги могут выполняться с использованием различных методов, от подходов на основе правил до продвинутых моделей глубокого обучения, таких как BERT или GPT. Выбор метода зависит от таких факторов, как специфика предметной области, доступные обучающие данные и требуемая точность.
"""


def test_entity_extraction_from_text(language: str, domain_type: str = "Knowledge Graph Construction"):
    """
    Test entity extraction from plain text.
    
    Args:
        language: Language code
        domain_type: Domain type for entity extraction
    """
    # Select sample text based on language
    sample_text = SAMPLE_TEXT_EN if language == "en" else SAMPLE_TEXT_RU
    
    logger.info(f"Testing entity extraction from text in {language}")
    logger.info(f"Domain type: {domain_type}")
    logger.info(f"Text length: {len(sample_text)} characters")
    
    # Extract entities
    result = extract_entities(
        text_or_segment=sample_text,
        language=language,
        domain_type=domain_type
    )
    
    if not result.success:
        logger.error(f"Entity extraction failed: {result.error}")
        return
    
    entities = result.value
    
    # Print results
    logger.info(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print_entity(entity)


def test_entity_extraction_from_segments(language: str, domain_type: str = "Knowledge Graph Construction"):
    """
    Test entity extraction from text segments.
    
    Args:
        language: Language code
        domain_type: Domain type for entity extraction
    """
    # Select sample text based on language
    sample_text = SAMPLE_TEXT_EN if language == "en" else SAMPLE_TEXT_RU
    
    logger.info(f"Testing entity extraction from segmented text in {language}")
    logger.info(f"Domain type: {domain_type}")
    
    # Segment text first
    logger.info("Segmenting text...")
    segmentation_result = segment_text(sample_text, language=language)
    
    if not segmentation_result.success:
        logger.error(f"Text segmentation failed: {segmentation_result.error}")
        return
    
    segments = segmentation_result.value
    logger.info(f"Created {len(segments.segments)} segments")
    
    # Extract entities from segments
    logger.info("Extracting entities from segments...")
    result = extract_entities(
        text_or_segment=segments,
        language=language,
        domain_type=domain_type
    )
    
    if not result.success:
        logger.error(f"Entity extraction failed: {result.error}")
        return
    
    registry = result.value
    
    # Print entity registry statistics
    logger.info(f"Extracted {registry.count()} unique entities")
    
    # Group entities by type
    entities_by_type = {}
    for entity in registry.all():
        if entity.entity_type not in entities_by_type:
            entities_by_type[entity.entity_type] = []
        entities_by_type[entity.entity_type].append(entity)
    
    # Print entity types and counts
    logger.info("Entity types:")
    for entity_type, entities in sorted(entities_by_type.items()):
        logger.info(f"  - {entity_type}: {len(entities)} entities")
    
    # Print top 5 entities by confidence
    top_entities = sorted(registry.all(), key=lambda e: e.confidence, reverse=True)[:5]
    logger.info("Top 5 entities by confidence:")
    for entity in top_entities:
        print_entity(entity)


def print_entity(entity: Entity):
    """
    Print entity information.
    
    Args:
        entity: Entity to print
    """
    print("\n" + "="*50)
    print(f"Entity: {entity.name}")
    print(f"Type: {entity.entity_type}")
    print(f"Confidence: {entity.confidence:.2f}")
    
    if entity.context:
        print("\nContext:")
        print(f"  \"{entity.context}\"")
    
    if entity.attributes:
        print("\nAttributes:")
        for key, value in entity.attributes.items():
            print(f"  - {key}: {value}")
    
    print("="*50)


def test_entity_extraction_with_custom_concepts(language: str):
    """
    Test entity extraction with custom concept list.
    
    Args:
        language: Language code
    """
    # Select sample text based on language
    sample_text = SAMPLE_TEXT_EN if language == "en" else SAMPLE_TEXT_RU
    
    # Define custom concepts
    custom_concepts = [
        "Process", "Method", "Tool", "Algorithm", 
        "Framework", "Technique", "System", "Concept"
    ]
    
    if language == "ru":
        custom_concepts = [
            "Процесс", "Метод", "Инструмент", "Алгоритм", 
            "Фреймворк", "Техника", "Система", "Концепция"
        ]
    
    logger.info(f"Testing entity extraction with custom concepts in {language}")
    logger.info(f"Custom concepts: {', '.join(custom_concepts)}")
    
    # Extract entities
    result = extract_entities(
        text_or_segment=sample_text,
        language=language,
        domain_type="AI and Knowledge Representation",
        key_concepts=custom_concepts
    )
    
    if not result.success:
        logger.error(f"Entity extraction failed: {result.error}")
        return
    
    entities = result.value
    
    # Print results
    logger.info(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print_entity(entity)


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test entity extraction functionality")
    
    # Define arguments
    parser.add_argument('--test', choices=['text', 'segments', 'custom'], 
                        default='text', help='Test to run')
    parser.add_argument('--language', choices=['en', 'ru'], 
                        default='en', help='Language to use')
    parser.add_argument('--domain', type=str, 
                        default='Knowledge Graph Construction', 
                        help='Domain type for entity extraction')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Run the selected test
    if args.test == 'text':
        test_entity_extraction_from_text(args.language, args.domain)
    elif args.test == 'segments':
        test_entity_extraction_from_segments(args.language, args.domain)
    elif args.test == 'custom':
        test_entity_extraction_with_custom_concepts(args.language)


if __name__ == "__main__":
    main() 