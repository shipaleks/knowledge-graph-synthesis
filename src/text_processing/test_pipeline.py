#!/usr/bin/env python3
"""
Test script for the text processing pipeline.

This script demonstrates how to use the complete text processing pipeline
to load, segment, and summarize text.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.result import Result
from src.utils.logger import setup_logging, get_logger
from src.config.app_config import AppConfig
from src.text_processing.pipeline import process_text, TextProcessingPipeline

# Set up logging
logger = get_logger(__name__)

# Sample text for testing
SAMPLE_TEXT_EN = """
# Knowledge Graphs: A Comprehensive Overview

## Introduction
Knowledge graphs represent a powerful way to organize and interconnect information, providing a structured representation of knowledge that combines the flexibility of graphs with the expressiveness of semantics. They are essential tools in modern information systems, enabling advanced data integration, reasoning, and discovery.

## Core Components
The fundamental building blocks of knowledge graphs include:

1. **Entities** - Representing real-world objects, concepts, or events (nodes in the graph)
2. **Relationships** - Expressing how entities are connected to each other (edges)
3. **Attributes** - Properties that describe entities and relationships in more detail
4. **Ontologies** - Formal specifications of conceptualizations that define entity types and relationship semantics

## Applications
Knowledge graphs find applications across numerous domains:

- **Search engines** use them to understand query semantics and enhance results with structured information
- **Recommendation systems** leverage knowledge graphs to model complex user preferences and item relationships
- **Scientific research** employs them to integrate heterogeneous data sources and discover new connections
- **Enterprise information systems** utilize knowledge graphs to create unified views of organizational knowledge
"""

SAMPLE_TEXT_RU = """
# Графы знаний: всесторонний обзор

## Введение
Графы знаний представляют собой мощный способ организации и взаимосвязи информации, обеспечивая структурированное представление знаний, которое сочетает гибкость графов с выразительностью семантики. Они являются важнейшими инструментами в современных информационных системах, обеспечивающими продвинутую интеграцию данных, рассуждения и обнаружение знаний.

## Основные компоненты
Фундаментальные строительные блоки графов знаний включают:

1. **Сущности** - представляющие реальные объекты, концепции или события (узлы в графе)
2. **Отношения** - выражающие, как сущности связаны друг с другом (рёбра)
3. **Атрибуты** - свойства, которые подробнее описывают сущности и отношения
4. **Онтологии** - формальные спецификации концептуализаций, определяющие типы сущностей и семантику отношений

## Применения
Графы знаний находят применение во многих областях:

- **Поисковые системы** используют их для понимания семантики запросов и улучшения результатов структурированной информацией
- **Рекомендательные системы** используют графы знаний для моделирования сложных предпочтений пользователей и связей между элементами
- **Научные исследования** применяют их для интеграции разнородных источников данных и обнаружения новых связей
- **Корпоративные информационные системы** используют графы знаний для создания единого представления организационных знаний
"""


def test_pipeline_with_sample_text(language: str):
    """
    Test the text processing pipeline with the sample text.
    
    Args:
        language: Language code
    """
    # Select sample text based on language
    sample_text = SAMPLE_TEXT_EN if language == "en" else SAMPLE_TEXT_RU
    
    logger.info(f"Testing text processing pipeline with sample {language} text ({len(sample_text)} chars)")
    
    # Process text
    result = process_text(
        input_text=sample_text,
        language=language,
        use_llm_segmentation=True,
        use_cache=True,
        save_intermediates=True,
        output_prefix="sample"
    )
    
    if not result.success:
        logger.error(f"Text processing failed: {result.error}")
        return
    
    # Show results summary
    results = result.value
    metadata = results["metadata"]
    
    logger.info("Text processing completed successfully")
    logger.info(f"Language: {metadata['language']}")
    logger.info(f"Text length: {metadata['text_length']} characters")
    logger.info(f"Segments created: {metadata['segment_count']}")
    logger.info(f"Summaries generated: {metadata['summary_count']}")
    logger.info(f"Processing time: {metadata['timing']['overall']:.2f} seconds")
    
    # Detailed timing breakdown
    logger.info("Processing time breakdown:")
    for step, time_taken in metadata["timing"].items():
        if step != "overall":
            logger.info(f"  - {step}: {time_taken:.2f} seconds")
    
    logger.info(f"Results saved to the output directory")


def test_pipeline_with_file(file_path: str, language: Optional[str] = None):
    """
    Test the text processing pipeline with a text file.
    
    Args:
        file_path: Path to the text file
        language: Language code (auto-detected if None)
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info(f"Testing text processing pipeline with file: {file_path}")
    
    # Process file
    result = process_text(
        input_text=path,
        language=language,
        use_llm_segmentation=True,
        use_cache=True,
        save_intermediates=True,
        output_prefix=path.stem
    )
    
    if not result.success:
        logger.error(f"File processing failed: {result.error}")
        return
    
    # Show results summary
    results = result.value
    metadata = results["metadata"]
    
    logger.info("File processing completed successfully")
    logger.info(f"Language: {metadata['language']}")
    logger.info(f"Text length: {metadata['text_length']} characters")
    logger.info(f"Segments created: {metadata['segment_count']}")
    logger.info(f"Summaries generated: {metadata['summary_count']}")
    logger.info(f"Processing time: {metadata['timing']['overall']:.2f} seconds")
    
    # Detailed timing breakdown
    logger.info("Processing time breakdown:")
    for step, time_taken in metadata["timing"].items():
        if step != "overall":
            logger.info(f"  - {step}: {time_taken:.2f} seconds")
    
    logger.info(f"Results saved to the output directory with prefix: {path.stem}")


def display_result_stats(results: Dict[str, Any]):
    """
    Display statistics about the processing results.
    
    Args:
        results: Processing results
    """
    print("\n" + "="*50)
    print("TEXT PROCESSING RESULTS")
    print("="*50)
    
    metadata = results["metadata"]
    print(f"Language: {metadata['language']}")
    print(f"Text length: {metadata['text_length']} characters")
    print(f"Segments created: {metadata['segment_count']}")
    print(f"Summaries generated: {metadata['summary_count']}")
    print(f"Total processing time: {metadata['timing']['overall']:.2f} seconds")
    
    print("\nTiming breakdown:")
    for step, time_taken in metadata["timing"].items():
        if step != "overall":
            print(f"  • {step}: {time_taken:.2f} seconds")
    
    print("\nSegment types:")
    segment_types = {}
    for seg_id, summary in results["summaries"].items():
        seg_type = summary.get("segment_type", "unknown")
        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
    
    for seg_type, count in segment_types.items():
        print(f"  • {seg_type}: {count}")
    
    print("="*50)


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test text processing pipeline")
    
    # Define argument groups
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sample', choices=['en', 'ru'], help='Use sample text in specified language')
    input_group.add_argument('--file', type=str, help='Path to text file to process')
    
    # Other options
    parser.add_argument('--language', type=str, choices=['en', 'ru'], help='Force specific language')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM-based segmentation')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--no-save', action='store_true', help='Disable saving intermediate results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    parser.add_argument('--output-prefix', type=str, help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Configure options
    use_llm = not args.no_llm
    use_cache = not args.no_cache
    save_intermediates = not args.no_save
    
    if args.sample:
        # Test with sample text
        test_pipeline_with_sample_text(args.sample)
    elif args.file:
        # Test with file
        test_pipeline_with_file(args.file, args.language)


if __name__ == "__main__":
    main() 