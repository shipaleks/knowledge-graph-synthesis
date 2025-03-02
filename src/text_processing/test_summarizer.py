#!/usr/bin/env python3
"""
Test script for text summarization functionality.

This script demonstrates how to use the text summarizer module
to create summaries for text segments.
"""

import os
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.result import Result
from src.utils.logger import setup_logging, get_logger
from src.config.app_config import AppConfig
from src.text_processing.segment import Segment, SegmentationResult
from src.text_processing.text_segmenter import TextSegmenter, segment_text
from src.text_processing.text_summarizer import TextSummarizer, summarize_text

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

## Construction Techniques
Building knowledge graphs typically involves several approaches:

1. **Manual curation** by domain experts
2. **Information extraction** from unstructured text
3. **Integration of existing structured data** sources
4. **Crowd-sourcing** and collaborative editing
5. **Machine learning** for automated knowledge acquisition

## Challenges
Despite their benefits, knowledge graphs face several challenges:

- **Quality assurance** and validation of extracted information
- **Scalability** issues with very large graphs
- **Integration** of heterogeneous and conflicting information
- **Reasoning** over incomplete and uncertain knowledge
- **Maintenance** and evolution over time

## Future Directions
The field of knowledge graphs continues to evolve, with promising research in:

1. **Neural-symbolic** approaches combining knowledge graphs with deep learning
2. **Temporal knowledge graphs** representing how knowledge changes over time
3. **Multimodal knowledge graphs** incorporating various data types beyond text
4. **Federated knowledge graphs** distributing knowledge across multiple sources
5. **Explainable AI** using knowledge graphs to provide transparent reasoning
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

## Методы построения
Создание графов знаний обычно включает несколько подходов:

1. **Ручное курирование** экспертами в предметной области
2. **Извлечение информации** из неструктурированного текста
3. **Интеграция существующих структурированных источников** данных
4. **Краудсорсинг** и совместное редактирование
5. **Машинное обучение** для автоматизированного приобретения знаний

## Проблемы
Несмотря на свои преимущества, графы знаний сталкиваются с несколькими проблемами:

- **Обеспечение качества** и проверка извлеченной информации
- **Проблемы масштабируемости** с очень большими графами
- **Интеграция** разнородной и противоречивой информации
- **Рассуждения** при неполных и неопределенных знаниях
- **Сопровождение** и эволюция с течением времени

## Будущие направления
Область графов знаний продолжает развиваться, с многообещающими исследованиями в:

1. **Нейро-символических** подходах, сочетающих графы знаний с глубоким обучением
2. **Временных графах знаний**, представляющих изменение знаний со временем
3. **Мультимодальных графах знаний**, включающих различные типы данных помимо текста
4. **Федеративных графах знаний**, распределяющих знания между несколькими источниками
5. **Объяснимом искусственном интеллекте**, использующем графы знаний для обеспечения прозрачных рассуждений
"""


def create_segments(text: str, language: str) -> Result[SegmentationResult]:
    """
    Create segments from sample text.
    
    Args:
        text: Text to segment
        language: Language code
    
    Returns:
        Result[SegmentationResult]: Segmentation result
    """
    return segment_text(text, language=language)


def create_segment_manually() -> Segment:
    """Create a simple segment for testing without segmentation."""
    section_text = """
    Knowledge graphs represent a powerful way to organize and interconnect information, 
    providing a structured representation of knowledge that combines the flexibility 
    of graphs with the expressiveness of semantics. They are essential tools in modern 
    information systems, enabling advanced data integration, reasoning, and discovery.
    """
    
    segment = Segment(
        id="test-segment-1",
        text=section_text.strip(),
        segment_type="section",
        level=1,
        title="Introduction to Knowledge Graphs"
    )
    
    return segment


def test_segment_summarization():
    """Test summarizing a single segment."""
    logger.info("Testing single segment summarization")
    
    # Create a test segment
    segment = create_segment_manually()
    
    # Create summarizer
    summarizer = TextSummarizer()
    
    # English summarization
    logger.info("Summarizing segment in English")
    result_en = summarizer.summarize_segment(segment, language="en")
    
    if result_en.success:
        logger.info("English summary generated successfully")
        print_summary(result_en.value)
    else:
        logger.error(f"Failed to generate English summary: {result_en.error}")
    
    # Russian summarization
    logger.info("Summarizing segment in Russian")
    result_ru = summarizer.summarize_segment(segment, language="ru")
    
    if result_ru.success:
        logger.info("Russian summary generated successfully")
        print_summary(result_ru.value)
    else:
        logger.error(f"Failed to generate Russian summary: {result_ru.error}")


def test_segmentation_and_summarization(language: str):
    """
    Test the complete process of segmentation and summarization.
    
    Args:
        language: Language code
    """
    logger.info(f"Testing full segmentation and summarization in {language}")
    
    # Select sample text based on language
    sample_text = SAMPLE_TEXT_EN if language == "en" else SAMPLE_TEXT_RU
    
    # Segment text
    segmentation_result = create_segments(sample_text, language)
    
    if not segmentation_result.success:
        logger.error(f"Segmentation failed: {segmentation_result.error}")
        return
    
    logger.info(f"Created {len(segmentation_result.value.segments)} segments")
    
    # Summarize all segments
    summary_result = summarize_text(segmentation_result.value, language)
    
    if not summary_result.success:
        logger.error(f"Summarization failed: {summary_result.error}")
        return
    
    # Print summaries
    logger.info(f"Generated {len(summary_result.value)} summaries")
    
    # Save summaries to output file
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"summaries_{language}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_result.value, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Summaries saved to {output_file}")
    
    # Print first summary as example
    first_summary_id = list(summary_result.value.keys())[0]
    first_summary = summary_result.value[first_summary_id]
    
    logger.info(f"Example summary for segment {first_summary_id}:")
    print_summary(first_summary)


def print_summary(summary: Dict[str, Any]):
    """
    Print a summary in a readable format.
    
    Args:
        summary: Summary data
    """
    print("\n" + "="*50)
    print(f"Summary for segment: {summary.get('segment_id', 'unknown')}")
    print(f"Type: {summary.get('segment_type', 'unknown')}")
    print(f"Role: {summary.get('role', 'unknown')}")
    print("-"*50)
    print(f"Summary: {summary.get('summary', '')}")
    print("-"*50)
    print("Key points:")
    for point in summary.get('key_points', []):
        print(f"  • {point}")
    print("-"*50)
    print("Keywords:")
    print(", ".join(summary.get('keywords', [])))
    print("="*50 + "\n")


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test text summarization functionality")
    parser.add_argument('--test', choices=['single', 'full-en', 'full-ru'], 
                        default='single', help='Test to run')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Run selected test
    if args.test == 'single':
        test_segment_summarization()
    elif args.test == 'full-en':
        test_segmentation_and_summarization("en")
    elif args.test == 'full-ru':
        test_segmentation_and_summarization("ru")


if __name__ == "__main__":
    main() 