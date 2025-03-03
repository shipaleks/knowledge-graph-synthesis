"""
Test module for the relationship extractor

This module provides test functions for the relationship extraction functionality.
It can be used to test extracting relationships from text and segments.
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path so we can import the modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config.app_config import AppConfig
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.entity_extractor import extract_entities
from src.knowledge.relationship import Relationship, RelationshipRegistry
from src.knowledge.relationship_extractor import extract_relationships
from src.text_processing.text_loader import load_text
from src.text_processing.text_segmenter import segment_text


# Sample texts for testing
SAMPLE_TEXT_EN = """
Knowledge graphs represent structured information about entities and their relationships. The process of building a knowledge graph involves several key steps:

First, entity extraction identifies important concepts, objects, or individuals mentioned in the text. For example, in AI literature, entities might include "artificial intelligence," "machine learning," and "neural networks."

Second, relationship extraction determines how these entities are connected. These connections might be hierarchical (e.g., "neural networks" are a "subset of" machine learning techniques) or causal (e.g., "overfitting" causes "poor generalization").

Third, entity resolution ensures that different mentions of the same entity are properly linked, even when they appear in different forms.

Fourth, knowledge integration combines the extracted information with existing knowledge bases to create a comprehensive and consistent graph.

Advanced techniques like BERT and GPT can significantly improve these steps by leveraging deep learning for more accurate extraction and linking.
"""

SAMPLE_TEXT_RU = """
Графы знаний представляют структурированную информацию о сущностях и их отношениях. Процесс построения графа знаний включает несколько ключевых этапов:

Во-первых, извлечение сущностей идентифицирует важные концепции, объекты или индивидуумы, упомянутые в тексте. Например, в литературе по ИИ сущностями могут быть "искусственный интеллект", "машинное обучение" и "нейронные сети".

Во-вторых, извлечение отношений определяет, как эти сущности связаны между собой. Эти связи могут быть иерархическими (например, "нейронные сети" являются "подмножеством" методов машинного обучения) или причинно-следственными (например, "переобучение" вызывает "плохую обобщающую способность").

В-третьих, разрешение сущностей гарантирует, что различные упоминания одной и той же сущности правильно связаны, даже когда они появляются в разных формах.

В-четвертых, интеграция знаний объединяет извлеченную информацию с существующими базами знаний для создания всеобъемлющего и согласованного графа.

Продвинутые методы, такие как BERT и GPT, могут значительно улучшить эти шаги, используя глубокое обучение для более точного извлечения и связывания.
"""


def test_relationship_extraction_from_text(language: str, domain_type: str = "Knowledge Graph Construction"):
    """
    Test relationship extraction from text.
    
    Args:
        language: Language of the text to test
        domain_type: Domain type for extraction context
    """
    # Load text based on language
    if language == "en":
        text = SAMPLE_TEXT_EN
    elif language == "ru":
        text = SAMPLE_TEXT_RU
    else:
        logger.error(f"Unsupported language: {language}")
        return
    
    logger.info(f"Testing relationship extraction from text in {language}")
    
    # First extract entities from the text
    entity_result = extract_entities(text, language=language, domain_type=domain_type)
    
    if not entity_result.success:
        logger.error(f"Failed to extract entities: {entity_result.error}")
        return
    
    entities = entity_result.value
    logger.info(f"Extracted {len(entities)} entities")
    
    # Print extracted entities
    logger.info("Extracted entities:")
    for i, entity in enumerate(entities):
        logger.info(f"{i+1}. {entity.name} ({entity.entity_type}, Confidence: {entity.confidence:.2f})")
    
    # Now extract relationships between entities
    relation_result = extract_relationships(text, entities, language=language, domain_type=domain_type)
    
    if not relation_result.success:
        logger.error(f"Failed to extract relationships: {relation_result.error}")
        return
    
    relationships = relation_result.value
    logger.info(f"Extracted {len(relationships)} relationships")
    
    # Print extracted relationships
    logger.info("Extracted relationships:")
    
    # Create entity ID to name mapping
    entity_map = {entity.id: entity.name for entity in entities}
    
    # Print relationships
    for i, rel in enumerate(relationships):
        source = entity_map.get(rel.source_entity, "Unknown")
        target = entity_map.get(rel.target_entity, "Unknown")
        logger.info(f"{i+1}. {source} --[{rel.relation_type}]--> {target} (Confidence: {rel.confidence:.2f})")
        if rel.context:
            logger.info(f"   Context: '{rel.context}'")
    
    # Save results to JSON file
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"relationships_{language}.json")
    
    # Create a result object with both entities and relationships
    result = {
        "entities": [entity.to_dict() for entity in entities],
        "relationships": []
    }
    
    # Add relationships with readable names
    for rel in relationships:
        rel_dict = rel.to_dict()
        rel_dict["source_name"] = entity_map.get(rel.source_entity, "Unknown")
        rel_dict["target_name"] = entity_map.get(rel.target_entity, "Unknown")
        result["relationships"].append(rel_dict)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filename}")


def test_relationship_extraction_from_segments(language: str, domain_type: str = "Knowledge Graph Construction"):
    """
    Test relationship extraction from segmented text.
    
    Args:
        language: Language of the text to test
        domain_type: Domain type for extraction context
    """
    # Load text based on language
    if language == "en":
        text = SAMPLE_TEXT_EN
    elif language == "ru":
        text = SAMPLE_TEXT_RU
    else:
        logger.error(f"Unsupported language: {language}")
        return
    
    logger.info(f"Testing relationship extraction from segments in {language}")
    
    # Segment the text
    segmentation_result = segment_text(text, language=language)
    
    if not segmentation_result.success:
        logger.error(f"Failed to segment text: {segmentation_result.error}")
        return
    
    # Get all segments
    segments = segmentation_result.value.segments
    logger.info(f"Text segmented into {len(segments)} segments")
    
    # Extract entities from the segmented text
    entity_result = extract_entities(segmentation_result.value, language=language, domain_type=domain_type)
    
    if not entity_result.success:
        logger.error(f"Failed to extract entities: {entity_result.error}")
        return
    
    entity_registry = entity_result.value
    entities = entity_registry.all()
    logger.info(f"Extracted {len(entities)} unique entities from segments")
    
    # Print extracted entities
    logger.info("Extracted entities:")
    for i, entity in enumerate(entities):
        logger.info(f"{i+1}. {entity.name} ({entity.entity_type}, Confidence: {entity.confidence:.2f})")
    
    # Extract relationships from segments
    relation_result = extract_relationships(
        segmentation_result.value, 
        entity_registry, 
        language=language, 
        domain_type=domain_type
    )
    
    if not relation_result.success:
        logger.error(f"Failed to extract relationships: {relation_result.error}")
        return
    
    relationship_registry = relation_result.value
    relationships = relationship_registry.all()
    logger.info(f"Extracted {len(relationships)} unique relationships from segments")
    
    # Print relationship statistics
    relation_types = {}
    for rel in relationships:
        if rel.relation_type not in relation_types:
            relation_types[rel.relation_type] = 0
        relation_types[rel.relation_type] += 1
    
    logger.info("Relationship types:")
    for rel_type, count in relation_types.items():
        logger.info(f"- {rel_type}: {count}")
    
    # Create entity ID to name mapping
    entity_map = {entity.id: entity.name for entity in entities}
    
    # Print top 5 relationships by confidence
    logger.info("Top 5 relationships by confidence:")
    sorted_relationships = sorted(relationships, key=lambda r: r.confidence, reverse=True)
    for i, rel in enumerate(sorted_relationships[:5]):
        source = entity_map.get(rel.source_entity, "Unknown")
        target = entity_map.get(rel.target_entity, "Unknown")
        logger.info(f"{i+1}. {source} --[{rel.relation_type}]--> {target} (Confidence: {rel.confidence:.2f})")
        if rel.context:
            logger.info(f"   Context: '{rel.context}'")
    
    # Save results to JSON file
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"relationships_segments_{language}.json")
    
    # Create a result object with both entities and relationships
    result = {
        "entities": [entity.to_dict() for entity in entities],
        "relationships": []
    }
    
    # Add relationships with readable names
    for rel in relationships:
        rel_dict = rel.to_dict()
        rel_dict["source_name"] = entity_map.get(rel.source_entity, "Unknown")
        rel_dict["target_name"] = entity_map.get(rel.target_entity, "Unknown")
        result["relationships"].append(rel_dict)
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filename}")


def print_relationship(relationship: Relationship, entity_map: Dict[str, str]):
    """
    Print details of a relationship.
    
    Args:
        relationship: Relationship to print
        entity_map: Mapping from entity ID to entity name
    """
    source = entity_map.get(relationship.source_entity, "Unknown")
    target = entity_map.get(relationship.target_entity, "Unknown")
    
    print(f"Relationship: {source} --[{relationship.relation_type}]--> {target}")
    print(f"  Confidence: {relationship.confidence:.2f}")
    print(f"  Strength: {relationship.strength:.2f}")
    print(f"  Bidirectional: {relationship.bidirectional}")
    if relationship.context:
        print(f"  Context: '{relationship.context}'")
    if relationship.attributes:
        print(f"  Attributes: {relationship.attributes}")
    print()


def test_relationship_extraction_with_custom_types(language: str):
    """
    Test relationship extraction with custom relationship types.
    
    Args:
        language: Language of the text to test
    """
    # Load text based on language
    if language == "en":
        text = SAMPLE_TEXT_EN
    elif language == "ru":
        text = SAMPLE_TEXT_RU
    else:
        logger.error(f"Unsupported language: {language}")
        return
    
    logger.info(f"Testing relationship extraction with custom types in {language}")
    
    # Define custom relation types
    custom_relation_types = [
        "is_part_of", "contains", "depends_on", "enables", 
        "causes", "is_used_for", "is_type_of", "succeeds",
        "improves", "contradicts", "complements", "implements"
    ]
    
    logger.info(f"Using custom relation types: {', '.join(custom_relation_types)}")
    
    # First extract entities from the text
    entity_result = extract_entities(text, language=language)
    
    if not entity_result.success:
        logger.error(f"Failed to extract entities: {entity_result.error}")
        return
    
    entities = entity_result.value
    logger.info(f"Extracted {len(entities)} entities")
    
    # Extract relationships with custom types
    relation_result = extract_relationships(
        text, 
        entities, 
        language=language,
        relation_types=custom_relation_types
    )
    
    if not relation_result.success:
        logger.error(f"Failed to extract relationships: {relation_result.error}")
        return
    
    relationships = relation_result.value
    logger.info(f"Extracted {len(relationships)} relationships using custom types")
    
    # Create entity ID to name mapping
    entity_map = {entity.id: entity.name for entity in entities}
    
    # Print relationships
    for i, rel in enumerate(relationships):
        source = entity_map.get(rel.source_entity, "Unknown")
        target = entity_map.get(rel.target_entity, "Unknown")
        logger.info(f"{i+1}. {source} --[{rel.relation_type}]--> {target} (Confidence: {rel.confidence:.2f})")
    
    # Count relation types used
    relation_type_counts = {}
    for rel in relationships:
        if rel.relation_type not in relation_type_counts:
            relation_type_counts[rel.relation_type] = 0
        relation_type_counts[rel.relation_type] += 1
    
    logger.info("Relation types used:")
    for rel_type, count in relation_type_counts.items():
        logger.info(f"- {rel_type}: {count}")


def main():
    """Main function for testing the relationship extractor."""
    parser = argparse.ArgumentParser(description="Test the relationship extractor")
    parser.add_argument("--test", choices=["text", "segments", "custom"], 
                        default="text", help="Type of test to run")
    parser.add_argument("--language", choices=["en", "ru"], default="en", 
                        help="Language to use for testing")
    
    args = parser.parse_args()
    
    if args.test == "text":
        test_relationship_extraction_from_text(language=args.language)
    elif args.test == "segments":
        test_relationship_extraction_from_segments(language=args.language)
    elif args.test == "custom":
        test_relationship_extraction_with_custom_types(language=args.language)
    else:
        logger.error(f"Unknown test type: {args.test}")


if __name__ == "__main__":
    main() 