"""
Test module for the coreference resolution

This module provides test functions for the coreference resolution functionality.
It can be used to test resolving coreferences between entities.
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
from src.knowledge.entity import Entity, EntityRegistry, create_entity
from src.knowledge.coreference_resolver import (
    CoreferenceResolver, 
    resolve_coreferences, 
    CoreferenceGroup
)


def create_test_entities() -> EntityRegistry:
    """
    Create a set of test entities with some coreference examples.
    
    Returns:
        EntityRegistry with test entities
    """
    registry = EntityRegistry()
    
    # Group 1: Different names for the same concept (Knowledge Graph)
    registry.add(create_entity(
        name="Knowledge Graph",
        entity_type="Concept",
        context="Knowledge graphs represent structured information about entities and their relationships.",
        attributes={"area": "Information Technology", "importance": "high"},
        confidence=1.0
    ))
    
    registry.add(create_entity(
        name="KG",
        entity_type="Concept",
        context="KGs are used to represent relationships between entities in a structured way.",
        attributes={"use_case": "Data Integration"},
        confidence=0.85
    ))
    
    registry.add(create_entity(
        name="knowledge graph",
        entity_type="Concept",
        context="A knowledge graph can be used to power recommendation systems.",
        attributes={"application": "Recommendation Systems"},
        confidence=0.9
    ))
    
    # Group 2: Different but related concepts (Neural Networks vs. Deep Learning)
    registry.add(create_entity(
        name="Neural Networks",
        entity_type="Technology",
        context="Neural networks are computational models inspired by the human brain.",
        attributes={"area": "Machine Learning"},
        confidence=0.95
    ))
    
    registry.add(create_entity(
        name="Deep Learning",
        entity_type="Technology",
        context="Deep learning uses neural networks with multiple layers.",
        attributes={"area": "Artificial Intelligence"},
        confidence=0.9
    ))
    
    registry.add(create_entity(
        name="neural net",
        entity_type="Technology",
        context="A neural net can be trained to recognize patterns in data.",
        attributes={"use_case": "Pattern Recognition"},
        confidence=0.8
    ))
    
    # Group 3: Slight name variations (Entity Extraction)
    registry.add(create_entity(
        name="Entity Extraction",
        entity_type="Process",
        context="Entity extraction identifies important concepts mentioned in text.",
        attributes={"step": 1, "importance": "high"},
        confidence=0.95
    ))
    
    registry.add(create_entity(
        name="entity extraction",
        entity_type="Process",
        context="entity extraction is a key part of building a knowledge graph.",
        attributes={"input": "text", "output": "entities"},
        confidence=0.9
    ))
    
    # Group 4: Ambiguous entities (different meanings for same term)
    registry.add(create_entity(
        name="Python",
        entity_type="Programming Language",
        context="Python is a popular programming language for data science.",
        attributes={"type": "interpreted", "paradigm": "multi-paradigm"},
        confidence=0.9
    ))
    
    registry.add(create_entity(
        name="Python",
        entity_type="Animal",
        context="The python is a type of large snake.",
        attributes={"class": "reptile", "diet": "carnivore"},
        confidence=0.85
    ))
    
    # Group 5: Entities that should not be merged
    registry.add(create_entity(
        name="BERT",
        entity_type="Technology",
        context="BERT is a transformer-based language model developed by Google.",
        attributes={"developer": "Google", "type": "language model"},
        confidence=0.95
    ))
    
    registry.add(create_entity(
        name="GPT",
        entity_type="Technology",
        context="GPT is a language model developed by OpenAI.",
        attributes={"developer": "OpenAI", "type": "language model"},
        confidence=0.95
    ))
    
    return registry


def test_coreference_resolution(language: str = "en"):
    """
    Test coreference resolution on test entities.
    
    Args:
        language: Language to use for testing
    """
    # Create test entities
    entity_registry = create_test_entities()
    
    logger.info(f"Testing coreference resolution in {language}")
    logger.info(f"Created {entity_registry.count()} test entities")
    
    # Print original entities
    logger.info("Original entities:")
    for i, entity in enumerate(entity_registry.all()):
        logger.info(f"{i+1}. {entity.name} ({entity.entity_type}, Confidence: {entity.confidence:.2f})")
    
    # Resolve coreferences
    result = resolve_coreferences(
        entity_registry, 
        language=language,
        similarity_threshold=0.6
    )
    
    if not result.success:
        logger.error(f"Failed to resolve coreferences: {result.error}")
        return
    
    resolved_registry = result.value
    
    # Print resolved entities
    logger.info(f"\nResolved entities: {resolved_registry.count()} (from {entity_registry.count()} original)")
    for i, entity in enumerate(resolved_registry.all()):
        logger.info(f"{i+1}. {entity.name} ({entity.entity_type}, Confidence: {entity.confidence:.2f})")
        
        # Print coreference info if available
        if "coreference_resolution" in entity.metadata:
            coref_info = entity.metadata["coreference_resolution"]
            original_names = coref_info.get("original_names", [])
            if len(original_names) > 1:
                logger.info(f"   Merged from: {', '.join(original_names)}")
                logger.info(f"   Confidence: {coref_info.get('resolution_confidence', 0.0):.2f}")
    
    # Save results to JSON file
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"coreference_{language}.json")
    
    # Create result object with original and resolved entities
    result_data = {
        "original_entities": [entity.to_dict() for entity in entity_registry.all()],
        "resolved_entities": [entity.to_dict() for entity in resolved_registry.all()],
        "resolution_summary": {
            "original_count": entity_registry.count(),
            "resolved_count": resolved_registry.count(),
            "merged_count": entity_registry.count() - resolved_registry.count()
        }
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filename}")


def test_custom_entity_group(language: str = "en"):
    """
    Test coreference resolution on a custom group of entities.
    
    Args:
        language: Language to use for testing
    """
    # Create a custom group of entities
    entities = [
        create_entity(
            name="Artificial Intelligence",
            entity_type="Concept",
            context="Artificial intelligence is a field of computer science focused on creating systems that can perform tasks requiring human intelligence.",
            attributes={"area": "Computer Science"},
            confidence=0.95
        ),
        create_entity(
            name="AI",
            entity_type="Concept",
            context="AI systems can learn from experience and adapt to new inputs.",
            attributes={"type": "Technology"},
            confidence=0.9
        ),
        create_entity(
            name="Machine Intelligence",
            entity_type="Concept",
            context="Machine intelligence refers to the ability of machines to mimic human cognitive functions.",
            attributes={"related_to": "Cognitive Science"},
            confidence=0.85
        ),
        create_entity(
            name="Computer Vision",
            entity_type="Field",
            context="Computer vision is a field of AI that trains computers to interpret visual data.",
            attributes={"application": "Image Processing"},
            confidence=0.9
        )
    ]
    
    logger.info(f"Testing coreference resolution on custom group in {language}")
    logger.info(f"Created {len(entities)} test entities")
    
    # Print original entities
    logger.info("Original entities:")
    for i, entity in enumerate(entities):
        logger.info(f"{i+1}. {entity.name} ({entity.entity_type}, Confidence: {entity.confidence:.2f})")
    
    # Create resolver
    resolver = CoreferenceResolver()
    
    # Resolve coreferences in group
    result = resolver.resolve_entity_group(entities, language)
    
    if not result.success:
        logger.error(f"Failed to resolve coreferences: {result.error}")
        return
    
    coreference_groups = result.value
    
    # Print coreference groups
    logger.info(f"\nIdentified {len(coreference_groups)} coreference groups:")
    for i, group in enumerate(coreference_groups):
        logger.info(f"Group {i+1}:")
        logger.info(f"  Canonical name: {group.canonical_name}")
        logger.info(f"  Merge decision: {group.merge_decision}")
        logger.info(f"  Reason: {group.reason}")
        logger.info(f"  Confidence: {group.confidence:.2f}")
        
        # Find original entities
        group_entities = [entity for entity in entities if entity.id in group.entity_ids]
        if group_entities:
            logger.info("  Entities in group:")
            for j, entity in enumerate(group_entities):
                logger.info(f"    {j+1}. {entity.name} ({entity.entity_type})")
    
    # Save results to JSON file
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, f"coreference_custom_{language}.json")
    
    # Create result object with entities and groups
    result_data = {
        "entities": [entity.to_dict() for entity in entities],
        "coreference_groups": [
            {
                "entity_ids": group.entity_ids,
                "canonical_name": group.canonical_name,
                "merge_decision": group.merge_decision,
                "reason": group.reason,
                "confidence": group.confidence,
                "combined_attributes": group.combined_attributes
            }
            for group in coreference_groups
        ]
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {filename}")


def main():
    """Main function for testing the coreference resolver."""
    parser = argparse.ArgumentParser(description="Test the coreference resolver")
    parser.add_argument("--test", choices=["full", "custom"], 
                        default="full", help="Type of test to run")
    parser.add_argument("--language", choices=["en", "ru"], default="en", 
                        help="Language to use for testing")
    
    args = parser.parse_args()
    
    if args.test == "full":
        test_coreference_resolution(language=args.language)
    elif args.test == "custom":
        test_custom_entity_group(language=args.language)
    else:
        logger.error(f"Unknown test type: {args.test}")


if __name__ == "__main__":
    main() 