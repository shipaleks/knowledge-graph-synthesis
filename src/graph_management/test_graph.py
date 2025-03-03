"""
Test module for the knowledge graph classes.

This module provides testing functionality for the knowledge graph classes,
including creation, manipulation, and visualization.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.relationship import Relationship, RelationshipRegistry
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_creator import (
    GraphCreator, create_graph, create_graph_from_extraction, create_graph_from_file
)

# Configure logger
logger = get_logger(__name__)


def test_create_simple_graph() -> KnowledgeGraph:
    """
    Create a simple example knowledge graph.
    
    Returns:
        KnowledgeGraph: A simple example graph
    """
    logger.info("Creating a simple example knowledge graph")
    
    # Create a graph
    graph = KnowledgeGraph(name="test_graph")
    
    # Create entities
    entity1 = Entity(
        name="Knowledge Graph", 
        entity_type="Concept",
        context="A knowledge graph represents information as nodes and edges",
        attributes={"domain": "Information Technology", "importance": "high"}
    )
    
    entity2 = Entity(
        name="Entity Extraction", 
        entity_type="Process",
        context="Entity extraction is the process of identifying entities in text",
        attributes={"step": 1, "difficulty": "medium"}
    )
    
    entity3 = Entity(
        name="Relationship Extraction", 
        entity_type="Process",
        context="Relationship extraction identifies connections between entities",
        attributes={"step": 2, "difficulty": "high"}
    )
    
    entity4 = Entity(
        name="Graph Creation", 
        entity_type="Process",
        context="Graph creation builds a graph from entities and relationships",
        attributes={"step": 3, "difficulty": "medium"}
    )
    
    entity5 = Entity(
        name="LLM", 
        entity_type="Technology",
        context="Large Language Models are used for NLP tasks",
        attributes={"type": "AI", "examples": ["GPT", "Claude"]}
    )
    
    # Add entities to graph
    graph.add_entity(entity1)
    graph.add_entity(entity2)
    graph.add_entity(entity3)
    graph.add_entity(entity4)
    graph.add_entity(entity5)
    
    # Create relationships
    rel1 = Relationship(
        source_entity=entity2.id,
        target_entity=entity1.id,
        relation_type="creates_part_of",
        context="Entity extraction creates part of a knowledge graph",
        strength=0.9,
        confidence=0.95
    )
    
    rel2 = Relationship(
        source_entity=entity3.id,
        target_entity=entity1.id,
        relation_type="creates_part_of",
        context="Relationship extraction creates part of a knowledge graph",
        strength=0.9,
        confidence=0.95
    )
    
    rel3 = Relationship(
        source_entity=entity4.id,
        target_entity=entity1.id,
        relation_type="creates",
        context="Graph creation creates a knowledge graph",
        strength=1.0,
        confidence=0.98
    )
    
    rel4 = Relationship(
        source_entity=entity2.id,
        target_entity=entity3.id,
        relation_type="precedes",
        context="Entity extraction precedes relationship extraction",
        strength=0.8,
        confidence=0.9
    )
    
    rel5 = Relationship(
        source_entity=entity3.id,
        target_entity=entity4.id,
        relation_type="precedes",
        context="Relationship extraction precedes graph creation",
        strength=0.8,
        confidence=0.9
    )
    
    rel6 = Relationship(
        source_entity=entity5.id,
        target_entity=entity2.id,
        relation_type="used_by",
        context="LLMs are used for entity extraction",
        strength=0.85,
        confidence=0.9,
        bidirectional=True
    )
    
    rel7 = Relationship(
        source_entity=entity5.id,
        target_entity=entity3.id,
        relation_type="used_by",
        context="LLMs are used for relationship extraction",
        strength=0.85,
        confidence=0.9,
        bidirectional=True
    )
    
    # Add relationships to graph
    graph.add_relationship(rel1)
    graph.add_relationship(rel2)
    graph.add_relationship(rel3)
    graph.add_relationship(rel4)
    graph.add_relationship(rel5)
    graph.add_relationship(rel6)
    graph.add_relationship(rel7)
    
    logger.info(f"Created graph with {len(graph.entity_map)} entities and {len(graph.relationship_map)} relationships")
    
    return graph


def test_graph_creation(output_dir: Union[str, Path] = "output") -> None:
    """
    Test creating and manipulating a knowledge graph.
    
    Args:
        output_dir: Directory to save output files
    """
    logger.info("Testing knowledge graph creation")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple graph
    graph = test_create_simple_graph()
    
    # Print graph statistics
    stats = graph.get_stats()
    logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")
    
    # Save graph to JSON
    json_path = output_dir / "test_graph.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_json())
    logger.info(f"Saved graph to {json_path}")
    
    # Visualize graph
    vis_path = output_dir / "test_graph.png"
    graph.visualize(output_path=vis_path)
    logger.info(f"Saved graph visualization to {vis_path}")
    
    # Test graph operations
    test_graph_operations(graph, output_dir)


def test_graph_operations(graph: KnowledgeGraph, output_dir: Union[str, Path]) -> None:
    """
    Test operations on a knowledge graph.
    
    Args:
        graph: The graph to operate on
        output_dir: Directory to save output files
    """
    logger.info("Testing knowledge graph operations")
    
    # Test getting entities by type
    concept_entities = graph.get_entities_by_type("Concept")
    process_entities = graph.get_entities_by_type("Process")
    logger.info(f"Found {len(concept_entities)} Concept entities and {len(process_entities)} Process entities")
    
    # Test getting relationships by type
    creates_relationships = graph.get_relationships_by_type("creates")
    precedes_relationships = graph.get_relationships_by_type("precedes")
    logger.info(f"Found {len(creates_relationships)} 'creates' relationships and {len(precedes_relationships)} 'precedes' relationships")
    
    # Test entity relationship retrieval
    llm_entity_id = None
    for entity in graph.entity_map.values():
        if entity.name == "LLM":
            llm_entity_id = entity.id
            break
    
    if llm_entity_id:
        llm_relationships = graph.get_entity_relationships(llm_entity_id)
        logger.info(f"LLM entity has {len(llm_relationships)} relationships")
    
    # Test merging entities
    # Create two new entities to merge
    entity1 = Entity(
        name="GPT", 
        entity_type="Technology",
        context="GPT is a large language model",
        attributes={"developer": "OpenAI", "type": "transformer"}
    )
    
    entity2 = Entity(
        name="Claude", 
        entity_type="Technology",
        context="Claude is a large language model",
        attributes={"developer": "Anthropic", "type": "transformer"}
    )
    
    # Add to graph
    graph.add_entity(entity1)
    graph.add_entity(entity2)
    
    # Create a relationship
    for entity in graph.entity_map.values():
        if entity.name == "LLM":
            llm_id = entity.id
            rel1 = Relationship(
                source_entity=entity1.id,
                target_entity=llm_id,
                relation_type="is_a",
                context="GPT is a type of LLM",
                strength=1.0,
                confidence=0.98
            )
            
            rel2 = Relationship(
                source_entity=entity2.id,
                target_entity=llm_id,
                relation_type="is_a",
                context="Claude is a type of LLM",
                strength=1.0,
                confidence=0.98
            )
            
            graph.add_relationship(rel1)
            graph.add_relationship(rel2)
            break
    
    # Save graph before merge
    json_path = Path(output_dir) / "test_graph_before_merge.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_json())
    
    vis_path = Path(output_dir) / "test_graph_before_merge.png"
    graph.visualize(output_path=vis_path)
    
    # Merge the entities
    merged_entity = Entity(
        name="Foundation Models", 
        entity_type="Technology",
        context="Foundation models are large pre-trained models like GPT and Claude",
        attributes={
            "examples": ["GPT", "Claude"],
            "type": "transformer",
            "developers": ["OpenAI", "Anthropic"]
        }
    )
    
    graph.merge_entities([entity1.id, entity2.id], merged_entity)
    logger.info(f"Merged GPT and Claude into a new entity: {merged_entity.name}")
    
    # Save graph after merge
    json_path = Path(output_dir) / "test_graph_after_merge.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(graph.to_json())
    
    vis_path = Path(output_dir) / "test_graph_after_merge.png"
    graph.visualize(output_path=vis_path)
    
    # Test loading from JSON
    logger.info("Testing loading graph from JSON")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_str = f.read()
    
    loaded_graph = KnowledgeGraph.from_json(json_str)
    logger.info(f"Loaded graph with {len(loaded_graph.entity_map)} entities and {len(loaded_graph.relationship_map)} relationships")
    
    # Verify loaded graph
    assert len(loaded_graph.entity_map) == len(graph.entity_map)
    assert len(loaded_graph.relationship_map) == len(graph.relationship_map)
    logger.info("Graph loaded successfully with matching entity and relationship counts")


def test_graph_creator_from_extraction_file(file_path: Union[str, Path], output_dir: Union[str, Path] = "output") -> None:
    """
    Test creating a graph from an extraction results file.
    
    Args:
        file_path: Path to extraction results JSON file
        output_dir: Directory to save output files
    """
    logger.info(f"Testing graph creation from extraction file: {file_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create graph from file
    graph_result = create_graph_from_file(file_path)
    
    if graph_result.success:
        graph = graph_result.value
        logger.info(f"Created graph with {len(graph.entity_map)} entities and {len(graph.relationship_map)} relationships")
        
        # Save graph to JSON
        output_name = Path(file_path).stem
        json_path = output_dir / f"{output_name}_graph.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(graph.to_json())
        logger.info(f"Saved graph to {json_path}")
        
        # Visualize graph
        vis_path = output_dir / f"{output_name}_graph.png"
        graph.visualize(output_path=vis_path)
        logger.info(f"Saved graph visualization to {vis_path}")
        
        # Print graph statistics
        stats = graph.get_stats()
        logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")
    else:
        logger.error(f"Failed to create graph: {graph_result.error}")


def main():
    """Main function for the test module."""
    parser = argparse.ArgumentParser(description="Test knowledge graph functionality")
    parser.add_argument("--test", choices=["simple", "extraction"], 
                        default="simple", help="Test to run")
    parser.add_argument("--file", type=str, help="Extraction results file for extraction test")
    parser.add_argument("--output", type=str, default="output", 
                        help="Output directory for test results")
    
    args = parser.parse_args()
    
    if args.test == "simple":
        test_graph_creation(args.output)
    elif args.test == "extraction":
        if not args.file:
            logger.error("Must provide --file for extraction test")
            return
        test_graph_creator_from_extraction_file(args.file, args.output)


if __name__ == "__main__":
    main() 