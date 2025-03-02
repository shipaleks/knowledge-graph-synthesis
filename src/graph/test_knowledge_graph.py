#!/usr/bin/env python3
"""
Test script for knowledge graph functionality.

This script demonstrates how to build and visualize knowledge graphs
from text summaries.
"""

import os
import json
import sys
import uuid
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.result import Result
from src.utils.logger import setup_logging, get_logger
from src.config.app_config import AppConfig
from src.text_processing.segment import Segment, SegmentationResult
from src.text_processing.text_segmenter import segment_text
from src.text_processing.text_summarizer import summarize_text
from src.graph.knowledge_graph import (
    KnowledgeGraph, Entity, Relationship, GraphBuilder, build_knowledge_graph
)

# Set up logging
logger = get_logger(__name__)

# Paths for input/output
OUTPUT_DIR = Path("./output")


def load_summaries(language: str) -> Dict[str, Dict[str, Any]]:
    """
    Load summaries from a file.
    
    Args:
        language: Language code
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of summaries
    """
    filename = f"summaries_{language}.json"
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        logger.error(f"Summaries file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
        
        logger.info(f"Loaded {len(summaries)} summaries from {file_path}")
        return summaries
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load summaries: {str(e)}")
        return {}


def load_segmentation(language: str) -> Optional[SegmentationResult]:
    """
    Load segmentation from a file.
    
    Args:
        language: Language code
        
    Returns:
        Optional[SegmentationResult]: Segmentation result or None
    """
    filename = "segments.json"
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        logger.warning(f"Segmentation file not found: {file_path}")
        return None
    
    try:
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create segments from data
        segments = []
        
        def process_segment_data(segment_data, parent_id=None):
            # Create segment
            segment = Segment(
                id=segment_data.get("id", str(uuid.uuid4())),
                text=segment_data["text"],
                segment_type=segment_data["segment_type"],
                level=segment_data.get("level", 1),
                title=segment_data.get("title"),
                parent_id=parent_id
            )
            
            segments.append(segment)
            
            # Process children
            for child_data in segment_data.get("children", []):
                child = process_segment_data(child_data, segment.id)
                segment.add_child(child.id)
            
            return segment
        
        # Process all segments
        for segment_data in data.get("segments", []):
            process_segment_data(segment_data)
        
        # Create segmentation result
        result = SegmentationResult(segments)
        
        logger.info(f"Loaded segmentation with {len(segments)} segments")
        return result
        
    except (json.JSONDecodeError, IOError, KeyError) as e:
        logger.error(f"Failed to load segmentation: {str(e)}")
        return None


def test_manual_graph_creation():
    """Test creating a knowledge graph manually."""
    logger.info("Testing manual graph creation")
    
    # Create a knowledge graph
    graph = KnowledgeGraph(name="Test Knowledge Graph")
    
    # Add entities
    entity1 = Entity(
        entity_id="e1",
        name="Knowledge Graph",
        entity_type="Concept",
        attributes={"definition": "A structured representation of knowledge"},
        salience=0.9
    )
    
    entity2 = Entity(
        entity_id="e2",
        name="Graph Database",
        entity_type="Tool",
        attributes={"example": "Neo4j"},
        salience=0.7
    )
    
    entity3 = Entity(
        entity_id="e3",
        name="Data Integration",
        entity_type="Process",
        salience=0.6
    )
    
    graph.add_entity(entity1)
    graph.add_entity(entity2)
    graph.add_entity(entity3)
    
    # Add relationships
    rel1 = Relationship(
        source_id="e2",
        target_id="e1",
        relationship_type="implements",
        weight=0.8
    )
    
    rel2 = Relationship(
        source_id="e1",
        target_id="e3",
        relationship_type="enables",
        weight=0.7
    )
    
    graph.add_relationship(rel1)
    graph.add_relationship(rel2)
    
    # Export to various formats
    logger.info(f"Created graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships")
    
    # Save JSON
    try:
        json_path = OUTPUT_DIR / "manual_graph.json"
        graph.save(json_path)
        logger.info(f"Graph saved to JSON at {json_path}")
    except Exception as e:
        logger.error(f"Failed to save graph to JSON: {str(e)}")
    
    # Export GraphML
    try:
        graphml_path = OUTPUT_DIR / "manual_graph.graphml"
        graph.export_to_graphml(graphml_path)
        logger.info(f"Graph exported to GraphML at {graphml_path}")
    except Exception as e:
        logger.error(f"Failed to export graph to GraphML: {str(e)}")
    
    # Export Cypher
    try:
        cypher_path = OUTPUT_DIR / "manual_graph.cypher"
        graph.export_to_cypher(cypher_path)
        logger.info(f"Graph exported to Cypher at {cypher_path}")
    except Exception as e:
        logger.error(f"Failed to export graph to Cypher: {str(e)}")
    
    # Visualize
    try:
        viz_path = OUTPUT_DIR / "manual_graph.png"
        graph.visualize(output_path=viz_path)
        logger.info(f"Graph visualization saved to {viz_path}")
    except Exception as e:
        logger.error(f"Failed to visualize graph: {str(e)}")


def test_graph_from_summaries(language: str):
    """
    Test building a knowledge graph from summaries.
    
    Args:
        language: Language code
    """
    logger.info(f"Testing knowledge graph creation from summaries ({language})")
    
    # Load summaries
    summaries = load_summaries(language)
    if not summaries:
        logger.error("No summaries found, cannot build graph")
        return
    
    # Load segmentation (optional)
    segmentation = load_segmentation(language)
    
    # Build graph
    result = build_knowledge_graph(summaries, segmentation, language)
    
    if not result.success:
        logger.error(f"Failed to build graph: {result.error}")
        return
    
    graph = result.value
    
    logger.info(f"Created graph with {len(graph.entities)} entities and {len(graph.relationships)} relationships")
    
    # Save JSON
    try:
        json_path = OUTPUT_DIR / f"knowledge_graph_{language}.json"
        graph.save(json_path)
        logger.info(f"Graph saved to JSON at {json_path}")
    except Exception as e:
        logger.error(f"Failed to save graph to JSON: {str(e)}")
    
    # Export GraphML
    try:
        graphml_path = OUTPUT_DIR / f"knowledge_graph_{language}.graphml"
        graph.export_to_graphml(graphml_path)
        logger.info(f"Graph exported to GraphML at {graphml_path}")
    except Exception as e:
        logger.error(f"Failed to export graph to GraphML: {str(e)}")
    
    # Export Cypher
    try:
        cypher_path = OUTPUT_DIR / f"knowledge_graph_{language}.cypher"
        graph.export_to_cypher(cypher_path)
        logger.info(f"Graph exported to Cypher at {cypher_path}")
    except Exception as e:
        logger.error(f"Failed to export graph to Cypher: {str(e)}")
    
    # Visualize
    try:
        viz_path = OUTPUT_DIR / f"knowledge_graph_{language}.png"
        graph.visualize(output_path=viz_path)
        logger.info(f"Graph visualization saved to {viz_path}")
    except Exception as e:
        logger.error(f"Failed to visualize graph: {str(e)}")


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Test knowledge graph functionality")
    parser.add_argument('--test', choices=['manual', 'from-summaries-en', 'from-summaries-ru'], 
                        default='manual', help='Test to run')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                        default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level=args.log_level)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run selected test
    if args.test == 'manual':
        test_manual_graph_creation()
    elif args.test == 'from-summaries-en':
        test_graph_from_summaries("en")
    elif args.test == 'from-summaries-ru':
        test_graph_from_summaries("ru")


if __name__ == "__main__":
    main() 