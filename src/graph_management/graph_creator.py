"""
Graph Creator Module

This module provides functionality to create knowledge graphs from extracted entities
and relationships.
"""

import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import uuid

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.timer import Timer
from src.config.app_config import AppConfig
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.relationship import Relationship, RelationshipRegistry
from src.graph_management.graph import KnowledgeGraph

# Configure logger
logger = get_logger(__name__)


class GraphCreator:
    """
    Creates knowledge graphs from extracted entities and relationships.
    
    This class provides methods to build a graph from extraction results
    and configure it with appropriate settings.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the graph creator.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or AppConfig()
    
    def create_from_registries(self, 
                              entity_registry: EntityRegistry,
                              relationship_registry: RelationshipRegistry,
                              graph_name: str = "knowledge_graph",
                              metadata: Optional[Dict[str, Any]] = None) -> Result[KnowledgeGraph]:
        """
        Create a knowledge graph from entity and relationship registries.
        
        Args:
            entity_registry: Registry containing entities
            relationship_registry: Registry containing relationships
            graph_name: Name for the graph
            metadata: Optional metadata for the graph
            
        Returns:
            Result[KnowledgeGraph]: Result containing the created graph or an error
        """
        try:
            logger.info(f"Creating knowledge graph '{graph_name}' from registries")
            logger.info(f"Entity count: {len(entity_registry.entities)}")
            logger.info(f"Relationship count: {len(relationship_registry._relationships)}")
            
            # Create a new graph
            graph = KnowledgeGraph(name=graph_name, config=self.config)
            
            # Add custom metadata if provided
            if metadata:
                graph.metadata.update(metadata)
            
            # Add all entities to the graph
            for entity_id, entity in entity_registry.entities.items():
                graph.add_entity(entity)
            
            # Add all relationships to the graph
            for rel_id, relationship in relationship_registry._relationships.items():
                graph.add_relationship(relationship)
            
            # Get graph statistics
            stats = graph.get_stats()
            logger.info(f"Graph created with {stats['entity_count']} entities and {stats['relationship_count']} relationships")
            
            return Result.ok(graph)
        except Exception as e:
            error_msg = f"Failed to create knowledge graph: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def create_from_extraction_results(self, 
                                      extraction_results: Dict[str, Any],
                                      graph_name: Optional[str] = None) -> Result[KnowledgeGraph]:
        """
        Create a knowledge graph from knowledge extraction results.
        
        Args:
            extraction_results: Results from the knowledge extraction pipeline
            graph_name: Optional name for the graph
            
        Returns:
            Result[KnowledgeGraph]: Result containing the created graph or an error
        """
        try:
            # Check if this is a graph file rather than extraction results
            if "entities" in extraction_results and isinstance(extraction_results["entities"], dict):
                # This appears to be a serialized graph
                try:
                    graph = KnowledgeGraph.from_dict(extraction_results)
                    return Result.ok(graph)
                except Exception as e:
                    error_msg = f"Failed to load graph from serialized data: {str(e)}"
                    logger.error(error_msg)
                    return Result.fail(error_msg)
            
            # Extract relevant data from results
            if "entities" not in extraction_results or "relationships" not in extraction_results:
                return Result.fail("Extraction results must contain 'entities' and 'relationships'")
            
            # Get entities and relationships
            entities_data = extraction_results["entities"]
            relationships_data = extraction_results["relationships"]
            
            # Check if entities is a list or a dict
            if isinstance(entities_data, dict):
                # Convert dict to list
                entities_data = list(entities_data.values())
            
            if isinstance(relationships_data, dict):
                # Convert dict to list
                relationships_data = list(relationships_data.values())
            
            # Create entity registry
            entity_registry = EntityRegistry()
            for entity_data in entities_data:
                # Check the structure of entity data
                if "type" in entity_data and "entity_type" not in entity_data:
                    entity_data["entity_type"] = entity_data["type"]
                
                entity = Entity(
                    id=entity_data.get("id", str(uuid.uuid4())),
                    name=entity_data["name"],
                    entity_type=entity_data["entity_type"],
                    context=entity_data.get("context"),
                    attributes=entity_data.get("attributes", {}),
                    metadata=entity_data.get("metadata", {}),
                    confidence=entity_data.get("confidence", 1.0)
                )
                entity_registry.add(entity)
            
            # Create relationship registry
            relationship_registry = RelationshipRegistry()
            for rel_data in relationships_data:
                # Check the structure of relationship data
                if "type" in rel_data and "relation_type" not in rel_data:
                    rel_data["relation_type"] = rel_data["type"]
                
                if "source" in rel_data and "source_entity" not in rel_data:
                    rel_data["source_entity"] = rel_data["source"]
                
                if "target" in rel_data and "target_entity" not in rel_data:
                    rel_data["target_entity"] = rel_data["target"]
                
                relationship = Relationship(
                    id=rel_data.get("id", str(uuid.uuid4())),
                    source_entity=rel_data["source_entity"],
                    target_entity=rel_data["target_entity"],
                    relation_type=rel_data["relation_type"],
                    context=rel_data.get("context"),
                    attributes=rel_data.get("attributes", {}),
                    metadata=rel_data.get("metadata", {}),
                    bidirectional=rel_data.get("bidirectional", False),
                    strength=rel_data.get("strength", 1.0),
                    confidence=rel_data.get("confidence", 1.0)
                )
                relationship_registry.add(relationship)
            
            # Set graph name
            if not graph_name:
                # Try to get domain type from extraction results
                domain_type = extraction_results.get("metadata", {}).get("domain_type", "domain")
                graph_name = f"{domain_type}_knowledge_graph"
            
            # Create metadata
            metadata = {
                "source": "knowledge_extraction",
                "domain_type": extraction_results.get("metadata", {}).get("domain_type"),
                "created_at": extraction_results.get("metadata", {}).get("timestamp"),
                "language": extraction_results.get("metadata", {}).get("language", "en"),
                "extraction_confidence": extraction_results.get("metadata", {}).get("confidence", 1.0)
            }
            
            # Create graph
            return self.create_from_registries(
                entity_registry=entity_registry,
                relationship_registry=relationship_registry,
                graph_name=graph_name,
                metadata=metadata
            )
        except Exception as e:
            error_msg = f"Failed to create knowledge graph from extraction results: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def create_from_json_file(self, 
                             file_path: Union[str, Path],
                             graph_name: Optional[str] = None) -> Result[KnowledgeGraph]:
        """
        Create a knowledge graph from a JSON file containing extraction results.
        
        Args:
            file_path: Path to the JSON file
            graph_name: Optional name for the graph
            
        Returns:
            Result[KnowledgeGraph]: Result containing the created graph or an error
        """
        try:
            logger.info(f"Creating knowledge graph from JSON file: {file_path}")
            
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                extraction_results = json.load(f)
            
            # Set graph name if not provided
            if not graph_name:
                # Use filename as graph name
                graph_name = Path(file_path).stem
            
            # Create graph from extraction results
            return self.create_from_extraction_results(
                extraction_results=extraction_results,
                graph_name=graph_name
            )
        except Exception as e:
            error_msg = f"Failed to create knowledge graph from JSON file: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)


def create_graph(
    entity_registry: EntityRegistry,
    relationship_registry: RelationshipRegistry,
    graph_name: str = "knowledge_graph",
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[AppConfig] = None
) -> Result[KnowledgeGraph]:
    """
    Convenience function to create a knowledge graph from registries.
    
    Args:
        entity_registry: Registry containing entities
        relationship_registry: Registry containing relationships
        graph_name: Name for the graph
        metadata: Optional metadata for the graph
        config: Optional application configuration
        
    Returns:
        Result[KnowledgeGraph]: Result containing the created graph or an error
    """
    creator = GraphCreator(config)
    return creator.create_from_registries(
        entity_registry=entity_registry,
        relationship_registry=relationship_registry,
        graph_name=graph_name,
        metadata=metadata
    )


def create_graph_from_extraction(
    extraction_results: Dict[str, Any],
    graph_name: Optional[str] = None,
    config: Optional[AppConfig] = None
) -> Result[KnowledgeGraph]:
    """
    Convenience function to create a knowledge graph from extraction results.
    
    Args:
        extraction_results: Results from the knowledge extraction pipeline
        graph_name: Optional name for the graph
        config: Optional application configuration
        
    Returns:
        Result[KnowledgeGraph]: Result containing the created graph or an error
    """
    creator = GraphCreator(config)
    return creator.create_from_extraction_results(
        extraction_results=extraction_results,
        graph_name=graph_name
    )


def create_graph_from_file(
    file_path: Union[str, Path],
    graph_name: Optional[str] = None,
    config: Optional[AppConfig] = None
) -> Result[KnowledgeGraph]:
    """
    Convenience function to create a knowledge graph from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        graph_name: Optional name for the graph
        config: Optional application configuration
        
    Returns:
        Result[KnowledgeGraph]: Result containing the created graph or an error
    """
    creator = GraphCreator(config)
    return creator.create_from_json_file(
        file_path=file_path,
        graph_name=graph_name
    ) 