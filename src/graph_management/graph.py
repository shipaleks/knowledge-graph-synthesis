"""
Knowledge Graph Module

This module provides the KnowledgeGraph class and related utilities for working with
knowledge graphs built from entities and relationships.
"""

import json
import uuid
import os
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.relationship import Relationship, RelationshipRegistry
from src.utils.result import Result
from src.utils.logger import get_logger
from src.config.app_config import AppConfig

# Configure logger
logger = get_logger(__name__)


class KnowledgeGraph:
    """
    Represents a knowledge graph built from entities and relationships.
    
    A knowledge graph is a structured representation of knowledge where entities
    are nodes and relationships are edges between nodes.
    """
    
    def __init__(self, name: str = "knowledge_graph", config: Optional[AppConfig] = None):
        """
        Initialize a new knowledge graph.
        
        Args:
            name: Name of the knowledge graph
            config: Optional application configuration
        """
        self.name = name
        self.config = config or AppConfig()
        self.graph = nx.DiGraph(name=name)
        self.entity_map: Dict[str, Entity] = {}
        self.relationship_map: Dict[str, Relationship] = {}
        self.metadata: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "created_at": None,
            "updated_at": None,
            "version": "1.0",
            "entity_count": 0,
            "relationship_count": 0,
            "domain_type": None
        }
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity node to the graph.
        
        Args:
            entity: The entity to add
        """
        if entity.id in self.entity_map:
            logger.warning(f"Entity {entity.id} already exists in the graph. Updating it.")
        
        # Add node to networkx graph
        self.graph.add_node(
            entity.id,
            label=entity.name,
            entity_type=entity.entity_type,
            attributes=entity.attributes,
            metadata=entity.metadata,
            confidence=entity.confidence,
            node_type="entity"
        )
        
        # Store entity in map
        self.entity_map[entity.id] = entity
        self.metadata["entity_count"] = len(self.entity_map)
        logger.debug(f"Added entity {entity.name} ({entity.id}) to the graph")
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship edge to the graph.
        
        Args:
            relationship: The relationship to add
        """
        # Check if source and target entities exist
        if relationship.source_entity not in self.entity_map:
            logger.warning(f"Source entity {relationship.source_entity} does not exist in the graph")
            return
        
        if relationship.target_entity not in self.entity_map:
            logger.warning(f"Target entity {relationship.target_entity} does not exist in the graph")
            return
        
        # Add edge to networkx graph
        self.graph.add_edge(
            relationship.source_entity,
            relationship.target_entity,
            id=relationship.id,
            relation_type=relationship.relation_type,
            attributes=relationship.attributes,
            metadata=relationship.metadata,
            bidirectional=relationship.bidirectional,
            strength=relationship.strength,
            confidence=relationship.confidence,
            context=relationship.context,
            edge_type="relationship"
        )
        
        # If bidirectional, add reverse edge as well
        if relationship.bidirectional:
            self.graph.add_edge(
                relationship.target_entity,
                relationship.source_entity,
                id=f"{relationship.id}_reverse",
                relation_type=relationship.relation_type,
                attributes=relationship.attributes,
                metadata=relationship.metadata,
                bidirectional=True,
                strength=relationship.strength,
                confidence=relationship.confidence,
                context=relationship.context,
                edge_type="relationship"
            )
        
        # Store relationship in map
        self.relationship_map[relationship.id] = relationship
        self.metadata["relationship_count"] = len(self.relationship_map)
        logger.debug(f"Added relationship {relationship.relation_type} from {relationship.source_entity} to {relationship.target_entity}")
    
    def remove_entity(self, entity_id: str) -> None:
        """
        Remove an entity and its relationships from the graph.
        
        Args:
            entity_id: ID of the entity to remove
        """
        if entity_id not in self.entity_map:
            logger.warning(f"Entity {entity_id} does not exist in the graph")
            return
        
        # Remove from networkx graph
        self.graph.remove_node(entity_id)
        
        # Remove from entity map
        del self.entity_map[entity_id]
        
        # Remove any relationships involving this entity
        relationships_to_remove = []
        for rel_id, relationship in self.relationship_map.items():
            if relationship.source_entity == entity_id or relationship.target_entity == entity_id:
                relationships_to_remove.append(rel_id)
        
        for rel_id in relationships_to_remove:
            del self.relationship_map[rel_id]
        
        self.metadata["entity_count"] = len(self.entity_map)
        self.metadata["relationship_count"] = len(self.relationship_map)
        logger.debug(f"Removed entity {entity_id} and {len(relationships_to_remove)} related relationships from the graph")
    
    def remove_relationship(self, relationship_id: str) -> None:
        """
        Remove a relationship from the graph.
        
        Args:
            relationship_id: ID of the relationship to remove
        """
        if relationship_id not in self.relationship_map:
            logger.warning(f"Relationship {relationship_id} does not exist in the graph")
            return
        
        relationship = self.relationship_map[relationship_id]
        
        # Remove from networkx graph
        self.graph.remove_edge(relationship.source_entity, relationship.target_entity)
        
        # If bidirectional, remove reverse edge too
        if relationship.bidirectional:
            try:
                self.graph.remove_edge(relationship.target_entity, relationship.source_entity)
            except nx.NetworkXError:
                pass  # Edge might not exist
        
        # Remove from relationship map
        del self.relationship_map[relationship_id]
        self.metadata["relationship_count"] = len(self.relationship_map)
        logger.debug(f"Removed relationship {relationship_id} from the graph")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            Entity: The entity if found, None otherwise
        """
        return self.entity_map.get(entity_id)
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """
        Get a relationship by ID.
        
        Args:
            relationship_id: ID of the relationship to retrieve
            
        Returns:
            Relationship: The relationship if found, None otherwise
        """
        return self.relationship_map.get(relationship_id)
    
    def get_relationships_between(self, source_id: str, target_id: str) -> List[Relationship]:
        """
        Get all relationships between two entities.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            
        Returns:
            List[Relationship]: List of relationships between the entities
        """
        if not self.graph.has_edge(source_id, target_id):
            return []
        
        relationships = []
        for rel_id, rel in self.relationship_map.items():
            if rel.source_entity == source_id and rel.target_entity == target_id:
                relationships.append(rel)
        
        return relationships
    
    def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """
        Get all relationships involving an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            List[Relationship]: List of relationships involving the entity
        """
        if entity_id not in self.entity_map:
            return []
        
        relationships = []
        for rel_id, rel in self.relationship_map.items():
            if rel.source_entity == entity_id or rel.target_entity == entity_id:
                relationships.append(rel)
        
        return relationships
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entities to retrieve
            
        Returns:
            List[Entity]: List of entities of the specified type
        """
        return [entity for entity in self.entity_map.values() if entity.entity_type == entity_type]
    
    def get_relationships_by_type(self, relation_type: str) -> List[Relationship]:
        """
        Get all relationships of a specific type.
        
        Args:
            relation_type: Type of relationships to retrieve
            
        Returns:
            List[Relationship]: List of relationships of the specified type
        """
        return [rel for rel in self.relationship_map.values() if rel.relation_type == relation_type]
    
    def merge_entities(self, entity_ids: List[str], new_entity: Entity) -> None:
        """
        Merge multiple entities into a new entity.
        
        Args:
            entity_ids: List of entity IDs to merge
            new_entity: The new entity to replace them with
        """
        # Add the new entity
        self.add_entity(new_entity)
        
        # Redirect relationships
        for entity_id in entity_ids:
            if entity_id not in self.entity_map:
                continue
                
            # Get all relationships involving this entity
            relationships = self.get_entity_relationships(entity_id)
            
            # Create new relationships with the new entity
            for rel in relationships:
                if rel.source_entity == entity_id:
                    new_rel = Relationship(
                        source_entity=new_entity.id,
                        target_entity=rel.target_entity,
                        relation_type=rel.relation_type,
                        context=rel.context,
                        attributes=rel.attributes.copy(),
                        metadata=rel.metadata.copy(),
                        bidirectional=rel.bidirectional,
                        strength=rel.strength,
                        confidence=rel.confidence
                    )
                    self.add_relationship(new_rel)
                
                elif rel.target_entity == entity_id:
                    new_rel = Relationship(
                        source_entity=rel.source_entity,
                        target_entity=new_entity.id,
                        relation_type=rel.relation_type,
                        context=rel.context,
                        attributes=rel.attributes.copy(),
                        metadata=rel.metadata.copy(),
                        bidirectional=rel.bidirectional,
                        strength=rel.strength,
                        confidence=rel.confidence
                    )
                    self.add_relationship(new_rel)
            
            # Remove the old entity
            self.remove_entity(entity_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the knowledge graph to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the knowledge graph
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entity_map.items()},
            "relationships": {rel_id: rel.to_dict() for rel_id, rel in self.relationship_map.items()}
        }
    
    def to_json(self, pretty: bool = True) -> str:
        """
        Convert the knowledge graph to a JSON string.
        
        Args:
            pretty: Whether to format the JSON with indentation
            
        Returns:
            str: JSON representation of the knowledge graph
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeGraph':
        """
        Create a knowledge graph from a dictionary.
        
        Args:
            data: Dictionary representation of a knowledge graph
            
        Returns:
            KnowledgeGraph: New knowledge graph instance
        """
        graph = cls(name=data.get("name", "knowledge_graph"))
        
        # Set metadata
        if "metadata" in data:
            graph.metadata = data["metadata"]
        
        # Add entities
        for entity_id, entity_data in data.get("entities", {}).items():
            entity = Entity(
                id=entity_id,
                name=entity_data["name"],
                entity_type=entity_data["entity_type"],
                context=entity_data.get("context"),
                attributes=entity_data.get("attributes", {}),
                metadata=entity_data.get("metadata", {}),
                confidence=entity_data.get("confidence", 1.0)
            )
            graph.add_entity(entity)
        
        # Add relationships
        for rel_id, rel_data in data.get("relationships", {}).items():
            relationship = Relationship(
                id=rel_id,
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
            graph.add_relationship(relationship)
        
        return graph
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeGraph':
        """
        Create a knowledge graph from a JSON string.
        
        Args:
            json_str: JSON representation of a knowledge graph
            
        Returns:
            KnowledgeGraph: New knowledge graph instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def visualize(self, output_path: Optional[Union[str, Path]] = None, 
                  show: bool = False, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Visualize the knowledge graph.
        
        Args:
            output_path: Path to save the visualization image
            show: Whether to display the visualization
            figsize: Figure size (width, height)
        """
        plt.figure(figsize=figsize)
        
        # Create node colors based on entity type
        entity_types = set(entity.entity_type for entity in self.entity_map.values())
        color_map = {}
        
        # Generate color map
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, entity_type in enumerate(entity_types):
            color_map[entity_type] = colors[i % len(colors)]
        
        # Set node colors
        node_colors = [color_map[self.entity_map[node_id].entity_type] for node_id in self.graph.nodes()]
        
        # Set node labels
        node_labels = {node_id: self.entity_map[node_id].name for node_id in self.graph.nodes()}
        
        # Set edge labels
        edge_labels = {(rel.source_entity, rel.target_entity): rel.relation_type 
                      for rel in self.relationship_map.values()}
        
        # Draw the graph
        pos = nx.spring_layout(self.graph, seed=42)  # Position nodes using spring layout
        
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels)
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, arrows=True)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        # Add legend for entity types
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=color, label=entity_type) 
                          for entity_type, color in color_map.items()]
        plt.legend(handles=legend_patches, loc='upper right')
        
        plt.title(f"Knowledge Graph: {self.name}")
        plt.axis('off')
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {output_path}")
        
        if show:
            plt.show()
        
        plt.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dict[str, Any]: Dictionary of graph statistics
        """
        entity_type_counts = {}
        for entity in self.entity_map.values():
            entity_type_counts[entity.entity_type] = entity_type_counts.get(entity.entity_type, 0) + 1
        
        relationship_type_counts = {}
        for rel in self.relationship_map.values():
            relationship_type_counts[rel.relation_type] = relationship_type_counts.get(rel.relation_type, 0) + 1
        
        # Calculate graph metrics
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph) if num_nodes > 1 else 0
        
        try:
            # These metrics may not work for all graphs
            avg_clustering = nx.average_clustering(self.graph)
            avg_shortest_path = nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph.to_undirected()) else None
        except:
            avg_clustering = None
            avg_shortest_path = None
        
        return {
            "entity_count": len(self.entity_map),
            "relationship_count": len(self.relationship_map),
            "entity_types": entity_type_counts,
            "relationship_types": relationship_type_counts,
            "graph_metrics": {
                "nodes": num_nodes,
                "edges": num_edges,
                "density": density,
                "avg_clustering": avg_clustering,
                "avg_shortest_path": avg_shortest_path,
                "connected": nx.is_connected(self.graph.to_undirected()) if num_nodes > 0 else False
            }
        } 