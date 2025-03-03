"""
Graph Visualization Module

This module provides functionality for visualizing knowledge graphs,
creating interactive HTML visualizations using PyVis.
"""

import os
import json
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from pathlib import Path
import networkx as nx
from pyvis.network import Network

from src.utils.result import Result
from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph

# Configure logger
logger = get_logger(__name__)

# Define color schemes
COLOR_SCHEMES = {
    "default": {
        "entity_colors": {
            "Person": "#4682B4",  # Steel Blue
            "Organization": "#D2691E",  # Chocolate
            "Location": "#228B22",  # Forest Green
            "Event": "#8A2BE2",  # Blue Violet
            "Concept": "#FF8C00",  # Dark Orange
            "Skill": "#20B2AA",  # Light Sea Green
            "Job": "#CD5C5C",  # Indian Red
            "Project": "#9370DB",  # Medium Purple
            "default": "#A9A9A9"  # Dark Gray
        },
        "relationship_colors": {
            "is-a": "#0000CD",  # Medium Blue
            "part-of": "#2E8B57",  # Sea Green
            "has": "#8B4513",  # Saddle Brown
            "related-to": "#4B0082",  # Indigo
            "causes": "#B22222",  # Fire Brick
            "works-for": "#1E90FF",  # Dodger Blue
            "knows": "#9932CC",  # Dark Orchid
            "has-skill": "#228B22",  # Forest Green
            "default": "#696969"  # Dim Gray
        }
    },
    "grayscale": {
        "entity_colors": {
            "default": "#555555"
        },
        "relationship_colors": {
            "default": "#999999"
        }
    },
    "pastel": {
        "entity_colors": {
            "Person": "#FFB6C1",  # Light Pink
            "Organization": "#FFD700",  # Gold
            "Location": "#98FB98",  # Pale Green
            "Event": "#DDA0DD",  # Plum
            "Concept": "#FFDAB9",  # Peach Puff
            "Skill": "#AFEEEE",  # Pale Turquoise
            "Job": "#FFA07A",  # Light Salmon
            "Project": "#D8BFD8",  # Thistle
            "default": "#E6E6FA"  # Lavender
        },
        "relationship_colors": {
            "default": "#B0C4DE"  # Light Steel Blue
        }
    }
}


class GraphVisualizer:
    """
    Provides methods for visualizing knowledge graphs.
    
    This class offers functionality for creating interactive HTML visualizations
    of knowledge graphs using PyVis.
    """
    
    def __init__(self, graph: KnowledgeGraph, config: Optional[AppConfig] = None):
        """
        Initialize the graph visualizer.
        
        Args:
            graph: The knowledge graph to visualize
            config: Optional application configuration
        """
        self.graph = graph
        self.config = config or AppConfig()
        self.network = None
        self.color_scheme = COLOR_SCHEMES["default"]
    
    def create_visualization(self, 
                           height: str = "800px", 
                           width: str = "100%",
                           directed: bool = True,
                           color_scheme: str = "default") -> Result[Network]:
        """
        Create a visualization of the knowledge graph.
        
        Args:
            height: Height of the visualization
            width: Width of the visualization
            directed: Whether to show the graph as directed
            color_scheme: Color scheme to use (default, grayscale, pastel)
            
        Returns:
            Result[Network]: Result containing the PyVis Network object or an error
        """
        try:
            # Set color scheme
            if color_scheme in COLOR_SCHEMES:
                self.color_scheme = COLOR_SCHEMES[color_scheme]
            else:
                logger.warning(f"Unknown color scheme: {color_scheme}, using default")
                self.color_scheme = COLOR_SCHEMES["default"]
            
            # Create a PyVis network
            self.network = Network(height=height, width=width, directed=directed, notebook=False)
            
            # Add all entities as nodes
            for entity_id, entity in self.graph.entity_map.items():
                self._add_entity_as_node(entity)
            
            # Add all relationships as edges
            for rel_id, relationship in self.graph.relationship_map.items():
                self._add_relationship_as_edge(relationship)
            
            return Result.ok(self.network)
        
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return Result.fail(f"Failed to create visualization: {str(e)}")
    
    def _add_entity_as_node(self, entity: Entity) -> None:
        """
        Add an entity as a node to the visualization.
        
        Args:
            entity: The entity to add as a node
        """
        # Get color based on entity type
        entity_colors = self.color_scheme["entity_colors"]
        color = entity_colors.get(entity.entity_type, entity_colors.get("default", "#A9A9A9"))
        
        # Create node labels with key attributes
        attributes_text = ""
        if entity.attributes:
            attributes = []
            # Display up to 3 key attributes
            for i, (key, value) in enumerate(entity.attributes.items()):
                if i < 3:
                    attributes.append(f"{key}: {value}")
            if attributes:
                attributes_text = "<br>" + "<br>".join(attributes)
        
        # Add confidence if available
        confidence_text = ""
        if hasattr(entity, 'confidence') and entity.confidence is not None:
            confidence = f"{entity.confidence:.2f}" if isinstance(entity.confidence, float) else entity.confidence
            confidence_text = f"<br>confidence: {confidence}"
        
        # Create label
        label = f"{entity.name}<br><i>{entity.entity_type}</i>{attributes_text}{confidence_text}"
        
        # Add node to network
        self.network.add_node(
            entity.id, 
            label=label, 
            title=label,
            color=color,
            shape="dot" if entity.entity_type == "Person" else "box",
            size=30 if hasattr(entity, 'confidence') and entity.confidence and entity.confidence > 0.8 else 20
        )
    
    def _add_relationship_as_edge(self, relationship: Relationship) -> None:
        """
        Add a relationship as an edge to the visualization.
        
        Args:
            relationship: The relationship to add as an edge
        """
        # Get color based on relationship type
        relationship_colors = self.color_scheme["relationship_colors"]
        color = relationship_colors.get(relationship.relation_type.lower(), relationship_colors.get("default", "#696969"))
        
        # Create edge label
        label = relationship.relation_type
        
        # Add confidence to title if available
        title = label
        if hasattr(relationship, 'confidence') and relationship.confidence is not None:
            confidence = f"{relationship.confidence:.2f}" if isinstance(relationship.confidence, float) else relationship.confidence
            title = f"{label} (confidence: {confidence})"
        
        # Adjust edge width based on confidence
        width = 1.0
        if hasattr(relationship, 'confidence') and relationship.confidence is not None:
            if isinstance(relationship.confidence, (int, float)):
                # Scale width between 1 and 5 based on confidence
                width = 1.0 + (relationship.confidence * 4.0)
        
        # Add edge to network
        self.network.add_edge(
            relationship.source_entity,
            relationship.target_entity,
            label=label,
            title=title,
            color=color,
            width=width
        )
        
        # If relationship is bidirectional, add reverse edge
        if hasattr(relationship, 'bidirectional') and relationship.bidirectional:
            self.network.add_edge(
                relationship.target_entity,
                relationship.source_entity,
                label=label,
                title=title,
                color=color,
                width=width
            )
    
    def filter_visualization(self, 
                          entity_types: Optional[List[str]] = None,
                          relationship_types: Optional[List[str]] = None,
                          min_confidence: Optional[float] = None) -> Result[Network]:
        """
        Filter the visualization to show only specific entity and relationship types.
        
        Args:
            entity_types: List of entity types to include (None for all)
            relationship_types: List of relationship types to include (None for all)
            min_confidence: Minimum confidence threshold (None for all)
            
        Returns:
            Result[Network]: Result containing the filtered network or an error
        """
        try:
            if not self.network:
                return Result.fail("No visualization has been created yet")
            
            # Create a new network with same properties
            filtered_network = Network(
                height=self.network.height,
                width=self.network.width,
                directed=True,
                notebook=False
            )
            
            # Keep track of entities to include
            included_entity_ids = set()
            
            # Filter entities by type and confidence
            for entity_id, entity in self.graph.entity_map.items():
                include = True
                
                # Filter by entity type
                if entity_types and entity.entity_type not in entity_types:
                    include = False
                
                # Filter by confidence
                if min_confidence is not None and hasattr(entity, 'confidence'):
                    if entity.confidence < min_confidence:
                        include = False
                
                if include:
                    included_entity_ids.add(entity_id)
                    self._add_entity_to_network(entity, filtered_network)
            
            # Filter relationships and only include those connecting included entities
            for rel_id, relationship in self.graph.relationship_map.items():
                include = True
                
                # Only include if both entities are included
                if (relationship.source_entity not in included_entity_ids or 
                    relationship.target_entity not in included_entity_ids):
                    include = False
                
                # Filter by relationship type
                if relationship_types and relationship.relation_type not in relationship_types:
                    include = False
                
                # Filter by confidence
                if min_confidence is not None and hasattr(relationship, 'confidence'):
                    if relationship.confidence < min_confidence:
                        include = False
                
                if include:
                    self._add_relationship_to_network(relationship, filtered_network)
            
            return Result.ok(filtered_network)
        
        except Exception as e:
            logger.error(f"Error filtering visualization: {str(e)}")
            return Result.fail(f"Failed to filter visualization: {str(e)}")
    
    def _add_entity_to_network(self, entity: Entity, network: Network) -> None:
        """
        Add an entity to a specific network.
        
        Args:
            entity: The entity to add
            network: The network to add the entity to
        """
        # Get color based on entity type
        entity_colors = self.color_scheme["entity_colors"]
        color = entity_colors.get(entity.entity_type, entity_colors.get("default", "#A9A9A9"))
        
        # Create label
        attributes_text = ""
        if entity.attributes:
            attributes = []
            for i, (key, value) in enumerate(entity.attributes.items()):
                if i < 3:
                    attributes.append(f"{key}: {value}")
            if attributes:
                attributes_text = "<br>" + "<br>".join(attributes)
        
        confidence_text = ""
        if hasattr(entity, 'confidence') and entity.confidence is not None:
            confidence = f"{entity.confidence:.2f}" if isinstance(entity.confidence, float) else entity.confidence
            confidence_text = f"<br>confidence: {confidence}"
        
        label = f"{entity.name}<br><i>{entity.entity_type}</i>{attributes_text}{confidence_text}"
        
        # Add node to network
        network.add_node(
            entity.id, 
            label=label, 
            title=label,
            color=color,
            shape="dot" if entity.entity_type == "Person" else "box",
            size=30 if hasattr(entity, 'confidence') and entity.confidence and entity.confidence > 0.8 else 20
        )
    
    def _add_relationship_to_network(self, relationship: Relationship, network: Network) -> None:
        """
        Add a relationship to a specific network.
        
        Args:
            relationship: The relationship to add
            network: The network to add the relationship to
        """
        # Get color based on relationship type
        relationship_colors = self.color_scheme["relationship_colors"]
        color = relationship_colors.get(relationship.relation_type.lower(), relationship_colors.get("default", "#696969"))
        
        # Create edge label
        label = relationship.relation_type
        
        # Add confidence to title if available
        title = label
        if hasattr(relationship, 'confidence') and relationship.confidence is not None:
            confidence = f"{relationship.confidence:.2f}" if isinstance(relationship.confidence, float) else relationship.confidence
            title = f"{label} (confidence: {confidence})"
        
        # Adjust edge width based on confidence
        width = 1.0
        if hasattr(relationship, 'confidence') and relationship.confidence is not None:
            if isinstance(relationship.confidence, (int, float)):
                # Scale width between 1 and 5 based on confidence
                width = 1.0 + (relationship.confidence * 4.0)
        
        # Add edge to network
        network.add_edge(
            relationship.source_entity,
            relationship.target_entity,
            label=label,
            title=title,
            color=color,
            width=width
        )
        
        # If relationship is bidirectional, add reverse edge
        if hasattr(relationship, 'bidirectional') and relationship.bidirectional:
            network.add_edge(
                relationship.target_entity,
                relationship.source_entity,
                label=label,
                title=title,
                color=color,
                width=width
            )
    
    def save_visualization(self, 
                         output_path: Union[str, Path],
                         filename: str = "knowledge_graph.html",
                         network: Optional[Network] = None) -> Result[str]:
        """
        Save the visualization to an HTML file.
        
        Args:
            output_path: Directory to save the visualization
            filename: Name of the HTML file
            network: Optional network to save (uses current network if None)
            
        Returns:
            Result[str]: Result containing the path to the saved file or an error
        """
        try:
            if network is None:
                if not self.network:
                    return Result.fail("No visualization has been created yet")
                network = self.network
            
            # Ensure output directory exists
            output_path = Path(output_path)
            os.makedirs(output_path, exist_ok=True)
            
            # Save the network to HTML
            full_path = output_path / filename
            network.save_graph(str(full_path))
            
            logger.info(f"Visualization saved to {full_path}")
            return Result.ok(str(full_path))
        
        except Exception as e:
            logger.error(f"Error saving visualization: {str(e)}")
            return Result.fail(f"Failed to save visualization: {str(e)}")


# Helper functions for common visualization operations

def visualize_graph(graph: KnowledgeGraph,
                   output_path: Union[str, Path] = "output",
                   filename: str = "knowledge_graph.html",
                   height: str = "800px",
                   width: str = "100%",
                   color_scheme: str = "default") -> Result[str]:
    """
    Visualize a knowledge graph and save to HTML.
    
    Args:
        graph: The knowledge graph to visualize
        output_path: Directory to save the visualization
        filename: Name of the HTML file
        height: Height of the visualization
        width: Width of the visualization
        color_scheme: Color scheme to use (default, grayscale, pastel)
        
    Returns:
        Result[str]: Result containing the path to the saved file or an error
    """
    visualizer = GraphVisualizer(graph)
    
    # Create visualization
    result = visualizer.create_visualization(
        height=height,
        width=width,
        directed=True,
        color_scheme=color_scheme
    )
    
    if not result.success:
        return result
    
    # Save visualization
    return visualizer.save_visualization(output_path=output_path, filename=filename)


def visualize_subgraph(graph: KnowledgeGraph,
                     entity_ids: List[str],
                     include_neighbors: bool = True,
                     output_path: Union[str, Path] = "output",
                     filename: str = "subgraph.html",
                     color_scheme: str = "default") -> Result[str]:
    """
    Visualize a subgraph of a knowledge graph.
    
    Args:
        graph: The knowledge graph to visualize
        entity_ids: List of entity IDs to include
        include_neighbors: Whether to include neighboring entities
        output_path: Directory to save the visualization
        filename: Name of the HTML file
        color_scheme: Color scheme to use (default, grayscale, pastel)
        
    Returns:
        Result[str]: Result containing the path to the saved file or an error
    """
    from src.graph_management.graph_query import GraphQuery
    
    # Extract subgraph
    query = GraphQuery(graph)
    subgraph = query.get_subgraph(
        entity_ids=entity_ids,
        include_neighbors=include_neighbors
    )
    
    # Visualize subgraph
    return visualize_graph(
        graph=subgraph,
        output_path=output_path,
        filename=filename,
        color_scheme=color_scheme
    )


def visualize_path(graph: KnowledgeGraph,
                 source_entity_id: str,
                 target_entity_id: str,
                 output_path: Union[str, Path] = "output",
                 filename: str = "path.html") -> Result[str]:
    """
    Visualize paths between two entities in a knowledge graph.
    
    Args:
        graph: The knowledge graph to visualize
        source_entity_id: ID of the source entity
        target_entity_id: ID of the target entity
        output_path: Directory to save the visualization
        filename: Name of the HTML file
        
    Returns:
        Result[str]: Result containing the path to the saved file or an error
    """
    from src.graph_management.graph_reasoning import reason_over_paths
    
    # Find paths between entities
    path_result = reason_over_paths(
        graph=graph,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id
    )
    
    if not path_result["success"]:
        return Result.fail(f"No paths found between {source_entity_id} and {target_entity_id}")
    
    # Get the first path
    if not path_result["paths"]:
        return Result.fail(f"No paths found between {source_entity_id} and {target_entity_id}")
    
    # Extract entities from the path for visualization
    entity_ids = []
    for item in path_result["paths"][0]:
        if "entity" in item:
            entity_ids.append(item["entity"].id)
    
    # Visualize the subgraph of the path
    return visualize_subgraph(
        graph=graph,
        entity_ids=entity_ids,
        include_neighbors=False,
        output_path=output_path,
        filename=filename
    )


def visualize_filtered_graph(graph: KnowledgeGraph,
                          entity_types: Optional[List[str]] = None,
                          relationship_types: Optional[List[str]] = None,
                          min_confidence: float = 0.6,
                          output_path: Union[str, Path] = "output",
                          filename: str = "filtered_graph.html",
                          color_scheme: str = "default") -> Result[str]:
    """
    Visualize a filtered view of a knowledge graph.
    
    Args:
        graph: The knowledge graph to visualize
        entity_types: List of entity types to include (None for all)
        relationship_types: List of relationship types to include (None for all)
        min_confidence: Minimum confidence threshold
        output_path: Directory to save the visualization
        filename: Name of the HTML file
        color_scheme: Color scheme to use (default, grayscale, pastel)
        
    Returns:
        Result[str]: Result containing the path to the saved file or an error
    """
    # Create visualization
    visualizer = GraphVisualizer(graph)
    result = visualizer.create_visualization(color_scheme=color_scheme)
    
    if not result.success:
        return result
    
    # Filter visualization
    filter_result = visualizer.filter_visualization(
        entity_types=entity_types,
        relationship_types=relationship_types,
        min_confidence=min_confidence
    )
    
    if not filter_result.success:
        return filter_result
    
    # Save visualization
    return visualizer.save_visualization(
        output_path=output_path,
        filename=filename,
        network=filter_result.value
    ) 