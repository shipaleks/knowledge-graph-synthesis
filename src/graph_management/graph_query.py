"""
Graph Query Module

This module provides functionality for querying and traversing knowledge graphs,
including filtering, searching, and graph algorithms.
"""

import re
from typing import Dict, List, Set, Optional, Any, Union, Tuple, Callable
from pathlib import Path

from src.utils.result import Result
from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph

# Configure logger
logger = get_logger(__name__)


class GraphQuery:
    """
    Provides methods for querying and traversing knowledge graphs.
    
    This class offers functionality for searching, filtering, and traversing
    knowledge graphs, as well as performing graph algorithms.
    """
    
    def __init__(self, graph: KnowledgeGraph, config: Optional[AppConfig] = None):
        """
        Initialize the graph query engine.
        
        Args:
            graph: The knowledge graph to query
            config: Optional application configuration
        """
        self.graph = graph
        self.config = config or AppConfig()
    
    def find_entities(self, 
                     query: Dict[str, Any],
                     limit: Optional[int] = None) -> List[Entity]:
        """
        Find entities matching a query.
        
        Args:
            query: Query dictionary with criteria to match
            limit: Maximum number of results to return
            
        Returns:
            List[Entity]: List of matching entities
        """
        results = []
        
        for entity_id, entity in self.graph.entity_map.items():
            if self._entity_matches_query(entity, query):
                results.append(entity)
                if limit and len(results) >= limit:
                    break
        
        return results
    
    def find_relationships(self, 
                          query: Dict[str, Any],
                          limit: Optional[int] = None) -> List[Relationship]:
        """
        Find relationships matching a query.
        
        Args:
            query: Query dictionary with criteria to match
            limit: Maximum number of results to return
            
        Returns:
            List[Relationship]: List of matching relationships
        """
        results = []
        
        for rel_id, relationship in self.graph.relationship_map.items():
            if self._relationship_matches_query(relationship, query):
                results.append(relationship)
                if limit and len(results) >= limit:
                    break
        
        return results
    
    def traverse(self,
                start_entity_id: str,
                max_depth: int = 2,
                direction: str = "outgoing",
                relationship_types: Optional[List[str]] = None,
                entity_types: Optional[List[str]] = None,
                visited: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Traverse the graph starting from a specified entity.
        
        Args:
            start_entity_id: ID of the entity to start traversal from
            max_depth: Maximum traversal depth
            direction: Direction of traversal (outgoing, incoming, both)
            relationship_types: Optional list of relationship types to follow
            entity_types: Optional list of entity types to include
            visited: Optional set of already visited entity IDs
            
        Returns:
            Dict[str, Any]: Tree representation of the traversal
        """
        # Check if start entity exists
        start_entity = self.graph.get_entity(start_entity_id)
        if not start_entity:
            logger.warning(f"Start entity {start_entity_id} not found")
            return {}
        
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
        
        # Create result tree
        result = {
            "entity": start_entity,
            "children": []
        }
        
        # Mark as visited
        visited.add(start_entity_id)
        
        # Stop if max depth reached
        if max_depth <= 0:
            return result
        
        # Get relationships based on direction
        relationships = []
        
        if direction in ["outgoing", "both"]:
            # Get outgoing relationships
            outgoing = self._get_outgoing_relationships(start_entity_id, relationship_types)
            relationships.extend([(rel, rel.target_entity, "outgoing") for rel in outgoing])
        
        if direction in ["incoming", "both"]:
            # Get incoming relationships
            incoming = self._get_incoming_relationships(start_entity_id, relationship_types)
            relationships.extend([(rel, rel.source_entity, "incoming") for rel in incoming])
        
        # Traverse children
        for rel, neighbor_id, rel_direction in relationships:
            # Skip if already visited
            if neighbor_id in visited:
                continue
            
            # Get neighbor entity
            neighbor = self.graph.get_entity(neighbor_id)
            if not neighbor:
                continue
            
            # Skip if entity type filter applied and type doesn't match
            if entity_types and neighbor.entity_type not in entity_types:
                continue
            
            # Recursively traverse
            child_result = self.traverse(
                neighbor_id,
                max_depth - 1,
                direction,
                relationship_types,
                entity_types,
                visited
            )
            
            if child_result:
                result["children"].append({
                    "relationship": rel,
                    "direction": rel_direction,
                    "child": child_result
                })
        
        return result
    
    def find_path(self,
                 source_entity_id: str,
                 target_entity_id: str,
                 max_depth: int = 5,
                 relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find paths between two entities.
        
        Args:
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            max_depth: Maximum path length
            relationship_types: Optional list of relationship types to follow
            
        Returns:
            List[Dict[str, Any]]: List of paths, each as a list of entities and relationships
        """
        # Check if entities exist
        if not self.graph.get_entity(source_entity_id):
            logger.warning(f"Source entity {source_entity_id} not found")
            return []
        
        if not self.graph.get_entity(target_entity_id):
            logger.warning(f"Target entity {target_entity_id} not found")
            return []
        
        # Use breadth-first search to find paths
        paths = []
        queue = [[(source_entity_id, None)]]  # (entity_id, relationship)
        visited = set([source_entity_id])
        
        while queue and len(paths) < 10:  # Limit to 10 paths
            path = queue.pop(0)
            current_entity_id = path[-1][0]
            
            # Check if reached target
            if current_entity_id == target_entity_id:
                # Convert path to result format
                formatted_path = []
                for i, (entity_id, relationship) in enumerate(path):
                    entity = self.graph.get_entity(entity_id)
                    formatted_path.append({"entity": entity})
                    
                    if i < len(path) - 1 and relationship:
                        formatted_path.append({"relationship": relationship})
                
                paths.append(formatted_path)
                continue
            
            # Stop if max depth reached
            if len(path) // 2 >= max_depth:
                continue
            
            # Get relationships
            current_entity = self.graph.get_entity(current_entity_id)
            
            # Get outgoing relationships
            outgoing = self._get_outgoing_relationships(current_entity_id, relationship_types)
            
            for rel in outgoing:
                next_entity_id = rel.target_entity
                
                # Skip if already visited in this path
                if next_entity_id in [p[0] for p in path]:
                    continue
                
                # Create new path
                new_path = path.copy()
                new_path.append((next_entity_id, rel))
                queue.append(new_path)
                
                # Mark as visited
                visited.add(next_entity_id)
        
        return paths
    
    def search_text(self,
                   text: str,
                   search_in: List[str] = ["name", "attributes", "context"],
                   entity_types: Optional[List[str]] = None,
                   case_sensitive: bool = False) -> Dict[str, List[Union[Entity, Relationship]]]:
        """
        Search for text in entity and relationship attributes.
        
        Args:
            text: Text to search for
            search_in: Fields to search in (name, attributes, context)
            entity_types: Optional list of entity types to filter by
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Dict[str, List[Union[Entity, Relationship]]]: Dictionary with 'entities' and 'relationships' keys
        """
        result = {
            "entities": [],
            "relationships": []
        }
        
        # Create regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(text, flags)
        
        # Search in entities
        for entity_id, entity in self.graph.entity_map.items():
            # Apply entity type filter if specified
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            if self._text_matches_object(pattern, entity, search_in):
                result["entities"].append(entity)
        
        # Search in relationships
        for rel_id, relationship in self.graph.relationship_map.items():
            if self._text_matches_object(pattern, relationship, search_in):
                result["relationships"].append(relationship)
        
        return result
    
    def get_subgraph(self, 
                    entity_ids: List[str], 
                    include_neighbors: bool = False,
                    max_relationships: int = 100) -> KnowledgeGraph:
        """
        Extract a subgraph containing specified entities.
        
        Args:
            entity_ids: List of entity IDs to include
            include_neighbors: Whether to include neighboring entities
            max_relationships: Maximum number of relationships to include
            
        Returns:
            KnowledgeGraph: Subgraph as a new knowledge graph
        """
        # Create a new graph
        subgraph = KnowledgeGraph(name=f"subgraph_of_{self.graph.name}", config=self.config)
        
        # Track entity IDs to include
        included_entities = set(entity_ids)
        
        # Add entities
        for entity_id in entity_ids:
            entity = self.graph.get_entity(entity_id)
            if entity:
                subgraph.add_entity(entity)
        
        # If include_neighbors, expand the set of included entities
        if include_neighbors:
            for entity_id in entity_ids:
                # Get relationships involving this entity
                relationships = self.graph.get_entity_relationships(entity_id)
                
                # Add neighbors
                for rel in relationships:
                    if rel.source_entity == entity_id:
                        included_entities.add(rel.target_entity)
                    else:
                        included_entities.add(rel.source_entity)
        
        # Add all remaining entities
        for entity_id in included_entities:
            if entity_id not in subgraph.entity_map:
                entity = self.graph.get_entity(entity_id)
                if entity:
                    subgraph.add_entity(entity)
        
        # Add relationships
        relationship_count = 0
        for rel_id, rel in self.graph.relationship_map.items():
            if rel.source_entity in included_entities and rel.target_entity in included_entities:
                subgraph.add_relationship(rel)
                relationship_count += 1
                
                if relationship_count >= max_relationships:
                    break
        
        return subgraph
    
    def filter_by_confidence(self,
                            min_confidence: float = 0.5,
                            apply_to: str = "both") -> KnowledgeGraph:
        """
        Filter the graph to include only high-confidence entities and relationships.
        
        Args:
            min_confidence: Minimum confidence score (0-1)
            apply_to: What to filter ("entities", "relationships", or "both")
            
        Returns:
            KnowledgeGraph: Filtered graph as a new knowledge graph
        """
        # Create a new graph
        filtered_graph = KnowledgeGraph(name=f"filtered_{self.graph.name}", config=self.config)
        
        # Add entities with confidence >= min_confidence
        if apply_to in ["entities", "both"]:
            for entity_id, entity in self.graph.entity_map.items():
                if entity.confidence >= min_confidence:
                    filtered_graph.add_entity(entity)
        else:
            # Add all entities if not filtering by entity confidence
            for entity_id, entity in self.graph.entity_map.items():
                filtered_graph.add_entity(entity)
        
        # Add relationships with confidence >= min_confidence
        if apply_to in ["relationships", "both"]:
            for rel_id, rel in self.graph.relationship_map.items():
                # Only add if both source and target entities exist in the filtered graph
                if (rel.source_entity in filtered_graph.entity_map and 
                    rel.target_entity in filtered_graph.entity_map and
                    rel.confidence >= min_confidence):
                    filtered_graph.add_relationship(rel)
        else:
            # Add all relationships if not filtering by relationship confidence
            for rel_id, rel in self.graph.relationship_map.items():
                # Only add if both source and target entities exist in the filtered graph
                if (rel.source_entity in filtered_graph.entity_map and 
                    rel.target_entity in filtered_graph.entity_map):
                    filtered_graph.add_relationship(rel)
        
        return filtered_graph
    
    def get_connected_components(self) -> List[KnowledgeGraph]:
        """
        Get connected components of the graph.
        
        Returns:
            List[KnowledgeGraph]: List of connected components as separate graphs
        """
        # Create undirected graph for component analysis
        undirected = self.graph.graph.to_undirected()
        
        # Get connected components
        components = []
        
        import networkx as nx
        for component in nx.connected_components(undirected):
            # Create a subgraph for this component
            component_graph = KnowledgeGraph(
                name=f"component_{len(components)+1}_of_{self.graph.name}",
                config=self.config
            )
            
            # Add entities
            for entity_id in component:
                entity = self.graph.get_entity(entity_id)
                if entity:
                    component_graph.add_entity(entity)
            
            # Add relationships
            for rel_id, rel in self.graph.relationship_map.items():
                if rel.source_entity in component and rel.target_entity in component:
                    component_graph.add_relationship(rel)
            
            components.append(component_graph)
        
        return components
    
    def sort_entities_by_metric(self, 
                               metric: Union[str, Callable[[Entity], float]], 
                               limit: Optional[int] = None,
                               reverse: bool = True) -> List[Tuple[Entity, float]]:
        """
        Sort entities by a specified metric.
        
        Args:
            metric: Metric to sort by, either a string naming a centrality measure or a function
            limit: Maximum number of results to return
            reverse: Whether to sort in descending order
            
        Returns:
            List[Tuple[Entity, float]]: List of entity-score pairs sorted by score
        """
        entities = list(self.graph.entity_map.values())
        
        # Calculate metric values
        if isinstance(metric, str):
            # Use NetworkX centrality algorithm
            import networkx as nx
            
            centrality_functions = {
                "degree": nx.degree_centrality,
                "closeness": nx.closeness_centrality,
                "betweenness": nx.betweenness_centrality,
                "eigenvector": nx.eigenvector_centrality,
                "pagerank": nx.pagerank
            }
            
            if metric not in centrality_functions:
                logger.warning(f"Unknown centrality metric: {metric}. Using degree centrality.")
                metric = "degree"
            
            # Calculate centrality
            centrality = centrality_functions[metric](self.graph.graph)
            
            # Create entity-score pairs
            entity_scores = [(self.graph.get_entity(entity_id), score) 
                            for entity_id, score in centrality.items()
                            if entity_id in self.graph.entity_map]
        
        else:
            # Use custom metric function
            entity_scores = [(entity, metric(entity)) for entity in entities]
        
        # Sort by score
        entity_scores.sort(key=lambda x: x[1], reverse=reverse)
        
        # Apply limit
        if limit:
            entity_scores = entity_scores[:limit]
        
        return entity_scores
    
    def _entity_matches_query(self, entity: Entity, query: Dict[str, Any]) -> bool:
        """
        Check if an entity matches a query.
        
        Args:
            entity: Entity to check
            query: Query dictionary with criteria to match
            
        Returns:
            bool: True if entity matches query, False otherwise
        """
        for key, value in query.items():
            # Handle special case for entity_type/type
            if key == "entity_type" or key == "type":
                if entity.entity_type != value:
                    return False
                continue
            
            # Handle special case for name
            if key == "name":
                if isinstance(value, str):
                    # Exact match
                    if entity.name != value:
                        return False
                elif isinstance(value, dict):
                    # Advanced name matching
                    if "contains" in value and value["contains"] not in entity.name:
                        return False
                    if "exact" in value and entity.name != value["exact"]:
                        return False
                    if "regex" in value and not re.search(value["regex"], entity.name):
                        return False
                continue
            
            # Handle special case for confidence
            if key == "confidence":
                if isinstance(value, (int, float)):
                    # Exact match
                    if entity.confidence != value:
                        return False
                elif isinstance(value, dict):
                    # Range matching
                    if "min" in value and entity.confidence < value["min"]:
                        return False
                    if "max" in value and entity.confidence > value["max"]:
                        return False
                continue
            
            # Handle special case for attributes
            if key == "attributes":
                for attr_key, attr_value in value.items():
                    if attr_key not in entity.attributes:
                        return False
                    
                    if attr_value != entity.attributes[attr_key]:
                        return False
                continue
            
            # Check if key is a direct attribute
            if key in entity.attributes:
                if entity.attributes[key] != value:
                    return False
        
        return True
    
    def _relationship_matches_query(self, relationship: Relationship, query: Dict[str, Any]) -> bool:
        """
        Check if a relationship matches a query.
        
        Args:
            relationship: Relationship to check
            query: Query dictionary with criteria to match
            
        Returns:
            bool: True if relationship matches query, False otherwise
        """
        for key, value in query.items():
            # Handle special case for relation_type/type
            if key == "relation_type" or key == "type":
                if relationship.relation_type != value:
                    return False
                continue
            
            # Handle special case for source_entity/source
            if key == "source_entity" or key == "source":
                if relationship.source_entity != value:
                    return False
                continue
            
            # Handle special case for target_entity/target
            if key == "target_entity" or key == "target":
                if relationship.target_entity != value:
                    return False
                continue
            
            # Handle special case for confidence
            if key == "confidence":
                if isinstance(value, (int, float)):
                    # Exact match
                    if relationship.confidence != value:
                        return False
                elif isinstance(value, dict):
                    # Range matching
                    if "min" in value and relationship.confidence < value["min"]:
                        return False
                    if "max" in value and relationship.confidence > value["max"]:
                        return False
                continue
            
            # Handle special case for strength
            if key == "strength":
                if isinstance(value, (int, float)):
                    # Exact match
                    if relationship.strength != value:
                        return False
                elif isinstance(value, dict):
                    # Range matching
                    if "min" in value and relationship.strength < value["min"]:
                        return False
                    if "max" in value and relationship.strength > value["max"]:
                        return False
                continue
            
            # Handle special case for bidirectional
            if key == "bidirectional":
                if relationship.bidirectional != value:
                    return False
                continue
            
            # Handle special case for attributes
            if key == "attributes":
                for attr_key, attr_value in value.items():
                    if attr_key not in relationship.attributes:
                        return False
                    
                    if attr_value != relationship.attributes[attr_key]:
                        return False
                continue
            
            # Check if key is a direct attribute
            if key in relationship.attributes:
                if relationship.attributes[key] != value:
                    return False
        
        return True
    
    def _get_outgoing_relationships(self, 
                                  entity_id: str, 
                                  relationship_types: Optional[List[str]] = None) -> List[Relationship]:
        """
        Get outgoing relationships from an entity.
        
        Args:
            entity_id: ID of the entity
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List[Relationship]: List of outgoing relationships
        """
        outgoing = []
        
        for rel_id, rel in self.graph.relationship_map.items():
            if rel.source_entity != entity_id:
                continue
            
            if relationship_types and rel.relation_type not in relationship_types:
                continue
            
            outgoing.append(rel)
        
        return outgoing
    
    def _get_incoming_relationships(self, 
                                   entity_id: str, 
                                   relationship_types: Optional[List[str]] = None) -> List[Relationship]:
        """
        Get incoming relationships to an entity.
        
        Args:
            entity_id: ID of the entity
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List[Relationship]: List of incoming relationships
        """
        incoming = []
        
        for rel_id, rel in self.graph.relationship_map.items():
            if rel.target_entity != entity_id:
                continue
            
            if relationship_types and rel.relation_type not in relationship_types:
                continue
            
            incoming.append(rel)
        
        return incoming
    
    def _text_matches_object(self, 
                           pattern: re.Pattern, 
                           obj: Union[Entity, Relationship], 
                           search_in: List[str]) -> bool:
        """
        Check if text pattern matches an object.
        
        Args:
            pattern: Compiled regex pattern
            obj: Entity or relationship to check
            search_in: Fields to search in
            
        Returns:
            bool: True if pattern matches object, False otherwise
        """
        # Check name if searching in name
        if "name" in search_in and hasattr(obj, "name"):
            if pattern.search(obj.name):
                return True
        
        # Check context if searching in context
        if "context" in search_in and obj.context:
            if pattern.search(obj.context):
                return True
        
        # Check attributes if searching in attributes
        if "attributes" in search_in:
            for key, value in obj.attributes.items():
                if isinstance(value, str) and pattern.search(value):
                    return True
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str) and pattern.search(item):
                            return True
        
        return False


# Convenience functions

def query_graph(graph: KnowledgeGraph, query: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, List]:
    """
    Convenience function to query a graph for entities and relationships.
    
    Args:
        graph: Knowledge graph to query
        query: Query dictionary with criteria to match
        limit: Maximum number of results to return
        
    Returns:
        Dict[str, List]: Dictionary with 'entities' and 'relationships' keys
    """
    query_engine = GraphQuery(graph)
    
    result = {
        "entities": query_engine.find_entities(query, limit),
        "relationships": query_engine.find_relationships(query, limit)
    }
    
    return result


def search_graph_text(graph: KnowledgeGraph, 
                     text: str, 
                     search_in: List[str] = ["name", "attributes", "context"],
                     entity_types: Optional[List[str]] = None,
                     case_sensitive: bool = False) -> Dict[str, List]:
    """
    Convenience function to search for text in a graph.
    
    Args:
        graph: Knowledge graph to search
        text: Text to search for
        search_in: Fields to search in (name, attributes, context)
        entity_types: Optional list of entity types to filter by
        case_sensitive: Whether to perform case-sensitive search
        
    Returns:
        Dict[str, List]: Dictionary with 'entities' and 'relationships' keys
    """
    query_engine = GraphQuery(graph)
    return query_engine.search_text(text, search_in, entity_types, case_sensitive)


def get_entity_neighborhood(graph: KnowledgeGraph, 
                           entity_id: str,
                           max_depth: int = 1,
                           relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to get the neighborhood of an entity.
    
    Args:
        graph: Knowledge graph to query
        entity_id: ID of the entity to get neighborhood for
        max_depth: Maximum traversal depth
        relationship_types: Optional list of relationship types to follow
        
    Returns:
        Dict[str, Any]: Tree representation of the neighborhood
    """
    query_engine = GraphQuery(graph)
    return query_engine.traverse(entity_id, max_depth, "both", relationship_types)


def find_paths(graph: KnowledgeGraph,
              source_entity_id: str,
              target_entity_id: str,
              max_depth: int = 3) -> List[Dict[str, Any]]:
    """
    Convenience function to find paths between two entities.
    
    Args:
        graph: Knowledge graph to query
        source_entity_id: ID of the source entity
        target_entity_id: ID of the target entity
        max_depth: Maximum path length
        
    Returns:
        List[Dict[str, Any]]: List of paths, each as a list of entities and relationships
    """
    query_engine = GraphQuery(graph)
    return query_engine.find_path(source_entity_id, target_entity_id, max_depth) 