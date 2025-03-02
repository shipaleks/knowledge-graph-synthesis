"""
Knowledge Graph Module

This module provides functionality for building and manipulating
knowledge graphs from text summaries.
"""

import os
import json
import time
import uuid
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.language import get_prompt_for_language
from src.config.app_config import AppConfig
from src.config.llm_config import LLMConfig
from src.llm import get_provider, LLMProvider
from src.text_processing.segment import Segment, SegmentationResult

# Configure logger
logger = get_logger(__name__)

# Entity extraction prompts for different languages
ENTITY_EXTRACTION_PROMPTS = {
    "en": """Extract entities and relationships from the following summarized text segment.

Segment: {text}
Segment type: {segment_type}
Segment ID: {segment_id}
Summary: {summary}
Key points: {key_points}
Keywords: {keywords}

Please identify:
1. Main entities (concepts, objects, people, organizations, etc.)
2. Attributes of these entities
3. Relationships between entities

Format your response as a structured JSON with the following schema:
{{
  "entities": [
    {{
      "id": "unique_id",
      "name": "entity_name",
      "type": "entity_type",
      "attributes": {{
        "attribute_name": "attribute_value"
      }},
      "salience": 0.9
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "relationship_type",
      "attributes": {{
        "attribute_name": "attribute_value"
      }},
      "weight": 0.8
    }}
  ]
}}

Ensure entities have meaningful types and relationships are directional and descriptive.""",

    "ru": """Извлеките сущности и отношения из следующего суммаризированного сегмента текста.

Сегмент: {text}
Тип сегмента: {segment_type}
ID сегмента: {segment_id}
Резюме: {summary}
Ключевые тезисы: {key_points}
Ключевые слова: {keywords}

Пожалуйста, определите:
1. Основные сущности (концепции, объекты, люди, организации и т.д.)
2. Атрибуты этих сущностей
3. Отношения между сущностями

Форматируйте ваш ответ как структурированный JSON по следующей схеме:
{{
  "entities": [
    {{
      "id": "unique_id",
      "name": "entity_name",
      "type": "entity_type",
      "attributes": {{
        "attribute_name": "attribute_value"
      }},
      "salience": 0.9
    }}
  ],
  "relationships": [
    {{
      "source": "source_entity_id",
      "target": "target_entity_id",
      "type": "relationship_type",
      "attributes": {{
        "attribute_name": "attribute_value"
      }},
      "weight": 0.8
    }}
  ]
}}

Убедитесь, что сущности имеют содержательные типы, а отношения направлены и описательны."""
}

# Schema for entity extraction response
ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "attributes": {"type": "object"},
                    "salience": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["id", "name", "type", "salience"]
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "attributes": {"type": "object"},
                    "weight": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["source", "target", "type", "weight"]
            }
        }
    },
    "required": ["entities", "relationships"]
}

# Entity and relationship types for standardization
COMMON_ENTITY_TYPES = {
    "en": ["Concept", "Person", "Organization", "Location", "Event", "Process", "Method", "Tool", "Property", "Field", "Document"],
    "ru": ["Концепция", "Человек", "Организация", "Местоположение", "Событие", "Процесс", "Метод", "Инструмент", "Свойство", "Область", "Документ"]
}

COMMON_RELATIONSHIP_TYPES = {
    "en": ["is_a", "part_of", "has_part", "related_to", "causes", "used_by", "created_by", "located_in", "occurs_in", "has_property", "example_of", "opposite_of", "similar_to"],
    "ru": ["является", "часть_от", "имеет_часть", "связан_с", "вызывает", "используется_для", "создан_кем", "находится_в", "происходит_в", "имеет_свойство", "пример_от", "противоположен", "подобен"]
}


class Entity:
    """
    Represents an entity in a knowledge graph.
    
    Entities are nodes in the graph that represent concepts, objects, people, etc.
    """
    
    def __init__(self, 
                entity_id: str, 
                name: str, 
                entity_type: str,
                attributes: Optional[Dict[str, Any]] = None, 
                salience: float = 0.5,
                source_segments: Optional[List[str]] = None):
        """
        Initialize an entity.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Name or label of the entity
            entity_type: Type or category of the entity
            attributes: Dictionary of attributes
            salience: Importance score between 0-1
            source_segments: List of segment IDs that mention this entity
        """
        self.id = entity_id
        self.name = name
        self.type = entity_type
        self.attributes = attributes or {}
        self.salience = salience
        self.source_segments = source_segments or []
    
    def add_source_segment(self, segment_id: str) -> None:
        """
        Add a segment ID to the sources for this entity.
        
        Args:
            segment_id: ID of the segment mentioning this entity
        """
        if segment_id not in self.source_segments:
            self.source_segments.append(segment_id)
    
    def update_salience(self, new_salience: float) -> None:
        """
        Update the salience score, using the maximum of current and new value.
        
        Args:
            new_salience: New salience score
        """
        self.salience = max(self.salience, new_salience)
    
    def merge_attributes(self, new_attributes: Dict[str, Any]) -> None:
        """
        Merge new attributes with existing ones.
        
        Args:
            new_attributes: New attributes to add
        """
        for key, value in new_attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "attributes": self.attributes,
            "salience": self.salience,
            "source_segments": self.source_segments
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an entity from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Entity: Entity object
        """
        return cls(
            entity_id=data["id"],
            name=data["name"],
            entity_type=data["type"],
            attributes=data.get("attributes", {}),
            salience=data.get("salience", 0.5),
            source_segments=data.get("source_segments", [])
        )
    
    def __eq__(self, other: object) -> bool:
        """
        Compare entities by ID.
        
        Args:
            other: Another entity
            
        Returns:
            bool: True if equal
        """
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """
        Hash based on ID.
        
        Returns:
            int: Hash value
        """
        return hash(self.id)


class Relationship:
    """
    Represents a relationship between entities in a knowledge graph.
    
    Relationships are edges in the graph that connect entities.
    """
    
    def __init__(self, 
                source_id: str, 
                target_id: str,
                relationship_type: str,
                attributes: Optional[Dict[str, Any]] = None,
                weight: float = 0.5,
                source_segments: Optional[List[str]] = None):
        """
        Initialize a relationship.
        
        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            relationship_type: Type of relationship
            attributes: Dictionary of attributes
            weight: Confidence/strength score between 0-1
            source_segments: List of segment IDs that mention this relationship
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = relationship_type
        self.attributes = attributes or {}
        self.weight = weight
        self.source_segments = source_segments or []
    
    def add_source_segment(self, segment_id: str) -> None:
        """
        Add a segment ID to the sources for this relationship.
        
        Args:
            segment_id: ID of the segment mentioning this relationship
        """
        if segment_id not in self.source_segments:
            self.source_segments.append(segment_id)
    
    def update_weight(self, new_weight: float) -> None:
        """
        Update the weight, using the maximum of current and new value.
        
        Args:
            new_weight: New weight value
        """
        self.weight = max(self.weight, new_weight)
    
    def merge_attributes(self, new_attributes: Dict[str, Any]) -> None:
        """
        Merge new attributes with existing ones.
        
        Args:
            new_attributes: New attributes to add
        """
        for key, value in new_attributes.items():
            if key not in self.attributes:
                self.attributes[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the relationship to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.type,
            "attributes": self.attributes,
            "weight": self.weight,
            "source_segments": self.source_segments
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """
        Create a relationship from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Relationship: Relationship object
        """
        return cls(
            source_id=data["source"],
            target_id=data["target"],
            relationship_type=data["type"],
            attributes=data.get("attributes", {}),
            weight=data.get("weight", 0.5),
            source_segments=data.get("source_segments", [])
        )
    
    def __eq__(self, other: object) -> bool:
        """
        Compare relationships by source, target, and type.
        
        Args:
            other: Another relationship
            
        Returns:
            bool: True if equal
        """
        if not isinstance(other, Relationship):
            return False
        return (self.source_id == other.source_id and 
                self.target_id == other.target_id and 
                self.type == other.type)
    
    def __hash__(self) -> int:
        """
        Hash based on source, target, and type.
        
        Returns:
            int: Hash value
        """
        return hash((self.source_id, self.target_id, self.type))


class KnowledgeGraph:
    """
    Represents a knowledge graph built from text segments.
    
    Provides methods for building, manipulating, and visualizing the graph.
    """
    
    def __init__(self, name: str = "Knowledge Graph"):
        """
        Initialize a knowledge graph.
        
        Args:
            name: Name of the graph
        """
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.metadata: Dict[str, Any] = {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "entity_count": 0,
            "relationship_count": 0
        }
        self.nx_graph: Optional[nx.DiGraph] = None
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the graph or update if it exists.
        
        Args:
            entity: Entity to add
            
        Returns:
            str: Entity ID
        """
        if entity.id in self.entities:
            # Update existing entity
            existing = self.entities[entity.id]
            existing.update_salience(entity.salience)
            existing.merge_attributes(entity.attributes)
            for segment_id in entity.source_segments:
                existing.add_source_segment(segment_id)
        else:
            # Add new entity
            self.entities[entity.id] = entity
            self.metadata["entity_count"] = len(self.entities)
            self.metadata["updated"] = datetime.now().isoformat()
        
        return entity.id
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """
        Add a relationship to the graph or update if it exists.
        
        Args:
            relationship: Relationship to add
            
        Returns:
            bool: True if added, False if source or target entity not found
        """
        # Check that both source and target entities exist
        if relationship.source_id not in self.entities or relationship.target_id not in self.entities:
            return False
        
        # Check if relationship already exists
        for existing in self.relationships:
            if existing == relationship:
                # Update existing relationship
                existing.update_weight(relationship.weight)
                existing.merge_attributes(relationship.attributes)
                for segment_id in relationship.source_segments:
                    existing.add_source_segment(segment_id)
                return True
        
        # Add new relationship
        self.relationships.append(relationship)
        self.metadata["relationship_count"] = len(self.relationships)
        self.metadata["updated"] = datetime.now().isoformat()
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Optional[Entity]: Entity or None if not found
        """
        return self.entities.get(entity_id)
    
    def get_entities_by_name(self, name: str) -> List[Entity]:
        """
        Get entities by name.
        
        Args:
            name: Entity name
            
        Returns:
            List[Entity]: List of matching entities
        """
        return [e for e in self.entities.values() if e.name.lower() == name.lower()]
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get entities by type.
        
        Args:
            entity_type: Entity type
            
        Returns:
            List[Entity]: List of matching entities
        """
        return [e for e in self.entities.values() if e.type.lower() == entity_type.lower()]
    
    def get_relationships(self, source_id: Optional[str] = None, target_id: Optional[str] = None, 
                         relationship_type: Optional[str] = None) -> List[Relationship]:
        """
        Get relationships with optional filtering.
        
        Args:
            source_id: Optional source entity ID filter
            target_id: Optional target entity ID filter
            relationship_type: Optional relationship type filter
            
        Returns:
            List[Relationship]: List of matching relationships
        """
        results = self.relationships
        
        if source_id:
            results = [r for r in results if r.source_id == source_id]
        if target_id:
            results = [r for r in results if r.target_id == target_id]
        if relationship_type:
            results = [r for r in results if r.type.lower() == relationship_type.lower()]
            
        return results
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert the knowledge graph to a NetworkX directed graph.
        
        Returns:
            nx.DiGraph: NetworkX graph
        """
        G = nx.DiGraph(name=self.name)
        
        # Add nodes (entities)
        for entity_id, entity in self.entities.items():
            G.add_node(entity_id, 
                      name=entity.name, 
                      type=entity.type, 
                      attributes=entity.attributes, 
                      salience=entity.salience)
        
        # Add edges (relationships)
        for rel in self.relationships:
            G.add_edge(rel.source_id, 
                      rel.target_id, 
                      type=rel.type, 
                      attributes=rel.attributes, 
                      weight=rel.weight)
        
        self.nx_graph = G
        return G
    
    def visualize(self, 
                 output_path: Optional[Path] = None, 
                 show: bool = False, 
                 figsize: Tuple[int, int] = (16, 9)) -> Optional[Figure]:
        """
        Visualize the knowledge graph.
        
        Args:
            output_path: Optional path to save the visualization
            show: Whether to display the visualization
            figsize: Figure size as (width, height) in inches
            
        Returns:
            Optional[Figure]: Matplotlib figure if show is True
        """
        if not self.nx_graph:
            self.to_networkx()
        
        G = self.nx_graph
        
        # Create node color map based on entity type
        node_types = list(set(nx.get_node_attributes(G, 'type').values()))
        color_map = plt.cm.get_cmap('tab20', len(node_types))
        type_to_color = {t: color_map(i) for i, t in enumerate(node_types)}
        
        node_colors = [type_to_color[G.nodes[n]['type']] for n in G.nodes]
        
        # Scale node sizes based on salience
        saliences = nx.get_node_attributes(G, 'salience')
        node_sizes = [300 + 700 * saliences[n] for n in G.nodes]
        
        # Scale edge widths based on weight
        weights = nx.get_edge_attributes(G, 'weight')
        edge_widths = [0.5 + 2.0 * weights[(u, v)] for u, v in G.edges]
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, arrows=True, arrowsize=10)
        
        # Draw labels
        node_labels = {n: G.nodes[n]['name'] for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        # Draw edge labels
        edge_labels = {(u, v): G.edges[u, v]['type'] for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
        # Add legend for node types
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=type_to_color[t], markersize=8, label=t) 
                         for t in node_types]
        plt.legend(handles=legend_handles, loc='upper right')
        
        plt.axis('off')
        plt.title(self.name)
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Knowledge graph visualization saved to {output_path}")
        
        if show:
            plt.show()
            return fig
        else:
            plt.close(fig)
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the knowledge graph to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relationships": [rel.to_dict() for rel in self.relationships]
        }
    
    def to_json(self, pretty: bool = True) -> str:
        """
        Convert the knowledge graph to JSON.
        
        Args:
            pretty: Whether to format the JSON for readability
            
        Returns:
            str: JSON representation
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def save(self, output_path: Path) -> None:
        """
        Save the knowledge graph to a JSON file.
        
        Args:
            output_path: Path to save the file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        logger.info(f"Knowledge graph saved to {output_path}")
    
    @classmethod
    def load(cls, input_path: Path) -> 'KnowledgeGraph':
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            KnowledgeGraph: Loaded graph
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph = cls(name=data.get("name", "Knowledge Graph"))
        graph.metadata = data.get("metadata", graph.metadata)
        
        # Load entities
        for entity_data in data.get("entities", []):
            entity = Entity.from_dict(entity_data)
            graph.entities[entity.id] = entity
        
        # Load relationships
        for rel_data in data.get("relationships", []):
            rel = Relationship.from_dict(rel_data)
            graph.relationships.append(rel)
        
        logger.info(f"Loaded knowledge graph from {input_path} with {len(graph.entities)} entities and {len(graph.relationships)} relationships")
        return graph
    
    def export_to_graphml(self, output_path: Path) -> None:
        """
        Export the knowledge graph to GraphML format.
        
        Args:
            output_path: Path to save the file
        """
        if not self.nx_graph:
            self.to_networkx()
        
        # Create a copy of the graph for export
        export_graph = nx.DiGraph(name=self.name)
        
        # Add nodes (entities) with converted attributes
        for node, data in self.nx_graph.nodes(data=True):
            # Convert dictionary attributes to strings to avoid GraphML type errors
            export_data = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    export_data[k] = json.dumps(v)
                else:
                    export_data[k] = v
            
            export_graph.add_node(node, **export_data)
        
        # Add edges (relationships) with converted attributes
        for u, v, data in self.nx_graph.edges(data=True):
            # Convert dictionary attributes to strings
            export_data = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    export_data[k] = json.dumps(v)
                else:
                    export_data[k] = v
            
            export_graph.add_edge(u, v, **export_data)
        
        nx.write_graphml(export_graph, output_path)
        logger.info(f"Knowledge graph exported to GraphML at {output_path}")
    
    def export_to_cypher(self, output_path: Path) -> None:
        """
        Export the knowledge graph to Cypher statements for Neo4j.
        
        Args:
            output_path: Path to save the file
        """
        cypher_statements = []
        
        # Create constraints
        cypher_statements.append("// Create constraints and indexes")
        cypher_statements.append("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;")
        cypher_statements.append("")
        
        # Create nodes
        cypher_statements.append("// Create entities")
        for entity in self.entities.values():
            attrs = ", ".join([f"{k}: {json.dumps(v)}" for k, v in entity.attributes.items()])
            attrs_str = f", {attrs}" if attrs else ""
            
            stmt = (f"CREATE (e:{entity.type} {{id: '{entity.id}', name: {json.dumps(entity.name)}, "
                   f"salience: {entity.salience}{attrs_str}}});")
            cypher_statements.append(stmt)
        
        cypher_statements.append("")
        
        # Create relationships
        cypher_statements.append("// Create relationships")
        for rel in self.relationships:
            attrs = ", ".join([f"{k}: {json.dumps(v)}" for k, v in rel.attributes.items()])
            attrs_str = f", {attrs}" if attrs else ""
            
            stmt = (f"MATCH (source:Entity {{id: '{rel.source_id}'}}), (target:Entity {{id: '{rel.target_id}'}})\n"
                   f"CREATE (source)-[:{rel.type} {{weight: {rel.weight}{attrs_str}}}]->(target);")
            cypher_statements.append(stmt)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(cypher_statements))
        
        logger.info(f"Knowledge graph exported to Cypher at {output_path}")


class GraphBuilder:
    """
    Class for building knowledge graphs from text summaries.
    
    Provides methods for extracting entities and relationships from summaries
    and building a knowledge graph from them.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the graph builder.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or AppConfig()
        self.llm_provider = None
        self.output_dir = Path("./output")
        self.output_dir.mkdir(exist_ok=True)
    
    def build_graph_from_summaries(self, 
                                 segment_summaries: Dict[str, Dict[str, Any]], 
                                 segmentation_result: Optional[SegmentationResult] = None,
                                 language: Optional[str] = None) -> Result[KnowledgeGraph]:
        """
        Build a knowledge graph from segment summaries.
        
        Args:
            segment_summaries: Dictionary mapping segment IDs to their summaries
            segmentation_result: Optional segmentation result for additional context
            language: Language code
            
        Returns:
            Result[KnowledgeGraph]: Knowledge graph or error
        """
        language = language or self.config.language
        
        # Create a new knowledge graph
        graph = KnowledgeGraph(name=f"Knowledge Graph ({language})")
        
        # Maps segment IDs to lists of entity IDs
        segment_entities: Dict[str, List[str]] = {}
        
        # Process each summary
        for segment_id, summary in segment_summaries.items():
            # Get the corresponding segment if available
            segment = None
            if segmentation_result:
                segment = segmentation_result.get_segment(segment_id)
            
            # Extract entities and relationships
            extraction_result = self._extract_entities_and_relationships(segment_id, summary, segment, language)
            
            if not extraction_result.success:
                logger.warning(f"Failed to extract entities for segment {segment_id}: {extraction_result.error}")
                continue
            
            extraction_data = extraction_result.value
            segment_entities[segment_id] = []
            
            # Add entities to the graph
            for entity_data in extraction_data.get("entities", []):
                # Generate a deterministic ID if not provided
                if not entity_data.get("id"):
                    entity_data["id"] = str(uuid.uuid4())
                
                # Create entity and add source segment
                entity = Entity.from_dict(entity_data)
                entity.add_source_segment(segment_id)
                
                # Add to graph
                graph.add_entity(entity)
                segment_entities[segment_id].append(entity.id)
            
            # Add relationships to the graph
            for rel_data in extraction_data.get("relationships", []):
                # Create relationship and add source segment
                rel = Relationship.from_dict(rel_data)
                rel.add_source_segment(segment_id)
                
                # Add to graph
                graph.add_relationship(rel)
        
        # Add segment-to-segment relationships based on common entities
        if segmentation_result:
            self._add_segment_relationships(graph, segmentation_result, segment_entities)
        
        # Build NetworkX graph
        graph.to_networkx()
        
        return Result.ok(graph)
    
    def _extract_entities_and_relationships(self, 
                                         segment_id: str,
                                         summary: Dict[str, Any],
                                         segment: Optional[Segment] = None,
                                         language: str = "en") -> Result[Dict[str, Any]]:
        """
        Extract entities and relationships from a segment summary.
        
        Args:
            segment_id: ID of the segment
            summary: Summary of the segment
            segment: Optional segment object
            language: Language code
            
        Returns:
            Result[Dict[str, Any]]: Extraction results or error
        """
        # Initialize LLM provider if needed
        if not self.llm_provider:
            # Get LLM provider from LLMConfig
            llm_config = LLMConfig()
            provider_name = llm_config.provider
            model_name = llm_config.model
            
            provider_result = get_provider(provider_name, model_name)
            if not provider_result.success:
                return Result.fail(f"Failed to initialize LLM provider: {provider_result.error}")
            self.llm_provider = provider_result.value
        
        # Get appropriate prompt template
        prompt_template = ENTITY_EXTRACTION_PROMPTS.get(language, ENTITY_EXTRACTION_PROMPTS["en"])
        
        # Get segment text and type
        segment_text = ""
        segment_type = "unknown"
        
        if segment:
            segment_text = segment.text
            segment_type = segment.segment_type
        else:
            segment_text = summary.get("text", "")
            segment_type = summary.get("segment_type", "unknown")
        
        # Format key points as string
        key_points_list = summary.get("key_points", [])
        key_points = "\n".join([f"- {point}" for point in key_points_list])
        
        # Format keywords as string
        keywords_list = summary.get("keywords", [])
        keywords = ", ".join(keywords_list)
        
        # Get summary text
        summary_text = summary.get("summary", "")
        
        # Format prompt with summary information
        prompt = prompt_template.format(
            text=segment_text,
            segment_type=segment_type,
            segment_id=segment_id,
            summary=summary_text,
            key_points=key_points,
            keywords=keywords
        )
        
        # Call LLM to extract entities and relationships
        try:
            logger.info(f"Extracting entities from segment {segment_id} (type: {segment_type})")
            
            # Generate structured JSON response
            response = self.llm_provider.generate_json(
                prompt=prompt,
                json_schema=ENTITY_EXTRACTION_SCHEMA,
                temperature=0.3
            )
            
            if not response.success:
                logger.error(f"LLM entity extraction failed: {response.error}")
                return Result.fail(f"Failed to extract entities: {response.error}")
            
            extraction_data = response.value
            
            entity_count = len(extraction_data.get("entities", []))
            rel_count = len(extraction_data.get("relationships", []))
            
            logger.info(f"Extracted {entity_count} entities and {rel_count} relationships from segment {segment_id}")
            return Result.ok(extraction_data)
            
        except Exception as e:
            logger.error(f"Error during entity extraction: {str(e)}")
            return Result.fail(f"Failed to extract entities: {str(e)}")
    
    def _add_segment_relationships(self, 
                                graph: KnowledgeGraph, 
                                segmentation_result: SegmentationResult,
                                segment_entities: Dict[str, List[str]]) -> None:
        """
        Add segment-to-segment relationships based on common entities.
        
        Args:
            graph: Knowledge graph
            segmentation_result: Segmentation result
            segment_entities: Dictionary mapping segment IDs to lists of entity IDs
        """
        # Create entities for segments if they don't exist
        for segment in segmentation_result.segments:
            # Skip if no entities extracted from this segment
            if segment.id not in segment_entities:
                continue
            
            # Create segment entity if it doesn't exist
            segment_entity_id = f"segment_{segment.id}"
            if segment_entity_id not in graph.entities:
                title = segment.title or f"Segment {segment.id}"
                entity = Entity(
                    entity_id=segment_entity_id,
                    name=title,
                    entity_type="TextSegment",
                    attributes={
                        "segment_type": segment.segment_type,
                        "level": segment.level,
                        "text_length": len(segment.text)
                    },
                    salience=0.7,
                    source_segments=[segment.id]
                )
                graph.add_entity(entity)
            
            # Create relationships to contained entities
            for entity_id in segment_entities.get(segment.id, []):
                rel = Relationship(
                    source_id=segment_entity_id,
                    target_id=entity_id,
                    relationship_type="contains",
                    weight=0.9,
                    source_segments=[segment.id]
                )
                graph.add_relationship(rel)
            
            # Create parent-child relationships between segments
            if segment.parent_id:
                parent_entity_id = f"segment_{segment.parent_id}"
                # Only if parent entity exists
                if parent_entity_id in graph.entities:
                    rel = Relationship(
                        source_id=parent_entity_id,
                        target_id=segment_entity_id,
                        relationship_type="contains",
                        weight=1.0,
                        source_segments=[segment.parent_id, segment.id]
                    )
                    graph.add_relationship(rel)


def build_knowledge_graph(segment_summaries: Dict[str, Dict[str, Any]],
                         segmentation_result: Optional[SegmentationResult] = None,
                         language: Optional[str] = None,
                         config: Optional[AppConfig] = None) -> Result[KnowledgeGraph]:
    """
    Convenience function to build a knowledge graph from summaries.
    
    Args:
        segment_summaries: Dictionary mapping segment IDs to their summaries
        segmentation_result: Optional segmentation result for additional context
        language: Language code
        config: Optional application configuration
        
    Returns:
        Result[KnowledgeGraph]: Knowledge graph or error
    """
    builder = GraphBuilder(config)
    return builder.build_graph_from_summaries(segment_summaries, segmentation_result, language) 