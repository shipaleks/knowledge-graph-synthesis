"""
Relationship Module

This module provides the Relationship class and related utilities for working with
relationships between entities extracted from text.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from src.knowledge.entity import Entity


@dataclass
class Relationship:
    """
    Represents a relationship between two entities extracted from text.
    
    A relationship connects two entities with a specific type and direction,
    capturing how they relate to each other in the text.
    """
    source_entity: str  # ID of the source entity
    target_entity: str  # ID of the target entity
    relation_type: str  # type of relationship (is-a, part-of, causes, etc.)
    context: Optional[str] = None  # source text context where relationship was found
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False  # whether the relationship applies in both directions
    strength: float = 1.0  # strength of the relationship between 0 and 1
    confidence: float = 1.0  # confidence score between 0 and 1
    
    def __post_init__(self):
        """Validate relationship after initialization."""
        if not self.source_entity:
            raise ValueError("Source entity ID cannot be empty")
        if not self.target_entity:
            raise ValueError("Target entity ID cannot be empty")
        if not self.relation_type:
            raise ValueError("Relationship type cannot be empty")
        if self.strength < 0 or self.strength > 1:
            raise ValueError("Strength must be between 0 and 1")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def reverse(self) -> 'Relationship':
        """
        Create a new relationship with source and target swapped.
        
        Returns:
            Relationship: A new reversed relationship
        """
        return Relationship(
            source_entity=self.target_entity,
            target_entity=self.source_entity,
            relation_type=self.relation_type,
            context=self.context,
            attributes=self.attributes.copy(),
            metadata=self.metadata.copy(),
            bidirectional=self.bidirectional,
            strength=self.strength,
            confidence=self.confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert relationship to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the relationship
        """
        return {
            "id": self.id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation_type": self.relation_type,
            "context": self.context,
            "attributes": self.attributes,
            "metadata": self.metadata,
            "bidirectional": self.bidirectional,
            "strength": self.strength,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """
        Create a relationship from a dictionary.
        
        Args:
            data: Dictionary representation of a relationship
            
        Returns:
            Relationship: Relationship instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source_entity=data["source_entity"],
            target_entity=data["target_entity"],
            relation_type=data["relation_type"],
            context=data.get("context"),
            attributes=data.get("attributes", {}),
            metadata=data.get("metadata", {}),
            bidirectional=data.get("bidirectional", False),
            strength=data.get("strength", 1.0),
            confidence=data.get("confidence", 1.0)
        )
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Convert relationship to JSON string.
        
        Args:
            pretty: Whether to format the JSON for readability
            
        Returns:
            str: JSON representation of the relationship
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Relationship':
        """
        Create a relationship from a JSON string.
        
        Args:
            json_str: JSON string representation of a relationship
            
        Returns:
            Relationship: Relationship instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class RelationshipRegistry:
    """
    Manages a collection of relationships with functionality for retrieval and manipulation.
    
    This class provides methods for adding, finding, and removing relationships,
    as well as serializing and deserializing the collection.
    """
    
    def __init__(self):
        """Initialize an empty relationship registry."""
        self._relationships: Dict[str, Relationship] = {}
        self._source_index: Dict[str, Set[str]] = {}  # source entity ID -> relationship IDs
        self._target_index: Dict[str, Set[str]] = {}  # target entity ID -> relationship IDs
        self._type_index: Dict[str, Set[str]] = {}    # relation type -> relationship IDs
    
    def add(self, relationship: Relationship) -> str:
        """
        Add a relationship to the registry.
        
        Args:
            relationship: The relationship to add
            
        Returns:
            str: The ID of the added relationship
        """
        # Store the relationship
        self._relationships[relationship.id] = relationship
        
        # Update indices
        if relationship.source_entity not in self._source_index:
            self._source_index[relationship.source_entity] = set()
        self._source_index[relationship.source_entity].add(relationship.id)
        
        if relationship.target_entity not in self._target_index:
            self._target_index[relationship.target_entity] = set()
        self._target_index[relationship.target_entity].add(relationship.id)
        
        if relationship.relation_type not in self._type_index:
            self._type_index[relationship.relation_type] = set()
        self._type_index[relationship.relation_type].add(relationship.id)
        
        return relationship.id
    
    def get(self, relationship_id: str) -> Optional[Relationship]:
        """
        Retrieve a relationship by ID.
        
        Args:
            relationship_id: ID of the relationship to retrieve
            
        Returns:
            Optional[Relationship]: The relationship if found, None otherwise
        """
        return self._relationships.get(relationship_id)
    
    def find_by_source(self, source_entity_id: str) -> List[Relationship]:
        """
        Find all relationships with the given source entity.
        
        Args:
            source_entity_id: ID of the source entity
            
        Returns:
            List[Relationship]: List of relationships with the given source
        """
        if source_entity_id not in self._source_index:
            return []
        
        return [self._relationships[rel_id] for rel_id in self._source_index[source_entity_id]]
    
    def find_by_target(self, target_entity_id: str) -> List[Relationship]:
        """
        Find all relationships with the given target entity.
        
        Args:
            target_entity_id: ID of the target entity
            
        Returns:
            List[Relationship]: List of relationships with the given target
        """
        if target_entity_id not in self._target_index:
            return []
        
        return [self._relationships[rel_id] for rel_id in self._target_index[target_entity_id]]
    
    def find_by_entity(self, entity_id: str) -> List[Relationship]:
        """
        Find all relationships involving the given entity (as source or target).
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            List[Relationship]: List of relationships involving the entity
        """
        result = []
        
        if entity_id in self._source_index:
            for rel_id in self._source_index[entity_id]:
                result.append(self._relationships[rel_id])
                
        if entity_id in self._target_index:
            for rel_id in self._target_index[entity_id]:
                if rel_id not in self._source_index.get(entity_id, set()):
                    result.append(self._relationships[rel_id])
        
        return result
    
    def find_by_type(self, relation_type: str) -> List[Relationship]:
        """
        Find all relationships of the given type.
        
        Args:
            relation_type: Type of relationship to find
            
        Returns:
            List[Relationship]: List of relationships of the given type
        """
        if relation_type not in self._type_index:
            return []
        
        return [self._relationships[rel_id] for rel_id in self._type_index[relation_type]]
    
    def remove(self, relationship_id: str) -> bool:
        """
        Remove a relationship from the registry.
        
        Args:
            relationship_id: ID of the relationship to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if relationship_id not in self._relationships:
            return False
        
        relationship = self._relationships[relationship_id]
        
        # Remove from indices
        if relationship.source_entity in self._source_index:
            self._source_index[relationship.source_entity].discard(relationship_id)
            if not self._source_index[relationship.source_entity]:
                del self._source_index[relationship.source_entity]
                
        if relationship.target_entity in self._target_index:
            self._target_index[relationship.target_entity].discard(relationship_id)
            if not self._target_index[relationship.target_entity]:
                del self._target_index[relationship.target_entity]
                
        if relationship.relation_type in self._type_index:
            self._type_index[relationship.relation_type].discard(relationship_id)
            if not self._type_index[relationship.relation_type]:
                del self._type_index[relationship.relation_type]
        
        # Remove from storage
        del self._relationships[relationship_id]
        
        return True
    
    def clear(self) -> None:
        """Clear all relationships from the registry."""
        self._relationships.clear()
        self._source_index.clear()
        self._target_index.clear()
        self._type_index.clear()
    
    def all(self) -> List[Relationship]:
        """
        Get all relationships in the registry.
        
        Returns:
            List[Relationship]: List of all relationships
        """
        return list(self._relationships.values())
    
    def count(self) -> int:
        """
        Get the number of relationships in the registry.
        
        Returns:
            int: Number of relationships
        """
        return len(self._relationships)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the registry
        """
        return {
            "relationships": [r.to_dict() for r in self._relationships.values()]
        }
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Convert registry to a JSON string.
        
        Args:
            pretty: Whether to format the JSON for readability
            
        Returns:
            str: JSON representation of the registry
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipRegistry':
        """
        Create a registry from a dictionary.
        
        Args:
            data: Dictionary representation of a registry
            
        Returns:
            RelationshipRegistry: RelationshipRegistry instance
        """
        registry = cls()
        
        for rel_data in data.get("relationships", []):
            relationship = Relationship.from_dict(rel_data)
            registry.add(relationship)
            
        return registry
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RelationshipRegistry':
        """
        Create a registry from a JSON string.
        
        Args:
            json_str: JSON string representation of a registry
            
        Returns:
            RelationshipRegistry: RelationshipRegistry instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


def create_relationship(
    source_entity: str,
    target_entity: str,
    relation_type: str,
    context: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    bidirectional: bool = False,
    strength: float = 1.0,
    confidence: float = 1.0
) -> Relationship:
    """
    Create a new relationship between entities.
    
    Args:
        source_entity: ID of the source entity
        target_entity: ID of the target entity
        relation_type: Type of relationship
        context: Text context where the relationship was found
        attributes: Additional attributes of the relationship
        metadata: Additional metadata for the relationship
        bidirectional: Whether the relationship is bidirectional
        strength: Strength of the relationship (0-1)
        confidence: Confidence in the relationship (0-1)
        
    Returns:
        Relationship: A new relationship instance
    """
    return Relationship(
        source_entity=source_entity,
        target_entity=target_entity,
        relation_type=relation_type,
        context=context,
        attributes=attributes or {},
        metadata=metadata or {},
        bidirectional=bidirectional,
        strength=strength,
        confidence=confidence
    ) 