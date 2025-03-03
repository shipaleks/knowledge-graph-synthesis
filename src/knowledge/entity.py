"""
Entity Module

This module provides the Entity class and related utilities for working with
entities extracted from text.
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field


@dataclass
class Entity:
    """
    Represents an entity extracted from text.
    
    An entity is a person, place, organization, concept, or other named item
    that can be identified in text.
    """
    name: str
    entity_type: str  # person, organization, concept, location, etc.
    context: Optional[str] = None  # source text context where entity was found
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # confidence score between 0 and 1
    
    def __post_init__(self):
        """Validate entity after initialization."""
        if not self.name:
            raise ValueError("Entity name cannot be empty")
        if not self.entity_type:
            raise ValueError("Entity type cannot be empty")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def merge(self, other: 'Entity') -> 'Entity':
        """
        Merge this entity with another entity.
        
        Args:
            other: Another entity to merge with
            
        Returns:
            Entity: A new merged entity
        """
        # Decide which name to use (prefer self's name if confidence is higher)
        name = self.name if self.confidence >= other.confidence else other.name
        
        # Use the higher confidence value
        confidence = max(self.confidence, other.confidence)
        
        # Combine contexts if they are different
        if self.context and other.context and self.context != other.context:
            context = f"{self.context}\n{other.context}"
        else:
            context = self.context or other.context
        
        # Merge attributes
        attributes = self.attributes.copy()
        for key, value in other.attributes.items():
            if key not in attributes:
                attributes[key] = value
            elif isinstance(attributes[key], list) and isinstance(value, list):
                # Combine lists without duplicates
                combined = attributes[key] + [v for v in value if v not in attributes[key]]
                attributes[key] = combined
            elif attributes[key] != value:
                # If conflicting non-list values, create a list with both values
                attributes[key] = [attributes[key], value]
        
        # Merge metadata
        metadata = {**other.metadata, **self.metadata}  # self's metadata takes precedence
        
        return Entity(
            name=name,
            entity_type=self.entity_type,  # Assuming entities being merged have the same type
            context=context,
            id=self.id,  # Keep the original ID
            attributes=attributes,
            metadata=metadata,
            confidence=confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entity to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entity
        """
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "context": self.context,
            "attributes": self.attributes,
            "metadata": self.metadata,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an entity from a dictionary.
        
        Args:
            data: Dictionary representation of an entity
            
        Returns:
            Entity: Entity instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            entity_type=data["entity_type"],
            context=data.get("context"),
            attributes=data.get("attributes", {}),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0)
        )
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Convert entity to JSON string.
        
        Args:
            pretty: Whether to format the JSON for readability
            
        Returns:
            str: JSON representation of the entity
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Entity':
        """
        Create an entity from a JSON string.
        
        Args:
            json_str: JSON string representation of an entity
            
        Returns:
            Entity: Entity instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class EntityRegistry:
    """Registry for managing and deduplicating entities."""
    
    def __init__(self):
        """Initialize an empty registry."""
        self.entities: Dict[str, Entity] = {}
        self.name_to_id: Dict[str, List[str]] = {}  # Map normalized names to entity IDs
    
    def add(self, entity: Entity) -> str:
        """
        Add an entity to the registry.
        
        Args:
            entity: Entity to add
            
        Returns:
            str: ID of the added or merged entity
        """
        # First check if we already have this exact entity by ID
        if entity.id in self.entities:
            # If so, merge them
            self.entities[entity.id] = self.entities[entity.id].merge(entity)
            return entity.id
        
        # Normalize the entity name for fuzzy matching
        norm_name = self._normalize_name(entity.name)
        
        # Check if we have entities with similar names
        if norm_name in self.name_to_id:
            # Check each entity with this normalized name
            for existing_id in self.name_to_id[norm_name]:
                existing = self.entities[existing_id]
                
                # If they have the same type, merge them
                if existing.entity_type == entity.entity_type:
                    self.entities[existing_id] = existing.merge(entity)
                    return existing_id
        
        # If no match was found, add as a new entity
        self.entities[entity.id] = entity
        
        # Update the name mapping
        if norm_name not in self.name_to_id:
            self.name_to_id[norm_name] = []
        self.name_to_id[norm_name].append(entity.id)
        
        return entity.id
    
    def get(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by its ID.
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Optional[Entity]: Entity or None if not found
        """
        return self.entities.get(entity_id)
    
    def find_by_name(self, name: str) -> List[Entity]:
        """
        Find entities with a given name.
        
        Args:
            name: Name to search for
            
        Returns:
            List[Entity]: List of matching entities
        """
        norm_name = self._normalize_name(name)
        if norm_name not in self.name_to_id:
            return []
        
        return [self.entities[entity_id] for entity_id in self.name_to_id[norm_name]]
    
    def find_by_type(self, entity_type: str) -> List[Entity]:
        """
        Find entities of a given type.
        
        Args:
            entity_type: Entity type to search for
            
        Returns:
            List[Entity]: List of matching entities
        """
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def remove(self, entity_id: str) -> bool:
        """
        Remove an entity from the registry.
        
        Args:
            entity_id: ID of the entity to remove
            
        Returns:
            bool: True if entity was removed, False if not found
        """
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        norm_name = self._normalize_name(entity.name)
        
        # Remove from name mapping
        if norm_name in self.name_to_id:
            self.name_to_id[norm_name].remove(entity_id)
            if not self.name_to_id[norm_name]:
                del self.name_to_id[norm_name]
        
        # Remove from entities dictionary
        del self.entities[entity_id]
        
        return True
    
    def clear(self) -> None:
        """Clear all entities from the registry."""
        self.entities = {}
        self.name_to_id = {}
    
    def all(self) -> List[Entity]:
        """
        Get all entities in the registry.
        
        Returns:
            List[Entity]: List of all entities
        """
        return list(self.entities.values())
    
    def count(self) -> int:
        """
        Get the number of entities in the registry.
        
        Returns:
            int: Number of entities
        """
        return len(self.entities)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert registry to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the registry
        """
        return {
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()}
        }
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Convert registry to JSON string.
        
        Args:
            pretty: Whether to format the JSON for readability
            
        Returns:
            str: JSON representation of the registry
        """
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRegistry':
        """
        Create a registry from a dictionary.
        
        Args:
            data: Dictionary representation of a registry
            
        Returns:
            EntityRegistry: Registry instance
        """
        registry = cls()
        entities_data = data.get("entities", {})
        
        for entity_id, entity_data in entities_data.items():
            entity = Entity.from_dict(entity_data)
            registry.entities[entity_id] = entity
            
            norm_name = registry._normalize_name(entity.name)
            if norm_name not in registry.name_to_id:
                registry.name_to_id[norm_name] = []
            registry.name_to_id[norm_name].append(entity_id)
        
        return registry
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EntityRegistry':
        """
        Create a registry from a JSON string.
        
        Args:
            json_str: JSON string representation of a registry
            
        Returns:
            EntityRegistry: Registry instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize an entity name for fuzzy matching.
        
        Args:
            name: Entity name
            
        Returns:
            str: Normalized name
        """
        # Simple normalization: lowercase and strip whitespace
        # This could be extended with more sophisticated methods
        return name.lower().strip()


def create_entity(
    name: str,
    entity_type: str,
    context: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0
) -> Entity:
    """
    Create a new entity.
    
    Args:
        name: Entity name
        entity_type: Entity type
        context: Source text context (optional)
        attributes: Entity attributes (optional)
        metadata: Entity metadata (optional)
        confidence: Confidence score between 0 and 1 (default: 1.0)
        
    Returns:
        Entity: New entity instance
    """
    return Entity(
        name=name,
        entity_type=entity_type,
        context=context,
        attributes=attributes or {},
        metadata=metadata or {},
        confidence=confidence
    ) 