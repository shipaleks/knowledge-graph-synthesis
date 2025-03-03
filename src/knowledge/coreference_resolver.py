"""
Coreference Resolution Module

This module provides functionality for resolving coreferences between entities
extracted from text, allowing the system to identify when different mentions
refer to the same entity.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass

from src.utils.result import Result
from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.config.llm_config import LLMConfig
from src.llm import get_provider
from src.knowledge.entity import Entity, EntityRegistry

# Configure logger
logger = get_logger(__name__)

# Coreference resolution prompts for different languages
COREFERENCE_PROMPTS = {
    "en": """
Analyze the following list of entities extracted from different text segments and determine which ones refer to the same concept.

Entities:
{entities}

For each group of potentially matching entities:

1. Determine if they are indeed the same entity (yes/no)
2. If yes, specify the canonical name for this entity
3. Explain the reason for the decision (lexical similarity, contextual similarity, semantic similarity)
4. Level of confidence in the decision (0-1)
5. Combined list of attributes from all mentions

Consider:
- Synonyms and spelling variations
- Hypernyms and hyponyms (genus-species relationships)
- Metonymy (designation of an object through a related concept)
- Anaphoric references (he, she, they, this, etc.)

Present the result in JSON format with array of entity groups, each having fields: entity_ids, canonical_name, merge_decision, reason, confidence, combined_attributes.
""",
    "ru": """
Проанализируйте следующий список сущностей, извлеченных из разных сегментов текста, и определите, какие из них относятся к одному и тому же концепту.

Сущности:
{entities}

Для каждой группы потенциально совпадающих сущностей:

1. Определите, действительно ли это одна и та же сущность (да/нет)
2. Если да, укажите каноническое имя для этой сущности
3. Объясните причину решения (лексическое сходство, контекстуальное сходство, семантическое сходство)
4. Уровень уверенности в решении (0-1)
5. Объединенный список атрибутов из всех упоминаний

Учитывайте:
- Синонимы и вариации написания
- Гиперонимы и гипонимы (родо-видовые отношения)
- Метонимию (обозначение объекта через связанное понятие)
- Анафорические ссылки (он, она, они, этот и т.д.)

Представьте результат в JSON-формате с массивом групп сущностей, каждая из которых имеет поля: entity_ids, canonical_name, merge_decision, reason, confidence, combined_attributes.
"""
}

# Schema for coreference resolution response validation
COREFERENCE_RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entity_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_ids": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "canonical_name": {"type": "string"},
                    "merge_decision": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                    "combined_attributes": {"type": "object"}
                },
                "required": ["entity_ids", "canonical_name", "merge_decision", "confidence"]
            }
        }
    },
    "required": ["entity_groups"]
}


@dataclass
class CoreferenceGroup:
    """
    Represents a group of entities that refer to the same concept.
    
    Contains information about the entities, the canonical representation,
    and the confidence in the coreference decision.
    """
    entity_ids: List[str]
    canonical_name: str
    merge_decision: bool
    reason: str
    confidence: float
    combined_attributes: Dict[str, Any]


class CoreferenceResolver:
    """
    Resolves coreferences between entities to identify when different
    mentions refer to the same entity.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the coreference resolver.
        
        Args:
            config: Application configuration
        """
        self.config = config or AppConfig()
        self.llm_provider = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def resolve_coreferences(self, 
                          entity_registry: EntityRegistry,
                          language: str = "en",
                          similarity_threshold: float = 0.7,
                          batch_size: int = 20) -> Result[EntityRegistry]:
        """
        Resolve coreferences in the given entity registry.
        
        Args:
            entity_registry: Registry containing entities to resolve
            language: Language of the entities
            similarity_threshold: Threshold for string similarity to consider potential matches
            batch_size: Number of entities to process in each batch
            
        Returns:
            Result containing a new entity registry with resolved coreferences
        """
        if entity_registry.count() == 0:
            return Result.ok(EntityRegistry())
        
        # Create a copy of the registry to work with
        resolved_registry = EntityRegistry()
        
        # Get all entities
        entities = entity_registry.all()
        
        # Group entities by type for more efficient processing
        entities_by_type: Dict[str, List[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)
        
        logger.info(f"Resolving coreferences for {entity_registry.count()} entities across {len(entities_by_type)} types")
        
        # Process each type separately
        for entity_type, type_entities in entities_by_type.items():
            logger.info(f"Processing {len(type_entities)} entities of type {entity_type}")
            
            # Skip if only one entity of this type
            if len(type_entities) <= 1:
                for entity in type_entities:
                    resolved_registry.add(entity)
                continue
            
            # Find potential matches based on name similarity
            potential_matches = self._find_potential_matches(type_entities, similarity_threshold)
            
            if not potential_matches:
                # No potential matches, add all entities as is
                for entity in type_entities:
                    resolved_registry.add(entity)
                continue
            
            # Process potential matches in batches
            for i in range(0, len(potential_matches), batch_size):
                batch = potential_matches[i:i+batch_size]
                
                # Resolve coreferences in this batch
                result = self._resolve_batch(batch, type_entities, language)
                
                if not result.success:
                    logger.error(f"Failed to resolve batch: {result.error}")
                    continue
                
                coreference_groups = result.value
                
                # Apply coreference resolutions
                self._apply_coreference_resolutions(
                    coreference_groups,
                    type_entities,
                    resolved_registry
                )
            
            # Add any remaining entities that weren't in any groups
            processed_ids = set()
            for group in potential_matches:
                processed_ids.update(group)
            
            for entity in type_entities:
                if entity.id not in processed_ids:
                    resolved_registry.add(entity)
        
        logger.info(f"Coreference resolution complete. Original entities: {entity_registry.count()}, Resolved entities: {resolved_registry.count()}")
        
        return Result.ok(resolved_registry)
    
    def resolve_entity_group(self,
                          entities: List[Entity],
                          language: str = "en") -> Result[List[CoreferenceGroup]]:
        """
        Resolve coreferences in a group of entities.
        
        Args:
            entities: List of entities to check for coreferences
            language: Language of the entities
            
        Returns:
            Result containing list of coreference groups
        """
        if len(entities) <= 1:
            return Result.ok([])
        
        # Create matching groups
        matching_group = set(entity.id for entity in entities)
        
        # Resolve the group
        return self._resolve_batch([matching_group], entities, language)
    
    def _find_potential_matches(self, 
                              entities: List[Entity], 
                              similarity_threshold: float) -> List[Set[str]]:
        """
        Find potential entity matches based on name similarity.
        
        Args:
            entities: List of entities to check
            similarity_threshold: Threshold for string similarity
            
        Returns:
            List of sets of entity IDs that potentially refer to the same entity
        """
        from difflib import SequenceMatcher
        
        potential_matches: List[Set[str]] = []
        processed_entities = set()
        
        # For each entity, find other entities with similar names
        for i, entity1 in enumerate(entities):
            if entity1.id in processed_entities:
                continue
            
            current_group = {entity1.id}
            
            for j, entity2 in enumerate(entities):
                if i == j or entity2.id in processed_entities:
                    continue
                
                # Calculate string similarity
                similarity = SequenceMatcher(None, 
                                          entity1.name.lower(), 
                                          entity2.name.lower()).ratio()
                
                if similarity >= similarity_threshold:
                    current_group.add(entity2.id)
            
            # Only add groups with more than one entity
            if len(current_group) > 1:
                potential_matches.append(current_group)
                processed_entities.update(current_group)
        
        return potential_matches
    
    def _resolve_batch(self,
                     entity_groups: List[Set[str]],
                     all_entities: List[Entity],
                     language: str) -> Result[List[CoreferenceGroup]]:
        """
        Resolve coreferences in a batch of entity groups using LLM.
        
        Args:
            entity_groups: List of sets of entity IDs to check
            all_entities: List of all entities
            language: Language of the entities
            
        Returns:
            Result containing list of coreference groups
        """
        # Initialize LLM provider if needed
        if not self.llm_provider:
            result = self._initialize_llm_provider()
            if not result.success:
                return Result.fail(result.error)
        
        # Create entity ID to entity mapping
        entity_map = {entity.id: entity for entity in all_entities}
        
        # Format entity groups for the prompt
        entities_text = ""
        for i, group in enumerate(entity_groups):
            entities_text += f"Group {i+1}:\n"
            for entity_id in group:
                entity = entity_map.get(entity_id)
                if entity:
                    attributes_str = ", ".join(f"{k}: {v}" for k, v in entity.attributes.items())
                    entities_text += (
                        f"  - ID: {entity.id}\n"
                        f"    Name: {entity.name}\n"
                        f"    Type: {entity.entity_type}\n"
                        f"    Context: {entity.context if entity.context else 'N/A'}\n"
                        f"    Attributes: {attributes_str if attributes_str else 'N/A'}\n"
                        f"    Confidence: {entity.confidence}\n\n"
                    )
        
        # Get prompt for language
        if language not in COREFERENCE_PROMPTS:
            return Result.fail(f"Language {language} not supported for coreference resolution")
        
        prompt = COREFERENCE_PROMPTS[language].format(entities=entities_text)
        
        try:
            # Call LLM with the prompt
            response = self.llm_provider.generate_json(
                prompt=prompt,
                json_schema=COREFERENCE_RESOLUTION_SCHEMA,
                temperature=0.2
            )
            
            if not response.success:
                return Result.fail(f"Failed to generate coreference resolutions: {response.error}")
            
            # Parse the response
            result_data = response.value
            
            # Convert to CoreferenceGroup objects
            coreference_groups = []
            for group_data in result_data.get("entity_groups", []):
                # Skip if merge decision is false
                if not group_data.get("merge_decision", False):
                    continue
                
                group = CoreferenceGroup(
                    entity_ids=group_data.get("entity_ids", []),
                    canonical_name=group_data.get("canonical_name", ""),
                    merge_decision=group_data.get("merge_decision", False),
                    reason=group_data.get("reason", ""),
                    confidence=group_data.get("confidence", 0.0),
                    combined_attributes=group_data.get("combined_attributes", {})
                )
                coreference_groups.append(group)
            
            return Result.ok(coreference_groups)
            
        except Exception as e:
            logger.error(f"Error resolving coreferences: {str(e)}")
            return Result.fail(f"Error resolving coreferences: {str(e)}")
    
    def _apply_coreference_resolutions(self,
                                    coreference_groups: List[CoreferenceGroup],
                                    entities: List[Entity],
                                    target_registry: EntityRegistry) -> None:
        """
        Apply coreference resolution results to the target registry.
        
        Args:
            coreference_groups: List of coreference groups
            entities: List of all entities
            target_registry: Target registry to add resolved entities to
        """
        # Create entity ID to entity mapping
        entity_map = {entity.id: entity for entity in entities}
        
        # Track processed entities
        processed_ids = set()
        
        # Process each coreference group
        for group in coreference_groups:
            # Skip groups that shouldn't be merged
            if not group.merge_decision or len(group.entity_ids) < 2:
                continue
            
            # Get entities in this group
            group_entities = [entity_map.get(entity_id) for entity_id in group.entity_ids 
                             if entity_id in entity_map]
            
            # Skip if no valid entities
            if not group_entities:
                continue
            
            # Mark these entities as processed
            processed_ids.update(group.entity_ids)
            
            # Merge entities
            merged_entity = self._merge_entities(
                group_entities,
                group.canonical_name,
                group.combined_attributes,
                group.confidence
            )
            
            # Add merged entity to target registry
            target_registry.add(merged_entity)
        
        # Add remaining entities that weren't in any groups
        for entity in entities:
            if entity.id not in processed_ids:
                target_registry.add(entity)
    
    def _merge_entities(self,
                      entities: List[Entity],
                      canonical_name: str,
                      combined_attributes: Dict[str, Any],
                      confidence: float) -> Entity:
        """
        Merge multiple entities into a single entity.
        
        Args:
            entities: List of entities to merge
            canonical_name: Canonical name for the merged entity
            combined_attributes: Combined attributes from all entities
            confidence: Confidence in the merge decision
            
        Returns:
            Merged entity
        """
        if not entities:
            raise ValueError("No entities to merge")
        
        # Start with the first entity
        merged = entities[0]
        
        # Merge with the rest
        for entity in entities[1:]:
            merged = merged.merge(entity)
        
        # Override with canonical name if provided
        if canonical_name:
            merged.name = canonical_name
        
        # Add or update attributes
        for key, value in combined_attributes.items():
            merged.attributes[key] = value
        
        # Update metadata to include coreference info
        merged.metadata["coreference_resolution"] = {
            "original_ids": [entity.id for entity in entities],
            "original_names": [entity.name for entity in entities],
            "resolution_confidence": confidence
        }
        
        # Set confidence
        merged.confidence = confidence
        
        return merged
    
    def _initialize_llm_provider(self) -> Result[bool]:
        """
        Initialize the LLM provider.
        
        Returns:
            Result indicating success or failure
        """
        try:
            # Get LLM configuration
            llm_config = LLMConfig()
            provider_name = llm_config.provider
            model_name = llm_config.model
            
            # Initialize LLM provider
            provider_result = get_provider(provider_name, model_name)
            
            if not provider_result.success:
                return Result.fail(f"Failed to initialize LLM provider: {provider_result.error}")
            
            self.llm_provider = provider_result.value
            logger.info(f"Initialized {provider_name} provider with model {model_name}")
            
            return Result.ok(True)
            
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {str(e)}")
            return Result.fail(f"Error initializing LLM provider: {str(e)}")


def resolve_coreferences(
    entity_registry: EntityRegistry,
    language: str = "en",
    similarity_threshold: float = 0.7,
    batch_size: int = 20,
    config: Optional[AppConfig] = None
) -> Result[EntityRegistry]:
    """
    Resolve coreferences in the given entity registry.
    
    Args:
        entity_registry: Registry containing entities to resolve
        language: Language of the entities
        similarity_threshold: Threshold for string similarity to consider potential matches
        batch_size: Number of entities to process in each batch
        config: Application configuration
        
    Returns:
        Result containing a new entity registry with resolved coreferences
    """
    resolver = CoreferenceResolver(config)
    return resolver.resolve_coreferences(
        entity_registry,
        language,
        similarity_threshold,
        batch_size
    ) 