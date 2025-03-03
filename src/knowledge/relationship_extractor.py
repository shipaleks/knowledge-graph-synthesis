"""
Relationship Extraction Module

This module provides functionality for extracting relationships between entities
from text using LLM-based approaches.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.language import is_language_supported
from src.config.app_config import AppConfig
from src.config.llm_config import LLMConfig
from src.llm import get_provider, LLMProvider
from src.text_processing.segment import Segment, SegmentationResult
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.relationship import Relationship, RelationshipRegistry, create_relationship

# Configure logger
logger = get_logger(__name__)

# Relationship extraction prompts for different languages
RELATIONSHIP_EXTRACTION_PROMPTS = {
    "en": """
Determine the relationships between the following entities in this text, considering the domain of "{domain_type}".

Text: {text}

Entities:
{entities}

Focus on the following relationship types relevant to this domain:
{relation_types}

For each pair of entities, determine:

1. Whether there is a direct relationship (yes/no)
2. Type of relationship (from the list above or suggest your own if necessary)
3. Direction (from which entity to which)
4. Context of the relationship (quote from text)
5. Strength of the relationship (0-1)
6. Confidence in the relationship's existence (0-1)

Do not indicate trivial or overly general relationships.
Consider both explicit and implicit relationships, but distinguish them by level of confidence.

Present the result in JSON format with array of relationships, each having fields: source_entity, target_entity, relation_type, context, bidirectional, strength, confidence.
""",
    "ru": """
Определите отношения между следующими сущностями в данном тексте, учитывая предметную область "{domain_type}".

Текст: {text}

Сущности:
{entities}

Фокусируйтесь на следующих типах отношений, релевантных для данной области:
{relation_types}

Для каждой пары сущностей определите:

1. Есть ли прямое отношение (да/нет)
2. Тип отношения (из списка выше или предложите свой, если необходимо)
3. Направление (от какой сущности к какой)
4. Контекст отношения (цитата из текста)
5. Сила отношения (0-1)
6. Уверенность в наличии отношения (0-1)

Не указывайте тривиальные или слишком общие отношения.
Учитывайте как явные, так и неявные отношения, но различайте их по уровню уверенности.

Представьте результат в JSON-формате с массивом отношений, каждое из которых имеет поля: source_entity, target_entity, relation_type, context, bidirectional, strength, confidence.
"""
}

# Default relationship types if not provided
DEFAULT_RELATION_TYPES = {
    "en": [
        "is_a", "part_of", "has_part", "causes", "caused_by",
        "related_to", "similar_to", "opposite_of", "depends_on",
        "used_for", "derived_from", "instance_of", "precedes",
        "follows", "enables", "prevents", "affects"
    ],
    "ru": [
        "является", "часть_от", "содержит", "вызывает", "вызвано_чем",
        "связан_с", "похож_на", "противоположен", "зависит_от",
        "используется_для", "получен_из", "экземпляр", "предшествует",
        "следует_за", "обеспечивает", "предотвращает", "влияет_на"
    ]
}

# Schema for relationship extraction response validation
RELATIONSHIP_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity": {"type": "string"},
                    "target_entity": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "context": {"type": "string"},
                    "bidirectional": {"type": "boolean"},
                    "strength": {"type": "number"},
                    "confidence": {"type": "number"}
                },
                "required": ["source_entity", "target_entity", "relation_type"]
            }
        }
    },
    "required": ["relationships"]
}


class RelationshipExtractor:
    """
    Extracts relationships between entities from text using LLM.
    """
    
    def __init__(self, 
                config: Optional[AppConfig] = None, 
                domain_type: Optional[str] = None, 
                relation_types: Optional[List[str]] = None):
        """
        Initialize the relationship extractor.
        
        Args:
            config: Application configuration
            domain_type: Domain type for extraction context
            relation_types: Types of relations to focus on
        """
        self.config = config or AppConfig()
        self.domain_type = domain_type or "Knowledge Graph Construction"
        self.relation_types = relation_types
        self.llm_provider = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def extract_from_text(self, 
                         text: str, 
                         entities: List[Entity],
                         language: Optional[str] = None) -> Result[List[Relationship]]:
        """
        Extract relationships from text using the provided entities.
        
        Args:
            text: Text to extract relationships from
            entities: List of entities to find relationships between
            language: Language of the text (auto-detected if not provided)
            
        Returns:
            Result containing list of extracted relationships or error
        """
        if not text:
            return Result.fail("Text cannot be empty")
        
        if not entities:
            return Result.fail("No entities provided for relationship extraction")
        
        # Detect language if not provided
        if language is None:
            from src.utils.language import detect_language
            language_result = detect_language(text)
            if not language_result.success:
                return Result.fail(f"Failed to detect language: {language_result.error}")
            language = language_result.value
        
        # Check if language is supported
        if not is_language_supported(language):
            closest = get_closest_supported_language(language)
            logger.warning(f"Language {language} not supported. Using {closest} instead.")
            language = closest
        
        logger.info(f"Extracting relationships from text of length {len(text)} in {language}")
        
        # Create entity mapping for prompt
        entity_text = "\n".join([f"{idx+1}. {entity.name} (ID: {entity.id}, Type: {entity.entity_type})" 
                              for idx, entity in enumerate(entities)])
        
        # Get relation types for language
        relation_types = self._get_relation_types_for_language(language)
        
        # Extract relationships using LLM
        return self._extract_with_llm(text, entities, entity_text, language)
    
    def extract_from_segment(self,
                           segment: Segment,
                           entities: List[Entity],
                           language: Optional[str] = None) -> Result[List[Relationship]]:
        """
        Extract relationships from a segment using the provided entities.
        
        Args:
            segment: Segment to extract relationships from
            entities: List of entities to find relationships between
            language: Language of the segment (auto-detected if not provided)
            
        Returns:
            Result containing list of extracted relationships or error
        """
        if not segment.text:
            return Result.fail("Segment text cannot be empty")
        
        if not entities:
            return Result.fail("No entities provided for relationship extraction")
        
        # Filter entities to those that appear in this segment
        segment_entities = [entity for entity in entities 
                           if entity.context and segment.text.find(entity.context) >= 0]
        
        if not segment_entities:
            logger.warning(f"No entities found in segment {segment.id}")
            return Result.ok([])
        
        return self.extract_from_text(segment.text, segment_entities, language)
    
    def extract_from_segments(self,
                             segmentation_result: SegmentationResult,
                             entity_registry: EntityRegistry,
                             language: Optional[str] = None) -> Result[RelationshipRegistry]:
        """
        Extract relationships from all segments in the segmentation result.
        
        Args:
            segmentation_result: Result of text segmentation
            entity_registry: Registry containing all extracted entities
            language: Language of the segments (auto-detected if not provided)
            
        Returns:
            Result containing registry of extracted relationships or error
        """
        if not segmentation_result.segments:
            return Result.fail("No segments provided for relationship extraction")
        
        if entity_registry.count() == 0:
            return Result.fail("No entities provided for relationship extraction")
        
        all_entities = entity_registry.all()
        
        relationship_registry = RelationshipRegistry()
        
        # Extract relationships from each segment
        for segment in segmentation_result.segments:
            logger.info(f"Extracting relationships from segment {segment.id}")
            
            # Extract relationships for this segment
            result = self.extract_from_segment(segment, all_entities, language)
            
            if not result.success:
                logger.warning(f"Failed to extract relationships from segment {segment.id}: {result.error}")
                continue
            
            # Add relationships to registry
            for relationship in result.value:
                relationship_registry.add(relationship)
        
        return Result.ok(relationship_registry)
    
    def _extract_with_llm(self, text: str, entities: List[Entity], entity_text: str, language: str) -> Result[List[Relationship]]:
        """
        Extract relationships from text using LLM.
        
        Args:
            text: Text to analyze
            entities: List of entities to find relationships between
            entity_text: Formatted text representation of entities for prompt
            language: Language of the text
            
        Returns:
            Result containing list of extracted relationships or error
        """
        try:
            # Initialize LLM provider if needed
            if not self.llm_provider:
                # Get LLM configuration
                llm_config = LLMConfig()
                provider_name = llm_config.provider
                model_name = llm_config.model
                
                # Initialize LLM provider
                provider_result = get_provider(provider_name, model_name)
                
                if not provider_result.success:
                    return Result.fail(f"LLM provider '{provider_name}' not found")
                
                self.llm_provider = provider_result.value
                
                logger.info(f"Initializing {provider_name} provider with model {model_name}")
            
            # Get relation types for language
            relation_types = self._get_relation_types_for_language(language)
            relation_types_text = ", ".join(relation_types)
            
            # Prepare prompt
            if language not in RELATIONSHIP_EXTRACTION_PROMPTS:
                return Result.fail(f"Language {language} not supported for relationship extraction")
            
            prompt_template = RELATIONSHIP_EXTRACTION_PROMPTS[language]
            prompt = prompt_template.format(
                domain_type=self.domain_type,
                text=text,
                entities=entity_text,
                relation_types=relation_types_text
            )
            
            # Call LLM
            logger.info(f"Calling LLM for relationship extraction (language: {language}, text length: {len(text)})")
            
            # Generate structured JSON response
            response = self.llm_provider.generate_json(
                prompt=prompt,
                json_schema=RELATIONSHIP_EXTRACTION_SCHEMA,
                temperature=0.2
            )
            
            if not response.success:
                return Result.fail(f"LLM generation failed: {response.error}")
            
            logger.info(f"LLM response received successfully")
            
            # Extract relationships from parsed data
            data = response.value
            relationships = []
            
            # Create entity ID mapping
            entity_map = {entity.name.lower(): entity.id for entity in entities}
            
            for rel_data in data.get("relationships", []):
                source_name = rel_data.get("source_entity", "").lower()
                target_name = rel_data.get("target_entity", "").lower()
                
                # Check if entities exist in the map
                if source_name not in entity_map:
                    logger.warning(f"Source entity '{source_name}' not found in entity map")
                    continue
                    
                if target_name not in entity_map:
                    logger.warning(f"Target entity '{target_name}' not found in entity map")
                    continue
                
                # Create relationship
                relationship = create_relationship(
                    source_entity=entity_map[source_name],
                    target_entity=entity_map[target_name],
                    relation_type=rel_data.get("relation_type", "related_to"),
                    context=rel_data.get("context"),
                    bidirectional=rel_data.get("bidirectional", False),
                    strength=rel_data.get("strength", 1.0),
                    confidence=rel_data.get("confidence", 1.0)
                )
                
                relationships.append(relationship)
            
            logger.info(f"Extracted {len(relationships)} relationships")
            
            return Result.ok(relationships)
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
            return Result.fail(f"Error extracting relationships: {str(e)}")
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """
        Extract JSON string from LLM response.
        
        Args:
            response: LLM response string
            
        Returns:
            JSON string if found, None otherwise
        """
        # Try to find JSON block in the response
        json_start = response.find("{")
        json_end = response.rfind("}")
        
        if json_start >= 0 and json_end >= 0:
            return response[json_start:json_end+1]
        
        return None
    
    def _get_relation_types_for_language(self, language: str) -> List[str]:
        """
        Get relation types for the specified language.
        
        Args:
            language: Language code (e.g., 'en', 'ru')
            
        Returns:
            List of relation types
        """
        if self.relation_types:
            return self.relation_types
        
        if language in DEFAULT_RELATION_TYPES:
            return DEFAULT_RELATION_TYPES[language]
        
        # Default to English if language not found
        return DEFAULT_RELATION_TYPES["en"]


def extract_relationships(
    text_or_segment: Union[str, Segment, SegmentationResult],
    entities: Union[List[Entity], EntityRegistry],
    language: Optional[str] = None,
    domain_type: Optional[str] = None,
    relation_types: Optional[List[str]] = None,
    config: Optional[AppConfig] = None
) -> Result[Union[List[Relationship], RelationshipRegistry]]:
    """
    Extract relationships from text, segment, or segmentation result.
    
    Args:
        text_or_segment: Text, segment, or segmentation result to extract relationships from
        entities: List of entities or entity registry to find relationships between
        language: Language of the text (auto-detected if not provided)
        domain_type: Domain type for extraction context
        relation_types: Types of relations to focus on
        config: Application configuration
        
    Returns:
        Result containing list of relationships, registry of relationships, or error
    """
    # Convert entity registry to list if needed
    entity_list = entities if isinstance(entities, list) else entities.all()
    
    # Create extractor
    extractor = RelationshipExtractor(config, domain_type, relation_types)
    
    # Process based on input type
    if isinstance(text_or_segment, str):
        return extractor.extract_from_text(text_or_segment, entity_list, language)
    
    elif isinstance(text_or_segment, Segment):
        return extractor.extract_from_segment(text_or_segment, entity_list, language)
    
    elif isinstance(text_or_segment, SegmentationResult):
        return extractor.extract_from_segments(text_or_segment, 
                                          entities if isinstance(entities, EntityRegistry) else EntityRegistry(), 
                                          language)
    
    else:
        return Result.fail(f"Unsupported input type: {type(text_or_segment)}")


# Import after function definitions to avoid circular imports
from src.utils.language import get_closest_supported_language 