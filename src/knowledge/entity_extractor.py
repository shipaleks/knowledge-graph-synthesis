"""
Entity Extraction Module

This module provides functionality for extracting entities from text
using LLM-based approaches.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.language import is_language_supported
from src.config.app_config import AppConfig
from src.config.llm_config import LLMConfig
from src.llm import get_provider, LLMProvider
from src.text_processing.segment import Segment, SegmentationResult
from src.knowledge.entity import Entity, EntityRegistry, create_entity

# Configure logger
logger = get_logger(__name__)

# Entity extraction prompts for different languages
ENTITY_EXTRACTION_PROMPTS = {
    "en": """
Extract significant entities from the following text, considering the domain of "{domain_type}".

Text: {text}

Focus on the following entity types relevant to this domain:
{key_concepts}

For each entity, specify:

1. Name (canonical form)
2. Entity type (from the list above or suggest your own if necessary)
3. Context of appearance (quote from text)
4. Attributes (properties, characteristics mentioned in the text)
5. Role in this segment (main topic, supporting example, definition, etc.)
6. Confidence in significance (0-1)

Do not extract general concepts or minor entities; focus on key concepts for this domain.
Keep in mind that the same entity may be mentioned under different names or in different forms.

Present the result in JSON format with array of entities, each having fields: name, type, context, attributes, role, confidence.
""",
    "ru": """
Извлеките значимые сущности из следующего текста, учитывая предметную область "{domain_type}".

Текст: {text}

Фокусируйтесь на следующих типах сущностей, релевантных для данной области:
{key_concepts}

Для каждой сущности укажите:

1. Название (каноническая форма)
2. Тип сущности (из списка выше или предложите свой, если необходимо)
3. Контекст появления (цитата из текста)
4. Атрибуты (свойства, характеристики, упомянутые в тексте)
5. Роль в данном сегменте (основная тема, вспомогательный пример, определение и т.д.)
6. Уверенность в значимости (0-1)

Не извлекайте общие концепции или малозначимые сущности, фокусируйтесь на ключевых понятиях для данной области.
Учитывайте, что одна и та же сущность может быть упомянута под разными именами или в разных формах.

Представьте результат в формате JSON с массивом сущностей, каждая из которых имеет поля: name, type, context, attributes, role, confidence.
"""
}

# Default domain-specific entity types if not provided
DEFAULT_KEY_CONCEPTS = {
    "en": [
        "Person", "Organization", "Location", "Concept", 
        "Process", "Method", "Technology", "Event",
        "Problem", "Solution", "Theory", "Argument"
    ],
    "ru": [
        "Персона", "Организация", "Местоположение", "Концепция", 
        "Процесс", "Метод", "Технология", "Событие",
        "Проблема", "Решение", "Теория", "Аргумент"
    ]
}

# Schema for entity extraction response validation
ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "context": {"type": "string"},
                    "attributes": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "role": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["name", "type"]
            }
        }
    },
    "required": ["entities"]
}


class EntityExtractor:
    """
    Class for extracting entities from text.
    
    Uses LLM to identify and extract entities from text segments.
    """
    
    def __init__(self, 
                config: Optional[AppConfig] = None, 
                domain_type: Optional[str] = None, 
                key_concepts: Optional[List[str]] = None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Optional application configuration
            domain_type: Domain type for contextualizing entity extraction
            key_concepts: List of key concept types to extract
        """
        self.config = config or AppConfig()
        self.llm_provider = None
        self.domain_type = domain_type or "general knowledge"
        self.key_concepts = key_concepts or []
        self.output_dir = Path(os.getenv("APP_OUTPUT_DIR", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_from_text(self, 
                         text: str, 
                         language: Optional[str] = None) -> Result[List[Entity]]:
        """
        Extract entities from a text string.
        
        Args:
            text: Text to extract entities from
            language: Language code (auto-detected if None)
            
        Returns:
            Result[List[Entity]]: List of extracted entities or error
        """
        # Set default language if not provided
        language = language or self.config.language
        
        # Verify language is supported
        if not is_language_supported(language):
            return Result.fail(f"Language '{language}' is not supported")
        
        # Extract entities using LLM
        extraction_result = self._extract_with_llm(text, language)
        
        if not extraction_result.success:
            return Result.fail(f"Failed to extract entities: {extraction_result.error}")
        
        return extraction_result
    
    def extract_from_segment(self,
                           segment: Segment,
                           language: Optional[str] = None) -> Result[List[Entity]]:
        """
        Extract entities from a text segment.
        
        Args:
            segment: Text segment to extract entities from
            language: Language code (auto-detected if None)
            
        Returns:
            Result[List[Entity]]: List of extracted entities or error
        """
        # Set default language if not provided
        language = language or self.config.language
        
        # Extract entities from segment text
        result = self.extract_from_text(segment.text, language)
        
        if not result.success:
            return result
        
        # Add segment metadata to entities
        entities = result.value
        for entity in entities:
            entity.metadata["segment_id"] = segment.id
            entity.metadata["segment_type"] = segment.segment_type
            entity.metadata["segment_level"] = segment.level
            
            # Add segment title if available
            if segment.title:
                entity.metadata["segment_title"] = segment.title
        
        return Result.ok(entities)
    
    def extract_from_segments(self,
                             segmentation_result: SegmentationResult,
                             language: Optional[str] = None) -> Result[EntityRegistry]:
        """
        Extract entities from all segments in a segmentation result.
        
        Args:
            segmentation_result: Result of text segmentation
            language: Language code (auto-detected if None)
            
        Returns:
            Result[EntityRegistry]: Registry with all extracted entities or error
        """
        # Set default language if not provided
        language = language or self.config.language
        
        # Create a registry for the extracted entities
        registry = EntityRegistry()
        
        # Track extraction statistics
        stats = {
            "processed_segments": 0,
            "extracted_entities": 0,
            "failed_segments": 0
        }
        
        # Process each segment
        for segment in segmentation_result.segments:
            logger.debug(f"Extracting entities from segment {segment.id}")
            result = self.extract_from_segment(segment, language)
            
            if result.success:
                entities = result.value
                for entity in entities:
                    registry.add(entity)
                
                stats["processed_segments"] += 1
                stats["extracted_entities"] += len(entities)
            else:
                logger.warning(f"Failed to extract entities from segment {segment.id}: {result.error}")
                stats["failed_segments"] += 1
        
        if stats["processed_segments"] == 0:
            return Result.fail("Failed to extract entities from any segments")
        
        logger.info(f"Extracted {registry.count()} unique entities from {stats['processed_segments']} segments")
        logger.info(f"Failed to process {stats['failed_segments']} segments")
        
        # Save entities to file if output directory exists
        if self.output_dir.exists():
            output_file = self.output_dir / f"entities_{language}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(registry.to_json(pretty=True))
            logger.debug(f"Saved entities to {output_file}")
        
        return Result.ok(registry)
    
    def _extract_with_llm(self, text: str, language: str) -> Result[List[Entity]]:
        """
        Extract entities from text using an LLM.
        
        Args:
            text: Text to extract entities from
            language: Language code
            
        Returns:
            Result[List[Entity]]: List of extracted entities or error
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
        
        # Get appropriate prompt for language
        prompt_template = ENTITY_EXTRACTION_PROMPTS.get(language, ENTITY_EXTRACTION_PROMPTS["en"])
        
        # Get key concepts for prompt
        key_concepts = self.key_concepts
        if not key_concepts:
            key_concepts = DEFAULT_KEY_CONCEPTS.get(language, DEFAULT_KEY_CONCEPTS["en"])
        
        key_concepts_text = "\n".join([f"- {concept}" for concept in key_concepts])
        
        # Format prompt with text and domain
        prompt = prompt_template.format(
            text=text,
            domain_type=self.domain_type,
            key_concepts=key_concepts_text
        )
        
        # Call LLM to get entities
        try:
            logger.info(f"Calling LLM for entity extraction (language: {language}, text length: {len(text)})")
            
            # Generate structured JSON response
            response = self.llm_provider.generate_json(
                prompt=prompt,
                json_schema=ENTITY_EXTRACTION_SCHEMA,
                temperature=0.2
            )
            
            if not response.success:
                logger.error(f"LLM entity extraction failed: {response.error}")
                return Result.fail(f"Failed to extract entities: {response.error}")
            
            # Convert JSON response to entities
            extraction_data = response.value
            entity_data_list = extraction_data.get("entities", [])
            
            if not entity_data_list:
                logger.warning("No entities found in text")
                return Result.ok([])
            
            # Convert each entity data to an Entity object
            entities = []
            for entity_data in entity_data_list:
                try:
                    # Get required fields
                    name = entity_data["name"]
                    entity_type = entity_data["type"]
                    
                    # Get optional fields
                    context = entity_data.get("context")
                    attributes = entity_data.get("attributes", {})
                    
                    # Convert attributes if it's not a dict
                    if not isinstance(attributes, dict):
                        attributes = {"value": attributes}
                    
                    # Add role to attributes if present
                    if "role" in entity_data:
                        attributes["role"] = entity_data["role"]
                    
                    # Get confidence or default to 1.0
                    confidence = entity_data.get("confidence", 1.0)
                    
                    # Create metadata
                    metadata = {
                        "extraction_time": time.time(),
                        "language": language,
                        "domain_type": self.domain_type
                    }
                    
                    # Create entity
                    entity = create_entity(
                        name=name,
                        entity_type=entity_type,
                        context=context,
                        attributes=attributes,
                        metadata=metadata,
                        confidence=confidence
                    )
                    
                    entities.append(entity)
                except KeyError as e:
                    logger.warning(f"Missing required field in entity data: {e}")
                except Exception as e:
                    logger.warning(f"Error creating entity from data: {e}")
            
            logger.info(f"Extracted {len(entities)} entities from text")
            return Result.ok(entities)
            
        except Exception as e:
            logger.error(f"Error during LLM entity extraction: {str(e)}")
            return Result.fail(f"Failed to extract entities: {str(e)}")


def extract_entities(
    text_or_segment: Union[str, Segment, SegmentationResult],
    language: Optional[str] = None,
    domain_type: Optional[str] = None,
    key_concepts: Optional[List[str]] = None,
    config: Optional[AppConfig] = None
) -> Result[Union[List[Entity], EntityRegistry]]:
    """
    Extract entities from text, segment, or segmentation result.
    
    Args:
        text_or_segment: Text, segment, or segmentation result to extract entities from
        language: Language code (auto-detected if None)
        domain_type: Domain type for contextualizing entity extraction
        key_concepts: List of key concept types to extract
        config: Optional application configuration
        
    Returns:
        Result[Union[List[Entity], EntityRegistry]]: 
            - List of entities if input is text or segment
            - EntityRegistry if input is segmentation result
            - Error result if extraction fails
    """
    extractor = EntityExtractor(
        config=config,
        domain_type=domain_type,
        key_concepts=key_concepts
    )
    
    if isinstance(text_or_segment, str):
        return extractor.extract_from_text(text_or_segment, language)
    elif isinstance(text_or_segment, Segment):
        return extractor.extract_from_segment(text_or_segment, language)
    elif isinstance(text_or_segment, SegmentationResult):
        return extractor.extract_from_segments(text_or_segment, language)
    else:
        return Result.fail(f"Unsupported input type: {type(text_or_segment)}") 