"""
Knowledge Extraction Pipeline Module

This module provides a complete knowledge extraction pipeline that integrates
entity extraction, relationship extraction, coreference resolution, and knowledge
verification into a cohesive workflow.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from src.utils.result import Result
from src.utils.logger import get_logger
from src.utils.timer import Timer
from src.config.app_config import AppConfig
from src.text_processing.segment import Segment, SegmentationResult
from src.text_processing.pipeline import TextProcessingPipeline
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.entity_extractor import EntityExtractor
from src.knowledge.relationship import Relationship, RelationshipRegistry
from src.knowledge.relationship_extractor import RelationshipExtractor
from src.knowledge.coreference_resolver import CoreferenceResolver
from src.knowledge.knowledge_verifier import KnowledgeVerifier

# Configure logger
logger = get_logger(__name__)


class KnowledgeExtractionPipeline:
    """
    A complete pipeline for extracting knowledge from text.
    
    Includes entity extraction, relationship extraction, coreference resolution,
    and knowledge verification with configuration options.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the knowledge extraction pipeline.
        
        Args:
            config: Optional application configuration
        """
        self.config = config or AppConfig()
        self.text_pipeline = TextProcessingPipeline(config)
        self.entity_extractor = EntityExtractor(config)
        self.relationship_extractor = RelationshipExtractor(config)
        self.coreference_resolver = CoreferenceResolver(config)
        self.knowledge_verifier = KnowledgeVerifier(config)
        
        self.output_dir = Path(os.getenv("APP_OUTPUT_DIR", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing options
        self.use_cache = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.save_intermediates = True
        self.domain_type = os.getenv("DOMAIN_TYPE", "general")
        self.relation_types = self._get_default_relation_types()
        
    def _get_default_relation_types(self) -> List[str]:
        """Get default relation types for extraction."""
        return [
            "is-a", "part-of", "has-property", "causes", "depends-on",
            "related-to", "similar-to", "opposite-of", "used-for",
            "created-by", "located-in", "occurs-at", "affects"
        ]
    
    def process(self, 
               input_text: Union[str, Dict[str, Any], Path], 
               language: Optional[str] = None,
               output_prefix: Optional[str] = None) -> Result[Dict[str, Any]]:
        """
        Process text through the complete knowledge extraction pipeline.
        
        Args:
            input_text: Text to process (string, dict, or file path)
            language: Language code (auto-detected if None)
            output_prefix: Prefix for output files
            
        Returns:
            Result[Dict[str, Any]]: Processing results or error
        """
        # Create timer to measure performance
        timer = Timer()
        
        # Start processing
        logger.info("Starting knowledge extraction pipeline")
        overall_timer = timer.start("overall")
        
        # Step 1: Process text through the text processing pipeline
        text_processing_timer = timer.start("text_processing")
        text_result = self.text_pipeline.process(input_text, language, output_prefix)
        timer.stop(text_processing_timer)
        
        if not text_result.success:
            return Result.fail(f"Failed to process text: {text_result.error}")
        
        text_data = text_result.value
        segments = SegmentationResult.from_dict(text_data["segments"])
        language = text_data["metadata"]["language"]
        
        logger.info(f"Text processed: {len(segments.segments)} segments, language: {language}")
        
        # Step 2: Extract entities from segments
        entity_timer = timer.start("entity_extraction")
        # Set domain type in the entity extractor
        self.entity_extractor.domain_type = self.domain_type
        entity_result = self.entity_extractor.extract_from_segments(
            segments,
            language=language
        )
        timer.stop(entity_timer)
        
        if not entity_result.success:
            return Result.fail(f"Failed to extract entities: {entity_result.error}")
        
        entity_registry = entity_result.value
        logger.info(f"Extracted {entity_registry.count()} entities")
        
        # Save entity results if enabled
        if self.save_intermediates:
            self._save_entities(entity_registry, output_prefix, language)
        
        # Step 3: Extract relationships between entities
        relationship_timer = timer.start("relationship_extraction")
        relationship_result = self.relationship_extractor.extract_from_segments(
            segments,
            entity_registry,
            language=language
        )
        timer.stop(relationship_timer)
        
        if not relationship_result.success:
            return Result.fail(f"Failed to extract relationships: {relationship_result.error}")
        
        relationship_registry = relationship_result.value
        logger.info(f"Extracted {relationship_registry.count()} relationships")
        
        # Save relationship results if enabled
        if self.save_intermediates:
            self._save_relationships(relationship_registry, output_prefix, language)
        
        # Step 4: Resolve coreferences between entities
        coreference_timer = timer.start("coreference_resolution")
        coreference_result = self.coreference_resolver.resolve_coreferences(
            entity_registry,
            language=language
        )
        timer.stop(coreference_timer)
        
        if not coreference_result.success:
            return Result.fail(f"Failed to resolve coreferences: {coreference_result.error}")
        
        resolved_entity_registry = coreference_result.value
        logger.info(f"Resolved entities: {resolved_entity_registry.count()} after coreference resolution")
        
        # Save resolved entity results if enabled
        if self.save_intermediates:
            self._save_resolved_entities(resolved_entity_registry, output_prefix, language)
        
        # Step 5: Verify knowledge graph
        verification_timer = timer.start("knowledge_verification")
        verification_result = self.knowledge_verifier.verify_knowledge_graph(
            resolved_entity_registry,
            relationship_registry,
            language=language
        )
        timer.stop(verification_timer)
        
        if not verification_result.success:
            return Result.fail(f"Failed to verify knowledge graph: {verification_result.error}")
        
        verification_data = verification_result.value
        logger.info(f"Knowledge graph verified: {len(verification_data.issues)} issues found")
        
        # Save verification results if enabled
        if self.save_intermediates:
            self._save_verification(verification_data, output_prefix, language)
        
        # Stop overall timer
        timer.stop(overall_timer)
        
        # Compile results
        timing_info = timer.get_all_timings()
        
        results = {
            "entities": resolved_entity_registry.to_dict(),
            "relationships": relationship_registry.to_dict(),
            "verification": verification_data.to_dict(),
            "metadata": {
                "language": language,
                "domain_type": self.domain_type,
                "entity_count": resolved_entity_registry.count(),
                "relationship_count": relationship_registry.count(),
                "issue_count": len(verification_data.issues),
                "is_valid": verification_data.is_valid,
                "timing": timing_info,
                "process_date": time.time()
            }
        }
        
        # Save complete results if enabled
        if self.save_intermediates:
            self._save_results(results, output_prefix, language)
        
        logger.info(f"Knowledge extraction complete in {timing_info['overall']:.2f} seconds")
        return Result.ok(results)
    
    def _save_entities(self, 
                     entity_registry: EntityRegistry, 
                     output_prefix: Optional[str] = None,
                     language: str = "en") -> None:
        """Save extracted entities to a file."""
        prefix = output_prefix or "knowledge"
        filename = self.output_dir / f"{prefix}_entities_{language}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(entity_registry.to_json(pretty=True))
            
        logger.info(f"Saved {entity_registry.count()} entities to {filename}")
    
    def _save_relationships(self, 
                          relationship_registry: RelationshipRegistry, 
                          output_prefix: Optional[str] = None,
                          language: str = "en") -> None:
        """Save extracted relationships to a file."""
        prefix = output_prefix or "knowledge"
        filename = self.output_dir / f"{prefix}_relationships_{language}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(relationship_registry.to_json(pretty=True))
            
        logger.info(f"Saved {relationship_registry.count()} relationships to {filename}")
    
    def _save_resolved_entities(self, 
                              entity_registry: EntityRegistry, 
                              output_prefix: Optional[str] = None,
                              language: str = "en") -> None:
        """Save resolved entities to a file."""
        prefix = output_prefix or "knowledge"
        filename = self.output_dir / f"{prefix}_resolved_entities_{language}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(entity_registry.to_json(pretty=True))
            
        logger.info(f"Saved {entity_registry.count()} resolved entities to {filename}")
    
    def _save_verification(self, 
                         verification_data: Any, 
                         output_prefix: Optional[str] = None,
                         language: str = "en") -> None:
        """Save verification results to a file."""
        prefix = output_prefix or "knowledge"
        filename = self.output_dir / f"{prefix}_verification_{language}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(verification_data.to_json(pretty=True))
            
        logger.info(f"Saved verification results with {len(verification_data.issues)} issues to {filename}")
    
    def _save_results(self, 
                    results: Dict[str, Any], 
                    output_prefix: Optional[str] = None,
                    language: str = "en") -> None:
        """Save complete results to a file."""
        prefix = output_prefix or "knowledge"
        filename = self.output_dir / f"{prefix}_extraction_{language}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved complete knowledge extraction results to {filename}")


def extract_knowledge(
    input_text: Union[str, Dict[str, Any], Path],
    language: Optional[str] = None,
    output_prefix: Optional[str] = None,
    config: Optional[AppConfig] = None
) -> Result[Dict[str, Any]]:
    """
    Extract knowledge from text.
    
    Convenience function that creates a KnowledgeExtractionPipeline and uses it
    to process the input text.
    
    Args:
        input_text: Text to process (string, dict, or file path)
        language: Language code (auto-detected if None)
        output_prefix: Prefix for output files
        config: Optional application configuration
        
    Returns:
        Result[Dict[str, Any]]: Extraction results or error
    """
    pipeline = KnowledgeExtractionPipeline(config)
    return pipeline.process(input_text, language, output_prefix) 