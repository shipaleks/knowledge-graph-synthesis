"""
Test Knowledge Verification Module

This module tests the functionality of the knowledge verification module.
"""

import argparse
import json
import logging
import uuid
import sys
import os
from typing import Dict, List, Any, Optional, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config.app_config import AppConfig
from src.utils.logger import configure_logging
from src.utils.result import Result
from src.knowledge.entity import Entity, EntityRegistry, create_entity
from src.knowledge.relationship import Relationship, RelationshipRegistry, create_relationship
from src.knowledge.knowledge_verifier import KnowledgeVerifier, VerificationIssue, VerificationSeverity, verify_knowledge_graph


def create_test_knowledge_graph() -> Tuple[EntityRegistry, RelationshipRegistry]:
    """
    Create a test knowledge graph with some entities and relationships.
    
    Returns:
        Tuple[EntityRegistry, RelationshipRegistry]: Entity and relationship registries
    """
    # Create entity registry
    entity_registry = EntityRegistry()
    
    # Add entities
    ai = create_entity("Artificial Intelligence", "Concept", 
                      "A field of computer science that creates systems capable of performing tasks that would typically require human intelligence.",
                      {"year_coined": 1956, "importance": "high"})
    
    ml = create_entity("Machine Learning", "Concept",
                     "A subset of AI that involves the development of algorithms that allow computers to learn from and make decisions or predictions based on data.",
                     {"year_emerged": 1959, "importance": "high"})
    
    dl = create_entity("Deep Learning", "Concept",
                     "A subset of machine learning that uses neural networks with many layers.",
                     {"year_emerged": 2010, "importance": "high"})
    
    nn = create_entity("Neural Networks", "Technology",
                     "Computing systems inspired by the biological neural networks that constitute animal brains.",
                     {"first_model_year": 1943, "complexity": "high"})
    
    rl = create_entity("Reinforcement Learning", "Method",
                     "A type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize a reward.",
                     {"application": "decision making", "complexity": "medium"})
    
    # Register entities
    entity_registry.add(ai)
    entity_registry.add(ml)
    entity_registry.add(dl)
    entity_registry.add(nn)
    entity_registry.add(rl)
    
    # Create relationship registry
    relationship_registry = RelationshipRegistry()
    
    # Add relationships
    rel1 = create_relationship(ml.id, ai.id, "is-a", 
                             "Machine Learning is a subset of Artificial Intelligence")
    
    rel2 = create_relationship(dl.id, ml.id, "is-a", 
                             "Deep Learning is a subset of Machine Learning")
    
    rel3 = create_relationship(rl.id, ml.id, "is-a", 
                             "Reinforcement Learning is a type of Machine Learning")
    
    rel4 = create_relationship(dl.id, nn.id, "uses", 
                             "Deep Learning uses Neural Networks")
    
    # Register relationships
    relationship_registry.add(rel1)
    relationship_registry.add(rel2)
    relationship_registry.add(rel3)
    relationship_registry.add(rel4)
    
    return entity_registry, relationship_registry


def create_contradictory_graph() -> Tuple[EntityRegistry, RelationshipRegistry]:
    """
    Create a knowledge graph with contradictions for testing verification.
    
    Returns:
        Tuple[EntityRegistry, RelationshipRegistry]: Entity and relationship registries
    """
    # Create entity registry
    entity_registry = EntityRegistry()
    
    # Add entities
    ai = create_entity("Artificial Intelligence", "Concept")
    ml = create_entity("Machine Learning", "Concept")
    dl = create_entity("Deep Learning", "Concept")
    nn = create_entity("Neural Networks", "Technology")
    rl = create_entity("Reinforcement Learning", "Method")
    
    # Register entities
    entity_registry.add(ai)
    entity_registry.add(ml)
    entity_registry.add(dl)
    entity_registry.add(nn)
    entity_registry.add(rl)
    
    # Create relationship registry
    relationship_registry = RelationshipRegistry()
    
    # Add contradictory relationships
    rel1 = create_relationship(ml.id, ai.id, "is-a", "Machine Learning is a subset of AI")
    rel2 = create_relationship(ai.id, ml.id, "is-a", "AI is a subset of Machine Learning") # Cyclic dependency
    
    rel3 = create_relationship(dl.id, ml.id, "is-a", "Deep Learning is a subset of ML")
    rel4 = create_relationship(dl.id, ml.id, "is-not-a", "Deep Learning is not ML") # Contradiction
    
    rel5 = create_relationship(nn.id, ai.id, "part-of", "Neural Networks are part of AI")
    rel6 = create_relationship(nn.id, ai.id, "separate-from", "Neural Networks are separate from AI") # Contradiction
    
    # Register relationships
    relationship_registry.add(rel1)
    relationship_registry.add(rel2)
    relationship_registry.add(rel3)
    relationship_registry.add(rel4)
    relationship_registry.add(rel5)
    relationship_registry.add(rel6)
    
    return entity_registry, relationship_registry


def create_cyclic_graph() -> Tuple[EntityRegistry, RelationshipRegistry]:
    """
    Create a knowledge graph with cyclic dependencies for testing verification.
    
    Returns:
        Tuple[EntityRegistry, RelationshipRegistry]: Entity and relationship registries
    """
    # Create entity registry
    entity_registry = EntityRegistry()
    
    # Add entities
    a = create_entity("Entity A", "Concept")
    b = create_entity("Entity B", "Concept")
    c = create_entity("Entity C", "Concept")
    d = create_entity("Entity D", "Concept")
    
    # Register entities
    entity_registry.add(a)
    entity_registry.add(b)
    entity_registry.add(c)
    entity_registry.add(d)
    
    # Create relationship registry
    relationship_registry = RelationshipRegistry()
    
    # Add relationships that form a cycle: A -> B -> C -> A
    rel1 = create_relationship(a.id, b.id, "part-of", "A is part of B")
    rel2 = create_relationship(b.id, c.id, "part-of", "B is part of C")
    rel3 = create_relationship(c.id, a.id, "part-of", "C is part of A") # Creates cycle
    rel4 = create_relationship(d.id, a.id, "part-of", "D is part of A") # Not part of cycle
    
    # Register relationships
    relationship_registry.add(rel1)
    relationship_registry.add(rel2)
    relationship_registry.add(rel3)
    relationship_registry.add(rel4)
    
    return entity_registry, relationship_registry


def create_dangling_reference_graph() -> Tuple[EntityRegistry, RelationshipRegistry]:
    """
    Create a knowledge graph with dangling references for testing verification.
    
    Returns:
        Tuple[EntityRegistry, RelationshipRegistry]: Entity and relationship registries
    """
    # Create entity registry
    entity_registry = EntityRegistry()
    
    # Add entities
    a = create_entity("Entity A", "Concept")
    b = create_entity("Entity B", "Concept")
    
    # Register entities
    entity_registry.add(a)
    entity_registry.add(b)
    
    # Create relationship registry
    relationship_registry = RelationshipRegistry()
    
    # Add relationships with dangling references
    rel1 = create_relationship(a.id, b.id, "related-to", "A is related to B")
    rel2 = create_relationship(a.id, str(uuid.uuid4()), "related-to", "A is related to non-existent entity")
    rel3 = create_relationship(str(uuid.uuid4()), b.id, "related-to", "Non-existent entity is related to B")
    
    # Register relationships
    relationship_registry.add(rel1)
    relationship_registry.add(rel2)
    relationship_registry.add(rel3)
    
    return entity_registry, relationship_registry


def test_cyclic_detection(language: str = "en") -> None:
    """
    Test the detection of cyclic dependencies.
    
    Args:
        language: Language for verification prompts
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing cyclic dependency detection...")
    
    # Create graph with cyclic dependencies
    entity_registry, relationship_registry = create_cyclic_graph()
    
    # Create verifier
    verifier = KnowledgeVerifier()
    
    # Verify graph
    result = verifier.verify_knowledge_graph(entity_registry, relationship_registry, language)
    
    if not result.success:
        logger.error(f"Verification failed: {result.error}")
        return
    
    verification_result = result.value
    
    # Check for cycle detection
    cycle_issues = [issue for issue in verification_result.issues 
                   if issue.issue_type == "cyclic_dependency"]
    
    if cycle_issues:
        logger.info(f"Successfully detected {len(cycle_issues)} cyclic dependencies:")
        for issue in cycle_issues:
            logger.info(f"  - Cycle: {issue.metadata.get('cycle_description', 'Unknown cycle')}")
            logger.info(f"  - Proposed solution: {issue.proposed_solution}")
            logger.info(f"  - Severity: {issue.severity.value}")
    else:
        logger.error("Failed to detect cyclic dependencies")


def test_contradiction_detection(language: str = "en") -> None:
    """
    Test the detection of contradictory relationships.
    
    Args:
        language: Language for verification prompts
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing contradiction detection...")
    
    # Create graph with contradictions
    entity_registry, relationship_registry = create_contradictory_graph()
    
    # Create verifier
    verifier = KnowledgeVerifier()
    
    # Verify graph
    result = verifier.verify_knowledge_graph(entity_registry, relationship_registry, language)
    
    if not result.success:
        logger.error(f"Verification failed: {result.error}")
        return
    
    verification_result = result.value
    
    # Check for contradiction detection
    contradiction_issues = [issue for issue in verification_result.issues 
                          if issue.issue_type in ["contradictory_relationships", "transitive_contradiction"]]
    
    if contradiction_issues:
        logger.info(f"Successfully detected {len(contradiction_issues)} contradictions:")
        for issue in contradiction_issues:
            entities_involved = []
            for rel_id in issue.involved_elements:
                rel = relationship_registry.get(rel_id)
                if rel:
                    source = entity_registry.get(rel.source_entity)
                    target = entity_registry.get(rel.target_entity)
                    if source and target:
                        entities_involved.append(f"{source.name} -> {target.name} ({rel.relation_type})")
            
            logger.info(f"  - Issue type: {issue.issue_type}")
            logger.info(f"  - Relationships: {', '.join(entities_involved)}")
            logger.info(f"  - Proposed solution: {issue.proposed_solution}")
            logger.info(f"  - Severity: {issue.severity.value}")
    else:
        logger.error("Failed to detect contradictions")


def test_dangling_reference_detection(language: str = "en") -> None:
    """
    Test the detection of dangling references.
    
    Args:
        language: Language for verification prompts
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing dangling reference detection...")
    
    # Create graph with dangling references
    entity_registry, relationship_registry = create_dangling_reference_graph()
    
    # Create verifier
    verifier = KnowledgeVerifier()
    
    # Verify graph
    result = verifier.verify_knowledge_graph(entity_registry, relationship_registry, language)
    
    if not result.success:
        logger.error(f"Verification failed: {result.error}")
        return
    
    verification_result = result.value
    
    # Check for dangling reference detection
    dangling_issues = [issue for issue in verification_result.issues 
                     if issue.issue_type == "dangling_reference"]
    
    if dangling_issues:
        logger.info(f"Successfully detected {len(dangling_issues)} dangling references:")
        for issue in dangling_issues:
            rel_id = issue.involved_elements[0] if issue.involved_elements else "Unknown"
            rel = relationship_registry.get(rel_id)
            if rel:
                logger.info(f"  - Relationship ID: {rel_id}")
                logger.info(f"  - Source: {rel.source_entity}")
                logger.info(f"  - Target: {rel.target_entity}")
                logger.info(f"  - Proposed solution: {issue.proposed_solution}")
                logger.info(f"  - Severity: {issue.severity.value}")
    else:
        logger.error("Failed to detect dangling references")


def test_llm_verification(language: str = "en") -> None:
    """
    Test the LLM-based verification of a knowledge graph.
    
    Args:
        language: Language for verification prompts
    """
    logger = logging.getLogger(__name__)
    logger.info("Testing LLM-based verification...")
    
    # Create simple test graph
    entity_registry, relationship_registry = create_test_knowledge_graph()
    
    # Create verifier
    verifier = KnowledgeVerifier()
    
    # Verify graph
    result = verifier.verify_knowledge_graph(entity_registry, relationship_registry, language)
    
    if not result.success:
        logger.error(f"Verification failed: {result.error}")
        return
    
    verification_result = result.value
    
    # Print results
    logger.info(f"Verification completed with {len(verification_result.issues)} issues:")
    
    if verification_result.issues:
        for i, issue in enumerate(verification_result.issues, 1):
            logger.info(f"Issue {i}:")
            logger.info(f"  - Type: {issue.issue_type}")
            
            # Handle different types in involved_elements
            elements_str = []
            for elem in issue.involved_elements:
                if isinstance(elem, str):
                    elements_str.append(elem)
                else:
                    elements_str.append(str(elem))
            
            logger.info(f"  - Elements: {', '.join(elements_str)}")
            logger.info(f"  - Solution: {issue.proposed_solution}")
            logger.info(f"  - Severity: {issue.severity.value}")
    else:
        logger.info("No issues found in the knowledge graph.")
    
    # Export to JSON
    filename = f"output/verification_results_{language}.json"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(verification_result.to_json(pretty=True))
    
    logger.info(f"Verification results saved to {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test knowledge verification module")
    parser.add_argument("--test", choices=["cyclic", "contradictions", "dangling", "llm", "all"], 
                      help="Test to run", default="all")
    parser.add_argument("--language", choices=["en", "ru"], help="Language for verification", default="en")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    logger = logging.getLogger(__name__)
    
    # Run tests
    if args.test == "cyclic" or args.test == "all":
        test_cyclic_detection(args.language)
    
    if args.test == "contradictions" or args.test == "all":
        test_contradiction_detection(args.language)
    
    if args.test == "dangling" or args.test == "all":
        test_dangling_reference_detection(args.language)
    
    if args.test == "llm" or args.test == "all":
        test_llm_verification(args.language)


if __name__ == "__main__":
    main() 