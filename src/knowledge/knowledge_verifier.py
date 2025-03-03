"""
Knowledge Verification Module

This module provides the KnowledgeVerifier class and related utilities for verifying
the consistency and validity of knowledge graphs.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Tuple

from src.config.app_config import AppConfig
from src.config.llm_config import LLMConfig
from src.llm.provider import LLMProvider, get_provider
from src.utils.result import Result
from src.utils.language import is_language_supported
from src.knowledge.entity import Entity, EntityRegistry
from src.knowledge.relationship import Relationship, RelationshipRegistry


class VerificationSeverity(str, Enum):
    """Severity levels for verification issues"""
    CRITICAL = "critical"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class VerificationIssue:
    """
    Represents an issue identified during knowledge graph verification.
    
    Contains information about the type of issue, affected elements,
    proposed solution, and severity level.
    """
    issue_type: str
    involved_elements: List[str]
    proposed_solution: str
    severity: VerificationSeverity
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue to a dictionary."""
        return {
            "issue_type": self.issue_type,
            "involved_elements": self.involved_elements,
            "proposed_solution": self.proposed_solution,
            "severity": self.severity.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationIssue':
        """Create an issue from a dictionary."""
        severity = data.get("severity", "medium")
        return cls(
            issue_type=data.get("issue_type", "unknown"),
            involved_elements=data.get("involved_elements", []),
            proposed_solution=data.get("proposed_solution", ""),
            severity=VerificationSeverity(severity),
            metadata=data.get("metadata", {})
        )


@dataclass
class VerificationResult:
    """
    Contains the results of a knowledge graph verification.
    
    Includes a list of identified issues and metadata about the verification process.
    """
    issues: List[VerificationIssue] = field(default_factory=list)
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue: VerificationIssue) -> None:
        """Add an issue to the verification result."""
        self.issues.append(issue)
        if issue.severity == VerificationSeverity.CRITICAL:
            self.is_valid = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the verification result to a dictionary."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "is_valid": self.is_valid,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create a verification result from a dictionary."""
        result = cls(
            is_valid=data.get("is_valid", True),
            metadata=data.get("metadata", {})
        )
        
        for issue_data in data.get("issues", []):
            issue = VerificationIssue.from_dict(issue_data)
            result.issues.append(issue)
            
        return result
    
    def to_json(self, pretty: bool = False) -> str:
        """Convert the verification result to a JSON string."""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VerificationResult':
        """Create a verification result from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class KnowledgeVerifier:
    """
    Verifies the consistency and validity of knowledge graphs.
    
    Uses a combination of rule-based checks and LLM-based verification
    to identify issues in knowledge graphs.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the knowledge verifier.
        
        Args:
            config: Application configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or AppConfig()
        self.llm_config = LLMConfig()
        self.provider: Optional[LLMProvider] = None
    
    def verify_knowledge_graph(self,
                               entity_registry: EntityRegistry,
                               relationship_registry: RelationshipRegistry,
                               language: str = "en") -> Result[VerificationResult]:
        """
        Verify a knowledge graph for consistency and validity.
        
        Args:
            entity_registry: Registry of entities in the graph
            relationship_registry: Registry of relationships in the graph
            language: Language code for verification prompts
            
        Returns:
            Result[VerificationResult]: Result containing verification issues if successful
        """
        if not is_language_supported(language):
            return Result.fail(f"Language {language} is not supported")
        
        # Initialize LLM provider if needed
        if not self.provider:
            provider_init_result = self._initialize_llm_provider()
            if not provider_init_result.success:
                return Result.fail(f"Failed to initialize LLM provider: {provider_init_result.error}")
        
        # Perform rule-based checks
        verification_result = VerificationResult()
        
        self._check_for_cyclic_dependencies(entity_registry, relationship_registry, verification_result)
        self._check_for_dangling_references(entity_registry, relationship_registry, verification_result)
        self._check_for_contradictory_relationships(entity_registry, relationship_registry, verification_result)
        
        # Perform LLM-based verification
        llm_verification_result = self._verify_with_llm(entity_registry, relationship_registry, language)
        
        if not llm_verification_result.success:
            return Result.fail(f"LLM verification failed: {llm_verification_result.error}")
        
        # Merge LLM-identified issues into our verification result
        for issue in llm_verification_result.value.issues:
            verification_result.add_issue(issue)
        
        verification_result.metadata["rule_checks_completed"] = True
        verification_result.metadata["llm_verification_completed"] = True
        
        return Result.ok(verification_result)
    
    def _check_for_cyclic_dependencies(self,
                                      entity_registry: EntityRegistry,
                                      relationship_registry: RelationshipRegistry,
                                      verification_result: VerificationResult) -> None:
        """
        Check for cyclic dependencies in hierarchical relationships.
        
        Args:
            entity_registry: Registry of entities
            relationship_registry: Registry of relationships
            verification_result: Verification result to update with issues
        """
        # Build a directed graph representation
        graph: Dict[str, List[str]] = {}
        hierarchical_relations = ["is-a", "part-of", "subclass-of", "contains", "includes"]
        
        # Only consider hierarchical relationships that could form cycles
        for rel in relationship_registry.all():
            if rel.relation_type.lower() in hierarchical_relations:
                if rel.source_entity not in graph:
                    graph[rel.source_entity] = []
                graph[rel.source_entity].append(rel.target_entity)
        
        # Function to detect cycles using DFS
        def detect_cycle(node: str, visited: Set[str], path: Set[str], cycle_path: List[str]) -> bool:
            visited.add(node)
            path.add(node)
            cycle_path.append(node)
            
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        if detect_cycle(neighbor, visited, path, cycle_path):
                            return True
                    elif neighbor in path:
                        # We found a cycle
                        cycle_path.append(neighbor)
                        return True
            
            path.remove(node)
            cycle_path.pop()
            return False
        
        # Check each node for cycles
        for node in graph.keys():
            if node not in set().union(*[set(visited) for visited in [set() for _ in range(len(graph))]]):
                visited: Set[str] = set()
                path: Set[str] = set()
                cycle_path: List[str] = []
                
                if detect_cycle(node, visited, path, cycle_path):
                    # Get the actual cycle
                    start_idx = cycle_path.index(cycle_path[-1])
                    actual_cycle = cycle_path[start_idx:-1]
                    
                    # Find the relationship IDs involved in the cycle
                    involved_relationships = []
                    for i in range(len(actual_cycle) - 1):
                        source = actual_cycle[i]
                        target = actual_cycle[i + 1]
                        
                        for rel in relationship_registry.all():
                            if rel.source_entity == source and rel.target_entity == target:
                                involved_relationships.append(rel.id)
                    
                    # Get human-readable entity names for the issue description
                    entity_names = []
                    for entity_id in actual_cycle:
                        entity = entity_registry.get(entity_id)
                        if entity:
                            entity_names.append(entity.name)
                        else:
                            entity_names.append(f"Unknown entity ({entity_id})")
                    
                    cycle_description = " -> ".join(entity_names)
                    
                    issue = VerificationIssue(
                        issue_type="cyclic_dependency",
                        involved_elements=involved_relationships,
                        proposed_solution=f"Remove one of the relationships in the cycle: {cycle_description}",
                        severity=VerificationSeverity.CRITICAL,
                        metadata={
                            "cycle_entities": actual_cycle,
                            "cycle_description": cycle_description
                        }
                    )
                    
                    verification_result.add_issue(issue)
    
    def _check_for_dangling_references(self,
                                      entity_registry: EntityRegistry,
                                      relationship_registry: RelationshipRegistry,
                                      verification_result: VerificationResult) -> None:
        """
        Check for relationships referencing non-existent entities.
        
        Args:
            entity_registry: Registry of entities
            relationship_registry: Registry of relationships
            verification_result: Verification result to update with issues
        """
        for relationship in relationship_registry.all():
            source_id = relationship.source_entity
            target_id = relationship.target_entity
            
            if not entity_registry.get(source_id):
                issue = VerificationIssue(
                    issue_type="dangling_reference",
                    involved_elements=[relationship.id],
                    proposed_solution=f"Remove relationship or create the missing source entity with ID: {source_id}",
                    severity=VerificationSeverity.CRITICAL
                )
                verification_result.add_issue(issue)
            
            if not entity_registry.get(target_id):
                issue = VerificationIssue(
                    issue_type="dangling_reference",
                    involved_elements=[relationship.id],
                    proposed_solution=f"Remove relationship or create the missing target entity with ID: {target_id}",
                    severity=VerificationSeverity.CRITICAL
                )
                verification_result.add_issue(issue)
    
    def _check_for_contradictory_relationships(self,
                                             entity_registry: EntityRegistry,
                                             relationship_registry: RelationshipRegistry,
                                             verification_result: VerificationResult) -> None:
        """
        Check for contradictory relationships between the same entities.
        
        Args:
            entity_registry: Registry of entities
            relationship_registry: Registry of relationships
            verification_result: Verification result to update with issues
        """
        # Define contradictory relationship types
        contradictions = {
            "is-a": ["is-not-a", "different-from"],
            "part-of": ["separate-from", "unrelated-to"],
            "causes": ["prevents", "unrelated-to"],
            "depends-on": ["independent-of"],
            "similar-to": ["different-from", "opposite-of"],
            "greater-than": ["less-than", "equal-to"],
            "before": ["after", "simultaneous-with"],
        }
        
        # Add symmetric contradictions
        for rel, contras in list(contradictions.items()):
            for contra in contras:
                if contra not in contradictions:
                    contradictions[contra] = []
                contradictions[contra].append(rel)
        
        # Group relationships by source-target pairs
        relationship_groups: Dict[Tuple[str, str], List[Relationship]] = {}
        
        for rel in relationship_registry.all():
            entity_pair = (rel.source_entity, rel.target_entity)
            # Also consider the reverse direction if the relationship is not directional
            reverse_pair = (rel.target_entity, rel.source_entity)
            
            if entity_pair not in relationship_groups:
                relationship_groups[entity_pair] = []
            relationship_groups[entity_pair].append(rel)
            
            if rel.bidirectional:
                if reverse_pair not in relationship_groups:
                    relationship_groups[reverse_pair] = []
                relationship_groups[reverse_pair].append(rel)
        
        # Check each group for contradictory relationships
        for entity_pair, relationships in relationship_groups.items():
            source_id, target_id = entity_pair
            source_entity = entity_registry.get(source_id)
            target_entity = entity_registry.get(target_id)
            
            if not source_entity or not target_entity:
                continue
            
            # Check for multiple contradictory relationship types
            checked_pairs = set()
            for i, rel1 in enumerate(relationships):
                for j, rel2 in enumerate(relationships[i+1:], i+1):
                    # Skip if we've already checked this pair
                    if (rel1.id, rel2.id) in checked_pairs or (rel2.id, rel1.id) in checked_pairs:
                        continue
                        
                    checked_pairs.add((rel1.id, rel2.id))
                    
                    # Check if these relationship types contradict each other
                    if rel1.relation_type in contradictions.get(rel2.relation_type, []) or \
                       rel2.relation_type in contradictions.get(rel1.relation_type, []):
                        issue = VerificationIssue(
                            issue_type="contradictory_relationships",
                            involved_elements=[rel1.id, rel2.id],
                            proposed_solution=(
                                f"Resolve the contradiction between relationships of types "
                                f"'{rel1.relation_type}' and '{rel2.relation_type}' between "
                                f"'{source_entity.name}' and '{target_entity.name}'"
                            ),
                            severity=VerificationSeverity.MEDIUM,
                            metadata={
                                "source_entity": source_id,
                                "target_entity": target_id,
                                "relationship_types": [rel1.relation_type, rel2.relation_type]
                            }
                        )
                        verification_result.add_issue(issue)
            
        # Check for semantic contradictions
        self._check_for_semantic_contradictions(entity_registry, relationship_registry, verification_result)
    
    def _check_for_semantic_contradictions(self,
                                         entity_registry: EntityRegistry,
                                         relationship_registry: RelationshipRegistry,
                                         verification_result: VerificationResult) -> None:
        """
        Check for semantically contradictory relationship patterns.
        
        Args:
            entity_registry: Registry of entities
            relationship_registry: Registry of relationships
            verification_result: Verification result to update with issues
        """
        # Check for transitive contradictions (A is-a B, B is-a C, but A is-not-a C)
        transitive_relations = ["is-a", "part-of", "subclass-of"]
        
        for transitive_rel in transitive_relations:
            # Build a graph for this relationship type
            graph: Dict[str, List[str]] = {}
            rel_map: Dict[Tuple[str, str], str] = {}  # Maps (source, target) to relationship ID
            
            for rel in relationship_registry.all():
                if rel.relation_type.lower() == transitive_rel:
                    if rel.source_entity not in graph:
                        graph[rel.source_entity] = []
                    graph[rel.source_entity].append(rel.target_entity)
                    rel_map[(rel.source_entity, rel.target_entity)] = rel.id
            
            # Find all transitive relationships
            for start_node in graph:
                # Use BFS to find all reachable nodes
                visited = {start_node}
                queue = [(start_node, [start_node])]
                
                while queue:
                    node, path = queue.pop(0)
                    
                    if node in graph:
                        for neighbor in graph[node]:
                            if neighbor not in visited:
                                new_path = path + [neighbor]
                                queue.append((neighbor, new_path))
                                visited.add(neighbor)
                                
                                # Check if there's a contradictory relationship directly between start and any node
                                # with distance > 1
                                if len(new_path) > 2:
                                    start_id = new_path[0]
                                    end_id = new_path[-1]
                                    
                                    # Look for contradictory relationships between start and end
                                    contradictory_rels = []
                                    for rel in relationship_registry.all():
                                        if rel.source_entity == start_id and rel.target_entity == end_id:
                                            if rel.relation_type.lower() in ["is-not-a", "different-from", "separate-from"]:
                                                contradictory_rels.append(rel)
                                                
                                    if contradictory_rels:
                                        # Get the IDs of all relationships in the transitive path
                                        path_rel_ids = []
                                        for i in range(len(new_path) - 1):
                                            source = new_path[i]
                                            target = new_path[i + 1]
                                            if (source, target) in rel_map:
                                                path_rel_ids.append(rel_map[(source, target)])
                                        
                                        # Create a verification issue
                                        for contra_rel in contradictory_rels:
                                            issue = VerificationIssue(
                                                issue_type="transitive_contradiction",
                                                involved_elements=path_rel_ids + [contra_rel.id],
                                                proposed_solution=(
                                                    f"Resolve the contradiction between the transitive relationship "
                                                    f"implied by the path and the direct relationship of type "
                                                    f"'{contra_rel.relation_type}'"
                                                ),
                                                severity=VerificationSeverity.MEDIUM,
                                                metadata={
                                                    "transitive_path": new_path,
                                                    "contradictory_relationship": contra_rel.id
                                                }
                                            )
                                            verification_result.add_issue(issue)
    
    def _verify_with_llm(self,
                        entity_registry: EntityRegistry,
                        relationship_registry: RelationshipRegistry,
                        language: str) -> Result[VerificationResult]:
        """
        Use LLM to verify the knowledge graph.
        
        Args:
            entity_registry: Registry of entities
            relationship_registry: Registry of relationships
            language: Language code for verification prompts
            
        Returns:
            Result[VerificationResult]: Result containing LLM-identified issues if successful
        """
        # Format the entities and relationships for the prompt
        entities_text = self._format_entities_for_prompt(entity_registry)
        relationships_text = self._format_relationships_for_prompt(relationship_registry, entity_registry)
        
        # Create the prompt for the LLM
        prompt = self._create_verification_prompt(entities_text, relationships_text, language)
        
        # Call the LLM
        llm_result = self.provider.generate_text(prompt, max_tokens=4000)
        
        if not llm_result.success:
            return Result.fail(f"LLM generation failed: {llm_result.error}")
        
        # Parse the LLM response
        return self._parse_llm_verification_response(llm_result.value)
    
    def _format_entities_for_prompt(self, entity_registry: EntityRegistry) -> str:
        """
        Format entities for inclusion in the verification prompt.
        
        Args:
            entity_registry: Registry of entities
            
        Returns:
            str: Formatted entities text
        """
        formatted_entities = []
        
        for entity in entity_registry.all():
            entity_str = (
                f"ID: {entity.id}\n"
                f"Name: {entity.name}\n"
                f"Type: {entity.entity_type}\n"
            )
            
            if entity.attributes:
                attributes_str = ", ".join(f"{k}: {v}" for k, v in entity.attributes.items())
                entity_str += f"Attributes: {attributes_str}\n"
            
            formatted_entities.append(entity_str)
        
        return "\n".join(formatted_entities)
    
    def _format_relationships_for_prompt(self,
                                        relationship_registry: RelationshipRegistry,
                                        entity_registry: EntityRegistry) -> str:
        """
        Format relationships for inclusion in the verification prompt.
        
        Args:
            relationship_registry: Registry of relationships
            entity_registry: Registry of entities
            
        Returns:
            str: Formatted relationships text
        """
        formatted_relationships = []
        
        for rel in relationship_registry.all():
            source_entity = entity_registry.get(rel.source_entity)
            target_entity = entity_registry.get(rel.target_entity)
            
            if not source_entity or not target_entity:
                continue
            
            rel_str = (
                f"ID: {rel.id}\n"
                f"Type: {rel.relation_type}\n"
                f"Source: {source_entity.name} (ID: {rel.source_entity})\n"
                f"Target: {target_entity.name} (ID: {rel.target_entity})\n"
                f"Bidirectional: {rel.bidirectional}\n"
            )
            
            if rel.attributes:
                attributes_str = ", ".join(f"{k}: {v}" for k, v in rel.attributes.items())
                rel_str += f"Attributes: {attributes_str}\n"
            
            formatted_relationships.append(rel_str)
        
        return "\n".join(formatted_relationships)
    
    def _create_verification_prompt(self,
                                  entities_text: str,
                                  relationships_text: str,
                                  language: str) -> str:
        """
        Create a prompt for LLM verification.
        
        Args:
            entities_text: Formatted entities text
            relationships_text: Formatted relationships text
            language: Language code for verification prompt
            
        Returns:
            str: Complete verification prompt
        """
        if language == "ru":
            return f"""
Проверьте следующий граф знаний на внутреннюю согласованность и логические противоречия.

Узлы графа:
{entities_text}

Рёбра графа:
{relationships_text}

Для каждого потенциального противоречия или несогласованности:

1. Опишите несогласованность (например, циклическая зависимость, противоречивые атрибуты)
2. Укажите вовлеченные узлы и рёбра
3. Предложите решение (например, удаление одного из рёбер, уточнение атрибута)
4. Оцените серьезность проблемы (критическая/средняя/низкая)

Также проверьте:
- Избыточные узлы и рёбра
- Отсутствующие важные связи, которые следуют из имеющихся
- Нарушения ограничений предметной области

Представьте результат в JSON-формате с полями: issue_type, involved_elements, proposed_solution, severity.
"""
        else:
            return f"""
Check the following knowledge graph for internal consistency and logical contradictions.

Graph nodes:
{entities_text}

Graph edges:
{relationships_text}

For each potential contradiction or inconsistency:

1. Describe the inconsistency (e.g., cyclic dependency, contradictory attributes)
2. Indicate the involved nodes and edges
3. Suggest a solution (e.g., removing one of the edges, clarifying an attribute)
4. Assess the severity of the problem (critical/medium/low)

Also check for:
- Redundant nodes and edges
- Missing important connections that follow from existing ones
- Violations of domain constraints

Present the result in JSON format with fields: issue_type, involved_elements, proposed_solution, severity.
"""
    
    def _parse_llm_verification_response(self, response: str) -> Result[VerificationResult]:
        """
        Parse the LLM's verification response.
        
        Args:
            response: Response from the LLM
            
        Returns:
            Result[VerificationResult]: Result containing parsed verification issues if successful
        """
        verification_result = VerificationResult()
        
        try:
            # Extract JSON from the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start == -1 or json_end == 0:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                return Result.fail("Failed to extract JSON from LLM response")
            
            json_text = response[json_start:json_end]
            issues_data = json.loads(json_text)
            
            # Handle both list and single object formats
            if isinstance(issues_data, dict):
                issues_data = [issues_data]
            
            for issue_data in issues_data:
                try:
                    issue = VerificationIssue.from_dict(issue_data)
                    verification_result.add_issue(issue)
                except Exception as e:
                    self.logger.warning(f"Failed to parse issue data: {str(e)}")
            
            return Result.ok(verification_result)
        except Exception as e:
            return Result.fail(f"Failed to parse LLM verification response: {str(e)}")
    
    def _initialize_llm_provider(self) -> Result[bool]:
        """
        Initialize the LLM provider.
        
        Returns:
            Result[bool]: Result indicating success or failure
        """
        try:
            provider_name = self.llm_config.provider
            model_name = self.llm_config.model
            
            self.logger.info(f"Initializing LLM provider {provider_name} with model {model_name}")
            
            provider_result = get_provider(provider_name, model_name)
            if not provider_result.success:
                return Result.fail(f"Failed to create provider: {provider_result.error}")
            
            self.provider = provider_result.value
            return Result.ok(True)
        except Exception as e:
            return Result.fail(f"Error initializing LLM provider: {str(e)}")


def verify_knowledge_graph(
    entity_registry: EntityRegistry,
    relationship_registry: RelationshipRegistry,
    language: str = "en",
    config: Optional[AppConfig] = None
) -> Result[VerificationResult]:
    """
    Verify a knowledge graph for consistency and validity.
    
    Convenience function that creates a KnowledgeVerifier and uses it to verify
    the graph.
    
    Args:
        entity_registry: Registry of entities in the graph
        relationship_registry: Registry of relationships in the graph
        language: Language code for verification prompts
        config: Application configuration
        
    Returns:
        Result[VerificationResult]: Result containing verification issues if successful
    """
    verifier = KnowledgeVerifier(config)
    return verifier.verify_knowledge_graph(entity_registry, relationship_registry, language) 