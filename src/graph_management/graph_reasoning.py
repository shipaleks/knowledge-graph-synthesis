"""
Graph Reasoning Module

This module provides functionality for advanced reasoning over knowledge graphs,
including path-based reasoning, conflict detection, and resolution.
"""

import re
from typing import Dict, List, Set, Optional, Any, Union, Tuple, Callable
import networkx as nx

from src.utils.result import Result
from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_query import GraphQuery

# Configure logger
logger = get_logger(__name__)


class GraphReasoning:
    """
    Provides methods for advanced reasoning over knowledge graphs.
    
    This class offers functionality for path-based reasoning, conflict detection,
    and resolution in knowledge graphs.
    """
    
    def __init__(self, graph: KnowledgeGraph, config: Optional[AppConfig] = None):
        """
        Initialize the graph reasoning engine.
        
        Args:
            graph: The knowledge graph to reason over
            config: Optional application configuration
        """
        self.graph = graph
        self.config = config or AppConfig()
        self.query = GraphQuery(graph, config)
    
    def reason_over_path(self, 
                        source_entity_id: str, 
                        target_entity_id: str,
                        max_depth: int = 5,
                        relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform reasoning over paths between two entities.
        
        Args:
            source_entity_id: ID of the source entity
            target_entity_id: ID of the target entity
            max_depth: Maximum path length
            relationship_types: Optional list of relationship types to follow
            
        Returns:
            Dict[str, Any]: Reasoning results including paths and inferences
        """
        # Find paths between entities
        paths = self.query.find_path(
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            max_depth=max_depth,
            relationship_types=relationship_types
        )
        
        if not paths:
            logger.warning(f"No paths found between {source_entity_id} and {target_entity_id}")
            return {
                "source_entity": source_entity_id,
                "target_entity": target_entity_id,
                "paths_found": 0,
                "paths": [],
                "inferences": [],
                "success": False
            }
        
        # Extract path inferences
        inferences = self._extract_path_inferences(paths)
        
        return {
            "source_entity": source_entity_id,
            "target_entity": target_entity_id,
            "paths_found": len(paths),
            "paths": paths,
            "inferences": inferences,
            "success": True
        }
    
    def _extract_path_inferences(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract inferences from paths between entities.
        
        Args:
            paths: List of paths, each containing entities and relationships
            
        Returns:
            List[Dict[str, Any]]: List of inferences derived from paths
        """
        inferences = []
        
        for path_idx, path in enumerate(paths):
            # Skip paths that are too short
            if len(path) < 3:  # Need at least source -> relationship -> target
                continue
                
            # Extract entities and relationships in order
            entities = []
            relationships = []
            
            for item in path:
                if "entity" in item:
                    entities.append(item["entity"])
                elif "relationship" in item:
                    relationships.append(item["relationship"])
            
            # Generate transitive relationship inference if a path has multiple entities connected
            if len(entities) >= 3 and len(relationships) >= 2:
                # Check for has-skill and required-for to infer a qualified-for relationship
                if relationships[0].relation_type.lower() == "has-skill" and relationships[1].relation_type.lower() == "required-for":
                    inference = {
                        "type": "transitive_relation",
                        "source_entity": entities[0].id,
                        "target_entity": entities[2].id,
                        "inferred_relation_type": "qualified-for",
                        "path_index": path_idx,
                        "confidence": min(relationships[0].confidence, relationships[1].confidence) * 0.9,
                        "explanation": f"Transitive relation: If {entities[0].name} has-skill {entities[1].name} "
                                    f"and {entities[1].name} required-for {entities[2].name}, "
                                    f"then {entities[0].name} qualified-for {entities[2].name}"
                    }
                    inferences.append(inference)
                # Check for specific relationship types that might imply transitive relations
                elif relationships[0].relation_type.lower() == relationships[1].relation_type.lower():
                    transitive_types = ["is-a", "part-of", "contains", "causes", "implies"]
                    rel_type = relationships[0].relation_type.lower()
                    
                    if rel_type in transitive_types:
                        inference = {
                            "type": "transitive_relation",
                            "source_entity": entities[0].id,
                            "target_entity": entities[2].id,
                            "inferred_relation_type": rel_type,
                            "path_index": path_idx,
                            "confidence": min(relationships[0].confidence, relationships[1].confidence) * 0.9,
                            "explanation": f"Transitive relation: If {entities[0].name} {rel_type} {entities[1].name} "
                                        f"and {entities[1].name} {rel_type} {entities[2].name}, "
                                        f"then {entities[0].name} {rel_type} {entities[2].name}"
                        }
                        inferences.append(inference)
            
            # Look for circular reasoning (cycles) in longer paths
            if len(entities) > 3:
                # Check if first and last entity are the same
                if entities[0].id == entities[-1].id:
                    inference = {
                        "type": "circular_reasoning",
                        "entities_involved": [e.id for e in entities],
                        "path_index": path_idx,
                        "confidence": 0.7,
                        "explanation": f"Circular reasoning detected: Path starts and ends with {entities[0].name}"
                    }
                    inferences.append(inference)
        
        return inferences
    
    def detect_conflicts(self, 
                        confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect logical conflicts in the knowledge graph.
        
        Args:
            confidence_threshold: Minimum confidence level for conflict detection
            
        Returns:
            List[Dict[str, Any]]: List of detected conflicts
        """
        conflicts = []
        
        # Detect contradictory relationships
        contradictory_relationships = self._detect_contradictory_relationships(confidence_threshold)
        conflicts.extend(contradictory_relationships)
        
        # Detect property value conflicts
        property_conflicts = self._detect_property_conflicts(confidence_threshold)
        conflicts.extend(property_conflicts)
        
        # Detect semantic inconsistencies
        semantic_conflicts = self._detect_semantic_inconsistencies(confidence_threshold)
        conflicts.extend(semantic_conflicts)
        
        return conflicts
    
    def _detect_contradictory_relationships(self, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Detect contradictory relationships in the graph.
        
        Args:
            confidence_threshold: Minimum confidence level for detection
            
        Returns:
            List[Dict[str, Any]]: List of contradictory relationship conflicts
        """
        conflicts = []
        
        # Define opposite relationship types
        opposites = {
            "works-for": ["not-affiliated-with", "unrelated-to"],
            "is-a": ["is-not-a", "different-from"],
            "part-of": ["separate-from", "not-part-of"],
            "contains": ["does-not-contain", "excludes"],
            "causes": ["prevents", "unrelated-to"],
            "before": ["after", "simultaneous-with"],
            "implies": ["contradicts", "unrelated-to"]
        }
        
        # Add inverse mappings
        for key, values in list(opposites.items()):
            for value in values:
                if value not in opposites:
                    opposites[value] = [key]
                else:
                    opposites[value].append(key)
        
        # Check all relationships for contradictions
        for rel_id, rel in self.graph.relationship_map.items():
            # Skip low confidence relationships
            if rel.confidence < confidence_threshold:
                continue
                
            rel_type = rel.relation_type.lower()
            
            # Skip if no known opposites
            if rel_type not in opposites:
                continue
                
            # Find contradicting relationships
            for other_rel_id, other_rel in self.graph.relationship_map.items():
                # Skip if same relationship or low confidence
                if rel_id == other_rel_id or other_rel.confidence < confidence_threshold:
                    continue
                    
                other_rel_type = other_rel.relation_type.lower()
                
                # Check if there's a contradiction between same entities
                if (rel.source_entity == other_rel.source_entity and
                    rel.target_entity == other_rel.target_entity and
                    other_rel_type in opposites.get(rel_type, [])):
                    
                    source_entity = self.graph.get_entity(rel.source_entity)
                    target_entity = self.graph.get_entity(rel.target_entity)
                    
                    if source_entity and target_entity:
                        conflict = {
                            "type": "contradictory_relationship",
                            "relationships": [rel_id, other_rel_id],
                            "source_entity": rel.source_entity,
                            "target_entity": rel.target_entity,
                            "explanation": f"Contradictory relationships: '{rel_type}' and '{other_rel_type}' "
                                        f"between {source_entity.name} and {target_entity.name}",
                            "severity": "high",
                            "confidence": min(rel.confidence, other_rel.confidence)
                        }
                        conflicts.append(conflict)
                    
        return conflicts
    
    def _detect_property_conflicts(self, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Detect conflicts in entity property values.
        
        Args:
            confidence_threshold: Minimum confidence level for detection
            
        Returns:
            List[Dict[str, Any]]: List of property value conflicts
        """
        conflicts = []
        
        # Get all entities with same name but different attribute values
        entities_by_name = {}
        for entity_id, entity in self.graph.entity_map.items():
            if entity.confidence < confidence_threshold:
                continue
                
            # Group by name
            if entity.name not in entities_by_name:
                entities_by_name[entity.name] = []
            entities_by_name[entity.name].append(entity)
        
        # Check for attribute conflicts within same-named entities
        for name, entities in entities_by_name.items():
            if len(entities) < 2:
                continue
                
            # Compare attributes between all pairs
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    entity1 = entities[i]
                    entity2 = entities[j]
                    
                    # Skip if different entity types
                    if entity1.entity_type != entity2.entity_type:
                        continue
                        
                    # Find conflicting attributes
                    conflicting_attrs = []
                    
                    # Check common attributes
                    common_attrs = set(entity1.attributes.keys()) & set(entity2.attributes.keys())
                    for attr in common_attrs:
                        if entity1.attributes[attr] != entity2.attributes[attr]:
                            conflicting_attrs.append({
                                "name": attr,
                                "value1": entity1.attributes[attr],
                                "value2": entity2.attributes[attr]
                            })
                    
                    if conflicting_attrs:
                        conflict = {
                            "type": "property_conflict",
                            "entities": [entity1.id, entity2.id],
                            "entity_name": name,
                            "entity_type": entity1.entity_type,
                            "conflicting_attributes": conflicting_attrs,
                            "explanation": f"Conflicting attribute values for entities with same name '{name}'",
                            "severity": "medium",
                            "confidence": min(entity1.confidence, entity2.confidence)
                        }
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_semantic_inconsistencies(self, confidence_threshold: float) -> List[Dict[str, Any]]:
        """
        Detect semantic inconsistencies in the graph.
        
        Args:
            confidence_threshold: Minimum confidence level for detection
            
        Returns:
            List[Dict[str, Any]]: List of semantic inconsistency conflicts
        """
        conflicts = []
        
        # Check for transitive relationship inconsistencies (e.g., circular hierarchies)
        transitive_relation_types = ["is-a", "part-of", "contains"]
        
        for rel_type in transitive_relation_types:
            # Build directed graph for each transitive relation type
            directed_graph = nx.DiGraph()
            
            # Add edges for this relationship type
            for rel_id, rel in self.graph.relationship_map.items():
                if rel.relation_type.lower() == rel_type and rel.confidence >= confidence_threshold:
                    directed_graph.add_edge(rel.source_entity, rel.target_entity)
            
            # Skip if graph is empty
            if len(directed_graph.nodes()) == 0:
                continue
            
            # Check for cycles
            try:
                cycles = list(nx.simple_cycles(directed_graph))
                for cycle in cycles:
                    if len(cycle) >= 2:  # Only consider cycles with at least 2 nodes
                        # Get entity names for readability
                        entity_names = []
                        for entity_id in cycle:
                            entity = self.graph.get_entity(entity_id)
                            if entity:
                                entity_names.append(entity.name)
                            else:
                                entity_names.append(entity_id)
                        
                        cycle_str = " -> ".join(entity_names) + " -> " + entity_names[0]
                        
                        conflict = {
                            "type": "circular_hierarchy",
                            "relation_type": rel_type,
                            "entities": cycle,
                            "explanation": f"Circular hierarchy detected with relation '{rel_type}': {cycle_str}",
                            "severity": "high",
                            "confidence": 0.9  # High confidence for cycle detection
                        }
                        conflicts.append(conflict)
            except nx.NetworkXError:
                # Error in cycle detection (possibly empty graph)
                pass
        
        return conflicts
    
    def resolve_conflicts(self, 
                         conflicts: List[Dict[str, Any]]) -> Result[Dict[str, Any]]:
        """
        Attempt to resolve detected conflicts in the graph.
        
        Args:
            conflicts: List of conflicts to resolve
            
        Returns:
            Result[Dict[str, Any]]: Result containing resolution actions and a new graph,
                                  or an error if resolution failed
        """
        if not conflicts:
            return Result.ok({
                "message": "No conflicts to resolve",
                "actions_taken": [],
                "graph": self.graph
            })
        
        # Create a copy of the graph to perform resolutions
        resolved_graph = KnowledgeGraph(name=f"resolved_{self.graph.name}", config=self.config)
        
        # Copy all entities
        for entity_id, entity in self.graph.entity_map.items():
            resolved_graph.add_entity(entity)
        
        # Copy all relationships
        for rel_id, rel in self.graph.relationship_map.items():
            resolved_graph.add_relationship(rel)
        
        # Track resolution actions
        actions = []
        
        # Resolve each conflict
        for conflict in conflicts:
            action = self._resolve_single_conflict(conflict, resolved_graph)
            if action:
                actions.append(action)
        
        return Result.ok({
            "message": f"Resolved {len(actions)} conflicts",
            "actions_taken": actions,
            "graph": resolved_graph
        })
    
    def _resolve_single_conflict(self, 
                               conflict: Dict[str, Any], 
                               graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Resolve a single conflict in the graph.
        
        Args:
            conflict: Conflict to resolve
            graph: Graph to modify for resolution
            
        Returns:
            Dict[str, Any]: Description of the action taken to resolve the conflict
        """
        conflict_type = conflict.get("type")
        
        if conflict_type == "contradictory_relationship":
            return self._resolve_contradictory_relationship(conflict, graph)
        elif conflict_type == "property_conflict":
            return self._resolve_property_conflict(conflict, graph)
        elif conflict_type == "circular_hierarchy":
            return self._resolve_circular_hierarchy(conflict, graph)
        else:
            logger.warning(f"Unknown conflict type: {conflict_type}")
            return None
    
    def _resolve_contradictory_relationship(self, 
                                          conflict: Dict[str, Any], 
                                          graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Resolve a contradictory relationship conflict.
        
        Args:
            conflict: Conflict to resolve
            graph: Graph to modify for resolution
            
        Returns:
            Dict[str, Any]: Description of the action taken
        """
        rel_ids = conflict.get("relationships", [])
        
        if len(rel_ids) < 2:
            return None
        
        # Get relationships
        relationships = []
        for rel_id in rel_ids:
            rel = self.graph.get_relationship(rel_id)
            if rel:
                relationships.append(rel)
        
        if len(relationships) < 2:
            return None
        
        # Keep the relationship with higher confidence
        relationships.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove all but the highest confidence relationship
        for rel in relationships[1:]:
            graph.remove_relationship(rel.id)
        
        return {
            "action_type": "remove_contradictory_relationships",
            "kept_relationship": relationships[0].id,
            "removed_relationships": [rel.id for rel in relationships[1:]],
            "explanation": f"Kept relationship {relationships[0].relation_type} with higher confidence "
                         f"({relationships[0].confidence:.2f}) and removed contradictory relationships"
        }
    
    def _resolve_property_conflict(self, 
                                 conflict: Dict[str, Any], 
                                 graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Resolve a property conflict between entities.
        
        Args:
            conflict: Conflict to resolve
            graph: Graph to modify for resolution
            
        Returns:
            Dict[str, Any]: Description of the action taken
        """
        entity_ids = conflict.get("entities", [])
        
        if len(entity_ids) < 2:
            return None
        
        # Get entities
        entities = []
        for entity_id in entity_ids:
            entity = self.graph.get_entity(entity_id)
            if entity:
                entities.append(entity)
        
        if len(entities) < 2:
            return None
        
        # Sort entities by confidence
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        # Merge attributes from the highest confidence entity into the others
        highest_confidence_entity = entities[0]
        conflicting_attrs = conflict.get("conflicting_attributes", [])
        
        for entity in entities[1:]:
            # Update entity in the graph with attributes from highest confidence entity
            updated_entity = graph.get_entity(entity.id)
            if updated_entity:
                for attr_conflict in conflicting_attrs:
                    attr_name = attr_conflict.get("name")
                    if attr_name:
                        updated_entity.attributes[attr_name] = highest_confidence_entity.attributes.get(attr_name)
        
        return {
            "action_type": "merge_entity_attributes",
            "source_entity": highest_confidence_entity.id,
            "target_entities": [entity.id for entity in entities[1:]],
            "updated_attributes": [attr.get("name") for attr in conflicting_attrs],
            "explanation": f"Updated attributes in entities with lower confidence to match "
                         f"values from highest confidence entity '{highest_confidence_entity.name}'"
        }
    
    def _resolve_circular_hierarchy(self, 
                                  conflict: Dict[str, Any], 
                                  graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Resolve a circular hierarchy conflict.
        
        Args:
            conflict: Conflict to resolve
            graph: Graph to modify for resolution
            
        Returns:
            Dict[str, Any]: Description of the action taken
        """
        entity_ids = conflict.get("entities", [])
        relation_type = conflict.get("relation_type")
        
        if not entity_ids or len(entity_ids) < 2 or not relation_type:
            return None
        
        # Find the relationship with lowest confidence in the cycle
        lowest_confidence = float('inf')
        lowest_rel_id = None
        
        # Check relationships in the cycle
        for i in range(len(entity_ids)):
            source_id = entity_ids[i]
            target_id = entity_ids[(i + 1) % len(entity_ids)]
            
            # Find the relationship
            for rel_id, rel in self.graph.relationship_map.items():
                if (rel.source_entity == source_id and 
                    rel.target_entity == target_id and
                    rel.relation_type.lower() == relation_type.lower()):
                    
                    if rel.confidence < lowest_confidence:
                        lowest_confidence = rel.confidence
                        lowest_rel_id = rel.id
        
        if not lowest_rel_id:
            return None
        
        # Remove the relationship with lowest confidence
        graph.remove_relationship(lowest_rel_id)
        
        return {
            "action_type": "break_circular_hierarchy",
            "removed_relationship": lowest_rel_id,
            "relation_type": relation_type,
            "entities": entity_ids,
            "explanation": f"Removed lowest confidence relationship in circular hierarchy "
                         f"of type '{relation_type}'"
        }
    
    def infer_new_relationships(self, 
                              min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Infer new relationships based on existing knowledge.
        
        Args:
            min_confidence: Minimum confidence for inferred relationships
            
        Returns:
            List[Dict[str, Any]]: List of inferred relationships
        """
        inferences = []
        
        # Infer transitive relationships
        transitive_inferences = self._infer_transitive_relationships(min_confidence)
        inferences.extend(transitive_inferences)
        
        # Infer symmetric relationships
        symmetric_inferences = self._infer_symmetric_relationships(min_confidence)
        inferences.extend(symmetric_inferences)
        
        # Infer inverse relationships
        inverse_inferences = self._infer_inverse_relationships(min_confidence)
        inferences.extend(inverse_inferences)
        
        return inferences
    
    def _infer_transitive_relationships(self, min_confidence: float) -> List[Dict[str, Any]]:
        """
        Infer transitive relationships in the graph.
        
        Args:
            min_confidence: Minimum confidence for inferred relationships
            
        Returns:
            List[Dict[str, Any]]: List of inferred transitive relationships
        """
        inferences = []
        
        # Define transitive relationship types
        transitive_types = ["is-a", "part-of", "contains", "implies", "causes"]
        
        # Special case for has-skill and required-for
        has_skill_required_for = {
            "source_type": "has-skill",
            "target_type": "required-for",
            "inferred_type": "qualified-for"
        }
        
        # Check for has-skill and required-for combination
        # First, find all has-skill relationships
        has_skill_rels = {}
        for rel_id, rel in self.graph.relationship_map.items():
            if rel.relation_type.lower() == "has-skill" and rel.confidence >= min_confidence:
                # Map person to skills
                if rel.source_entity not in has_skill_rels:
                    has_skill_rels[rel.source_entity] = []
                
                has_skill_rels[rel.source_entity].append({
                    "skill_id": rel.target_entity,
                    "confidence": rel.confidence
                })
        
        # Then, find all required-for relationships
        required_for_rels = {}
        for rel_id, rel in self.graph.relationship_map.items():
            if rel.relation_type.lower() == "required-for" and rel.confidence >= min_confidence:
                # Map skill to jobs
                if rel.source_entity not in required_for_rels:
                    required_for_rels[rel.source_entity] = []
                
                required_for_rels[rel.source_entity].append({
                    "job_id": rel.target_entity,
                    "confidence": rel.confidence
                })
        
        # Now, infer qualified-for relationships
        for person_id, skills in has_skill_rels.items():
            for skill_info in skills:
                skill_id = skill_info["skill_id"]
                skill_confidence = skill_info["confidence"]
                
                if skill_id in required_for_rels:
                    for job_info in required_for_rels[skill_id]:
                        job_id = job_info["job_id"]
                        job_confidence = job_info["confidence"]
                        
                        # Compute combined confidence
                        combined_confidence = skill_confidence * job_confidence * 0.9
                        
                        if combined_confidence >= min_confidence:
                            person_entity = self.graph.get_entity(person_id)
                            job_entity = self.graph.get_entity(job_id)
                            skill_entity = self.graph.get_entity(skill_id)
                            
                            if person_entity and job_entity and skill_entity:
                                inference = {
                                    "type": "transitive_relationship",
                                    "relation_type": "qualified-for",
                                    "source_entity": person_id,
                                    "target_entity": job_id,
                                    "source_name": person_entity.name,
                                    "target_name": job_entity.name,
                                    "confidence": combined_confidence,
                                    "explanation": f"Inferred 'qualified-for' relationship: {person_entity.name} "
                                                 f"has-skill {skill_entity.name} and {skill_entity.name} "
                                                 f"required-for {job_entity.name}, therefore {person_entity.name} "
                                                 f"qualified-for {job_entity.name}"
                                }
                                inferences.append(inference)
        
        # Process standard transitive relationships
        for rel_type in transitive_types:
            # Build directed graph for each transitive relation type
            directed_graph = nx.DiGraph()
            
            # Map to store original relationship data
            rel_data = {}
            
            # Add edges for this relationship type
            for rel_id, rel in self.graph.relationship_map.items():
                if rel.relation_type.lower() == rel_type:
                    directed_graph.add_edge(rel.source_entity, rel.target_entity)
                    
                    # Store relationship data
                    key = (rel.source_entity, rel.target_entity)
                    rel_data[key] = {
                        "id": rel.id,
                        "confidence": rel.confidence,
                        "attributes": rel.attributes
                    }
            
            # Skip if graph is empty
            if len(directed_graph.nodes()) == 0:
                continue
            
            # For each pair of nodes, find paths of length > 1
            for source in directed_graph.nodes():
                for target in directed_graph.nodes():
                    # Skip self-relationships
                    if source == target:
                        continue
                        
                    # Skip if direct relationship already exists
                    if directed_graph.has_edge(source, target):
                        continue
                    
                    # Find all simple paths from source to target
                    try:
                        paths = list(nx.all_simple_paths(directed_graph, source, target, cutoff=3))
                        
                        if paths:
                            # Calculate confidence based on path relationships
                            path_confidences = []
                            
                            for path in paths:
                                if len(path) < 2:
                                    continue
                                    
                                # Calculate path confidence as product of relationship confidences
                                confidence = 1.0
                                for i in range(len(path) - 1):
                                    key = (path[i], path[i+1])
                                    rel_confidence = rel_data.get(key, {}).get("confidence", 0.5)
                                    confidence *= rel_confidence
                                
                                # Apply decay factor based on path length
                                confidence *= 0.9 ** (len(path) - 1)
                                
                                path_confidences.append(confidence)
                            
                            # Use maximum confidence from all paths
                            if path_confidences:
                                max_confidence = max(path_confidences)
                                
                                # Only include if confidence meets threshold
                                if max_confidence >= min_confidence:
                                    source_entity = self.graph.get_entity(source)
                                    target_entity = self.graph.get_entity(target)
                                    
                                    if source_entity and target_entity:
                                        inference = {
                                            "type": "transitive_relationship",
                                            "relation_type": rel_type,
                                            "source_entity": source,
                                            "target_entity": target,
                                            "source_name": source_entity.name,
                                            "target_name": target_entity.name,
                                            "confidence": max_confidence,
                                            "paths_count": len(paths),
                                            "explanation": f"Inferred '{rel_type}' relationship from {len(paths)} transitive paths"
                                        }
                                        inferences.append(inference)
                    
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # No path exists
                        pass
        
        return inferences
    
    def _infer_symmetric_relationships(self, min_confidence: float) -> List[Dict[str, Any]]:
        """
        Infer symmetric relationships in the graph.
        
        Args:
            min_confidence: Minimum confidence for inferred relationships
            
        Returns:
            List[Dict[str, Any]]: List of inferred symmetric relationships
        """
        inferences = []
        
        # Define symmetric relationship types
        symmetric_types = ["similar-to", "related-to", "connected-with", "interacts-with"]
        
        # Find symmetric relationships with only one direction
        for rel_id, rel in self.graph.relationship_map.items():
            if rel.relation_type.lower() in symmetric_types and rel.confidence >= min_confidence:
                # Check if reverse relationship exists
                reverse_exists = False
                
                for other_rel_id, other_rel in self.graph.relationship_map.items():
                    if (other_rel.source_entity == rel.target_entity and
                        other_rel.target_entity == rel.source_entity and
                        other_rel.relation_type.lower() == rel.relation_type.lower()):
                        reverse_exists = True
                        break
                
                # If reverse doesn't exist, suggest it
                if not reverse_exists:
                    source_entity = self.graph.get_entity(rel.target_entity)
                    target_entity = self.graph.get_entity(rel.source_entity)
                    
                    if source_entity and target_entity:
                        inference = {
                            "type": "symmetric_relationship",
                            "relation_type": rel.relation_type,
                            "source_entity": rel.target_entity,
                            "target_entity": rel.source_entity,
                            "source_name": source_entity.name,
                            "target_name": target_entity.name,
                            "confidence": rel.confidence * 0.95,  # Slightly lower confidence for the inference
                            "based_on_relationship": rel.id,
                            "explanation": f"Inferred symmetric '{rel.relation_type}' relationship in reverse direction"
                        }
                        inferences.append(inference)
        
        return inferences
    
    def _infer_inverse_relationships(self, min_confidence: float) -> List[Dict[str, Any]]:
        """
        Infer inverse relationships in the graph.
        
        Args:
            min_confidence: Minimum confidence for inferred relationships
            
        Returns:
            List[Dict[str, Any]]: List of inferred inverse relationships
        """
        inferences = []
        
        # Define inverse relationship pairs
        inverse_pairs = {
            "contains": "part-of",
            "broader-than": "narrower-than",
            "causes": "caused-by",
            "precedes": "follows",
            "parent-of": "child-of",
            "works-for": "employs"
        }
        
        # Add inverse mapping
        for rel1, rel2 in list(inverse_pairs.items()):
            inverse_pairs[rel2] = rel1
        
        # Find relationships where inverse could be inferred
        for rel_id, rel in self.graph.relationship_map.items():
            rel_type = rel.relation_type.lower()
            
            if rel_type in inverse_pairs and rel.confidence >= min_confidence:
                inverse_type = inverse_pairs[rel_type]
                
                # Check if inverse relationship exists
                inverse_exists = False
                
                for other_rel_id, other_rel in self.graph.relationship_map.items():
                    if (other_rel.source_entity == rel.target_entity and
                        other_rel.target_entity == rel.source_entity and
                        other_rel.relation_type.lower() == inverse_type):
                        inverse_exists = True
                        break
                
                # If inverse doesn't exist, suggest it
                if not inverse_exists:
                    source_entity = self.graph.get_entity(rel.target_entity)
                    target_entity = self.graph.get_entity(rel.source_entity)
                    
                    if source_entity and target_entity:
                        inference = {
                            "type": "inverse_relationship",
                            "relation_type": inverse_type,
                            "source_entity": rel.target_entity,
                            "target_entity": rel.source_entity,
                            "source_name": source_entity.name,
                            "target_name": target_entity.name,
                            "confidence": rel.confidence * 0.95,  # Slightly lower confidence for the inference
                            "based_on_relationship": rel.id,
                            "explanation": f"Inferred inverse '{inverse_type}' relationship from '{rel.relation_type}'"
                        }
                        inferences.append(inference)
        
        return inferences


# Helper functions for common reasoning patterns

def reason_over_paths(graph: KnowledgeGraph, 
                     source_entity_id: str,
                     target_entity_id: str,
                     max_depth: int = 5) -> Dict[str, Any]:
    """
    Perform reasoning over paths between two entities.
    
    Args:
        graph: Knowledge graph to reason over
        source_entity_id: ID of the source entity
        target_entity_id: ID of the target entity
        max_depth: Maximum path length
        
    Returns:
        Dict[str, Any]: Reasoning results
    """
    reasoner = GraphReasoning(graph)
    return reasoner.reason_over_path(source_entity_id, target_entity_id, max_depth)


def detect_graph_conflicts(graph: KnowledgeGraph,
                          confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect conflicts in the knowledge graph.
    
    Args:
        graph: Knowledge graph to analyze
        confidence_threshold: Minimum confidence level for conflict detection
        
    Returns:
        List[Dict[str, Any]]: List of detected conflicts
    """
    reasoner = GraphReasoning(graph)
    return reasoner.detect_conflicts(confidence_threshold)


def resolve_graph_conflicts(graph: KnowledgeGraph,
                           confidence_threshold: float = 0.5) -> Result[Dict[str, Any]]:
    """
    Detect and resolve conflicts in the knowledge graph.
    
    Args:
        graph: Knowledge graph to analyze and resolve
        confidence_threshold: Minimum confidence level for conflict detection
        
    Returns:
        Result[Dict[str, Any]]: Result containing resolution actions and a new graph
    """
    reasoner = GraphReasoning(graph)
    conflicts = reasoner.detect_conflicts(confidence_threshold)
    
    if not conflicts:
        return Result.ok({
            "message": "No conflicts detected",
            "actions_taken": [],
            "graph": graph
        })
    
    return reasoner.resolve_conflicts(conflicts)


def infer_new_knowledge(graph: KnowledgeGraph,
                       min_confidence: float = 0.7) -> List[Dict[str, Any]]:
    """
    Infer new relationships based on existing knowledge.
    
    Args:
        graph: Knowledge graph to analyze
        min_confidence: Minimum confidence for inferred relationships
        
    Returns:
        List[Dict[str, Any]]: List of inferred relationships
    """
    reasoner = GraphReasoning(graph)
    return reasoner.infer_new_relationships(min_confidence) 