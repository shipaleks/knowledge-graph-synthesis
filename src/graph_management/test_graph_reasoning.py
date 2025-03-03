"""
Test module for Graph Reasoning functionality.

This module contains tests for the GraphReasoning class and related functions.
"""

import unittest
from typing import Dict, List, Any
import networkx as nx

from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_reasoning import GraphReasoning, reason_over_paths, detect_graph_conflicts, resolve_graph_conflicts, infer_new_knowledge


class TestGraphReasoning(unittest.TestCase):
    """Test cases for the GraphReasoning class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AppConfig()
        self.graph = self._create_test_graph()
        self.reasoner = GraphReasoning(self.graph, self.config)
    
    def test_reason_over_path(self):
        """Test reasoning over paths between entities."""
        # Test path reasoning between related entities
        result = self.reasoner.reason_over_path(
            source_entity_id="person1",
            target_entity_id="company1"
        )
        
        self.assertTrue(result["success"])
        self.assertGreater(result["paths_found"], 0)
        
        # Test path reasoning between unrelated entities
        result = self.reasoner.reason_over_path(
            source_entity_id="person1",
            target_entity_id="nonexistent"
        )
        
        self.assertFalse(result["success"])
        self.assertEqual(result["paths_found"], 0)
        
        # Helper function should give similar results
        helper_result = reason_over_paths(
            graph=self.graph,
            source_entity_id="person1",
            target_entity_id="company1"
        )
        
        self.assertTrue(helper_result["success"])
        self.assertGreater(helper_result["paths_found"], 0)
        
    def test_extract_path_inferences(self):
        """Test extraction of inferences from paths."""
        # Create a simple path with transitive relationship
        person = self.graph.get_entity("person1")
        skill = self.graph.get_entity("skill1")
        job = self.graph.get_entity("job1")
        
        rel1 = self.graph.get_relationship("rel_person1_skill1")
        rel2 = self.graph.get_relationship("rel_skill1_job1")
        
        path = [
            {"entity": person},
            {"relationship": rel1},
            {"entity": skill},
            {"relationship": rel2},
            {"entity": job}
        ]
        
        inferences = self.reasoner._extract_path_inferences([path])
        
        # Should extract a transitive relation inference
        self.assertGreater(len(inferences), 0)
        self.assertEqual(inferences[0]["type"], "transitive_relation")
        self.assertEqual(inferences[0]["source_entity"], "person1")
        self.assertEqual(inferences[0]["target_entity"], "job1")
    
    def test_detect_conflicts(self):
        """Test detection of conflicts in the graph."""
        # Create conflicting relationships in the graph
        self._add_contradictory_relationships()
        
        conflicts = self.reasoner.detect_conflicts()
        
        # Should detect contradictory relationships
        self.assertGreater(len(conflicts), 0)
        
        # Helper function should give similar results
        helper_conflicts = detect_graph_conflicts(self.graph)
        self.assertGreater(len(helper_conflicts), 0)
    
    def test_resolve_conflicts(self):
        """Test resolution of conflicts in the graph."""
        # Create conflicting relationships in the graph
        self._add_contradictory_relationships()
        
        # Detect conflicts
        conflicts = self.reasoner.detect_conflicts()
        
        # Resolve conflicts
        result = self.reasoner.resolve_conflicts(conflicts)
        
        self.assertTrue(result.success)
        self.assertGreater(len(result.value["actions_taken"]), 0)
        
        # The resolved graph should have fewer relationships
        resolved_graph = result.value["graph"]
        self.assertLess(len(resolved_graph.relationship_map), len(self.graph.relationship_map))
        
        # Helper function should give similar results
        helper_result = resolve_graph_conflicts(self.graph)
        self.assertTrue(helper_result.success)
    
    def test_infer_new_relationships(self):
        """Test inference of new relationships."""
        # Test inference of new relationships
        inferences = self.reasoner.infer_new_relationships()
        
        # Should infer some relationships
        self.assertGreater(len(inferences), 0)
        
        # Helper function should give similar results
        helper_inferences = infer_new_knowledge(self.graph)
        self.assertGreater(len(helper_inferences), 0)
    
    def test_infer_transitive_relationships(self):
        """Test inference of transitive relationships."""
        # Test transitive relationship inference
        inferences = self.reasoner._infer_transitive_relationships(min_confidence=0.6)
        
        # Check for transitive inferences
        transitive_inferences = [inf for inf in inferences if inf["type"] == "transitive_relationship"]
        self.assertGreater(len(transitive_inferences), 0)
    
    def test_infer_symmetric_relationships(self):
        """Test inference of symmetric relationships."""
        # Add a symmetric relationship
        rel = Relationship(
            id="rel_symmetric",
            source_entity="person1",
            target_entity="person2",
            relation_type="similar-to",
            confidence=0.9
        )
        self.graph.add_relationship(rel)
        
        # Test symmetric relationship inference
        inferences = self.reasoner._infer_symmetric_relationships(min_confidence=0.7)
        
        # Should infer the reverse direction
        self.assertGreater(len(inferences), 0)
        self.assertEqual(inferences[0]["type"], "symmetric_relationship")
        self.assertEqual(inferences[0]["source_entity"], "person2")
        self.assertEqual(inferences[0]["target_entity"], "person1")
    
    def test_infer_inverse_relationships(self):
        """Test inference of inverse relationships."""
        # Add a relationship with a known inverse
        rel = Relationship(
            id="rel_inverse",
            source_entity="company1",
            target_entity="person1",
            relation_type="contains",
            confidence=0.9
        )
        self.graph.add_relationship(rel)
        
        # Test inverse relationship inference
        inferences = self.reasoner._infer_inverse_relationships(min_confidence=0.7)
        
        # Should infer the inverse relationship
        inverse_inferences = [inf for inf in inferences if inf["type"] == "inverse_relationship"]
        self.assertGreater(len(inverse_inferences), 0)
        
        # At least one inference should be "part-of" (inverse of "contains")
        part_of_inferences = [inf for inf in inverse_inferences if inf["relation_type"] == "part-of"]
        self.assertGreater(len(part_of_inferences), 0)
    
    def test_detect_contradictory_relationships(self):
        """Test detection of contradictory relationships."""
        # Add contradictory relationships
        self._add_contradictory_relationships()
        
        # Detect contradictory relationships
        conflicts = self.reasoner._detect_contradictory_relationships(confidence_threshold=0.5)
        
        # Should detect contradictions
        self.assertGreater(len(conflicts), 0)
        self.assertEqual(conflicts[0]["type"], "contradictory_relationship")
    
    def test_detect_property_conflicts(self):
        """Test detection of property conflicts."""
        # Add entities with conflicting properties
        entity1 = Entity(
            id="conflicting1",
            name="ConflictEntity",
            entity_type="test",
            attributes={"size": "large", "color": "red"},
            confidence=0.8
        )
        
        entity2 = Entity(
            id="conflicting2",
            name="ConflictEntity",
            entity_type="test",
            attributes={"size": "small", "color": "red"},
            confidence=0.7
        )
        
        self.graph.add_entity(entity1)
        self.graph.add_entity(entity2)
        
        # Detect property conflicts
        conflicts = self.reasoner._detect_property_conflicts(confidence_threshold=0.5)
        
        # Should detect conflicts
        self.assertGreater(len(conflicts), 0)
        self.assertEqual(conflicts[0]["type"], "property_conflict")
    
    def test_detect_semantic_inconsistencies(self):
        """Test detection of semantic inconsistencies."""
        # Create a circular hierarchy
        rel1 = Relationship(
            id="circular1",
            source_entity="entity1",
            target_entity="entity2",
            relation_type="is-a",
            confidence=0.8
        )
        
        rel2 = Relationship(
            id="circular2",
            source_entity="entity2",
            target_entity="entity3",
            relation_type="is-a",
            confidence=0.8
        )
        
        rel3 = Relationship(
            id="circular3",
            source_entity="entity3",
            target_entity="entity1",
            relation_type="is-a",
            confidence=0.8
        )
        
        self.graph.add_relationship(rel1)
        self.graph.add_relationship(rel2)
        self.graph.add_relationship(rel3)
        
        # Detect semantic inconsistencies
        conflicts = self.reasoner._detect_semantic_inconsistencies(confidence_threshold=0.5)
        
        # Should detect circular hierarchy
        self.assertGreater(len(conflicts), 0)
        circular_conflicts = [c for c in conflicts if c["type"] == "circular_hierarchy"]
        self.assertGreater(len(circular_conflicts), 0)
    
    def test_resolve_contradictory_relationship(self):
        """Test resolution of contradictory relationships."""
        # Add contradictory relationships
        self._add_contradictory_relationships()
        
        # Detect contradictory relationships
        conflicts = self.reasoner._detect_contradictory_relationships(confidence_threshold=0.5)
        
        # Create a copy of the graph for resolution
        resolved_graph = KnowledgeGraph(name="resolved", config=self.config)
        
        # Copy all entities and relationships
        for entity_id, entity in self.graph.entity_map.items():
            resolved_graph.add_entity(entity)
        
        for rel_id, rel in self.graph.relationship_map.items():
            resolved_graph.add_relationship(rel)
        
        # Resolve the first conflict
        action = self.reasoner._resolve_contradictory_relationship(conflicts[0], resolved_graph)
        
        # Should have produced an action
        self.assertIsNotNone(action)
        self.assertEqual(action["action_type"], "remove_contradictory_relationships")
        
        # Should have removed the lower confidence relationship
        self.assertGreater(len(action["removed_relationships"]), 0)
    
    def test_resolve_property_conflict(self):
        """Test resolution of property conflicts."""
        # Add entities with conflicting properties
        entity1 = Entity(
            id="conflicting1",
            name="ConflictEntity",
            entity_type="test",
            attributes={"size": "large", "color": "red"},
            confidence=0.8
        )
        
        entity2 = Entity(
            id="conflicting2",
            name="ConflictEntity",
            entity_type="test",
            attributes={"size": "small", "color": "red"},
            confidence=0.7
        )
        
        self.graph.add_entity(entity1)
        self.graph.add_entity(entity2)
        
        # Detect property conflicts
        conflicts = self.reasoner._detect_property_conflicts(confidence_threshold=0.5)
        
        # Create a copy of the graph for resolution
        resolved_graph = KnowledgeGraph(name="resolved", config=self.config)
        
        # Copy all entities and relationships
        for entity_id, entity in self.graph.entity_map.items():
            resolved_graph.add_entity(entity)
        
        for rel_id, rel in self.graph.relationship_map.items():
            resolved_graph.add_relationship(rel)
        
        # Resolve the first conflict
        action = self.reasoner._resolve_property_conflict(conflicts[0], resolved_graph)
        
        # Should have produced an action
        self.assertIsNotNone(action)
        self.assertEqual(action["action_type"], "merge_entity_attributes")
        
        # The entity with lower confidence should have its attributes updated
        updated_entity = resolved_graph.get_entity("conflicting2")
        self.assertEqual(updated_entity.attributes["size"], "large")
    
    def _create_test_graph(self) -> KnowledgeGraph:
        """Create a test graph for use in tests."""
        graph = KnowledgeGraph(name="test_graph", config=self.config)
        
        # Add entities
        entities = [
            Entity(id="person1", name="John Doe", entity_type="Person", 
                  attributes={"age": 30}, confidence=0.9),
            Entity(id="person2", name="Jane Smith", entity_type="Person", 
                  attributes={"age": 35}, confidence=0.9),
            Entity(id="company1", name="Acme Corp", entity_type="Organization", 
                  attributes={"industry": "Tech"}, confidence=0.9),
            Entity(id="skill1", name="Programming", entity_type="Skill", 
                  confidence=0.9),
            Entity(id="job1", name="Software Developer", entity_type="Job", 
                  confidence=0.9),
            Entity(id="entity1", name="Entity 1", entity_type="Test", 
                  confidence=0.9),
            Entity(id="entity2", name="Entity 2", entity_type="Test", 
                  confidence=0.9),
            Entity(id="entity3", name="Entity 3", entity_type="Test", 
                  confidence=0.9)
        ]
        
        for entity in entities:
            graph.add_entity(entity)
        
        # Add relationships
        relationships = [
            Relationship(id="rel_person1_person2", source_entity="person1", 
                        target_entity="person2", relation_type="knows", confidence=0.8),
            Relationship(id="rel_person1_company1", source_entity="person1", 
                        target_entity="company1", relation_type="works-for", confidence=0.9),
            Relationship(id="rel_person1_skill1", source_entity="person1", 
                        target_entity="skill1", relation_type="has-skill", confidence=0.9),
            Relationship(id="rel_skill1_job1", source_entity="skill1", 
                        target_entity="job1", relation_type="required-for", confidence=0.8),
            Relationship(id="rel_company1_job1", source_entity="company1", 
                        target_entity="job1", relation_type="offers", confidence=0.7)
        ]
        
        for relationship in relationships:
            graph.add_relationship(relationship)
        
        return graph
    
    def _add_contradictory_relationships(self):
        """Add contradictory relationships to the test graph."""
        # Add a relationship
        rel1 = Relationship(
            id="rel_contradictory1",
            source_entity="person1",
            target_entity="company1",
            relation_type="works-for",
            confidence=0.9
        )
        
        # Add a contradictory relationship
        rel2 = Relationship(
            id="rel_contradictory2",
            source_entity="person1",
            target_entity="company1",
            relation_type="not-affiliated-with",
            confidence=0.7
        )
        
        self.graph.add_relationship(rel1)
        self.graph.add_relationship(rel2)


if __name__ == "__main__":
    unittest.main() 