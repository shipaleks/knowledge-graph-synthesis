"""
Test Graph Query Module

This module tests the graph query functionality.
"""

import unittest
import networkx as nx
from typing import Dict, List, Any, Optional

from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_query import (
    GraphQuery,
    query_graph,
    search_graph_text,
    get_entity_neighborhood,
    find_paths
)


class TestGraphQuery(unittest.TestCase):
    """Test cases for the GraphQuery class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AppConfig()
        self.graph = self._create_test_graph()
        self.query_engine = GraphQuery(self.graph, self.config)

    def test_find_entities(self):
        """Test find_entities method."""
        # Test finding entities by type
        results = self.query_engine.find_entities({"entity_type": "person"})
        self.assertEqual(len(results), 2)
        self.assertEqual(set([e.name for e in results]), {"Alice", "Bob"})

        # Test finding entities by name
        results = self.query_engine.find_entities({"name": "Alice"})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Alice")

        # Test finding entities by attribute
        results = self.query_engine.find_entities({"attributes": {"age": "30"}})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Alice")

        # Test finding entities with limit
        results = self.query_engine.find_entities({"entity_type": "location"}, limit=1)
        self.assertEqual(len(results), 1)

        # Test finding entities with complex query
        results = self.query_engine.find_entities({
            "entity_type": "person",
            "attributes": {"role": "engineer"}
        })
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Bob")

    def test_find_relationships(self):
        """Test find_relationships method."""
        # Test finding relationships by type
        results = self.query_engine.find_relationships({"relation_type": "works_at"})
        self.assertEqual(len(results), 2)

        # Test finding relationships by source entity
        results = self.query_engine.find_relationships({"source_entity": "alice"})
        self.assertEqual(len(results), 2)
        
        # Test finding relationships by target entity
        results = self.query_engine.find_relationships({"target_entity": "company_x"})
        self.assertEqual(len(results), 2)

        # Test finding relationships with limit
        results = self.query_engine.find_relationships({"relation_type": "works_at"}, limit=1)
        self.assertEqual(len(results), 1)

    def test_traverse(self):
        """Test traverse method."""
        # Test basic traversal from Alice
        result = self.query_engine.traverse("alice", max_depth=1)
        self.assertEqual(result["entity"].name, "Alice")
        self.assertEqual(len(result["children"]), 2)

        # Test traversal with relationship type filter
        result = self.query_engine.traverse("alice", max_depth=1, relationship_types=["works_at"])
        self.assertEqual(len(result["children"]), 1)
        self.assertEqual(result["children"][0]["relationship"].relation_type, "works_at")

        # Test traversal with entity type filter
        result = self.query_engine.traverse("alice", max_depth=1, entity_types=["organization"])
        self.assertEqual(len(result["children"]), 1)
        self.assertEqual(result["children"][0]["child"]["entity"].entity_type, "organization")

        # Test traversal with direction filter (incoming)
        result = self.query_engine.traverse("company_x", direction="incoming")
        self.assertEqual(result["entity"].name, "Company X")
        self.assertTrue(all(child["direction"] == "incoming" for child in result["children"]))

    def test_find_path(self):
        """Test find_path method."""
        # Test finding path between Alice and Company X
        paths = self.query_engine.find_path("alice", "company_x")
        self.assertTrue(len(paths) > 0)
        # Verify the first and last entity in the path
        self.assertEqual(paths[0][0]["entity"].name, "Alice")
        self.assertEqual(paths[0][-1]["entity"].name, "Company X")

        # Test finding path between Alice and Project Y
        paths = self.query_engine.find_path("alice", "project_y")
        self.assertTrue(len(paths) > 0)
        # Verify the path length (should have multiple entities)
        self.assertGreater(len(paths[0]), 3)

        # Test finding path with max depth limit
        paths = self.query_engine.find_path("alice", "project_y", max_depth=1)
        self.assertEqual(len(paths), 0)  # No path within depth 1

    def test_search_text(self):
        """Test search_text method."""
        # Test searching for text in names
        results = self.query_engine.search_text("Alice")
        self.assertTrue(len(results["entities"]) == 1)
        self.assertEqual(results["entities"][0].name, "Alice")

        # Test searching for text in attributes
        results = self.query_engine.search_text("engineer")
        self.assertTrue(len(results["entities"]) == 1)
        self.assertEqual(results["entities"][0].name, "Bob")

        # Test searching for text in context
        for entity in self.graph.entity_map.values():
            if entity.id == "alice":
                entity.context = "This is a context with special keywords for testing."
                break
        
        results = self.query_engine.search_text("special keywords")
        self.assertTrue(len(results["entities"]) == 1)
        self.assertEqual(results["entities"][0].id, "alice")

        # Test case insensitive search
        results = self.query_engine.search_text("ALICE", case_sensitive=False)
        self.assertTrue(len(results["entities"]) == 1)

        # Test case sensitive search
        results = self.query_engine.search_text("ALICE", case_sensitive=True)
        self.assertTrue(len(results["entities"]) == 0)

    def test_get_subgraph(self):
        """Test get_subgraph method."""
        # Test getting subgraph with just Alice
        subgraph = self.query_engine.get_subgraph(["alice"])
        self.assertEqual(len(subgraph.entity_map), 1)
        self.assertEqual(len(subgraph.relationship_map), 0)

        # Test getting subgraph with Alice and include neighbors
        subgraph = self.query_engine.get_subgraph(["alice"], include_neighbors=True)
        self.assertGreater(len(subgraph.entity_map), 1)
        self.assertGreater(len(subgraph.relationship_map), 0)

        # Test getting subgraph with multiple entities
        subgraph = self.query_engine.get_subgraph(["alice", "bob"])
        self.assertEqual(len(subgraph.entity_map), 2)

    def test_filter_by_confidence(self):
        """Test filter_by_confidence method."""
        # Set different confidence scores for testing
        for entity in self.graph.entity_map.values():
            if entity.id == "alice":
                entity.confidence = 0.8
            elif entity.id == "bob":
                entity.confidence = 0.3
            else:
                entity.confidence = 0.4  # Set all other entities to below threshold
        
        for rel in self.graph.relationship_map.values():
            if rel.source_entity == "alice" and rel.target_entity == "company_x":
                rel.confidence = 0.9
            else:
                rel.confidence = 0.4
        
        # Test filtering entities by confidence
        filtered = self.query_engine.filter_by_confidence(min_confidence=0.5, apply_to="entities")
        self.assertEqual(len(filtered.entity_map), 1)  # Only Alice
        self.assertEqual(list(filtered.entity_map.values())[0].name, "Alice")

        # Test filtering relationships by confidence
        filtered = self.query_engine.filter_by_confidence(min_confidence=0.5, apply_to="relationships")
        self.assertEqual(len(filtered.relationship_map), 1)  # Only Alice->Company X

        # Test filtering both entities and relationships
        filtered = self.query_engine.filter_by_confidence(min_confidence=0.5, apply_to="both")
        self.assertEqual(len(filtered.entity_map), 1)  # Only Alice
        self.assertEqual(len(filtered.relationship_map), 0)  # No relationships (one was filtered out because Bob is filtered out)

    def test_get_connected_components(self):
        """Test get_connected_components method."""
        # Create a disconnected graph
        self.graph.add_entity(Entity(
            id="charlie",
            name="Charlie",
            entity_type="person",
            confidence=0.9
        ))
        self.graph.add_entity(Entity(
            id="company_z",
            name="Company Z",
            entity_type="organization",
            confidence=0.9
        ))
        self.graph.add_relationship(Relationship(
            id="charlie_works_at_z",
            source_entity="charlie",
            target_entity="company_z",
            relation_type="works_at",
            confidence=0.9
        ))
        
        # Get connected components
        components = self.query_engine.get_connected_components()
        self.assertEqual(len(components), 2)  # Two separate connected components
        
        # Validate the components
        self.assertIn(len(components[0].entity_map), [2, 5])
        self.assertIn(len(components[1].entity_map), [2, 5])

    def test_sort_entities_by_metric(self):
        """Test sort_entities_by_metric method."""
        # Test sorting by centrality
        results = self.query_engine.sort_entities_by_metric("degree")
        self.assertTrue(len(results) > 0)
        
        # Check that results are in descending order
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Test sorting with limit
        results = self.query_engine.sort_entities_by_metric("degree", limit=2)
        self.assertEqual(len(results), 2)
        
        # Test sorting by custom metric
        def custom_metric(entity):
            return len(entity.name)
        
        results = self.query_engine.sort_entities_by_metric(custom_metric)
        self.assertTrue(len(results) > 0)
        entity_names = [entity.name for entity, _ in results]
        self.assertIn("Company X", entity_names[:2])  # Should be one of the top results

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test query_graph
        results = query_graph(self.graph, {"entity_type": "person"})
        self.assertEqual(len(results["entities"]), 2)
        
        # Test search_graph_text
        results = search_graph_text(self.graph, "Alice")
        self.assertEqual(len(results["entities"]), 1)
        
        # Test get_entity_neighborhood
        neighborhood = get_entity_neighborhood(self.graph, "alice")
        self.assertEqual(neighborhood["entity"].name, "Alice")
        self.assertTrue(len(neighborhood["children"]) > 0)
        
        # Test find_paths
        paths = find_paths(self.graph, "alice", "company_x")
        self.assertTrue(len(paths) > 0)

    def _create_test_graph(self) -> KnowledgeGraph:
        """Create a test graph for testing."""
        graph = KnowledgeGraph(name="test_graph", config=self.config)
        
        # Add entities
        graph.add_entity(Entity(
            id="alice",
            name="Alice",
            entity_type="person",
            confidence=0.9,
            attributes={"age": "30", "role": "manager"}
        ))
        
        graph.add_entity(Entity(
            id="bob",
            name="Bob",
            entity_type="person",
            confidence=0.9,
            attributes={"age": "25", "role": "engineer"}
        ))
        
        graph.add_entity(Entity(
            id="company_x",
            name="Company X",
            entity_type="organization",
            confidence=0.9,
            attributes={"industry": "tech"}
        ))
        
        graph.add_entity(Entity(
            id="city_a",
            name="City A",
            entity_type="location",
            confidence=0.9,
            attributes={"country": "USA"}
        ))
        
        graph.add_entity(Entity(
            id="project_y",
            name="Project Y",
            entity_type="project",
            confidence=0.9,
            attributes={"status": "active"}
        ))
        
        # Add relationships
        graph.add_relationship(Relationship(
            id="alice_works_at_x",
            source_entity="alice",
            target_entity="company_x",
            relation_type="works_at",
            confidence=0.9
        ))
        
        graph.add_relationship(Relationship(
            id="bob_works_at_x",
            source_entity="bob",
            target_entity="company_x",
            relation_type="works_at",
            confidence=0.9
        ))
        
        graph.add_relationship(Relationship(
            id="alice_lives_in_a",
            source_entity="alice",
            target_entity="city_a",
            relation_type="lives_in",
            confidence=0.9
        ))
        
        graph.add_relationship(Relationship(
            id="company_x_located_in_a",
            source_entity="company_x",
            target_entity="city_a",
            relation_type="located_in",
            confidence=0.9
        ))
        
        graph.add_relationship(Relationship(
            id="bob_works_on_y",
            source_entity="bob",
            target_entity="project_y",
            relation_type="works_on",
            confidence=0.9
        ))
        
        graph.add_relationship(Relationship(
            id="company_x_develops_y",
            source_entity="company_x",
            target_entity="project_y",
            relation_type="develops",
            confidence=0.9
        ))
        
        return graph


if __name__ == "__main__":
    unittest.main() 