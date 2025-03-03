"""
Test module for Graph Visualization functionality.

This module contains tests for the GraphVisualizer class and related functions.
"""

import unittest
import os
from pathlib import Path
from typing import Dict, Any

from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_visualizer import (
    GraphVisualizer, visualize_graph, visualize_subgraph,
    visualize_path, visualize_filtered_graph
)


class TestGraphVisualizer(unittest.TestCase):
    """Test cases for the GraphVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AppConfig()
        self.graph = self._create_test_graph()
        self.visualizer = GraphVisualizer(self.graph, self.config)
        self.output_dir = Path("output/tests/visualizations")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_create_visualization(self):
        """Test creating a visualization."""
        # Create a visualization
        result = self.visualizer.create_visualization()
        
        # Check that the visualization was created successfully
        self.assertTrue(result.success)
        self.assertIsNotNone(result.value)
        self.assertIsNotNone(self.visualizer.network)
    
    def test_save_visualization(self):
        """Test saving a visualization."""
        # Create a visualization
        self.visualizer.create_visualization()
        
        # Save the visualization
        result = self.visualizer.save_visualization(
            output_path=self.output_dir,
            filename="test_graph.html"
        )
        
        # Check that the visualization was saved successfully
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test_graph.html")))
    
    def test_filter_visualization(self):
        """Test filtering a visualization."""
        # Create a visualization
        self.visualizer.create_visualization()
        
        # Filter the visualization to show only Person entities
        result = self.visualizer.filter_visualization(
            entity_types=["Person"],
            min_confidence=0.7
        )
        
        # Check that the filter was applied successfully
        self.assertTrue(result.success)
        self.assertIsNotNone(result.value)
        
        # Save the filtered visualization
        save_result = self.visualizer.save_visualization(
            output_path=self.output_dir,
            filename="filtered_graph.html",
            network=result.value
        )
        
        self.assertTrue(save_result.success)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "filtered_graph.html")))
    
    def test_visualize_graph_helper(self):
        """Test the visualize_graph helper function."""
        # Use the helper function to visualize the graph
        result = visualize_graph(
            graph=self.graph,
            output_path=self.output_dir,
            filename="helper_test.html",
            color_scheme="pastel"
        )
        
        # Check that the visualization was created successfully
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "helper_test.html")))
    
    def test_visualize_subgraph_helper(self):
        """Test the visualize_subgraph helper function."""
        # Use the helper function to visualize a subgraph
        result = visualize_subgraph(
            graph=self.graph,
            entity_ids=["person1", "skill1"],
            include_neighbors=True,
            output_path=self.output_dir,
            filename="subgraph_test.html"
        )
        
        # Check that the visualization was created successfully
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "subgraph_test.html")))
    
    def test_visualize_filtered_graph_helper(self):
        """Test the visualize_filtered_graph helper function."""
        # Use the helper function to visualize a filtered graph
        result = visualize_filtered_graph(
            graph=self.graph,
            entity_types=["Person", "Skill"],
            min_confidence=0.7,
            output_path=self.output_dir,
            filename="filtered_graph_helper.html",
            color_scheme="grayscale"
        )
        
        # Check that the visualization was created successfully
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "filtered_graph_helper.html")))
    
    def _create_test_graph(self) -> KnowledgeGraph:
        """Create a test graph for use in tests."""
        graph = KnowledgeGraph(name="test_graph", config=self.config)
        
        # Add entities
        entities = [
            Entity(id="person1", name="John Doe", entity_type="Person", 
                  attributes={"age": 30}, confidence=0.9),
            Entity(id="person2", name="Jane Smith", entity_type="Person", 
                  attributes={"age": 35}, confidence=0.7),
            Entity(id="company1", name="Acme Corp", entity_type="Organization", 
                  attributes={"industry": "Tech"}, confidence=0.9),
            Entity(id="skill1", name="Programming", entity_type="Skill", 
                  confidence=0.8),
            Entity(id="job1", name="Software Developer", entity_type="Job", 
                  confidence=0.9)
        ]
        
        for entity in entities:
            graph.add_entity(entity)
        
        # Add relationships
        relationships = [
            Relationship(id="rel1", source_entity="person1", target_entity="person2", 
                        relation_type="knows", confidence=0.8),
            Relationship(id="rel2", source_entity="person1", target_entity="company1", 
                        relation_type="works-for", confidence=0.9),
            Relationship(id="rel3", source_entity="person1", target_entity="skill1", 
                        relation_type="has-skill", confidence=0.9),
            Relationship(id="rel4", source_entity="skill1", target_entity="job1", 
                        relation_type="required-for", confidence=0.8)
        ]
        
        for relationship in relationships:
            graph.add_relationship(relationship)
        
        return graph


if __name__ == "__main__":
    unittest.main() 