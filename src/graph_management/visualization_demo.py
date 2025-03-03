"""
Demonstration script for the Graph Visualization module.

This script demonstrates the functionalities of the Graph Visualization module
by creating sample knowledge graphs and generating various visualizations.
"""

import os
from pathlib import Path

from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_visualizer import (
    GraphVisualizer, visualize_graph, visualize_subgraph,
    visualize_path, visualize_filtered_graph
)
from src.graph_management.graph_query import GraphQuery
from src.graph_management.graph_reasoning import reason_over_paths


def create_demo_graph() -> KnowledgeGraph:
    """Create a demonstration knowledge graph."""
    config = AppConfig()
    graph = KnowledgeGraph(name="demo_graph", config=config)
    
    # Add entities
    entities = [
        Entity(id="person1", name="John Doe", entity_type="Person", 
              attributes={"age": 30, "occupation": "Developer"}, confidence=0.9),
        Entity(id="person2", name="Jane Smith", entity_type="Person", 
              attributes={"age": 35, "occupation": "Manager"}, confidence=0.8),
        Entity(id="person3", name="Bob Johnson", entity_type="Person", 
              attributes={"age": 42, "occupation": "CTO"}, confidence=0.95),
        Entity(id="company1", name="Acme Corp", entity_type="Organization", 
              attributes={"industry": "Tech", "location": "San Francisco"}, confidence=0.9),
        Entity(id="company2", name="TechStart", entity_type="Organization", 
              attributes={"industry": "Tech", "location": "Boston"}, confidence=0.85),
        Entity(id="skill1", name="Python", entity_type="Skill", 
              attributes={"category": "Programming"}, confidence=0.9),
        Entity(id="skill2", name="Machine Learning", entity_type="Skill", 
              attributes={"category": "AI"}, confidence=0.8),
        Entity(id="skill3", name="Data Analysis", entity_type="Skill", 
              attributes={"category": "Data Science"}, confidence=0.85),
        Entity(id="project1", name="Web Platform", entity_type="Project", 
              attributes={"status": "In Progress"}, confidence=0.9),
        Entity(id="project2", name="Mobile App", entity_type="Project", 
              attributes={"status": "Planning"}, confidence=0.75),
        Entity(id="location1", name="San Francisco", entity_type="Location", 
              attributes={"country": "USA"}, confidence=0.9),
        Entity(id="location2", name="Boston", entity_type="Location", 
              attributes={"country": "USA"}, confidence=0.9),
        Entity(id="tool1", name="Git", entity_type="Tool", 
              attributes={"category": "Version Control"}, confidence=0.95),
        Entity(id="concept1", name="Agile Development", entity_type="Concept", 
              attributes={"category": "Methodology"}, confidence=0.9)
    ]
    
    for entity in entities:
        graph.add_entity(entity)
    
    # Add relationships
    relationships = [
        # Person relationships
        Relationship(id="rel1", source_entity="person1", target_entity="person2", 
                    relation_type="knows", confidence=0.8),
        Relationship(id="rel2", source_entity="person2", target_entity="person3", 
                    relation_type="reports-to", confidence=0.9),
        Relationship(id="rel3", source_entity="person1", target_entity="company1", 
                    relation_type="works-for", confidence=0.9),
        Relationship(id="rel4", source_entity="person2", target_entity="company1", 
                    relation_type="works-for", confidence=0.9),
        Relationship(id="rel5", source_entity="person3", target_entity="company1", 
                    relation_type="works-for", confidence=0.95),
        
        # Skills relationships
        Relationship(id="rel6", source_entity="person1", target_entity="skill1", 
                    relation_type="has-skill", confidence=0.9),
        Relationship(id="rel7", source_entity="person1", target_entity="skill2", 
                    relation_type="has-skill", confidence=0.7),
        Relationship(id="rel8", source_entity="person2", target_entity="skill3", 
                    relation_type="has-skill", confidence=0.85),
        Relationship(id="rel9", source_entity="person3", target_entity="skill1", 
                    relation_type="has-skill", confidence=0.8),
        Relationship(id="rel10", source_entity="person3", target_entity="skill2", 
                    relation_type="has-skill", confidence=0.9),
        
        # Project relationships
        Relationship(id="rel11", source_entity="person1", target_entity="project1", 
                    relation_type="works-on", confidence=0.9),
        Relationship(id="rel12", source_entity="person2", target_entity="project1", 
                    relation_type="manages", confidence=0.9),
        Relationship(id="rel13", source_entity="person1", target_entity="project2", 
                    relation_type="works-on", confidence=0.75),
        Relationship(id="rel14", source_entity="company1", target_entity="project1", 
                    relation_type="owns", confidence=0.9),
        Relationship(id="rel15", source_entity="company1", target_entity="project2", 
                    relation_type="owns", confidence=0.9),
        
        # Location relationships
        Relationship(id="rel16", source_entity="company1", target_entity="location1", 
                    relation_type="located-in", confidence=0.9),
        Relationship(id="rel17", source_entity="company2", target_entity="location2", 
                    relation_type="located-in", confidence=0.9),
        
        # Tool relationships
        Relationship(id="rel18", source_entity="person1", target_entity="tool1", 
                    relation_type="uses", confidence=0.9),
        Relationship(id="rel19", source_entity="project1", target_entity="tool1", 
                    relation_type="uses", confidence=0.9),
        
        # Concept relationships
        Relationship(id="rel20", source_entity="person2", target_entity="concept1", 
                    relation_type="advocates", confidence=0.8),
        Relationship(id="rel21", source_entity="project1", target_entity="concept1", 
                    relation_type="follows", confidence=0.85)
    ]
    
    for relationship in relationships:
        graph.add_relationship(relationship)
    
    return graph


def demonstrate_basic_visualization(graph: KnowledgeGraph, output_dir: Path):
    """
    Demonstrate basic graph visualization.
    
    Args:
        graph: Knowledge graph to visualize
        output_dir: Directory to save visualizations
    """
    print("\n=== Basic Visualization ===")
    
    # Create and save a basic visualization
    result = visualize_graph(
        graph=graph,
        output_path=output_dir,
        filename="basic_visualization.html"
    )
    
    if result.success:
        print(f"Basic visualization created: {result.value}")
    else:
        print(f"Failed to create basic visualization: {result.value}")
    
    # Create visualizations with different color schemes
    for scheme in ["default", "pastel", "grayscale"]:
        result = visualize_graph(
            graph=graph,
            output_path=output_dir,
            filename=f"{scheme}_visualization.html",
            color_scheme=scheme
        )
        
        if result.success:
            print(f"{scheme.capitalize()} visualization created: {result.value}")


def demonstrate_filtered_visualization(graph: KnowledgeGraph, output_dir: Path):
    """
    Demonstrate filtered graph visualization.
    
    Args:
        graph: Knowledge graph to visualize
        output_dir: Directory to save visualizations
    """
    print("\n=== Filtered Visualization ===")
    
    # Create a visualization filtered by entity type
    result = visualize_filtered_graph(
        graph=graph,
        entity_types=["Person", "Skill"],
        output_path=output_dir,
        filename="person_skill_visualization.html"
    )
    
    if result.success:
        print(f"Person-Skill visualization created: {result.value}")
    
    # Create a visualization filtered by relationship type
    result = visualize_filtered_graph(
        graph=graph,
        relationship_types=["works-for", "manages"],
        output_path=output_dir,
        filename="work_relationships_visualization.html"
    )
    
    if result.success:
        print(f"Work relationships visualization created: {result.value}")
    
    # Create a visualization filtered by confidence
    result = visualize_filtered_graph(
        graph=graph,
        min_confidence=0.85,
        output_path=output_dir,
        filename="high_confidence_visualization.html"
    )
    
    if result.success:
        print(f"High confidence visualization created: {result.value}")


def demonstrate_subgraph_visualization(graph: KnowledgeGraph, output_dir: Path):
    """
    Demonstrate subgraph visualization.
    
    Args:
        graph: Knowledge graph to visualize
        output_dir: Directory to save visualizations
    """
    print("\n=== Subgraph Visualization ===")
    
    # Create a subgraph visualization centered around a person
    result = visualize_subgraph(
        graph=graph,
        entity_ids=["person1"],
        include_neighbors=True,
        output_path=output_dir,
        filename="person1_subgraph.html"
    )
    
    if result.success:
        print(f"Person1 subgraph visualization created: {result.value}")
    
    # Create a subgraph visualization for a project and its connections
    result = visualize_subgraph(
        graph=graph,
        entity_ids=["project1"],
        include_neighbors=True,
        output_path=output_dir,
        filename="project1_subgraph.html",
        color_scheme="pastel"
    )
    
    if result.success:
        print(f"Project1 subgraph visualization created: {result.value}")


def demonstrate_path_visualization(graph: KnowledgeGraph, output_dir: Path):
    """
    Demonstrate path visualization.
    
    Args:
        graph: Knowledge graph to visualize
        output_dir: Directory to save visualizations
    """
    print("\n=== Path Visualization ===")
    
    # Create a visualization of paths between two entities
    result = visualize_path(
        graph=graph,
        source_entity_id="person1",
        target_entity_id="person3",
        output_path=output_dir,
        filename="person1_to_person3_path.html"
    )
    
    if result.success:
        print(f"Path visualization created: {result.value}")
    else:
        print(f"Failed to create path visualization: {result.value}")


def main():
    """Run the demonstration."""
    print("=== Graph Visualization Demonstration ===")
    
    # Create output directory
    output_dir = Path("output/demos/visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a demo graph
    graph = create_demo_graph()
    print(f"Created graph with {len(graph.entity_map)} entities and {len(graph.relationship_map)} relationships")
    
    # Demonstrate various visualization functions
    demonstrate_basic_visualization(graph, output_dir)
    demonstrate_filtered_visualization(graph, output_dir)
    demonstrate_subgraph_visualization(graph, output_dir)
    demonstrate_path_visualization(graph, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 