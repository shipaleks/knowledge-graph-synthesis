"""
Demonstration script for the Graph Reasoning module.

This script demonstrates the functionalities of the Graph Reasoning module
by creating a sample knowledge graph and performing various reasoning operations.
"""

import json

from src.config.app_config import AppConfig
from src.knowledge.entity import Entity
from src.knowledge.relationship import Relationship
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_reasoning import (
    GraphReasoning, reason_over_paths, detect_graph_conflicts,
    resolve_graph_conflicts, infer_new_knowledge
)

def create_demo_graph() -> KnowledgeGraph:
    """Create a demonstration knowledge graph."""
    config = AppConfig()
    graph = KnowledgeGraph(name="demo_graph", config=config)
    
    # Add entities
    entities = [
        Entity(id="person1", name="John Doe", entity_type="Person", 
              attributes={"age": 30, "email": "john@example.com"}, confidence=0.9),
        Entity(id="person2", name="Jane Smith", entity_type="Person", 
              attributes={"age": 35, "email": "jane@example.com"}, confidence=0.9),
        Entity(id="company1", name="Acme Corp", entity_type="Organization", 
              attributes={"industry": "Tech", "size": "Large"}, confidence=0.9),
        Entity(id="skill1", name="Programming", entity_type="Skill", 
              attributes={"type": "Technical"}, confidence=0.9),
        Entity(id="skill2", name="Project Management", entity_type="Skill", 
              attributes={"type": "Management"}, confidence=0.9),
        Entity(id="job1", name="Software Developer", entity_type="Job", 
              attributes={"level": "Senior", "salary_range": "High"}, confidence=0.9),
        Entity(id="job2", name="Product Manager", entity_type="Job", 
              attributes={"level": "Manager", "salary_range": "High"}, confidence=0.9),
        Entity(id="project1", name="Web Application", entity_type="Project", 
              attributes={"status": "In Progress"}, confidence=0.9)
    ]
    
    for entity in entities:
        graph.add_entity(entity)
    
    # Add relationships
    relationships = [
        Relationship(id="rel1", source_entity="person1", target_entity="person2", 
                    relation_type="knows", confidence=0.8),
        Relationship(id="rel2", source_entity="person1", target_entity="company1", 
                    relation_type="works-for", confidence=0.9),
        Relationship(id="rel3", source_entity="person2", target_entity="company1", 
                    relation_type="works-for", confidence=0.9),
        Relationship(id="rel4", source_entity="person1", target_entity="skill1", 
                    relation_type="has-skill", confidence=0.9),
        Relationship(id="rel5", source_entity="person1", target_entity="skill2", 
                    relation_type="has-skill", confidence=0.7),
        Relationship(id="rel6", source_entity="person2", target_entity="skill2", 
                    relation_type="has-skill", confidence=0.9),
        Relationship(id="rel7", source_entity="skill1", target_entity="job1", 
                    relation_type="required-for", confidence=0.9),
        Relationship(id="rel8", source_entity="skill2", target_entity="job2", 
                    relation_type="required-for", confidence=0.9),
        Relationship(id="rel9", source_entity="company1", target_entity="project1", 
                    relation_type="owns", confidence=0.9),
        Relationship(id="rel10", source_entity="person1", target_entity="project1", 
                    relation_type="works-on", confidence=0.8)
    ]
    
    for relationship in relationships:
        graph.add_relationship(relationship)
    
    # Add conflicting relationships for demonstration
    contradictory_rel = Relationship(
        id="rel_contradiction",
        source_entity="person1",
        target_entity="company1",
        relation_type="not-affiliated-with",
        confidence=0.6
    )
    graph.add_relationship(contradictory_rel)
    
    return graph

def demonstrate_path_reasoning(graph: KnowledgeGraph):
    """Demonstrate path-based reasoning."""
    print("\n=== Path-Based Reasoning ===\n")
    
    # Find paths and reason over them
    path_reasoning = reason_over_paths(
        graph=graph,
        source_entity_id="person1",
        target_entity_id="job1",
        max_depth=3
    )
    
    print(f"Paths found: {path_reasoning['paths_found']}")
    print("\nInferences:")
    for inference in path_reasoning["inferences"]:
        print(f"- {inference['explanation']}")

def demonstrate_conflict_detection(graph: KnowledgeGraph):
    """Demonstrate conflict detection."""
    print("\n=== Conflict Detection ===\n")
    
    # Detect conflicts
    conflicts = detect_graph_conflicts(graph)
    
    print(f"Conflicts found: {len(conflicts)}")
    for conflict in conflicts:
        print(f"\nConflict type: {conflict['type']}")
        print(f"Explanation: {conflict['explanation']}")
        print(f"Severity: {conflict['severity']}")

def demonstrate_conflict_resolution(graph: KnowledgeGraph):
    """Demonstrate conflict resolution."""
    print("\n=== Conflict Resolution ===\n")
    
    # Resolve conflicts
    result = resolve_graph_conflicts(graph)
    
    print(f"Message: {result.value['message']}")
    for action in result.value["actions_taken"]:
        print(f"\nAction type: {action['action_type']}")
        print(f"Explanation: {action['explanation']}")

def demonstrate_inference(graph: KnowledgeGraph):
    """Demonstrate relationship inference."""
    print("\n=== Relationship Inference ===\n")
    
    # Infer new relationships
    inferences = infer_new_knowledge(graph)
    
    print(f"New inferences: {len(inferences)}")
    for inference in inferences:
        print(f"\nInference type: {inference['type']}")
        print(f"Relation type: {inference['relation_type']}")
        source_name = inference.get('source_name', 'Unknown')
        target_name = inference.get('target_name', 'Unknown')
        print(f"Relationship: {source_name} -> {inference['relation_type']} -> {target_name}")
        print(f"Confidence: {inference['confidence']:.2f}")
        print(f"Explanation: {inference['explanation']}")

def main():
    """Run the demonstration."""
    print("=== Graph Reasoning Demonstration ===\n")
    
    # Create a demo graph
    graph = create_demo_graph()
    print(f"Created graph with {len(graph.entity_map)} entities and {len(graph.relationship_map)} relationships")
    
    # Demonstrate path reasoning
    demonstrate_path_reasoning(graph)
    
    # Demonstrate conflict detection
    demonstrate_conflict_detection(graph)
    
    # Demonstrate conflict resolution
    demonstrate_conflict_resolution(graph)
    
    # Demonstrate inference
    demonstrate_inference(graph)


if __name__ == "__main__":
    main() 