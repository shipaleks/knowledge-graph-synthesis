"""
Graph Management Package

This package provides functionality for managing and working with knowledge graphs.
"""

from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_storage import (
    GraphStorage, save_graph, load_graph, 
    export_graph_to_format
)
from src.graph_management.graph_query import (
    GraphQuery, query_graph, search_graph_text,
    get_entity_neighborhood, find_paths
)
from src.graph_management.graph_reasoning import (
    GraphReasoning, reason_over_paths, detect_graph_conflicts,
    resolve_graph_conflicts, infer_new_knowledge
)

__all__ = [
    'KnowledgeGraph',
    'GraphStorage',
    'save_graph',
    'load_graph',
    'export_graph_to_format',
    'GraphQuery',
    'query_graph',
    'search_graph_text',
    'get_entity_neighborhood',
    'find_paths',
    'GraphReasoning',
    'reason_over_paths',
    'detect_graph_conflicts',
    'resolve_graph_conflicts',
    'infer_new_knowledge'
]
