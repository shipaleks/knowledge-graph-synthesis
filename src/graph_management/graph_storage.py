"""
Graph Storage Module

This module provides functionality for storing, loading, and versioning knowledge graphs.
It supports saving graphs to different formats and loading them back with version tracking.
"""

import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from src.utils.result import Result
from src.utils.logger import get_logger
from src.config.app_config import AppConfig
from src.graph_management.graph import KnowledgeGraph

# Configure logger
logger = get_logger(__name__)


class GraphStorage:
    """
    Manages the storage, retrieval, and versioning of knowledge graphs.
    
    This class provides methods to save graphs to disk, load them back,
    and maintain version history.
    """
    
    def __init__(self, storage_dir: Union[str, Path], config: Optional[AppConfig] = None):
        """
        Initialize the graph storage manager.
        
        Args:
            storage_dir: Directory to store graphs
            config: Optional application configuration
        """
        self.storage_dir = Path(storage_dir)
        self.config = config or AppConfig()
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.graphs_dir = self.storage_dir / "graphs"
        self.versions_dir = self.storage_dir / "versions"
        self.metadata_dir = self.storage_dir / "metadata"
        
        self.graphs_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
    
    def save(self, graph: KnowledgeGraph, create_version: bool = True) -> Result[str]:
        """
        Save a knowledge graph to storage.
        
        Args:
            graph: The knowledge graph to save
            create_version: Whether to create a new version
            
        Returns:
            Result[str]: Result containing the path to the saved graph or an error
        """
        try:
            # Update metadata
            timestamp = datetime.now().isoformat()
            graph.metadata["updated_at"] = timestamp
            if not graph.metadata.get("created_at"):
                graph.metadata["created_at"] = timestamp
            
            # Base filename for the graph
            base_name = self._sanitize_name(graph.name)
            
            # Save current version
            graph_path = self.graphs_dir / f"{base_name}.json"
            with open(graph_path, 'w', encoding='utf-8') as f:
                f.write(graph.to_json())
            
            # Create version if requested
            if create_version:
                version_id = f"{base_name}_{int(time.time())}"
                version_path = self.versions_dir / f"{version_id}.json"
                
                # Copy the file
                shutil.copy2(graph_path, version_path)
                
                # Create or update version metadata
                self._update_version_metadata(graph.name, version_id, graph)
                
                logger.info(f"Created new version {version_id} for graph {graph.name}")
            
            logger.info(f"Saved graph {graph.name} to {graph_path}")
            return Result.ok(str(graph_path))
        
        except Exception as e:
            error_msg = f"Failed to save graph {graph.name}: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def load(self, graph_name: str, version_id: Optional[str] = None) -> Result[KnowledgeGraph]:
        """
        Load a knowledge graph from storage.
        
        Args:
            graph_name: Name of the graph to load
            version_id: Optional version ID to load a specific version
            
        Returns:
            Result[KnowledgeGraph]: Result containing the loaded graph or an error
        """
        try:
            base_name = self._sanitize_name(graph_name)
            
            # Determine which file to load
            if version_id:
                graph_path = self.versions_dir / f"{version_id}.json"
                if not graph_path.exists():
                    return Result.fail(f"Version {version_id} of graph {graph_name} not found")
            else:
                graph_path = self.graphs_dir / f"{base_name}.json"
                if not graph_path.exists():
                    return Result.fail(f"Graph {graph_name} not found")
            
            # Load the graph
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_json = f.read()
            
            graph = KnowledgeGraph.from_json(graph_json)
            logger.info(f"Loaded graph {graph_name} from {graph_path}")
            
            return Result.ok(graph)
        
        except Exception as e:
            error_msg = f"Failed to load graph {graph_name}: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def delete(self, graph_name: str, delete_versions: bool = False) -> Result[bool]:
        """
        Delete a knowledge graph from storage.
        
        Args:
            graph_name: Name of the graph to delete
            delete_versions: Whether to delete all versions as well
            
        Returns:
            Result[bool]: Result indicating success or failure
        """
        try:
            base_name = self._sanitize_name(graph_name)
            
            # Delete current version
            graph_path = self.graphs_dir / f"{base_name}.json"
            if graph_path.exists():
                os.remove(graph_path)
                logger.info(f"Deleted graph {graph_name}")
            else:
                logger.warning(f"Graph {graph_name} not found, nothing to delete")
            
            # Delete versions if requested
            if delete_versions:
                # Get version metadata
                metadata_path = self.metadata_dir / f"{base_name}_versions.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        versions = json.load(f)
                    
                    # Delete each version
                    for version_id in versions.get("versions", []):
                        version_path = self.versions_dir / f"{version_id}.json"
                        if version_path.exists():
                            os.remove(version_path)
                            logger.info(f"Deleted version {version_id} of graph {graph_name}")
                    
                    # Delete metadata file
                    os.remove(metadata_path)
            
            return Result.ok(True)
        
        except Exception as e:
            error_msg = f"Failed to delete graph {graph_name}: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def list_graphs(self) -> Result[List[Dict[str, Any]]]:
        """
        List all available graphs.
        
        Returns:
            Result[List[Dict[str, Any]]]: Result containing information about available graphs
        """
        try:
            graphs = []
            
            for graph_path in self.graphs_dir.glob("*.json"):
                try:
                    # Get basic info without loading the full graph
                    with open(graph_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract relevant information
                    graph_info = {
                        "name": data.get("name", graph_path.stem),
                        "entity_count": len(data.get("entities", {})),
                        "relationship_count": len(data.get("relationships", {})),
                        "created_at": data.get("metadata", {}).get("created_at"),
                        "updated_at": data.get("metadata", {}).get("updated_at"),
                        "file_path": str(graph_path)
                    }
                    
                    # Get version count
                    base_name = self._sanitize_name(graph_info["name"])
                    metadata_path = self.metadata_dir / f"{base_name}_versions.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            versions = json.load(f)
                        graph_info["version_count"] = len(versions.get("versions", []))
                    else:
                        graph_info["version_count"] = 0
                    
                    graphs.append(graph_info)
                
                except Exception as e:
                    logger.warning(f"Error processing graph file {graph_path}: {str(e)}")
            
            return Result.ok(graphs)
        
        except Exception as e:
            error_msg = f"Failed to list graphs: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def list_versions(self, graph_name: str) -> Result[List[Dict[str, Any]]]:
        """
        List all versions of a graph.
        
        Args:
            graph_name: Name of the graph
            
        Returns:
            Result[List[Dict[str, Any]]]: Result containing information about available versions
        """
        try:
            base_name = self._sanitize_name(graph_name)
            metadata_path = self.metadata_dir / f"{base_name}_versions.json"
            
            if not metadata_path.exists():
                return Result.ok([])  # No versions available
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                versions_data = json.load(f)
            
            versions = []
            for version_id in versions_data.get("versions", []):
                version_info = versions_data.get("version_info", {}).get(version_id, {})
                versions.append({
                    "version_id": version_id,
                    "created_at": version_info.get("created_at"),
                    "entity_count": version_info.get("entity_count", 0),
                    "relationship_count": version_info.get("relationship_count", 0),
                    "description": version_info.get("description", "")
                })
            
            # Sort by creation time, newest first
            versions.sort(key=lambda v: v.get("created_at", ""), reverse=True)
            
            return Result.ok(versions)
        
        except Exception as e:
            error_msg = f"Failed to list versions for graph {graph_name}: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def export_graph(self, 
                     graph_name: str, 
                     output_path: Union[str, Path], 
                     format: str = "json", 
                     version_id: Optional[str] = None) -> Result[str]:
        """
        Export a graph to a specific format.
        
        Args:
            graph_name: Name of the graph to export
            output_path: Path to save the exported graph
            format: Format to export (json, graphml, cypher)
            version_id: Optional version ID to export a specific version
            
        Returns:
            Result[str]: Result containing the path to the exported file or an error
        """
        try:
            # Load the graph
            graph_result = self.load(graph_name, version_id)
            if not graph_result.success:
                return Result.fail(graph_result.error)
            
            graph = graph_result.value
            output_path = Path(output_path)
            
            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export in the requested format
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(graph.to_json())
            elif format.lower() == "graphml":
                # Convert to NetworkX GraphML format
                import networkx as nx
                nx.write_graphml(graph.graph, output_path)
            elif format.lower() == "cypher":
                # Generate Cypher queries for Neo4j
                cypher = self._generate_cypher(graph)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cypher)
            else:
                return Result.fail(f"Unsupported export format: {format}")
            
            logger.info(f"Exported graph {graph_name} to {output_path} in {format} format")
            return Result.ok(str(output_path))
        
        except Exception as e:
            error_msg = f"Failed to export graph {graph_name}: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def import_graph(self, 
                     file_path: Union[str, Path], 
                     graph_name: Optional[str] = None,
                     format: str = "json") -> Result[KnowledgeGraph]:
        """
        Import a graph from a file.
        
        Args:
            file_path: Path to the file to import
            graph_name: Optional name for the imported graph
            format: Format of the file (json, graphml)
            
        Returns:
            Result[KnowledgeGraph]: Result containing the imported graph or an error
        """
        try:
            file_path = Path(file_path)
            
            # Set graph name if not provided
            if not graph_name:
                graph_name = file_path.stem
            
            # Import from the specified format
            if format.lower() == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    graph_json = f.read()
                
                graph = KnowledgeGraph.from_json(graph_json)
                graph.name = graph_name  # Override name if provided
            
            elif format.lower() == "graphml":
                # Import from NetworkX GraphML format
                import networkx as nx
                nx_graph = nx.read_graphml(file_path)
                
                # Create a new knowledge graph
                graph = KnowledgeGraph(name=graph_name, config=self.config)
                
                # TODO: Convert NetworkX graph to KnowledgeGraph
                # This is a complex conversion that depends on the exact GraphML structure
                # For now, we'll return a failure
                return Result.fail("GraphML import not fully implemented yet")
            
            else:
                return Result.fail(f"Unsupported import format: {format}")
            
            # Save the imported graph
            save_result = self.save(graph, create_version=False)
            if not save_result.success:
                return Result.fail(f"Failed to save imported graph: {save_result.error}")
            
            logger.info(f"Imported graph from {file_path} as {graph_name}")
            return Result.ok(graph)
        
        except Exception as e:
            error_msg = f"Failed to import graph from {file_path}: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def add_version_description(self, 
                               graph_name: str, 
                               version_id: str, 
                               description: str) -> Result[bool]:
        """
        Add a description to a graph version.
        
        Args:
            graph_name: Name of the graph
            version_id: ID of the version
            description: Description to add
            
        Returns:
            Result[bool]: Result indicating success or failure
        """
        try:
            base_name = self._sanitize_name(graph_name)
            metadata_path = self.metadata_dir / f"{base_name}_versions.json"
            
            if not metadata_path.exists():
                return Result.fail(f"No version metadata found for graph {graph_name}")
            
            # Load version metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                versions_data = json.load(f)
            
            # Check if version exists
            if version_id not in versions_data.get("versions", []):
                return Result.fail(f"Version {version_id} not found for graph {graph_name}")
            
            # Update description
            if "version_info" not in versions_data:
                versions_data["version_info"] = {}
            if version_id not in versions_data["version_info"]:
                versions_data["version_info"][version_id] = {}
            
            versions_data["version_info"][version_id]["description"] = description
            
            # Save updated metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(versions_data, f, indent=2)
            
            logger.info(f"Added description to version {version_id} of graph {graph_name}")
            return Result.ok(True)
        
        except Exception as e:
            error_msg = f"Failed to add version description: {str(e)}"
            logger.error(error_msg)
            return Result.fail(error_msg)
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in filenames."""
        return name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    def _update_version_metadata(self, graph_name: str, version_id: str, graph: KnowledgeGraph) -> None:
        """Update version metadata for a graph."""
        base_name = self._sanitize_name(graph_name)
        metadata_path = self.metadata_dir / f"{base_name}_versions.json"
        
        # Load existing metadata or create new
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {"versions": [], "version_info": {}}
        
        # Add this version
        if version_id not in metadata["versions"]:
            metadata["versions"].append(version_id)
        
        # Add version info
        if "version_info" not in metadata:
            metadata["version_info"] = {}
        
        metadata["version_info"][version_id] = {
            "created_at": datetime.now().isoformat(),
            "entity_count": len(graph.entity_map),
            "relationship_count": len(graph.relationship_map)
        }
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_cypher(self, graph: KnowledgeGraph) -> str:
        """Generate Cypher queries for Neo4j from a knowledge graph."""
        cypher_queries = []
        
        # Create constraints
        cypher_queries.append("// Create constraints")
        cypher_queries.append("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;")
        
        # Create nodes
        cypher_queries.append("\n// Create nodes")
        for entity_id, entity in graph.entity_map.items():
            # Escape properties for Cypher
            name = entity.name.replace("'", "\\'")
            entity_type = entity.entity_type.replace("'", "\\'")
            
            # Create attribute string
            attrs = {}
            for key, value in entity.attributes.items():
                if isinstance(value, str):
                    attrs[key] = f"'{value.replace('\'', '\\\'')}'"
                elif isinstance(value, (list, dict)):
                    attrs[key] = f"'{json.dumps(value).replace('\'', '\\\'')}'"
                else:
                    attrs[key] = str(value)
            
            attr_str = ", ".join([f"{k}: {v}" for k, v in attrs.items()])
            if attr_str:
                attr_str = f", {attr_str}"
            
            # Create node
            cypher_queries.append(
                f"CREATE (e:{entity_type} {{id: '{entity_id}', name: '{name}', confidence: {entity.confidence}{attr_str}}})")
        
        # Create relationships
        cypher_queries.append("\n// Create relationships")
        for rel_id, rel in graph.relationship_map.items():
            # Escape properties for Cypher
            rel_type = rel.relation_type.replace(" ", "_").replace("'", "\\'").upper()
            
            # Create attribute string
            attrs = {}
            for key, value in rel.attributes.items():
                if isinstance(value, str):
                    attrs[key] = f"'{value.replace('\'', '\\\'')}'"
                elif isinstance(value, (list, dict)):
                    attrs[key] = f"'{json.dumps(value).replace('\'', '\\\'')}'"
                else:
                    attrs[key] = str(value)
            
            attrs["id"] = f"'{rel_id}'"
            attrs["confidence"] = str(rel.confidence)
            attrs["strength"] = str(rel.strength)
            
            attr_str = ", ".join([f"{k}: {v}" for k, v in attrs.items()])
            
            # Create relationship
            cypher_queries.append(
                f"MATCH (a:Entity {{id: '{rel.source_entity}'}}), (b:Entity {{id: '{rel.target_entity}'}}) "
                f"CREATE (a)-[r:{rel_type} {{{attr_str}}}]->(b)")
        
        return "\n".join(cypher_queries)


# Convenience functions

def save_graph(
    graph: KnowledgeGraph,
    storage_dir: Union[str, Path] = "graphs",
    create_version: bool = True,
    config: Optional[AppConfig] = None
) -> Result[str]:
    """
    Save a knowledge graph to storage.
    
    Args:
        graph: The knowledge graph to save
        storage_dir: Directory to store graphs
        create_version: Whether to create a new version
        config: Optional application configuration
        
    Returns:
        Result[str]: Result containing the path to the saved graph or an error
    """
    storage = GraphStorage(storage_dir, config)
    return storage.save(graph, create_version)


def load_graph(
    graph_name: str,
    storage_dir: Union[str, Path] = "graphs",
    version_id: Optional[str] = None,
    config: Optional[AppConfig] = None
) -> Result[KnowledgeGraph]:
    """
    Load a knowledge graph from storage.
    
    Args:
        graph_name: Name of the graph to load
        storage_dir: Directory to load graphs from
        version_id: Optional version ID to load a specific version
        config: Optional application configuration
        
    Returns:
        Result[KnowledgeGraph]: Result containing the loaded graph or an error
    """
    storage = GraphStorage(storage_dir, config)
    return storage.load(graph_name, version_id)


def list_available_graphs(
    storage_dir: Union[str, Path] = "graphs",
    config: Optional[AppConfig] = None
) -> Result[List[Dict[str, Any]]]:
    """
    List all available graphs in storage.
    
    Args:
        storage_dir: Directory to list graphs from
        config: Optional application configuration
        
    Returns:
        Result[List[Dict[str, Any]]]: Result containing information about available graphs
    """
    storage = GraphStorage(storage_dir, config)
    return storage.list_graphs()


def export_graph_to_format(
    graph_name: str,
    output_path: Union[str, Path],
    format: str = "json",
    storage_dir: Union[str, Path] = "graphs",
    version_id: Optional[str] = None,
    config: Optional[AppConfig] = None
) -> Result[str]:
    """
    Export a graph to a specific format.
    
    Args:
        graph_name: Name of the graph to export
        output_path: Path to save the exported graph
        format: Format to export (json, graphml, cypher)
        storage_dir: Directory to load graphs from
        version_id: Optional version ID to export a specific version
        config: Optional application configuration
        
    Returns:
        Result[str]: Result containing the path to the exported file or an error
    """
    storage = GraphStorage(storage_dir, config)
    return storage.export_graph(graph_name, output_path, format, version_id) 