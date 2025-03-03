# Graph Visualization Module

This document describes the Graph Visualization module, which is part of the Knowledge Graph Synthesis System. The module provides functionality for creating interactive visualizations of knowledge graphs using PyVis.

## Overview

The Graph Visualization module enables the system to create interactive HTML visualizations of knowledge graphs, including:

1. **Basic Visualization**: Creating complete graph visualizations with customizable appearance
2. **Filtered Visualization**: Showing only specific entity or relationship types
3. **Subgraph Visualization**: Visualizing portions of the graph centered around specific entities
4. **Path Visualization**: Highlighting paths between entities

## Components

### `GraphVisualizer` Class

The main class that provides methods for visualizing knowledge graphs.

#### Methods:

- `create_visualization`: Creates a visualization of the knowledge graph
- `filter_visualization`: Filters the visualization to show only specific entity and relationship types
- `save_visualization`: Saves the visualization to an HTML file

### Helper Functions

The module also provides helper functions for common visualization operations:

- `visualize_graph`: Creates and saves a basic graph visualization
- `visualize_subgraph`: Creates and saves a visualization of a subgraph
- `visualize_path`: Creates and saves a visualization of paths between entities
- `visualize_filtered_graph`: Creates and saves a filtered visualization

## Visualization Features

### Color Schemes

The module supports multiple color schemes for visualizations:

- **Default**: A standard color scheme with distinct colors for different entity and relationship types
- **Grayscale**: A monochrome color scheme for simpler visualizations
- **Pastel**: A soft color scheme with pastel colors

### Entity Representation

Entities are represented as nodes in the visualization with the following features:

- Different shapes based on entity type (e.g., circles for Person entities, boxes for others)
- Color coding based on entity type
- Labels showing entity name, type, and key attributes
- Size variation based on confidence score

### Relationship Representation

Relationships are represented as edges in the visualization with the following features:

- Directional arrows showing relationship direction
- Labels showing relationship type
- Color coding based on relationship type
- Width variation based on confidence score

## Usage Examples

Here are some examples of using the Graph Visualization module:

### Basic Visualization

```python
from src.graph_management.graph import KnowledgeGraph
from src.graph_management.graph_visualizer import visualize_graph

# Create or load a knowledge graph
graph = KnowledgeGraph(name="my_graph")
# ... add entities and relationships ...

# Create and save a basic visualization
result = visualize_graph(
    graph=graph,
    output_path="output/visualizations",
    filename="my_graph.html",
    color_scheme="default"
)

if result.success:
    print(f"Visualization saved to: {result.value}")
```

### Filtered Visualization

```python
from src.graph_management.graph_visualizer import visualize_filtered_graph

# Create a visualization showing only Person and Organization entities
result = visualize_filtered_graph(
    graph=graph,
    entity_types=["Person", "Organization"],
    min_confidence=0.7,
    output_path="output/visualizations",
    filename="filtered_graph.html"
)
```

### Subgraph Visualization

```python
from src.graph_management.graph_visualizer import visualize_subgraph

# Create a visualization centered around a specific entity
result = visualize_subgraph(
    graph=graph,
    entity_ids=["entity_id_1"],
    include_neighbors=True,
    output_path="output/visualizations",
    filename="subgraph.html"
)
```

### Path Visualization

```python
from src.graph_management.graph_visualizer import visualize_path

# Create a visualization of paths between two entities
result = visualize_path(
    graph=graph,
    source_entity_id="entity_id_1",
    target_entity_id="entity_id_2",
    output_path="output/visualizations",
    filename="path.html"
)
```

## Integration with Other Modules

The Graph Visualization module integrates with:

- **Graph Management**: Uses the KnowledgeGraph class for graph data
- **Graph Query**: Uses the GraphQuery class for subgraph extraction
- **Graph Reasoning**: Uses the path-finding functionality for path visualization

## Technical Implementation

The module uses PyVis, a Python library for interactive network visualizations, which is built on top of the vis.js JavaScript library. The visualizations are saved as HTML files that can be viewed in any modern web browser.

Key technical features include:

- HTML output with embedded JavaScript for interactivity
- Force-directed graph layout for automatic arrangement
- Interactive features like zooming, panning, and node dragging
- Tooltips showing detailed information on hover

## Future Enhancements

Possible future enhancements to the Graph Visualization module:

1. Additional layout algorithms for different visualization styles
2. Time-based visualization for temporal knowledge graphs
3. Integration with web-based dashboards
4. Export to additional formats (e.g., SVG, PNG)
5. Advanced filtering and highlighting options
6. Clustering visualization for large graphs 