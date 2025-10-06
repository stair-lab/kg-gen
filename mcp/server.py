"""
This is the MCP server for kg-gen agent memory.
Run the server locally with: `kggen mcp` or `fastmcp run server.py`

You can specify configuration via environment variables:
- KG_MODEL (default: "openai/gpt-4o")
- KG_API_KEY (optional, can also use OPENAI_API_KEY)
- KG_STORAGE_PATH (default: "./kg_memory.json")
- KG_CLEAR_MEMORY (default: "false", set to "true" to clear memory on startup)

CLI Examples:
- kggen mcp (clears memory by default, uses ./kg_memory.json relative to current directory)
- kggen mcp --keep-memory (preserves existing memory)
- kggen mcp --model gemini/gemini-2.0-flash --storage-path ./custom.json
- kggen mcp --storage-path /absolute/path/to/memory.json (absolute path)

Environment Examples:
- KG_MODEL=gemini/gemini-2.0-flash fastmcp run server.py
- KG_CLEAR_MEMORY=true fastmcp run server.py

The server provides tools for agent memory:
- add_memories: Extract and store memories from unstructured text
- retrieve_relevant_memories: Retrieve relevant memories for a query
- visualize_memories: Generate HTML visualization of the memory graph
- get_memory_stats: Get statistics about stored memories
"""

import os
import json
from typing import Optional
from pathlib import Path
from fastmcp import FastMCP

from kg_gen import KGGen, Graph

# Global variables
kg_gen_instance = None
memory_graph = None
storage_path = None


def initialize_kg_gen():
    """Initialize KGGen with environment configuration."""
    global kg_gen_instance, memory_graph, storage_path

    # Get configuration from environment variables
    model = os.environ.get("KG_MODEL", "openai/gpt-4o")
    api_key = os.environ.get("KG_API_KEY") or os.environ.get("OPENAI_API_KEY")
    storage_path = os.environ.get("KG_STORAGE_PATH", "./kg_memory.json")
    clear_memory = os.environ.get("KG_CLEAR_MEMORY", "false").lower() == "true"

    # Ensure storage path is absolute for consistent behavior
    if not os.path.isabs(storage_path):
        storage_path = os.path.abspath(storage_path)

    print(f"Initializing KGGen with model: {model}")
    print(f"Using storage path: {storage_path}")

    # Clear existing memory if requested
    if clear_memory and os.path.exists(storage_path):
        try:
            os.remove(storage_path)
            print(f"Cleared existing memory file: {storage_path}")
        except Exception as e:
            print(f"Warning: Could not clear memory file: {e}")

    # Initialize KGGen
    kg_gen_instance = KGGen(model=model, temperature=0.0, api_key=api_key)

    # Load existing memory graph if it exists
    load_memory_graph()

    return kg_gen_instance


def load_memory_graph():
    """Load existing memory graph from storage."""
    global memory_graph, storage_path

    if os.path.exists(storage_path):
        try:
            with open(storage_path, "r") as f:
                graph_dict = json.load(f)

            memory_graph = Graph(
                entities=set(graph_dict.get("entities", [])),
                relations=set(tuple(rel) for rel in graph_dict.get("relations", [])),
                edges=set(graph_dict.get("edges", [])),
                entity_clusters={
                    k: set(v) for k, v in graph_dict.get("entity_clusters", {}).items()
                }
                if graph_dict.get("entity_clusters")
                else None,
                edge_clusters={
                    k: set(v) for k, v in graph_dict.get("edge_clusters", {}).items()
                }
                if graph_dict.get("edge_clusters")
                else None,
            )
            print(
                f"Loaded existing memory graph with {len(memory_graph.entities)} entities and {len(memory_graph.relations)} relations"
            )
        except Exception as e:
            print(f"Error loading memory graph: {e}")
            memory_graph = Graph(entities=set(), relations=set(), edges=set())
    else:
        memory_graph = Graph(entities=set(), relations=set(), edges=set())


def save_memory_graph():
    """Save memory graph to storage."""
    global memory_graph, storage_path

    if memory_graph is None:
        return False

    try:
        graph_dict = {
            "entities": list(memory_graph.entities),
            "relations": list(memory_graph.relations),
            "edges": list(memory_graph.edges),
            "entity_clusters": {
                k: list(v) for k, v in memory_graph.entity_clusters.items()
            }
            if memory_graph.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in memory_graph.edge_clusters.items()}
            if memory_graph.edge_clusters
            else None,
        }

        with open(storage_path, "w") as f:
            json.dump(graph_dict, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving memory graph: {e}")
        return False


# Initialize on module load
initialize_kg_gen()

mcp = FastMCP(name="KGGen")


@mcp.tool
def add_memories(text: str) -> str:
    """
    Extract and store memories from unstructured text.

    Args:
        text: Unstructured text to extract memories from

    Returns:
        Summary of extracted memories
    """
    global kg_gen_instance, memory_graph
    if kg_gen_instance is None:
        initialize_kg_gen()

    try:
        # Generate graph from text
        new_graph = kg_gen_instance.generate(input_data=text)

        # Merge with existing memory graph
        if memory_graph is None:
            memory_graph = new_graph
        else:
            memory_graph = kg_gen_instance.aggregate([memory_graph, new_graph])

        # Save to storage
        success = save_memory_graph()

        result = f"Successfully extracted and stored memories from text.\n"
        result += f"New memories: {len(new_graph.entities)} entities, {len(new_graph.relations)} relations\n"
        result += f"Total memories: {len(memory_graph.entities)} entities, {len(memory_graph.relations)} relations\n"
        result += f"Storage: {'Saved successfully' if success else 'Failed to save'}"

        return result

    except Exception as e:
        return f"Error extracting memories: {str(e)}"


@mcp.tool
def retrieve_relevant_memories(query: str) -> str:
    """
    Retrieve relevant memories for a query.

    Args:
        query: Query to find relevant memories for

    Returns:
        Relevant memories as text
    """
    global memory_graph

    if memory_graph is None or len(memory_graph.entities) == 0:
        return "No memories stored yet. Use add_memories to store some memories first."

    try:
        # Simple keyword-based retrieval for now
        # Find entities and relations that contain query terms
        query_lower = query.lower()
        relevant_entities = [
            e for e in memory_graph.entities if query_lower in e.lower()
        ]
        relevant_relations = [
            r
            for r in memory_graph.relations
            if any(query_lower in str(part).lower() for part in r)
        ]

        if not relevant_entities and not relevant_relations:
            return f"No relevant memories found for query: '{query}'"

        result = f"Relevant memories for '{query}':\n\n"

        if relevant_entities:
            result += f"Related entities ({len(relevant_entities)}):\n"
            for entity in relevant_entities[:10]:  # Limit to top 10
                result += f"- {entity}\n"
            result += "\n"

        if relevant_relations:
            result += f"Related facts ({len(relevant_relations)}):\n"
            for relation in relevant_relations[:10]:  # Limit to top 10
                result += f"- {relation[0]} {relation[1]} {relation[2]}\n"

        return result

    except Exception as e:
        return f"Error retrieving memories: {str(e)}"


@mcp.tool
def visualize_memories(output_filename: str = "memory_graph.html") -> str:
    """
    Generate HTML visualization of the memory graph.

    Args:
        output_filename: Name for the output HTML file

    Returns:
        Path to the generated visualization
    """
    global kg_gen_instance, memory_graph

    if memory_graph is None or len(memory_graph.entities) == 0:
        return (
            "No memories to visualize. Use add_memories to store some memories first."
        )

    try:
        # Create output path
        output_path = os.path.abspath(output_filename)

        # Generate visualization
        KGGen.visualize(memory_graph, output_path, open_in_browser=False)

        return f"Memory graph visualization saved to: {output_path}\n\nVisualization contains {len(memory_graph.entities)} entities and {len(memory_graph.relations)} relations.\nOpen the HTML file in your browser to view the interactive graph."

    except Exception as e:
        return f"Error generating visualization: {str(e)}"


@mcp.tool
def get_memory_stats() -> str:
    """
    Get statistics about stored memories.
    """
    global memory_graph, storage_path

    if memory_graph is None:
        return "No memory graph loaded."

    stats = f"""Memory Statistics:
- Total Entities: {len(memory_graph.entities)}
- Total Relations: {len(memory_graph.relations)}
- Edge Types: {len(memory_graph.edges)}
- Storage Path: {storage_path}
- Entity Clusters: {len(memory_graph.entity_clusters) if memory_graph.entity_clusters else 0}
- Edge Clusters: {len(memory_graph.edge_clusters) if memory_graph.edge_clusters else 0}"""

    return stats


if __name__ == "__main__":
    mcp.run()
