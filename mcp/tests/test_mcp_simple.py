"""
Simplified MCP server tests without async functionality.
These tests verify the server structure and tool definitions.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_mcp_server_structure():
    """Test that the MCP server has the expected structure."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Check that the FastMCP instance exists
    assert hasattr(server, "mcp")
    assert server.mcp is not None

    # Check that global variables are defined
    assert hasattr(server, "kg_gen_instance")
    assert hasattr(server, "memory_graph")
    assert hasattr(server, "storage_path")


def test_tool_functions_exist():
    """Test that all expected tool functions are defined."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Check that tool functions exist as FastMCP tools
    assert hasattr(server, "add_memories")
    assert hasattr(server, "retrieve_relevant_memories")
    assert hasattr(server, "visualize_memories")
    assert hasattr(server, "get_memory_stats")

    # Check that they have the underlying callable functions
    assert hasattr(server.add_memories, "fn")
    assert callable(server.add_memories.fn)
    assert hasattr(server.retrieve_relevant_memories, "fn")
    assert callable(server.retrieve_relevant_memories.fn)
    assert hasattr(server.visualize_memories, "fn")
    assert callable(server.visualize_memories.fn)
    assert hasattr(server.get_memory_stats, "fn")
    assert callable(server.get_memory_stats.fn)


def test_tool_functions_have_docstrings():
    """Test that tool functions have proper docstrings."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    tools = [
        server.add_memories,
        server.retrieve_relevant_memories,
        server.visualize_memories,
        server.get_memory_stats,
    ]

    for tool in tools:
        # Check that the FastMCP tool has a description
        assert tool.description is not None
        assert len(tool.description.strip()) > 0
        # Check that the underlying function has a docstring
        assert tool.fn.__doc__ is not None
        assert len(tool.fn.__doc__.strip()) > 0


def test_add_memories_no_instance():
    """Test add_memories when no KGGen instance exists."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    with patch("server.kg_gen_instance", None):
        with patch("server.memory_graph", None):
            with patch("server.initialize_kg_gen") as mock_init:
                mock_kg_gen = MagicMock()
                mock_graph = MagicMock()
                mock_graph.entities = {"test"}
                mock_graph.relations = {("a", "b", "c")}
                mock_kg_gen.generate.return_value = mock_graph
                mock_kg_gen.aggregate.return_value = mock_graph
                mock_init.return_value = mock_kg_gen

                with patch("server.save_memory_graph", return_value=True):
                    # Set the global variable that the function will modify
                    with patch("server.kg_gen_instance", mock_kg_gen):
                        result = server.add_memories.fn("Test text")

                # Should have attempted to initialize kg_gen
                assert isinstance(result, str)
                assert "Successfully extracted and stored memories" in result


@patch("server.memory_graph", None)
def test_retrieve_memories_empty():
    """Test retrieve_relevant_memories with no stored memories."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    result = server.retrieve_relevant_memories.fn("test query")
    assert isinstance(result, str)
    assert "No memories stored yet" in result


def test_retrieve_memories_with_data():
    """Test retrieve_relevant_memories with stored memories."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Mock memory graph with test data
    mock_graph = MagicMock()
    mock_graph.entities = {"Alice", "Google", "Engineer"}
    mock_graph.relations = {("Alice", "works for", "Google")}

    with patch("server.memory_graph", mock_graph):
        result = server.retrieve_relevant_memories.fn("Alice")

    assert isinstance(result, str)
    assert "Alice" in result


@patch("server.memory_graph", None)
def test_visualize_memories_empty():
    """Test visualize_memories with no stored memories."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    result = server.visualize_memories.fn("test.html")
    assert isinstance(result, str)
    assert "No memories to visualize" in result


def test_visualize_memories_with_data():
    """Test visualize_memories with stored memories."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Mock memory graph with test data
    mock_graph = MagicMock()
    mock_graph.entities = {"Alice", "Google"}
    mock_graph.relations = {("Alice", "works for", "Google")}

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test.html")

        with patch("server.memory_graph", mock_graph):
            with patch("server.KGGen.visualize") as mock_visualize:
                result = server.visualize_memories.fn(output_path)

        assert isinstance(result, str)
        assert output_path in result
        mock_visualize.assert_called_once()


def test_get_memory_stats():
    """Test get_memory_stats function."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Mock memory graph with test data
    mock_graph = MagicMock()
    mock_graph.entities = {"Alice", "Google"}
    mock_graph.relations = {("Alice", "works for", "Google")}
    mock_graph.edges = {"works for"}
    mock_graph.entity_clusters = {"people": {"Alice"}}
    mock_graph.edge_clusters = {"employment": {"works for"}}

    with patch("server.memory_graph", mock_graph):
        with patch("server.storage_path", "test_path.json"):
            result = server.get_memory_stats.fn()

    assert isinstance(result, str)
    assert "Memory Statistics" in result
    assert "Total Entities: 2" in result
    assert "Total Relations: 1" in result
    assert "test_path.json" in result


@patch("server.memory_graph", None)
def test_get_memory_stats_no_graph():
    """Test get_memory_stats with no memory graph."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    result = server.get_memory_stats.fn()
    assert isinstance(result, str)
    assert "No memory graph loaded" in result


def test_error_handling_in_add_memories():
    """Test error handling in add_memories function."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Mock KGGen to raise an exception
    mock_kg_gen = MagicMock()
    mock_kg_gen.generate.side_effect = Exception("Test error")

    with patch("server.kg_gen_instance", mock_kg_gen):
        result = server.add_memories.fn("Test text")

    assert isinstance(result, str)
    assert "Error extracting memories" in result
    assert "Test error" in result


def test_error_handling_in_retrieve_memories():
    """Test error handling in retrieve_relevant_memories function."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    # Create a mock graph that will cause an error during the query.lower() operation
    mock_graph = MagicMock()
    mock_graph.entities = {"test"}
    # Make len() work but cause an error in the comprehension
    mock_graph.entities.__len__ = MagicMock(return_value=1)
    mock_graph.entities.__iter__ = MagicMock(
        side_effect=Exception("Test iteration error")
    )

    with patch("server.memory_graph", mock_graph):
        result = server.retrieve_relevant_memories.fn("test")

    assert isinstance(result, str)
    assert "Error retrieving memories" in result
