"""
Test the MCP server functionality of kg-gen.
This tests the server integration, tools, and memory management.
"""

import pytest
import asyncio
import os
import tempfile
import json
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.fixture
def mcp_server_path():
    """Get the path to the MCP server file."""
    return Path(__file__).parent.parent / "server.py"


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


async def init_mcp_server(storage_path=None):
    """Initialize MCP server with optional storage path."""
    server_path = Path(__file__).parent.parent / "server.py"

    # Set up environment
    env = os.environ.copy()
    env["KG_MODEL"] = "openai/gpt-4o-mini"  # Use cheaper model for testing
    env["KG_API_KEY"] = env.get("OPENAI_API_KEY", "test-key")
    if storage_path:
        env["KG_STORAGE_PATH"] = storage_path

    # Create server parameters
    server_params = StdioServerParameters(
        command="fastmcp", args=["run", str(server_path)], env=env
    )

    return server_params


@pytest.mark.asyncio
async def test_add_memories_tool(temp_storage_dir):
    """Test the add_memories tool functionality."""
    storage_path = os.path.join(temp_storage_dir, "test_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test adding memories
            text = "John Smith works for Google. Google is located in Mountain View. Jane Doe is a software engineer at Microsoft."
            result = await session.call_tool("add_memories", {"text": text})

            # Check response
            response_text = result.content[0].text
            assert "Successfully extracted and stored memories" in response_text
            assert "entities" in response_text
            assert "relations" in response_text

            # Verify storage file was created
            assert os.path.exists(storage_path)


@pytest.mark.asyncio
async def test_retrieve_relevant_memories_tool(temp_storage_dir):
    """Test the retrieve_relevant_memories tool functionality."""
    storage_path = os.path.join(temp_storage_dir, "test_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # First add some memories
            text = "Alice works for TechCorp. Bob also works for TechCorp. TechCorp is headquartered in San Francisco."
            await session.call_tool("add_memories", {"text": text})

            # Now retrieve relevant memories
            query = "Who works at TechCorp?"
            result = await session.call_tool(
                "retrieve_relevant_memories", {"query": query}
            )

            # Check response
            response_text = result.content[0].text
            assert isinstance(response_text, str)
            assert (
                "Alice" in response_text
                or "Bob" in response_text
                or "TechCorp" in response_text
            )


@pytest.mark.asyncio
async def test_retrieve_memories_empty_storage(temp_storage_dir):
    """Test retrieving memories when no memories are stored."""
    storage_path = os.path.join(temp_storage_dir, "empty_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Try to retrieve memories without adding any
            query = "test query"
            result = await session.call_tool(
                "retrieve_relevant_memories", {"query": query}
            )

            # Check response
            response_text = result.content[0].text
            assert "No memories stored yet" in response_text


@pytest.mark.asyncio
async def test_get_memory_stats_tool(temp_storage_dir):
    """Test the get_memory_stats tool functionality."""
    storage_path = os.path.join(temp_storage_dir, "test_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # First add some memories
            text = "Tesla is an electric car company. Elon Musk is the CEO of Tesla."
            await session.call_tool("add_memories", {"text": text})

            # Get memory stats
            result = await session.call_tool("get_memory_stats", {})

            # Check response
            response_text = result.content[0].text
            assert "Memory Statistics" in response_text
            assert "Total Entities" in response_text
            assert "Total Relations" in response_text
            assert "Storage Path" in response_text


@pytest.mark.asyncio
async def test_visualize_memories_tool(temp_storage_dir):
    """Test the visualize_memories tool functionality."""
    storage_path = os.path.join(temp_storage_dir, "test_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # First add some memories
            text = (
                "Apple Inc. is a technology company. Tim Cook is the CEO of Apple Inc."
            )
            await session.call_tool("add_memories", {"text": text})

            # Generate visualization
            output_filename = os.path.join(temp_storage_dir, "test_visualization.html")
            result = await session.call_tool(
                "visualize_memories", {"output_filename": output_filename}
            )

            # Check response
            response_text = result.content[0].text
            assert "Memory graph visualization saved to" in response_text
            assert output_filename in response_text

            # Verify HTML file was created
            assert os.path.exists(output_filename)


@pytest.mark.asyncio
async def test_visualize_memories_empty_storage(temp_storage_dir):
    """Test visualizing memories when no memories are stored."""
    storage_path = os.path.join(temp_storage_dir, "empty_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Try to visualize without adding any memories
            result = await session.call_tool("visualize_memories", {})

            # Check response
            response_text = result.content[0].text
            assert "No memories to visualize" in response_text


@pytest.mark.asyncio
async def test_memory_persistence(temp_storage_dir):
    """Test that memories persist across server restarts."""
    storage_path = os.path.join(temp_storage_dir, "persistent_memory.json")

    # First session: add memories
    server_params1 = await init_mcp_server(storage_path)
    async with stdio_client(server_params1) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            text = "Netflix is a streaming service. Reed Hastings founded Netflix."
            await session.call_tool("add_memories", {"text": text})

    # Second session: verify memories are loaded
    server_params2 = await init_mcp_server(storage_path)
    async with stdio_client(server_params2) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Query for previously stored memories
            query = "Netflix"
            result = await session.call_tool(
                "retrieve_relevant_memories", {"query": query}
            )

            response_text = result.content[0].text
            assert "Netflix" in response_text


@pytest.mark.asyncio
async def test_memory_aggregation(temp_storage_dir):
    """Test that multiple memory additions are properly aggregated."""
    storage_path = os.path.join(temp_storage_dir, "aggregated_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Add first batch of memories
            text1 = "SpaceX is a space company. Elon Musk founded SpaceX."
            result1 = await session.call_tool("add_memories", {"text": text1})

            # Get stats after first addition
            stats1 = await session.call_tool("get_memory_stats", {})
            stats1_text = stats1.content[0].text

            # Add second batch of memories
            text2 = "SpaceX launched Falcon Heavy. Falcon Heavy is a rocket."
            result2 = await session.call_tool("add_memories", {"text": text2})

            # Get stats after second addition
            stats2 = await session.call_tool("get_memory_stats", {})
            stats2_text = stats2.content[0].text

            # Verify that memories were aggregated (should have more entities/relations)
            assert "Total Entities" in stats1_text
            assert "Total Entities" in stats2_text

            # Query for memories from both batches
            query = "SpaceX"
            result = await session.call_tool(
                "retrieve_relevant_memories", {"query": query}
            )
            response_text = result.content[0].text

            # Should find entities from both batches
            assert "SpaceX" in response_text


@pytest.mark.asyncio
async def test_error_handling_invalid_text(temp_storage_dir):
    """Test error handling for invalid input."""
    storage_path = os.path.join(temp_storage_dir, "error_test_memory.json")
    server_params = await init_mcp_server(storage_path)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test with empty text
            result = await session.call_tool("add_memories", {"text": ""})
            response_text = result.content[0].text

            # Should handle gracefully
            assert isinstance(response_text, str)

            # Test with very short text
            result = await session.call_tool("add_memories", {"text": "Hi"})
            response_text = result.content[0].text

            # Should handle gracefully
            assert isinstance(response_text, str)
