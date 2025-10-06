"""
Pytest configuration and shared fixtures for MCP server tests.
"""

import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_api_key():
    """Get API key for testing, with helpful error if missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def mcp_server_script():
    """Get the path to the MCP server script."""
    return Path(__file__).parent.parent / "server.py"


@pytest.fixture(autouse=True)
def reset_server_globals():
    """Reset server global variables before each test."""
    import sys

    # Add the parent directory to path so we can import server
    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    # Import and reset server globals
    try:
        import server

        server.kg_gen_instance = None
        server.memory_graph = None
        server.storage_path = None
    except ImportError:
        pass  # Server module not available in some test contexts

    yield  # Run the test

    # Cleanup after test
    try:
        import server

        server.kg_gen_instance = None
        server.memory_graph = None
        server.storage_path = None
    except ImportError:
        pass
