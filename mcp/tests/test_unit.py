"""
Unit tests for core MCP server functions.
These tests focus on individual functions without the full MCP client setup.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the server module
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from server import (
    initialize_kg_gen,
    load_memory_graph,
    save_memory_graph,
    memory_graph,
    storage_path,
)


@pytest.fixture
def temp_storage_file():
    """Create a temporary file for storage testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
    return {
        "entities": ["Alice", "Google", "Software Engineer"],
        "relations": [
            ["Alice", "works for", "Google"],
            ["Alice", "is a", "Software Engineer"],
        ],
        "edges": ["works for", "is a"],
        "entity_clusters": {
            "people": ["Alice"],
            "companies": ["Google"],
            "roles": ["Software Engineer"],
        },
        "edge_clusters": {"employment": ["works for"], "identity": ["is a"]},
    }


class TestInitializeKGGen:
    """Test the initialize_kg_gen function."""

    @patch.dict(
        os.environ,
        {
            "KG_MODEL": "test-model",
            "KG_API_KEY": "test-key",
            "KG_STORAGE_PATH": "./test_storage.json",
        },
    )
    @patch("server.KGGen")
    @patch("server.load_memory_graph")
    def test_initialize_with_env_vars(self, mock_load, mock_kg_gen):
        """Test initialization with environment variables."""
        mock_instance = MagicMock()
        mock_kg_gen.return_value = mock_instance

        result = initialize_kg_gen()

        # Verify KGGen was called with correct parameters
        mock_kg_gen.assert_called_once_with(
            model="test-model", temperature=0.0, api_key="test-key"
        )

        # Verify load_memory_graph was called
        mock_load.assert_called_once()

        assert result == mock_instance

    @patch.dict(os.environ, {}, clear=True)
    @patch("server.KGGen")
    @patch("server.load_memory_graph")
    def test_initialize_with_defaults(self, mock_load, mock_kg_gen):
        """Test initialization with default values."""
        mock_instance = MagicMock()
        mock_kg_gen.return_value = mock_instance

        result = initialize_kg_gen()

        # Verify KGGen was called with defaults
        mock_kg_gen.assert_called_once_with(
            model="openai/gpt-4o", temperature=0.0, api_key=None
        )

        mock_load.assert_called_once()
        assert result == mock_instance


class TestLoadMemoryGraph:
    """Test the load_memory_graph function."""

    def test_load_existing_graph(self, temp_storage_file, sample_graph_data):
        """Test loading an existing graph from storage."""
        # Write sample data to temp file
        with open(temp_storage_file, "w") as f:
            json.dump(sample_graph_data, f)

        # Patch the global storage_path
        with patch("server.storage_path", temp_storage_file):
            with patch("server.Graph") as mock_graph:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance

                load_memory_graph()

                # Verify Graph was created with correct data
                mock_graph.assert_called_once()
                call_args = mock_graph.call_args[1]

                assert call_args["entities"] == set(sample_graph_data["entities"])
                assert call_args["relations"] == {
                    tuple(rel) for rel in sample_graph_data["relations"]
                }
                assert call_args["edges"] == set(sample_graph_data["edges"])

    def test_load_nonexistent_file(self, temp_storage_file):
        """Test loading when storage file doesn't exist."""
        # Ensure file doesn't exist
        if os.path.exists(temp_storage_file):
            os.unlink(temp_storage_file)

        with patch("server.storage_path", temp_storage_file):
            with patch("server.Graph") as mock_graph:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance

                load_memory_graph()

                # Verify empty Graph was created
                mock_graph.assert_called_once_with(
                    entities=set(), relations=set(), edges=set()
                )

    def test_load_corrupted_file(self, temp_storage_file):
        """Test loading when storage file is corrupted."""
        # Write invalid JSON to file
        with open(temp_storage_file, "w") as f:
            f.write("invalid json content")

        with patch("server.storage_path", temp_storage_file):
            with patch("server.Graph") as mock_graph:
                mock_graph_instance = MagicMock()
                mock_graph.return_value = mock_graph_instance

                load_memory_graph()

                # Should create empty graph on error
                mock_graph.assert_called_once_with(
                    entities=set(), relations=set(), edges=set()
                )


class TestSaveMemoryGraph:
    """Test the save_memory_graph function."""

    def test_save_graph_success(self, temp_storage_file):
        """Test successfully saving a graph."""
        mock_graph = MagicMock()
        mock_graph.entities = {"Alice", "Google"}
        mock_graph.relations = {("Alice", "works for", "Google")}
        mock_graph.edges = {"works for"}
        mock_graph.entity_clusters = {"people": {"Alice"}}
        mock_graph.edge_clusters = {"employment": {"works for"}}

        with patch("server.storage_path", temp_storage_file):
            with patch("server.memory_graph", mock_graph):
                result = save_memory_graph()

        assert result is True

        # Verify file was written
        assert os.path.exists(temp_storage_file)

        # Verify content
        with open(temp_storage_file, "r") as f:
            data = json.load(f)

        assert set(data["entities"]) == {"Alice", "Google"}
        assert data["relations"] == [["Alice", "works for", "Google"]]
        assert set(data["edges"]) == {"works for"}

    def test_save_graph_no_graph(self, temp_storage_file):
        """Test saving when no graph exists."""
        with patch("server.storage_path", temp_storage_file):
            with patch("server.memory_graph", None):
                result = save_memory_graph()

        assert result is False

    def test_save_graph_permission_error(self, temp_storage_file):
        """Test saving when file write fails."""
        mock_graph = MagicMock()
        mock_graph.entities = {"Alice"}
        mock_graph.relations = set()
        mock_graph.edges = set()
        mock_graph.entity_clusters = None
        mock_graph.edge_clusters = None

        # Use a directory path to force a write error
        invalid_path = "/invalid/path/file.json"

        with patch("server.storage_path", invalid_path):
            with patch("server.memory_graph", mock_graph):
                result = save_memory_graph()

        assert result is False


class TestMemoryGraphOperations:
    """Test memory graph operations."""

    def test_graph_with_none_clusters(self, temp_storage_file, sample_graph_data):
        """Test loading graph with None clusters."""
        # Modify sample data to have None clusters
        graph_data = sample_graph_data.copy()
        graph_data["entity_clusters"] = None
        graph_data["edge_clusters"] = None

        with open(temp_storage_file, "w") as f:
            json.dump(graph_data, f)

        with patch("server.storage_path", temp_storage_file):
            with patch("server.Graph") as mock_graph:
                load_memory_graph()

                call_args = mock_graph.call_args[1]
                assert call_args["entity_clusters"] is None
                assert call_args["edge_clusters"] is None

    def test_graph_with_empty_clusters(self, temp_storage_file, sample_graph_data):
        """Test loading graph with empty clusters."""
        # Modify sample data to have empty clusters
        graph_data = sample_graph_data.copy()
        graph_data["entity_clusters"] = {}
        graph_data["edge_clusters"] = {}

        with open(temp_storage_file, "w") as f:
            json.dump(graph_data, f)

        with patch("server.storage_path", temp_storage_file):
            with patch("server.Graph") as mock_graph:
                load_memory_graph()

                call_args = mock_graph.call_args[1]
                # Empty dictionaries get converted to None by the server logic
                assert call_args["entity_clusters"] is None
                assert call_args["edge_clusters"] is None


class TestEnvironmentHandling:
    """Test environment variable handling."""

    def test_openai_api_key_fallback(self):
        """Test fallback to OPENAI_API_KEY when KG_API_KEY is not set."""
        test_env = {"OPENAI_API_KEY": "openai-key", "KG_MODEL": "test-model"}

        with patch.dict(os.environ, test_env, clear=True):
            with patch("server.KGGen") as mock_kg_gen:
                with patch("server.load_memory_graph"):
                    initialize_kg_gen()

                    # Should use OPENAI_API_KEY as fallback
                    mock_kg_gen.assert_called_once_with(
                        model="test-model", temperature=0.0, api_key="openai-key"
                    )

    def test_kg_api_key_priority(self):
        """Test that KG_API_KEY takes priority over OPENAI_API_KEY."""
        test_env = {
            "KG_API_KEY": "kg-key",
            "OPENAI_API_KEY": "openai-key",
            "KG_MODEL": "test-model",
        }

        with patch.dict(os.environ, test_env, clear=True):
            with patch("server.KGGen") as mock_kg_gen:
                with patch("server.load_memory_graph"):
                    initialize_kg_gen()

                    # Should use KG_API_KEY
                    mock_kg_gen.assert_called_once_with(
                        model="test-model", temperature=0.0, api_key="kg-key"
                    )
