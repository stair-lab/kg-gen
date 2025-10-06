"""
Test that memory JSON files are actually created and used properly.
"""

import pytest
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_memory_file_creation_and_loading():
    """Test that memory files are created when saving and loaded when initializing."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server
    from kg_gen.models import Graph

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "test_memory.json")

        # Mock the KGGen to avoid API calls
        with patch("server.KGGen") as mock_kg_gen:
            mock_instance = MagicMock()
            mock_kg_gen.return_value = mock_instance

            # Set up environment for testing
            original_storage = server.storage_path
            server.storage_path = storage_path

            try:
                # Create a test graph
                test_graph = Graph(
                    entities={"Harry Potter", "Hogwarts"},
                    relations={("Harry Potter", "attends", "Hogwarts")},
                    edges={"attends"},
                )
                server.memory_graph = test_graph

                # Test saving
                success = server.save_memory_graph()
                assert success, "Memory graph should save successfully"

                # Verify file was created
                assert os.path.exists(storage_path), (
                    f"Memory file should be created at {storage_path}"
                )

                # Verify file contents
                with open(storage_path, "r") as f:
                    saved_data = json.load(f)

                assert "entities" in saved_data
                assert "relations" in saved_data
                assert "edges" in saved_data
                assert "Harry Potter" in saved_data["entities"]
                assert "Hogwarts" in saved_data["entities"]
                assert ["Harry Potter", "attends", "Hogwarts"] in saved_data[
                    "relations"
                ]

                # Test loading
                server.memory_graph = None  # Reset
                server.load_memory_graph()

                # Verify data was loaded correctly
                assert server.memory_graph is not None
                assert "Harry Potter" in server.memory_graph.entities
                assert "Hogwarts" in server.memory_graph.entities
                assert (
                    "Harry Potter",
                    "attends",
                    "Hogwarts",
                ) in server.memory_graph.relations

            finally:
                # Restore original storage path
                server.storage_path = original_storage


def test_memory_file_clearing_on_startup():
    """Test that memory files are cleared when KG_CLEAR_MEMORY is set."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "test_memory.json")

        # Create a fake memory file
        fake_memory = {
            "entities": ["Old Entity"],
            "relations": [["Old Entity", "has", "old data"]],
            "edges": ["has"],
        }
        with open(storage_path, "w") as f:
            json.dump(fake_memory, f)

        assert os.path.exists(storage_path), "Test memory file should exist"

        # Mock KGGen to avoid API calls
        with patch("server.KGGen") as mock_kg_gen:
            mock_instance = MagicMock()
            mock_kg_gen.return_value = mock_instance

            # Test with clearing enabled
            with patch.dict(
                os.environ, {"KG_STORAGE_PATH": storage_path, "KG_CLEAR_MEMORY": "true"}
            ):
                # Reset globals
                original_storage = server.storage_path
                original_graph = server.memory_graph
                original_instance = server.kg_gen_instance

                try:
                    server.initialize_kg_gen()

                    # File should be deleted
                    assert not os.path.exists(storage_path), (
                        "Memory file should be cleared on startup"
                    )

                finally:
                    # Restore globals
                    server.storage_path = original_storage
                    server.memory_graph = original_graph
                    server.kg_gen_instance = original_instance


def test_memory_file_preservation_without_clearing():
    """Test that memory files are preserved when KG_CLEAR_MEMORY is not set."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "test_memory.json")

        # Create a fake memory file
        fake_memory = {
            "entities": ["Preserved Entity"],
            "relations": [["Preserved Entity", "has", "preserved data"]],
            "edges": ["has"],
        }
        with open(storage_path, "w") as f:
            json.dump(fake_memory, f)

        assert os.path.exists(storage_path), "Test memory file should exist"

        # Mock KGGen to avoid API calls
        with patch("server.KGGen") as mock_kg_gen:
            mock_instance = MagicMock()
            mock_kg_gen.return_value = mock_instance

            # Test without clearing (default behavior)
            with patch.dict(os.environ, {"KG_STORAGE_PATH": storage_path}, clear=False):
                # Remove KG_CLEAR_MEMORY if it exists
                if "KG_CLEAR_MEMORY" in os.environ:
                    del os.environ["KG_CLEAR_MEMORY"]

                # Reset globals
                original_storage = server.storage_path
                original_graph = server.memory_graph
                original_instance = server.kg_gen_instance

                try:
                    server.initialize_kg_gen()

                    # File should still exist
                    assert os.path.exists(storage_path), (
                        "Memory file should be preserved when not clearing"
                    )

                    # Memory should be loaded
                    assert server.memory_graph is not None
                    assert "Preserved Entity" in server.memory_graph.entities

                finally:
                    # Restore globals
                    server.storage_path = original_storage
                    server.memory_graph = original_graph
                    server.kg_gen_instance = original_instance


def test_storage_path_configuration():
    """Test that storage path can be configured via environment variable."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server

    custom_path = "/custom/path/to/memory.json"

    # Mock KGGen to avoid API calls
    with patch("server.KGGen") as mock_kg_gen:
        mock_instance = MagicMock()
        mock_kg_gen.return_value = mock_instance

        # Mock os.path.exists to avoid file system checks
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {"KG_STORAGE_PATH": custom_path}):
                # Reset globals
                original_storage = server.storage_path
                original_graph = server.memory_graph
                original_instance = server.kg_gen_instance

                try:
                    server.initialize_kg_gen()

                    # Storage path should be set to custom path
                    assert server.storage_path == custom_path

                finally:
                    # Restore globals
                    server.storage_path = original_storage
                    server.memory_graph = original_graph
                    server.kg_gen_instance = original_instance
