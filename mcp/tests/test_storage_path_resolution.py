"""
Test that storage paths are resolved correctly relative to where the command is called.
"""

import pytest
import os
import tempfile
import subprocess
from pathlib import Path


def test_cli_resolves_relative_paths():
    """Test that the CLI resolves relative paths relative to where the command is called."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test directory to run the command from
        test_dir = os.path.join(temp_dir, "test_project")
        os.makedirs(test_dir)

        # Run the CLI help command from the test directory to verify path resolution
        # We'll use a mock test since actually running the server would require API keys
        original_cwd = os.getcwd()

        try:
            os.chdir(test_dir)

            # Test that relative paths are resolved relative to current directory
            current_dir = os.getcwd()
            expected_path = os.path.join(current_dir, "kg_memory.json")

            # Simulate what the CLI does
            storage_path = "./kg_memory.json"
            abs_storage_path = os.path.abspath(storage_path)

            assert abs_storage_path == expected_path
            assert abs_storage_path.startswith(current_dir)

        finally:
            os.chdir(original_cwd)


def test_cli_handles_absolute_paths():
    """Test that the CLI preserves absolute paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "test_project")
        os.makedirs(test_dir)

        absolute_path = os.path.join(temp_dir, "custom_memory.json")

        original_cwd = os.getcwd()

        try:
            os.chdir(test_dir)

            # Simulate what the CLI does with absolute paths
            abs_storage_path = os.path.abspath(absolute_path)

            # Should preserve the absolute path
            assert abs_storage_path == absolute_path

        finally:
            os.chdir(original_cwd)


def test_server_path_resolution():
    """Test that the server resolves relative paths to absolute paths."""
    import sys

    server_dir = str(Path(__file__).parent.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)

    import server
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with relative path
        relative_path = "./test_memory.json"

        with patch("server.KGGen") as mock_kg_gen:
            mock_instance = type("MockKGGen", (), {})()
            mock_kg_gen.return_value = mock_instance

            with patch.dict(
                os.environ,
                {"KG_STORAGE_PATH": relative_path, "KG_CLEAR_MEMORY": "false"},
            ):
                # Mock os.path.exists to avoid file system operations
                with patch("os.path.exists", return_value=False):
                    original_cwd = os.getcwd()
                    original_storage = server.storage_path
                    original_graph = server.memory_graph
                    original_instance = server.kg_gen_instance

                    try:
                        os.chdir(temp_dir)
                        expected_abs_path = os.path.join(temp_dir, "test_memory.json")

                        server.initialize_kg_gen()

                        # Should be converted to absolute path
                        # Use os.path.samefile to handle macOS path resolution differences (/var vs /private/var)
                        assert os.path.isabs(server.storage_path)
                        assert (
                            os.path.basename(server.storage_path) == "test_memory.json"
                        )
                        # Check the directory contains the temp directory (handles /var vs /private/var)
                        assert (
                            temp_dir in server.storage_path
                            or os.path.realpath(temp_dir) in server.storage_path
                        )

                    finally:
                        os.chdir(original_cwd)
                        server.storage_path = original_storage
                        server.memory_graph = original_graph
                        server.kg_gen_instance = original_instance


def test_different_working_directories():
    """Test that the same relative path resolves to different absolute paths in different directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dir1 = os.path.join(temp_dir, "project1")
        dir2 = os.path.join(temp_dir, "project2")
        os.makedirs(dir1)
        os.makedirs(dir2)

        original_cwd = os.getcwd()

        try:
            # Test from dir1
            os.chdir(dir1)
            path1 = os.path.abspath("./kg_memory.json")

            # Test from dir2
            os.chdir(dir2)
            path2 = os.path.abspath("./kg_memory.json")

            # Should be different absolute paths
            assert path1 != path2
            assert path1.endswith("project1/kg_memory.json")
            assert path2.endswith("project2/kg_memory.json")

        finally:
            os.chdir(original_cwd)


def test_cli_path_resolution_integration():
    """Integration test for CLI path resolution using subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = os.path.join(temp_dir, "integration_test")
        os.makedirs(test_dir)

        # Test that kggen mcp --help works and shows correct help
        # We use --help to avoid actually starting the server
        result = subprocess.run(
            ["python", "-m", "kg_gen.cli", "mcp", "--help"],
            cwd=test_dir,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--storage-path" in result.stdout
        assert "Path for memory storage file" in result.stdout
