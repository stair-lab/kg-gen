#!/usr/bin/env python3
"""
Command line interface for kg-gen.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import importlib.util


def check_and_install_mcp_dependencies():
    """Check if MCP dependencies are installed, and install them if not."""
    # Check if fastmcp is available
    fastmcp_spec = importlib.util.find_spec("fastmcp")
    mcp_spec = importlib.util.find_spec("mcp")

    if fastmcp_spec is None or mcp_spec is None:
        print("MCP dependencies not found. Installing them now...")
        print("This is a one-time setup for MCP functionality.")
        print()

        try:
            # Try to install MCP dependencies
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "fastmcp>=2.10.6",
                    "mcp>=1.12.1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            print("✅ MCP dependencies installed successfully!")
            print()
            return True

        except subprocess.CalledProcessError as e:
            print("❌ Failed to install MCP dependencies:")
            print(f"Error: {e.stderr}")
            print()
            print("Please install manually:")
            print("  pip install fastmcp>=2.10.6 mcp>=1.12.1")
            print("or")
            print("  pip install 'kg-gen[mcp]'")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during installation: {e}")
            print()
            print("Please install manually:")
            print("  pip install fastmcp>=2.10.6 mcp>=1.12.1")
            return False

    return True


def run_mcp():
    """Start the MCP server using fastmcp."""
    # Check and install MCP dependencies if needed
    if not check_and_install_mcp_dependencies():
        return 1

    # Get the path to the server.py file
    server_path = Path(__file__).parent.parent.parent / "mcp" / "server.py"

    if not server_path.exists():
        print(f"Error: MCP server file not found at {server_path}")
        return 1

    print("Starting kg-gen MCP server...")
    print(f"Server path: {server_path}")
    print("Use Ctrl+C to stop the server")
    print()

    try:
        # Run fastmcp with the server script
        result = subprocess.run(["fastmcp", "run", str(server_path)], check=False)
        return result.returncode
    except FileNotFoundError:
        print("Error: fastmcp not found even after installation attempt.")
        print("Please try installing manually:")
        print("  pip install fastmcp>=2.10.6 mcp>=1.12.1")
        return 1
    except KeyboardInterrupt:
        print("\nMCP server stopped.")
        return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="kggen",
        description="kg-gen: Extract knowledge graphs using LLMs from any text",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # MCP subcommand
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument(
        "--model", help="Model to use (e.g., openai/gpt-4o)", default=None
    )
    mcp_parser.add_argument(
        "--storage-path", help="Path for memory storage file", default=None
    )
    mcp_parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="Keep existing memory instead of clearing it on startup",
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "mcp":
        # Set environment variables if provided
        if args.model:
            os.environ["KG_MODEL"] = args.model

        # Handle storage path - resolve relative paths relative to where command was called
        storage_path = args.storage_path or "./kg_memory.json"
        # Convert to absolute path relative to current working directory (where user called command)
        abs_storage_path = os.path.abspath(storage_path)
        os.environ["KG_STORAGE_PATH"] = abs_storage_path

        # Clear memory by default unless --keep-memory is specified
        if not args.keep_memory:
            os.environ["KG_CLEAR_MEMORY"] = "true"

        return run_mcp()

    return 0


if __name__ == "__main__":
    sys.exit(main())
