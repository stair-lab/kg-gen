#!/usr/bin/env python3
"""
Run all MCP server tests for kg-gen.
This script runs all test files in the tests directory.
"""

import sys
import pytest
from pathlib import Path


def main():
    """Run all tests in the current directory."""
    # Get the tests directory
    tests_dir = Path(__file__).parent

    # Run pytest on all test_*.py files
    print("Running all kg-gen MCP server tests...")
    print("-" * 50)

    # Run tests with verbose output
    args = [
        str(tests_dir),
        "-v",  # Verbose
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "--color=yes",  # Colored output
    ]

    # Add asyncio mode if pytest-asyncio is available
    try:
        import pytest_asyncio

        args.append("--asyncio-mode=auto")
    except ImportError:
        print("Note: pytest-asyncio not available, some async tests may fail")

    exit_code = pytest.main(args)

    if exit_code == 0:
        print("\n✅ All MCP server tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
