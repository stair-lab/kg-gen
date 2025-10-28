#!/bin/bash
# Script to run the dedup_demo.py script in Docker
# Usage: ./run_dedup_docker.sh <graph path>[additional_args...]

set -e

# Default script to run
SCRIPT=dedup_demo.py

# Build the Docker image
echo "🐳 Building Docker container for kg-gen scripts..."
docker build -f Dockerfile -t kg-gen-dedup-demo .
echo "✅ Docker image built successfully!"

# Check if the script exists
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Error: $SCRIPT not found in current directory"
    echo "   Available Python scripts:"
    ls -1 *.py 2>/dev/null || echo "   No Python scripts found"
    exit 1
fi

# Create output and logs directories if they don't exist
mkdir -p output logs

# Run the container with volume mounts
echo "🚀 Running $SCRIPT in Docker container with arguments: $@"
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/logs:/workspace/logs" \
    --env-file .env \
    kg-gen-dedup-demo python "$SCRIPT" "$@"

echo "✅ Script execution completed! Check the output/ directory for results."
