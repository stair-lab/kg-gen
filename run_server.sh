#!/bin/bash
# Script to run the kg-gen FastAPI server in Docker
# Usage: ./run_server.sh

set -e

# Build the Docker image
echo "🐳 Building Docker container for kg-gen scripts..."
docker build -f Dockerfile -t kg-gen-dedup-demo .
echo "✅ Docker image built successfully!"

# Run the container with volume mounts
echo "🚀 Running kg-gen FastAPI server in Docker container"
docker run --rm \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/logs:/workspace/logs" \
    -p 8000:8000 \
    --env-file .env \
    kg-gen-dedup-demo uvicorn app.server:app --reload --host 0.0.0.0 --port 8000
