#!/usr/bin/env python3
"""
Upload a knowledge graph (graph.json) to Neo4j (experiments script).

Usage examples:

  python experiments/wikiqa/_4_upload_neo4j.py \
    --graph-file path/to/graph.json \
    --uri bolt://localhost:7687 \
    --username neo4j \
    --password your-password \
    --graph-name my_graph \
    --clear-existing

  # AuraDB example
  python experiments/wikiqa/_4_upload_neo4j.py \
    --graph-file path/to/graph.json \
    --uri neo4j+s://<instance-id>.databases.neo4j.io \
    --username neo4j \
    --password your-aura-password \
    --graph-name my_graph

"""

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import the package from local src
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kg_gen.models import Graph  # type: ignore
from kg_gen.utils.neo4j_integration import (  # type: ignore
    upload_to_neo4j,
    Neo4jUploader,
)


def load_graph_from_file(file_path: str) -> Graph:
    """Load a Graph from a JSON file, converting lists to sets."""
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert lists back to sets expected by Graph model
    if isinstance(data.get("entities"), list):
        data["entities"] = set(data["entities"])
    if isinstance(data.get("relations"), list):
        data["relations"] = set(tuple(triple) for triple in data["relations"])
    if isinstance(data.get("edges"), list):
        data["edges"] = set(data["edges"])

    # Optional clusters
    if data.get("entity_clusters"):
        data["entity_clusters"] = {
            k: set(v) for k, v in data["entity_clusters"].items()
        }
    if data.get("edge_clusters"):
        data["edge_clusters"] = {k: set(v) for k, v in data["edge_clusters"].items()}

    return Graph(**data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a graph.json to Neo4j")
    parser.add_argument("--graph-file", required=True, help="Path to graph.json")
    parser.add_argument(
        "--uri", required=True, help="Neo4j URI (bolt://... or neo4j+s://...)"
    )
    parser.add_argument("--username", required=True, help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name")
    parser.add_argument("--graph-name", help="Optional graph name label/property")
    parser.add_argument(
        "--clear-existing", action="store_true", help="Clear DB before upload"
    )

    args = parser.parse_args()

    graph_path = Path(args.graph_file)
    if not graph_path.exists():
        print(f"❌ Graph file not found: {graph_path}")
        return 1

    try:
        graph = load_graph_from_file(str(graph_path))
    except Exception as e:
        print(f"❌ Failed to load graph: {e}")
        return 1

    print(
        f"📦 Loaded graph: {len(graph.entities)} entities, {len(graph.relations)} relations"
    )
    print(f"🔗 Uploading to Neo4j @ {args.uri} (db={args.database}) ...")

    ok = upload_to_neo4j(
        graph=graph,
        uri=args.uri,
        username=args.username,
        password=args.password,
        database=args.database,
        graph_name=args.graph_name,
        clear_existing=args.clear_existing,
    )

    if not ok:
        print("Upload failed")
        return 1

    print("Upload succeeded")

    # Show quick stats
    try:
        uploader = Neo4jUploader(args.uri, args.username, args.password, args.database)
        if uploader.connect():
            stats = uploader.get_graph_stats()
            print(
                f"Stats — nodes: {stats.get('node_count')}, "
                f"relationships: {stats.get('relationship_count')}"
            )
            uploader.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
