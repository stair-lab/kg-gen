import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
import os
from typing import Tuple
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
import logging
from pathlib import Path
from src.kg_gen import Graph
import dspy
from concurrent.futures import ThreadPoolExecutor
import faiss
import numpy as np
from scipy.spatial.distance import cdist
import time


# Set up logging safely
logger = logging.getLogger("kg_rag")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


class KGAssistedRAG:
    def __init__(self, kg_path: str, output_folder: str):
        """
        Initialize KG-assisted RAG with cached embeddings, BM25 tokens, and text chunk store.
        """
        self.output_folder = output_folder
        load_dotenv()

        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        self.kg = kg_data
        self.nodes: list[str] = kg_data.get("entities")
        self.edges: list[str] = kg_data.get("edges")
        self.node_clusters: list[list[str]] = []
        self.edge_clusters: list[list[str]] = []
        print(f"Loaded {len(self.nodes)} nodes from knowledge graph")
        print(f"Sample node: {self.nodes[0] if self.nodes else 'No nodes found'}")
        print(f"Loaded {len(self.edges)} edges from knowledge graph")
        print(f"Sample edge: {self.edges[0] if self.edges else 'No edges found'}")

        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Cache embeddings and BM25 tokens for nodes
        self.node_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Define cache file paths for nodes
        node_embeddings_cache_path = os.path.join(
            self.output_folder, "node_embeddings.npy"
        )
        node_bm25_tokens_cache_path = os.path.join(
            self.output_folder, "node_bm25_tokens.json"
        )

        # Check if cached node embeddings exist
        if os.path.exists(node_embeddings_cache_path):
            print(f"Loading node embeddings from cache: {node_embeddings_cache_path}")
            self.node_embeddings = np.load(node_embeddings_cache_path)
        else:
            print("Generating node embeddings...")
            self.node_embeddings = self.node_encoder.encode(
                self.nodes, show_progress_bar=True
            )
            # Save embeddings to cache
            np.save(node_embeddings_cache_path, self.node_embeddings)
            print(f"Saved node embeddings to cache: {node_embeddings_cache_path}")

        # Check if cached BM25 tokens for nodes exist
        if os.path.exists(node_bm25_tokens_cache_path):
            print(f"Loading node BM25 tokens from cache: {node_bm25_tokens_cache_path}")
            with open(node_bm25_tokens_cache_path, "r") as f:
                self.node_bm25_tokenized = json.load(f)
        else:
            print("Generating node BM25 tokens...")
            self.node_bm25_tokenized = [text.lower().split() for text in self.nodes]
            # Save tokens to cache
            with open(node_bm25_tokens_cache_path, "w") as f:
                json.dump(self.node_bm25_tokenized, f)
            print(f"Saved node BM25 tokens to cache: {node_bm25_tokens_cache_path}")

        # Always rebuild BM25 from tokens (it's fast and simpler than serializing the object)
        self.node_bm25 = BM25Okapi(self.node_bm25_tokenized)

        # Cache embeddings and BM25 tokens for edges
        self.edge_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Define cache file paths for edges
        edge_embeddings_cache_path = os.path.join(
            self.output_folder, "edge_embeddings.npy"
        )
        edge_bm25_tokens_cache_path = os.path.join(
            self.output_folder, "edge_bm25_tokens.json"
        )

        # Check if cached edge embeddings exist
        if os.path.exists(edge_embeddings_cache_path):
            print(f"Loading edge embeddings from cache: {edge_embeddings_cache_path}")
            self.edge_embeddings = np.load(edge_embeddings_cache_path)
        else:
            print("Generating edge embeddings...")
            self.edge_embeddings = self.edge_encoder.encode(
                self.edges, show_progress_bar=True
            )
            # Save embeddings to cache
            np.save(edge_embeddings_cache_path, self.edge_embeddings)
            print(f"Saved edge embeddings to cache: {edge_embeddings_cache_path}")

        # Check if cached BM25 tokens for edges exist
        if os.path.exists(edge_bm25_tokens_cache_path):
            print(f"Loading edge BM25 tokens from cache: {edge_bm25_tokens_cache_path}")
            with open(edge_bm25_tokens_cache_path, "r") as f:
                self.edge_bm25_tokenized = json.load(f)
        else:
            print("Generating edge BM25 tokens...")
            self.edge_bm25_tokenized = [text.lower().split() for text in self.edges]
            # Save tokens to cache
            with open(edge_bm25_tokens_cache_path, "w") as f:
                json.dump(self.edge_bm25_tokenized, f)
            print(f"Saved edge BM25 tokens to cache: {edge_bm25_tokens_cache_path}")

        # Always rebuild BM25 from tokens
        self.edge_bm25 = BM25Okapi(self.edge_bm25_tokenized)

    def get_relevant_items(
        self, query: str, top_k: int = 50, type: str = "node"
    ) -> list[str]:
        """
        Use rank fusion of BM25 + embedding to retrieve top-k nodes.
        """
        query_tokens = query.lower().split()

        # BM25
        bm25_scores = (
            self.node_bm25.get_scores(query_tokens)
            if type == "node"
            else self.edge_bm25.get_scores(query_tokens)
        )

        # Embedding
        encoder = self.node_encoder if type == "node" else self.edge_encoder
        query_embedding = encoder.encode([query], show_progress_bar=False)
        embeddings = self.node_embeddings if type == "node" else self.edge_embeddings
        embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()

        # Rank fusion (equal weighting)
        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        items = self.nodes if type == "node" else self.edges
        top_items = [items[i] for i in top_indices]

        return top_items

    def cluster(self):
        cluster_size = 128

        embedding_sets = {"node": self.node_embeddings, "edge": self.edge_embeddings}

        for embedding_type, embeddings in embedding_sets.items():
            # Check if clusters already exist
            if embedding_type == "node":
                node_clusters_path = os.path.join(
                    self.output_folder, "node_clusters.json"
                )
                if os.path.exists(node_clusters_path):
                    print(
                        f"Node clusters already exist at {node_clusters_path}, loading existing clusters"
                    )
                    with open(node_clusters_path, "r") as f:
                        self.node_clusters = json.load(f)
                    continue
            elif embedding_type == "edge":
                edge_clusters_path = os.path.join(
                    self.output_folder, "edge_clusters.json"
                )
                if os.path.exists(edge_clusters_path):
                    print(
                        f"Edge clusters already exist at {edge_clusters_path}, loading existing clusters"
                    )
                    with open(edge_clusters_path, "r") as f:
                        self.edge_clusters = json.load(f)
                    continue
            # Step 1: Cluster centers with FAISS
            d = embeddings.shape[1]
            num_clusters = len(embeddings) // cluster_size

            kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=True, gpu=True)
            kmeans.train(embeddings.astype(np.float32))
            centroids = kmeans.centroids

            # Step 2: Assign each point to nearest centroid (with 25 max per cluster)
            distances = cdist(embeddings, centroids)  # (300000, num_clusters)
            assignments = np.argsort(distances, axis=1)

            # Initialize cluster tracking
            clusters = [[] for _ in range(num_clusters)]
            assigned = np.full(len(embeddings), False)

            for rank in range(num_clusters):
                for i in range(len(embeddings)):
                    if assigned[i]:
                        continue
                    cluster_id = assignments[i, rank]
                    if len(clusters[cluster_id]) < cluster_size:
                        clusters[cluster_id].append(i)
                        assigned[i] = True

            unassigned = np.where(~assigned)[0]

            # Add unassigned items as their own cluster if any exist
            if len(unassigned) > 0:
                print(
                    f"Adding {len(unassigned)} unassigned items as a separate cluster"
                )
                clusters.append(unassigned.tolist())
            else:
                print("No unassigned items to add as a cluster")

            # Save clusters to JSON files
            cluster_type = embedding_type  # 'node' or 'edge'
            clusters_path = os.path.join(
                self.output_folder, f"{cluster_type}_clusters.json"
            )

            # Print debug information about clusters
            print(f"Number of {cluster_type} clusters: {len(clusters)}")
            print(f"First cluster size: {len(clusters[0])}")
            print(f"First few items in first cluster: {clusters[0][:5]}")
            print(f"Last cluster size: {len(clusters[-1])}")
            print(
                f"Distribution of cluster sizes: {[len(clust) for clust in clusters[:5]]}..."
            )

            # Convert clusters to JSON-serializable format - save names instead of indices
            if cluster_type == "node":
                print("Converting node indices to node names...")
                clusters_data = [
                    [self.nodes[idx] for idx in cluster] for cluster in clusters
                ]
                print(
                    f"Sample of first cluster after conversion: {clusters_data[0][:3]}"
                )
                # Add node clusters to self
                self.node_clusters = clusters_data
            else:  # edge
                print("Processing edge clusters...")
                clusters_data = [
                    [self.edges[idx] for idx in cluster] for cluster in clusters
                ]
                print(f"Edge clusters data is empty: {len(clusters_data) == 0}")
                # Add edge clusters to self
                self.edge_clusters = clusters_data

            # Save to file
            os.makedirs(self.output_folder, exist_ok=True)
            with open(clusters_path, "w") as f:
                json.dump(clusters_data, f, indent=4)

            print(f"{cluster_type.capitalize()} clusters saved to {clusters_path}")

    def deduplicate_cluster(
        self, cluster: list[str], type: str = "node"
    ) -> tuple[set, dict[str, list[str]]]:
        cluster = cluster.copy()

        items = set()
        item_clusters = {}
        plural_type = "entities" if type == "node" else "edges"
        singular_type = "entity" if type == "node" else "edge"

        print(f"Starting deduplication of {len(cluster)} {plural_type} in cluster")

        processed_count = 0
        while len(cluster) > 0:
            processed_count += 1
            item = cluster.pop()

            print(
                f"[{processed_count}/{processed_count + len(cluster)}] Processing {singular_type}: '{item}'"
            )

            relevant_items = self.get_relevant_items(item, 16, type)

            print(f"  Found {len(relevant_items)} relevant {plural_type} for '{item}'")
            if len(relevant_items) > 0:
                print(
                    f"  Sample relevant items: {relevant_items[:3]}"
                    + ("..." if len(relevant_items) > 3 else "")
                )

            class Deduplicate(dspy.Signature):
                __doc__ = f"""Find duplicate {plural_type} for the item and an alias that best represents the duplicates. Duplicates are those that are the same in meaning, such as with variation in tense, plural form, stem form, case, abbreviation, shorthand. Return an empty list if there are none. 
                """
                item: str = dspy.InputField()
                set: list[str] = dspy.InputField()
                duplicates: list[str] = dspy.OutputField(
                    description="Exact matches to items in {plural_type} set"
                )
                alias: str = dspy.OutputField(
                    description=f"Best {singular_type} name to represent the duplicates, ideally from the {plural_type} set"
                )

            deduplicate = dspy.Predict(Deduplicate)
            result = deduplicate(item=item, set=relevant_items)
            items.add(result.alias)

            # Filter duplicates to only include those that exist in the cluster
            duplicates = [dup for dup in result.duplicates if dup in cluster]

            if len(duplicates) > 0:
                print(f"  ✓ Found {len(duplicates)} duplicates for '{item}'")
                print(
                    f"  → Using alias '{result.alias}' to represent: '{item}' and {duplicates}"
                )
                item_clusters[result.alias] = {item}
                for duplicate in duplicates:
                    cluster.remove(duplicate)
                    item_clusters[result.alias].add(duplicate)
            else:
                print(f"  ✗ No duplicates found for '{item}', keeping as is")
                item_clusters[item] = {item}

        print(
            f"Deduplication complete: {len(items)} unique {plural_type} from original {processed_count}"
        )

        return items, item_clusters

    def deduplicate(self) -> Graph:
        lm = dspy.LM(
            model="gemini/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY")
        )
        dspy.configure(lm=lm)

        # Check if intermediate progress exists and load it
        progress_path = os.path.join(self.output_folder, "dedup_progress.json")
        entities = set()
        edges = set()
        entity_clusters = {}
        edge_clusters = {}

        if os.path.exists(progress_path):
            try:
                print(f"Found existing progress file at {progress_path}. Loading...")
                with open(progress_path, "r") as f:
                    progress = json.load(f)

                entities = set(progress.get("entities", []))
                edges = set(progress.get("edges", []))

                # Convert lists back to sets in the dictionaries
                entity_clusters = {
                    k: set(v) for k, v in progress.get("entity_clusters", {}).items()
                }
                edge_clusters = {
                    k: set(v) for k, v in progress.get("edge_clusters", {}).items()
                }

                print(f"Loaded {len(entities)} entities, {len(edges)} edges")
                print(
                    f"Loaded {len(entity_clusters)} entity clusters and {len(edge_clusters)} edge clusters"
                )
            except Exception as e:
                print(f"Error loading progress file: {e}")
                print("Starting deduplication from scratch")
                entities = set()
                edges = set()
                entity_clusters = {}
                edge_clusters = {}

        pool = ThreadPoolExecutor(max_workers=64)

        # Process node clusters in parallel
        node_futures = []
        for i, cluster in enumerate(self.node_clusters):
            node_futures.append(pool.submit(self.deduplicate_cluster, cluster, "node"))

            # Save progress every 10 clusters
            if (i + 1) % 10 == 0:
                print(
                    f"Submitted {i + 1}/{len(self.node_clusters)} node clusters for processing"
                )

        # Process edge clusters in parallel
        edge_futures = []
        for i, cluster in enumerate(self.edge_clusters):
            edge_futures.append(pool.submit(self.deduplicate_cluster, cluster, "edge"))

            # Save progress every 10 clusters
            if (i + 1) % 10 == 0:
                print(
                    f"Submitted {i + 1}/{len(self.edge_clusters)} edge clusters for processing"
                )

        # Collect results from node futures
        for i, future in enumerate(node_futures):
            try:
                cluster_entities, cluster_entity_map = future.result()
                entities.update(cluster_entities)
                entity_clusters.update(cluster_entity_map)

                # Save progress every 10 processed results
                print(f"Processed {i + 1}/{len(node_futures)} node cluster results")
                self._save_intermediate_progress(
                    entities, edges, entity_clusters, edge_clusters
                )

            except Exception as e:
                print(f"Error processing node cluster {i}: {e}")

        # Collect results from edge futures
        for i, future in enumerate(edge_futures):
            try:
                cluster_edges, cluster_edge_map = future.result()
                edges.update(cluster_edges)
                edge_clusters.update(cluster_edge_map)

                # Save progress every 10 processed results
                # if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(edge_futures)} edge cluster results")
                self._save_intermediate_progress(
                    entities, edges, entity_clusters, edge_clusters
                )

            except Exception as e:
                print(f"Error processing edge cluster {i}: {e}")

        # Final save of all progress
        self._save_intermediate_progress(
            entities, edges, entity_clusters, edge_clusters
        )
        print("Finished processing all clusters")

        # Update relations based on clusters
        relations: set[tuple[str, str, str]] = set()

        for s, p, o in self.kg.get("relations"):
            # Look up subject in entity clusters
            if s not in entities:
                for rep, cluster in entity_clusters.items():
                    if s in cluster:
                        s = rep
                        break

            # Look up predicate in edge clusters
            if p not in edges:
                for rep, cluster in edge_clusters.items():
                    if p in cluster:
                        p = rep
                        break

            # Look up object in entity clusters
            if o not in entities:
                for rep, cluster in entity_clusters.items():
                    if o in cluster:
                        o = rep
                        break

            relations.add((s, p, o))

        # Create new Graph instance with deduplicated data
        deduped_kg = Graph(
            entities=entities,
            edges=edges,
            relations=relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
            entities_chunk_ids=self.kg.get("entities_chunk_ids"),
            relations_chunk_ids=self.kg.get("relations_chunk_ids"),
            edges_chunk_ids=self.kg.get("edges_chunk_ids"),
        )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Save deduplicated knowledge graph
        output_path = os.path.join(self.output_folder, "kg.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entities": list(deduped_kg.entities),
                    "edges": list(deduped_kg.edges),
                    "relations": [list(r) for r in deduped_kg.relations],
                    "entity_clusters": {
                        k: list(v) for k, v in deduped_kg.entity_clusters.items()
                    }
                    if deduped_kg.entity_clusters
                    else None,
                    "edge_clusters": {
                        k: list(v) for k, v in deduped_kg.edge_clusters.items()
                    }
                    if deduped_kg.edge_clusters
                    else None,
                    "entities_chunk_ids": deduped_kg.entities_chunk_ids,
                    "relations_chunk_ids": deduped_kg.relations_chunk_ids,
                    "edges_chunk_ids": deduped_kg.edges_chunk_ids,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved deduplicated knowledge graph to {output_path}")

        return deduped_kg

    def _save_intermediate_progress(
        self, entities, edges, entity_clusters, edge_clusters
    ):
        """Save intermediate progress during deduplication"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_folder, exist_ok=True)

            # Save intermediate progress
            progress = {
                "entities": list(entities),
                "edges": list(edges),
                "entity_clusters": {k: list(v) for k, v in entity_clusters.items()},
                "edge_clusters": {k: list(v) for k, v in edge_clusters.items()},
            }

            progress_path = os.path.join(self.output_folder, "dedup_progress.json")
            with open(progress_path, "w") as f:
                json.dump(progress, f)

            print(f"Saved intermediate progress to {progress_path}")
        except Exception as e:
            print(f"Error saving intermediate progress: {e}")


if __name__ == "__main__":
    # Set paths
    kg_paths = [
        # "tests/data/wiki_qa/aggregated/articles_1_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_40k_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_400k_ch_kg.json",
        "tests/data/wiki_qa/aggregated/articles_4m_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_20m_ch_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_all_kg.json",
        # "tests/data/wiki_qa/aggregated/articles_w_context_kg.json",
    ]

    for kg_path in kg_paths:
        print("Path:", kg_path)
        start_time = time.time()
        output_folder = f"{kg_path}_deduplicated".replace(".json", "")
        # Initialize KGAssistedRAG with proper paths
        rag = KGAssistedRAG(kg_path=kg_path, output_folder=output_folder)

        # Test query
        test_query = "Winter Olympics"

        # Test get_relevant_nodes and edges
        nodes = rag.get_relevant_items(test_query, 50, "node")
        edges = rag.get_relevant_items(test_query, 50, "edge")
        print(f"\nQuery: {test_query}")
        print(f"\nRelevant Nodes:")
        for n in nodes:
            print(n)
        print(f"\nRelevant Edges:")
        for e in edges:
            print(e)

        rag.cluster()

        rag.deduplicate()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"\nProcessing time for {kg_path}: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)"
        )
        # Log processing time to a separate log file
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        kg_name = Path(kg_path).stem
        log_file = log_dir / f"{kg_name}_processing_time.log"

        with open(log_file, "a") as f:
            f.write(f"{kg_name},{elapsed_time:.2f},{elapsed_time / 60:.2f}\n")

        print(f"Processing time logged to {log_file}")
