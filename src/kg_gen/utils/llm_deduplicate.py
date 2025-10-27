from typing import List
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import dspy
from kg_gen.models import Graph
import logging
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans


class LLMDeduplicate:
    graph: Graph
    nodes: list[str]
    edges: list[str]
    node_clusters: list[list[str]]
    edge_clusters: list[list[str]]
    retrieval_model: SentenceTransformer
    lm: dspy.LM

    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, retrieval_model: SentenceTransformer, lm: dspy.LM, graph: Graph):
        """
        Initialize KG-assisted RAG with cached embeddings, BM25 tokens, and text chunk store.
        """
        self.graph = graph
        self.nodes = list(graph.entities)
        self.edges = list(graph.edges)
        self.node_clusters = graph.entity_clusters or []
        self.edge_clusters = graph.edge_clusters or []
        self.retrieval_model = retrieval_model
        self.lm = lm

        # Embeddings and BM25 tokens for nodes
        self.node_embeddings = retrieval_model.encode(self.nodes, show_progress_bar=True)
        self.node_bm25_tokenized = [text.lower().split() for text in self.nodes]

        # Always rebuild BM25 from tokens (it's fast and simpler than serializing the object)
        self.node_bm25 = BM25Okapi(self.node_bm25_tokenized)

        # Embeddings and BM25 tokens for edges
        self.edge_embeddings = retrieval_model.encode(self.edges, show_progress_bar=True)
        self.edge_bm25_tokenized = [text.lower().split() for text in self.edges]

        # Always rebuild BM25 from tokens
        self.edge_bm25 = BM25Okapi(self.edge_bm25_tokenized)

        dspy.configure(lm=lm)

    def get_relevant_items(self, query: str, top_k: int = 50, type: str = "node") -> list[str]:
        """
        Use rank fusion of BM25 + embedding to retrieve top-k nodes.
        """
        query_tokens = query.lower().split()

        # BM25
        bm25_scores = self.node_bm25.get_scores(
            query_tokens) if type == "node" else self.edge_bm25.get_scores(query_tokens)

        # Embedding
        query_embedding = self.retrieval_model.encode([query], show_progress_bar=False)
        embeddings = self.node_embeddings if type == "node" else self.edge_embeddings
        embedding_scores = cosine_similarity(
            query_embedding, embeddings).flatten()

        # Rank fusion (equal weighting)
        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        items = self.nodes if type == "node" else self.edges
        top_items = [items[i] for i in top_indices]

        return top_items

    def cluster(self):
        cluster_size = 128

        embedding_sets = {
            'node': self.node_embeddings,
            'edge': self.edge_embeddings
        }

        for embedding_type, embeddings in embedding_sets.items():
            n_samples = len(embeddings)
            num_clusters = max(1, n_samples // cluster_size)

            # Step 1: Cluster centers
            kmeans = KMeans(
                n_clusters=num_clusters,
                init="random",
                n_init=1,
                max_iter=20,
                tol=0.0,
                algorithm="lloyd",
                verbose=True,
            )
            kmeans.fit(embeddings.astype(np.float32))
            centroids = kmeans.cluster_centers_

            # Step 2: Assign each point to nearest centroid (with 25 max per cluster)
            distances = cdist(embeddings, centroids)
            assignments = np.argsort(distances, axis=1)

            # Initialize cluster tracking
            clusters: List[List[int]] = [[] for _ in range(num_clusters)]
            assigned = np.zeros(n_samples, dtype=bool)

            for rank in range(num_clusters):
                for i in range(n_samples):
                    if assigned[i]:
                        continue
                    cluster_id = assignments[i, rank]
                    if len(clusters[cluster_id]) < cluster_size:
                        clusters[cluster_id].append(i)
                        assigned[i] = True

            unassigned = np.where(~assigned)[0]

            # Add unassigned items as their own cluster if any exist
            if len(unassigned) > 0:
                self.logger.debug(
                    "Adding %s unassigned items as a separate cluster", len(unassigned))
                clusters.append(unassigned.tolist())
            else:
                self.logger.debug("No unassigned items to add as a cluster")

            # Save clusters to JSON files
            cluster_type = embedding_type  # 'node' or 'edge'

            # Print debug information about clusters
            self.logger.debug("Number of %s clusters: %s",
                              cluster_type, len(clusters))
            self.logger.debug("First cluster size: %s", len(clusters[0]))
            self.logger.debug(
                "First few items in first cluster: %s", clusters[0][:5])
            self.logger.debug("Last cluster size: %s", len(clusters[-1]))
            self.logger.debug("Distribution of cluster sizes: %s...", [
                len(clust) for clust in clusters[:5]])

            # Convert clusters to JSON-serializable format - save names instead of indices
            if cluster_type == "node":
                self.logger.debug("Converting node indices to node names...")
                clusters_data = [[self.nodes[idx]
                                  for idx in cluster] for cluster in clusters]
                self.logger.debug(
                    "Sample of first cluster after conversion: %s", clusters_data[0][:3])
                # Add node clusters to self
                self.node_clusters = clusters_data
            else:  # edge
                self.logger.debug("Processing edge clusters...")
                clusters_data = [[self.edges[idx]
                                  for idx in cluster] for cluster in clusters]
                self.logger.debug(
                    "Edge clusters data is empty: %s", len(clusters_data) == 0)
                # Add edge clusters to self
                self.edge_clusters = clusters_data

    def deduplicate_cluster(self, cluster: list[str], type: str = "node") -> tuple[set, dict[str, list[str]]]:
        cluster = cluster.copy()

        items = set()
        item_clusters = {}
        plural_type = "entities" if type == "node" else "edges"
        singular_type = "entity" if type == "node" else "edge"

        self.logger.info(
            "Starting deduplication of %s %s in cluster", len(cluster), plural_type)

        processed_count = 0
        while len(cluster) > 0:
            processed_count += 1
            item = cluster.pop()

            self.logger.debug(
                "[%s/%s] Processing %s: '%s'", processed_count, len(cluster), singular_type, item)

            relevant_items = self.get_relevant_items(item, 16, type)

            self.logger.debug(
                "  Found %s relevant %s for '%s'", len(relevant_items), plural_type, item)
            if len(relevant_items) > 0:
                self.logger.debug("  Sample relevant items: %s%s", relevant_items[:3], ("..." if len(
                    relevant_items) > 3 else ""))

            class Deduplicate(dspy.Signature):
                __doc__ = f"""Find duplicate {plural_type} for the item and an alias that best represents the duplicates. Duplicates are those that are the same in meaning, such as with variation in tense, plural form, stem form, case, abbreviation, shorthand. Return an empty list if there are none. 
                """
                item: str = dspy.InputField()
                set: list[str] = dspy.InputField()
                duplicates: list[str] = dspy.OutputField(
                    description="Exact matches to items in {plural_type} set")
                alias: str = dspy.OutputField(
                    description=f"Best {singular_type} name to represent the duplicates, ideally from the {plural_type} set")

            
            # with dspy.context(lm=self.lm):
            deduplicate = dspy.Predict(Deduplicate)
            result = deduplicate(item=item, set=relevant_items)
            items.add(result.alias)

            # Filter duplicates to only include those that exist in the cluster
            duplicates = [dup for dup in result.duplicates if dup in cluster]

            if len(duplicates) > 0:
                self.logger.debug(
                    "  ✓ Found %s duplicates for '%s'", len(duplicates), item)
                self.logger.info(
                    "  → Using alias '%s' to represent: '%s' and %s", result.alias, item, duplicates)
                item_clusters[result.alias] = {item}
                for duplicate in duplicates:
                    cluster.remove(duplicate)
                    item_clusters[result.alias].add(duplicate)
            else:
                self.logger.debug(
                    "  ✗ No duplicates found for '%s', keeping as is", item)
                item_clusters[item] = {item}

        self.logger.debug(
            "Deduplication complete: %s unique %s from original %s", len(items), plural_type, processed_count)

        return items, item_clusters

    def deduplicate(self) -> Graph:
        # Check if intermediate progress exists and load it
        entities = set()
        edges = set()
        entity_clusters = {}
        edge_clusters = {}

        pool = ThreadPoolExecutor(max_workers=64)

        # Process node clusters in parallel
        node_futures = []
        cnt_nodes = 0
        for i, cluster in enumerate(self.node_clusters):
            cnt_nodes += len(cluster)
            node_futures.append(pool.submit(
                self.deduplicate_cluster, cluster, "node"))

        # Process edge clusters in parallel
        edge_futures = []
        cnt_edges = 0
        for i, cluster in enumerate(self.edge_clusters):
            cnt_edges += len(cluster)
            edge_futures.append(pool.submit(
                self.deduplicate_cluster, cluster, "edge"))

        # Collect results from node futures
        for i, future in enumerate(node_futures):
            try:
                cluster_entities, cluster_entity_map = future.result()
                entities.update(cluster_entities)
                entity_clusters.update(cluster_entity_map)
            except Exception as e:
                self.logger.error("Error processing node cluster %s: %s", i, e)

        # Collect results from edge futures
        for i, future in enumerate(edge_futures):
            try:
                cluster_edges, cluster_edge_map = future.result()
                edges.update(cluster_edges)
                edge_clusters.update(cluster_edge_map)
            except Exception as e:
                self.logger.error("Error processing edge cluster %s: %s", i, e)

        self.logger.info(
            "Finished processing all clusters with %s nodes and %s edges LLM calls", cnt_nodes, cnt_edges)

        # Update relations based on clusters
        relations: set[tuple[str, str, str]] = set()

        for s, p, o in self.graph.relations:
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
        deduped_graph = Graph(
            entities=entities,
            edges=edges,
            relations=relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
        )

        return deduped_graph
