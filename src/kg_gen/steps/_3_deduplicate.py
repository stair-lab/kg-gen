from enum import Enum
from kg_gen.models import Graph
from kg_gen.utils.deduplicate import deduplicate_graph
from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from kg_gen.deduplication import (
    EntityDeduplicator,
    RelationDeduplicator,
    DeduplicationMethod as DedupMethod,
)
from sentence_transformers import SentenceTransformer
import dspy
import logging

logger = logging.getLogger(__name__)


class DeduplicationStrategy(Enum):
    """Strategy for initial deduplication before LLM clustering."""
    DETERMINISTIC = "deterministic"  # Fast fuzzy hash-based matching
    SEMANTIC_HASH = "semantic_hash"   # Existing semantic hashing method
    BOTH = "both"                     # Run deterministic first, then semantic hash


def _apply_deterministic_deduplication_to_graph(graph: Graph) -> Graph:
    """
    Apply fast deterministic (fuzzy hash-based) deduplication to a graph.
    
    This removes obvious duplicates (case/whitespace variations) using
    hash-based matching. Note: This is "fuzzy" due to normalization, not
    fully deterministic.
    
    Args:
        graph: Graph to deduplicate
        
    Returns:
        New Graph with deduplicated entities and edges, with relations updated
    """
    # Deduplicate entities
    entity_dedup = EntityDeduplicator(method=DedupMethod.DETERMINISTIC)
    unique_entities, entity_map = entity_dedup.deduplicate(list(graph.entities))
    
    # Deduplicate edges
    edge_dedup = RelationDeduplicator(method=DedupMethod.DETERMINISTIC)
    unique_edges, edge_map = edge_dedup.deduplicate(list(graph.edges))
    
    # Update relations to use canonical entity/edge names
    updated_relations = set()
    for subject, predicate, obj in graph.relations:
        # Map to canonical forms
        canonical_subject = entity_map.get(subject, subject)
        canonical_predicate = edge_map.get(predicate, predicate)
        canonical_obj = entity_map.get(obj, obj)
        
        # Only add if all components are in the deduplicated sets
        if (canonical_subject in unique_entities and 
            canonical_predicate in unique_edges and 
            canonical_obj in unique_entities):
            updated_relations.add((canonical_subject, canonical_predicate, canonical_obj))
    
    # Create new graph with deduplicated components
    return Graph(
        entities=set(unique_entities),
        edges=set(unique_edges),
        relations=updated_relations,
        entity_clusters=graph.entity_clusters,
        edge_clusters=graph.edge_clusters,
    )


def dedup_cluster_graph(
    retrieval_model: SentenceTransformer,
    lm: dspy.LM,
    graph: Graph,
    dedup_strategy: DeduplicationStrategy = DeduplicationStrategy.SEMANTIC_HASH,
) -> Graph:
    """
    Deduplicate and cluster a knowledge graph.
    
    This function performs a two-step process:
    1. Initial deduplication (configurable strategy)
    2. LLM-based semantic clustering and deduplication
    
    Args:
        retrieval_model: SentenceTransformer for embeddings
        lm: Language model for LLM verification
        graph: Input graph to deduplicate and cluster
        dedup_strategy: Strategy for initial deduplication step
        
    Returns:
        Deduplicated and clustered graph
    """
    # Handle empty graph early - return immediately
    if len(graph.entities) == 0 and len(graph.edges) == 0:
        logger.info("Empty graph detected, returning empty graph with empty clusters")
        return Graph(
            entities=set(),
            edges=set(),
            relations=set(),
            entity_clusters={},
            edge_clusters={},
        )
    
    # Step 1: Apply initial deduplication based on strategy
    if dedup_strategy == DeduplicationStrategy.DETERMINISTIC:
        logger.info("Using deterministic (fuzzy hash) deduplication")
        deduplicated_graph = _apply_deterministic_deduplication_to_graph(graph)
    elif dedup_strategy == DeduplicationStrategy.SEMANTIC_HASH:
        logger.info("Using semantic hashing deduplication")
        deduplicated_graph = deduplicate_graph(graph)
    elif dedup_strategy == DeduplicationStrategy.BOTH:
        logger.info("Using both deterministic and semantic hashing deduplication")
        # Run deterministic first, then semantic hash
        deduplicated_graph = _apply_deterministic_deduplication_to_graph(graph)
        deduplicated_graph = deduplicate_graph(deduplicated_graph)
    else:
        raise ValueError(f"Unknown deduplication strategy: {dedup_strategy}")

    # Step 2: LLM-based semantic clustering and deduplication
    logger.info("Applying LLM-based clustering and deduplication")
    llm_deduplicate = LLMDeduplicate(retrieval_model, lm, deduplicated_graph)
    llm_deduplicate.cluster()
    llm_deduplicated_graph = llm_deduplicate.deduplicate()
    return llm_deduplicated_graph
