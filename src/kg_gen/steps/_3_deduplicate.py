from kg_gen.models import Graph
from kg_gen.utils.deduplicate import run_semhash_deduplication
from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from sentence_transformers import SentenceTransformer
import dspy
import enum


class DeduplicateMethod(enum.Enum):
    SEMHASH = "semhash"  # Deduplicate using deterministic rules and semantic hashing
    LM_BASED = (
        "lm_based"  # Deduplicate using KNN clustering + Intra cluster LM deduplication
    )
    FULL = "full"  # Deduplicate using both semantic hashing and KNN clustering + Intra cluster LM deduplication


def run_deduplication(
    lm: dspy.LM,
    graph: Graph,
    method: DeduplicateMethod = DeduplicateMethod.FULL,
    retrieval_model: SentenceTransformer | None = None,
    semhash_similarity_threshold: float = 0.95,
) -> Graph:
    if method != DeduplicateMethod.SEMHASH and retrieval_model is None:
        raise ValueError("No retrieval model provided")

    if method == DeduplicateMethod.SEMHASH:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
    elif method == DeduplicateMethod.LM_BASED:
        llm_deduplicate = LLMDeduplicate(retrieval_model, lm, graph)
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()
    elif method == DeduplicateMethod.FULL:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
        llm_deduplicate = LLMDeduplicate(retrieval_model, lm, deduplicated_graph)
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()

    return deduplicated_graph
