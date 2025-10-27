from kg_gen.models import Graph
from kg_gen.utils.deduplicate import deduplicate_graph
from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from sentence_transformers import SentenceTransformer
import dspy


def dedup_cluster_graph(retrieval_model: SentenceTransformer, lm: dspy.LM, graph: Graph) -> Graph:

    # Deduplicate the graph using semantic hashing
    deduplicated_graph = deduplicate_graph(graph)

    # Deduplicate the semantic deduplicated graph using LLM
    llm_deduplicate = LLMDeduplicate(retrieval_model, lm, deduplicated_graph)
    llm_deduplicate.cluster()
    llm_deduplicated_graph = llm_deduplicate.deduplicate()
    return llm_deduplicated_graph
