import os
import time
import logging
from dotenv import load_dotenv
import dspy
from sentence_transformers import SentenceTransformer
import typer

from kg_gen.utils.llm_deduplicate import LLMDeduplicate
from kg_gen.utils.deduplicate import deduplicate_graph
from kg_gen.models import Graph

app = typer.Typer()


@app.command()
def main(
    graph_path: str,
    retrieval_model: str = "sentence-transformers/all-mpnet-base-v2",
    temperature: float = 1.0,
    log_level: str = "INFO",
):
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    load_dotenv()
    graph = Graph.from_file(graph_path)

    # Create the output folder
    os.makedirs("output", exist_ok=True)

    lm = dspy.LM(
        model=os.environ.get("LLM_MODEL"),
        api_key=os.environ.get("LLM_API_KEY"),
        temperature=temperature,
    )

    transformer_model = SentenceTransformer(retrieval_model)

    # Deduplicate the graph using semantic hashing
    start_time = time.time()
    deduplicated_graph = deduplicate_graph(graph)
    end_time = time.time()
    semhash_time = end_time - start_time
    deduplicated_graph.to_file("output/deduplicated_graph.json")

    # Deduplicate the original graph using LLM
    start_time = time.time()
    llm_deduplicate = LLMDeduplicate(transformer_model, lm, graph)
    start_cluster_time = time.time()
    llm_deduplicate.cluster()
    end_cluster_time = time.time()
    cluster_time = end_cluster_time - start_cluster_time
    llm_deduplicated_graph = llm_deduplicate.deduplicate()
    end_time = time.time()
    llm_time = end_time - start_time
    llm_deduplicated_graph.to_file("output/llm_deduplicated_graph1.json")

    # Deduplicate the semantic deduplicated graph using LLM
    start_time = time.time()
    llm_deduplicate = LLMDeduplicate(transformer_model, lm, deduplicated_graph)
    start_cluster_time = time.time()
    llm_deduplicate.cluster()
    end_cluster_time = time.time()
    cluster_time2 = end_cluster_time - start_cluster_time
    llm_deduplicated_graph2 = llm_deduplicate.deduplicate()
    llm_deduplicated_graph2.to_file("output/llm_deduplicated_graph2.json")
    end_time = time.time()
    llm_time2 = end_time - start_time

    graph.stats("Original graph")
    deduplicated_graph.stats("Deduplicated graph")
    llm_deduplicated_graph.stats("LLM deduplicated original graph")
    llm_deduplicated_graph2.stats("LLM deduplicated deduplicated graph")

    logger.info("Semantic hashing time: %s seconds", semhash_time)
    logger.info(
        "LLM time: %s seconds, cluster time: %s seconds", llm_time, cluster_time
    )
    logger.info(
        "LLM time2: %s seconds, cluster time2: %s seconds", llm_time2, cluster_time2
    )


if __name__ == "__main__":
    app()
