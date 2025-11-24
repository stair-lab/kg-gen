from dotenv import load_dotenv
import dspy
from datasets import load_dataset
from kg_gen.steps._3_deduplicate import DeduplicateMethod
import numpy as np
import networkx as nx
from kg_gen.kg_gen import KGGen
import json
import sys
import os
from typing import Literal
import typer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the src directory to Python path to import from source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

load_dotenv()

# Configure DSPy with OpenAI
lm = dspy.LM(
    model="openai/gpt-5",
    api_key=os.getenv("OPENAI_API_KEY"),
    reasoning_effort="high",
    temperature=1.0,
    max_tokens=16000,
)
dspy.configure(lm=lm)


# Define DSPy signature for evaluation
class EvaluateResponse(dspy.Signature):
    """Determine whether the context contains the information stated in the correct answer. Respond with 1 if yes, 0 if no."""

    context: str = dspy.InputField(desc="The context to evaluate")
    correct_answer: str = dspy.InputField(desc="The correct answer to check for")
    evaluation: int = dspy.OutputField(
        desc="1 if context contains the correct answer, 0 otherwise"
    )


# Create DSPy module for evaluation
class ResponseEvaluator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateResponse)

    def forward(self, context, correct_answer):
        return self.evaluate(context=context, correct_answer=correct_answer)


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    evaluator = ResponseEvaluator()
    result = evaluator.forward(context=context, correct_answer=correct_answer)
    return result.evaluation


def evaluate_accuracy(
    kggen: KGGen,
    queries: list[dict],
    node_embeddings: dict[str, np.ndarray],
    graph: nx.DiGraph,
    output_file: str,
):
    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
    )
    correct = 0
    results = []

    for query in queries:
        *_, context_text = kggen.retrieve(query, node_embeddings, graph)
        evaluation = gpt_evaluate_response(query, context_text)
        result = {
            "correct_answer": query,
            "retrieved_context": context_text,
            "evaluation": evaluation,
        }
        results.append(result)
        correct += evaluation

    accuracy = correct / len(queries)
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    # Save results to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def process_single_evaluation(
    i: int,
    data: dict | str,
    queries: list[dict],
    kggen: KGGen,
    evaluation_model: str,
    model_name: str,
    reasoning_effort: str | None,
    temperature: float,
    deduplication_method: Literal["semhash", "full"] | None = "full",
) -> tuple[int, bool, str]:
    """Process a single evaluation task. Returns (index, success, message)."""
    # Build directory name based on evaluation model
    if evaluation_model == "local":
        # Build directory name from model config
        dir_name = model_name.replace("/", "-")
        if reasoning_effort:
            dir_name += f"-{reasoning_effort}"
        dir_name += f"-{temperature}"
        if deduplication_method:
            dir_name += f"-{deduplication_method}"
    else:
        # For pre-generated KGs from HuggingFace dataset
        dir_name = f"hf-{evaluation_model}"

    output_file = f"experiments/MINE/results/{dir_name}/results_{i}.json"
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if not deduplication_method:
            method = None
        else:
            method = (
                DeduplicateMethod.SEMHASH
                if deduplication_method == "semhash"
                else DeduplicateMethod.FULL
            )

        if evaluation_model == "local":
            # Generate the graph from text
            graph = kggen.generate(data, deduplication_method=method)
            kg_output_file = f"experiments/MINE/results/{dir_name}/kg_{i}.json"
            KGGen.export_graph(graph, kg_output_file)
        else:
            graph = kggen.from_dict(data)

        nxGraph = kggen.to_nx(graph)
        node_embeddings, _ = kggen.generate_embeddings(nxGraph)
        evaluate_accuracy(
            kggen,
            queries,
            node_embeddings,
            nxGraph,
            output_file,
        )
        return (i, True, f"Successfully processed {output_file}")
    except Exception as e:
        return (i, False, f"Error processing {output_file}: {str(e)}")


def main(
    model: str = "openai/gpt-5-nano",
    api_key_env: str = "OPENAI_API_KEY",
    api_base_url: str | None = None,
    # local: means re-run the KG generation step
    # kggen: means use the KG generated and saved in the huggingface dataset, same for graphrag and openie
    evaluation_model: Literal["local", "kggen", "graphrag", "openie"] = "local",
    reasoning_effort: str = None,
    temperature: float = 1.0,
    deduplication_method: Literal["semhash", "full"] | None = "semhash",
    max_workers: int = 64,
):
    # Load data from Hugging Face (with local fallback)
    dataset = load_dataset("josancamon/kg-gen-MINE-evaluation-dataset")["train"]
    queries = [item["generated_queries"] for item in dataset.to_list()]

    if evaluation_model == "local":
        kg_data = [item["essay_content"] for item in dataset.to_list()]
    elif evaluation_model == "kggen":
        kg_data = [item["kggen"] for item in dataset.to_list()]
    elif evaluation_model == "graphrag":
        kg_data = [item["graphrag_kg"] for item in dataset.to_list()]
    elif evaluation_model == "openie":
        kg_data = [item["openie_kg"] for item in dataset.to_list()]

    kggen = KGGen(
        retrieval_model="all-MiniLM-L6-v2",
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        model=model,
        api_key=os.getenv(api_key_env),
        api_base=api_base_url,
        max_tokens=64000,
    )
    valid_pairs = [
        (kg, queries) for kg, queries in zip(kg_data, queries) if kg is not None
    ]

    print(f"Processing {len(valid_pairs)} evaluations with {max_workers} workers...")

    # Process evaluations in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_evaluation,
                i,
                kg,
                queries,
                kggen,
                evaluation_model,
                model,
                reasoning_effort,
                temperature,
                deduplication_method,
            ): i
            for i, (kg, queries) in enumerate(valid_pairs)
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            completed += 1
            i, success, message = future.result()
            status = "✓" if success else "✗"
            print(f"[{completed}/{len(valid_pairs)}] {status} {message}")

    print(f"\nCompleted all {len(valid_pairs)} evaluations!")


if __name__ == "__main__":
    typer.run(main)
    # uv run experiments/MINE/_1_evaluation.py --model together_ai/openai/gpt-oss-20b --reasoning-effort low --deduplication-method semhash --max-workers 110 --api-base-url https://api.together.xyz/v1 --api-key-env TOGETHER_API_KEY
