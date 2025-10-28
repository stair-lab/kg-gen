from dotenv import load_dotenv
import dspy
from datasets import load_dataset
import numpy as np
import networkx as nx
from kg_gen.kg_gen import KGGen
import json
import sys
import os
from typing import Literal

# Add the src directory to Python path to import from source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

load_dotenv()

# Configure DSPy with OpenAI
lm = dspy.LM(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("LLM_API_KEY"),
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


# Initialize the evaluator
evaluator = ResponseEvaluator()


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    """Evaluate if the context contains the correct answer using DSPy with GPT-5-nano."""
    result = evaluator.forward(context=context, correct_answer=correct_answer)
    return int(result.evaluation)


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


def main(evaluation_model: Literal["local", "kggen", "graphrag", "openie"] = "local"):
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

    kggen = KGGen(retrieval_model="all-MiniLM-L6-v2")
    valid_pairs = [
        (kg, queries) for kg, queries in zip(kg_data, queries) if kg is not None
    ]

    for i, (kg, queries) in enumerate(valid_pairs):
        output_file = f"experiments/MINE/results/{evaluation_model}/results_{i}.json"
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            if evaluation_model == "local":
                # Generate the graph from text
                graph = kggen.generate(
                    kg,
                    model=os.getenv("LLM_MODEL"),
                    api_key=os.getenv("LLM_API_KEY"),
                    cluster=True,
                    temperature=1.0,
                )
            else:
                graph = kggen.from_dict(kg)

            nxGraph = kggen.to_nx(graph)
            node_embeddings, _ = kggen.generate_embeddings(nxGraph)
            evaluate_accuracy(
                kggen,
                queries,
                node_embeddings,
                nxGraph,
                output_file,
            )
        except Exception as e:
            print(f"Error processing file {output_file}: {str(e)}, skipping...")


if __name__ == "__main__":
    main()
