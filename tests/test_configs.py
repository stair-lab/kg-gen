import os
from src.kg_gen import KGGen
from dotenv import load_dotenv
import pytest
from fixtures import kg

API_BASE = os.getenv("API_BASE")

# Custom API BAse
def test_custom_api_base(kg: KGGen):
    # Generate a simple graph
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."

    graph = kg.generate(
        input_data=text,
        api_base=API_BASE,
    )
    print(graph)


def test_gen_clus_agg(kg: KGGen):
    # Test texts
    text1 = "Linda is Joshua's mother. Ben is Josh's brother. Andrew is Josh's father."
    text2 = "Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt. Josh also goes by Joshua."

    # Generate individual graphs
    graph1 = kg.generate(
        input_data=text1,
        context="Family relationships",
    )

    graph2 = kg.generate(
        input_data=text2,
        context="Family relationships",
    )

    # Aggregate the graphs
    combined_graph = kg.aggregate([graph1, graph2])

    # Cluster the combined graph
    clustered_graph = kg.cluster(
        combined_graph,
        context="Family relationships",
    )

    # Print results
    print("\nGraph 1:")
    print("Entities:", graph1.entities)
    print("Relations:", graph1.relations)
    print("Edges:", graph1.edges)

    print("\nGraph 2:")
    print("Entities:", graph2.entities)
    print("Relations:", graph2.relations)
    print("Edges:", graph2.edges)

    print("\nCombined Graph:")
    print("Entities:", combined_graph.entities)
    print("Relations:", combined_graph.relations)
    print("Edges:", combined_graph.edges)

    print("\nClustered Combined Graph:")
    print("Entities:", clustered_graph.entities)
    print("Relations:", clustered_graph.relations)
    print("Edges:", clustered_graph.edges)
    print("Entity Clusters:", clustered_graph.entity_clusters)
    print("Edge Clusters:", clustered_graph.edge_clusters)


def test_multiple_models(kg: KGGen):
    # Test text input
    text = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father. Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt."
    # Test with different models and their corresponding API keys
    model_configs = [
        {"model": "openai/gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")},
        {
            "model": "anthropic/claude-3-5-sonnet-20240620",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        {"model": "gemini/gemini-pro", "api_key": os.getenv("GEMINI_API_KEY")},
    ]

    for config in model_configs:
        print(f"\nTesting with model: {config['model']}")
        try:
            graph = kg.generate(
                input_data=text, model=config["model"], api_key=config["api_key"]
            )
            print(graph)
        except Exception as e:
            print(f"Error with {config['model']}: {str(e)}")
