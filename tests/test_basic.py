from src.kg_gen import KGGen
from fixtures import kg


def match_subset(set1: set[str], set2: set[str]) -> bool:
    """
    Check if set1 is a subset of set2, with fuzzy matching for similar names.
    set1, must be the expected set, and should contain less words for an entity, so fuzzy matching works in case.
    """
    diff = set1 - set2

    if len(diff) == 0:
        return True

    # Check for fuzzy matches for remaining entities
    for entity1 in diff:
        found_match = False
        for entity2 in set2:
            if entity1.lower() in entity2.lower() or entity2.lower() in entity1.lower():
                found_match = True
                break
        if not found_match:
            return False

    return True


def test_basic(kg: KGGen):
    # Generate a simple graph
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."

    graph = kg.generate(input_data=text)

    expected_entities = {
        "Harry",
        "James Potter",
        "Lily Potter",
        "Ginny",
        "James Sirius",
        "Albus",
        "Lily Luna",
    }
    print(graph)
    assert match_subset(expected_entities, graph.entities)


def test_clustered(kg: KGGen):
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

    # # Aggregate the graphs
    combined_graph = kg.aggregate([graph1, graph2])

    # Cluster the combined graph
    clustered_graph = kg.cluster(
        combined_graph,
        context="Family relationships",
    )
    expected_entities = {"Linda", "Joshua", "Josh", "Ben", "Andrew", "Judy"}
    expected_edges = {
        "is mother of",
        "is brother of",
        "is father of",
        "is sister of",
        "is nephew of",
        "is aunt of",
    }
    # TODO: with gpt-5 temperature 1.0, it makes tests not deterministic, thus `is brother of` could be `is isbling of`.
    # print(clustered_graph)
    print("entities:", clustered_graph.entities)
    print("edges:", clustered_graph.edges)
    assert match_subset(expected_entities, clustered_graph.entities)
    assert match_subset(expected_edges, clustered_graph.edges)

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


def test_conversation(kg: KGGen):
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    graph = kg.generate(
        input_data=messages,
    )
    expected_entities = {"France", "Paris"}
    assert match_subset(expected_entities, graph.entities)
    print(graph)
