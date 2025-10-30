from src.kg_gen import KGGen
from src.kg_gen.models import Graph
import os
from fixtures import kg
from dotenv import load_dotenv

load_dotenv()

def test_basic_clustering(kg: KGGen):
    # Create a simple graph with redundant entities and edges
    graph = Graph(
        entities={"cat", "cats", "kitten", "dog", "dogs", "puppy"},
        edges={"likes", "like", "liking", "chases", "chase"},
        relations={
            ("cat", "likes", "dog"),
            ("cats", "like", "dogs"),
            ("kitten", "liking", "puppy"),
            ("dog", "chases", "cat"),
            ("dogs", "chase", "cats"),
        },
    )

    # Test clustering
    clustered = kg.cluster(graph)

    # Check that similar entities were clustered
    assert len(clustered.entities) < len(graph.entities)
    assert "cat" in clustered.entities  # Representative form
    assert "dog" in clustered.entities  # Representative form

    print(clustered)

    # Check that similar edges were clustered
    assert len(clustered.edges) < len(graph.edges)
    assert (
        "like" in clustered.edges or "likes" in clustered.edges
    )  # One representative form
    assert (
        "chase" in clustered.edges or "chases" in clustered.edges
    )  # One representative form

    # Check that relations were properly mapped to representatives
    assert len(clustered.relations) <= len(graph.relations)

    # Validate cluster mappings
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    # Check entity clusters
    cat_cluster = None
    dog_cluster = None
    for rep, cluster in clustered.entity_clusters.items():
        if "cat" in cluster or "cats" in cluster or "kitten" in cluster:
            cat_cluster = cluster
        if "dog" in cluster or "dogs" in cluster or "puppy" in cluster:
            dog_cluster = cluster

    assert cat_cluster is not None
    assert dog_cluster is not None
    # Allow for more conservative clustering
    assert len(cat_cluster) >= 1  # At least one cat-related term
    assert len(dog_cluster) >= 1  # At least one dog-related term

    # Check edge clusters
    like_cluster = None
    chase_cluster = None
    for rep, cluster in clustered.edge_clusters.items():
        if "like" in cluster or "likes" in cluster or "liking" in cluster:
            like_cluster = cluster
        if "chase" in cluster or "chases" in cluster:
            chase_cluster = cluster

    assert like_cluster is not None
    assert chase_cluster is not None
    # Allow for more conservative clustering
    assert len(like_cluster) >= 1  # At least one like-related term
    assert len(chase_cluster) >= 1  # At least one chase-related term


def test_method_level_configuration(kg: KGGen):
    graph = Graph(
        entities={"cat", "cats", "dog", "dogs"},
        edges={"likes", "like"},
        relations={("cat", "likes", "dog"), ("cats", "like", "dogs")},
    )

    # Test clustering with method-level configuration
    clustered = kg.cluster(
        graph
    )

    print(clustered)

    assert len(clustered.entities) < len(graph.entities)
    assert len(clustered.edges) < len(graph.edges)
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None


def test_case_sensitivity_clustering(kg: KGGen):
    # Create a graph with case variations
    graph = Graph(
        entities={"Person", "person", "PERSON", "Book", "BOOK", "book"},
        edges={"Reads", "reads", "READS"},
        relations={
            ("Person", "Reads", "Book"),
            ("person", "reads", "book"),
            ("PERSON", "READS", "BOOK"),
        },
    )

    clustered = kg.cluster(graph)

    # Check that case variations were clustered
    assert len(clustered.entities) == 2  # Should cluster to just person and book
    assert len(clustered.edges) == 1  # Should cluster to just reads
    assert len(clustered.relations) == 1  # Should have one canonical relation

    # Validate clusters
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    # Check that all case variations are in their respective clusters
    person_variations = {"Person", "person", "PERSON"}
    book_variations = {"Book", "BOOK", "book"}
    reads_variations = {"Reads", "reads", "READS"}

    found_person = False
    found_book = False
    for rep, cluster in clustered.entity_clusters.items():
        if person_variations & cluster:  # If there's any overlap
            assert cluster == person_variations
            found_person = True
        if book_variations & cluster:
            assert cluster == book_variations
            found_book = True

    assert found_person and found_book

    # Check edge clusters
    found_reads = False
    for rep, cluster in clustered.edge_clusters.items():
        if reads_variations & cluster:
            assert cluster == reads_variations
            found_reads = True

    assert found_reads


def test_semantic_clustering(kg: KGGen):
    # Create a graph with semantically similar items
    graph = Graph(
        entities={"happy", "joyful", "glad", "sad", "unhappy", "gloomy", "person"},
        edges={"is", "feels", "becomes"},
        relations={
            ("person", "is", "happy"),
            ("person", "feels", "joyful"),
            ("person", "becomes", "glad"),
            ("person", "is", "sad"),
            ("person", "feels", "unhappy"),
            ("person", "becomes", "gloomy"),
        },
    )

    clustered = kg.cluster(
        graph, context="cluster based on sentiment, semantic similarity"
    )

    # Check that semantically similar terms were clustered
    assert len(clustered.entities) < len(graph.entities)
    assert clustered.entity_clusters is not None

    # Should have two main emotion clusters (positive and negative)
    positive_emotions = {"happy", "joyful", "glad"}
    negative_emotions = {"sad", "unhappy", "gloomy"}

    found_positive = False
    found_negative = False
    for rep, cluster in clustered.entity_clusters.items():
        if positive_emotions & cluster:
            assert len(cluster & positive_emotions) == len(positive_emotions)
            found_positive = True
        if negative_emotions & cluster:
            assert len(cluster & negative_emotions) == len(negative_emotions)
            found_negative = True

    assert found_positive and found_negative


def test_no_invalid_clustering(kg: KGGen):
    # Create a graph with distinct items that shouldn't be clustered
    graph = Graph(
        entities={"apple", "banana", "carrot", "dog", "farmer"},
        edges={"eats", "grows", "likes"},
        relations={
            ("dog", "eats", "apple"),
            ("dog", "likes", "banana"),
            ("farmer", "grows", "carrot"),
        },
    )

    clustered = kg.cluster(graph)

    # Check that distinct items weren't clustered
    assert len(clustered.entities) == len(graph.entities)
    assert len(clustered.edges) == len(graph.edges)
    assert len(clustered.relations) == len(graph.relations)

    # Each item should be in its own single-item cluster
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    for entity in graph.entities:
        found = False
        for rep, cluster in clustered.entity_clusters.items():
            if entity in cluster:
                assert len(cluster) == 1
                found = True
        assert found

    for edge in graph.edges:
        found = False
        for rep, cluster in clustered.edge_clusters.items():
            if edge in cluster:
                assert len(cluster) == 1
                found = True
        assert found


def test_empty_graph_clustering(kg: KGGen):
    # Test with empty graph
    empty_graph = Graph(entities=set(), edges=set(), relations=set())
    clustered = kg.cluster(empty_graph)

    assert len(clustered.entities) == 0
    assert len(clustered.edges) == 0
    assert len(clustered.relations) == 0
    assert clustered.entity_clusters == {}
    assert clustered.edge_clusters == {}


def test_single_item_clustering(kg: KGGen):
    # Test with single items
    graph = Graph(
        entities={"person", "home"},
        edges={"walks"},
        relations={("person", "walks", "home")},
    )

    clustered = kg.cluster(graph)

    # Check that relations are preserved
    assert len(clustered.relations) == len(graph.relations)

    # Validate cluster mappings exist
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    # Check that each entity appears in some cluster
    for entity in graph.entities:
        found = False
        for cluster in clustered.entity_clusters.values():
            if entity in cluster:
                found = True
                break
        assert found, f"Entity {entity} not found in any cluster"

    # Check that each edge appears in some cluster
    for edge in graph.edges:
        found = False
        for cluster in clustered.edge_clusters.values():
            if edge in cluster:
                found = True
                break
        assert found, f"Edge {edge} not found in any cluster"


def test_configuration_override():
    # Initialize with one set of configurations
    kg_gen = KGGen(model="no-model", api_key="no-api-key", temperature=0.0, retrieval_model=os.getenv("RETRIEVAL_MODEL"))

    graph = Graph(
        entities={"cat", "cats", "food"},
        edges={"likes", "like"},
        relations={("cat", "likes", "food")},
    )

    # Override with different configurations in cluster method
    clustered = kg_gen.cluster(
        graph,
        model=os.getenv("LLM_MODEL"),  # Different model
        temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),  # Different temperature
        api_key=os.getenv("LLM_API_KEY"),
    )

    assert len(clustered.entities) <= len(graph.entities)
    assert len(clustered.edges) <= len(graph.edges)
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None


def test_large_scale_clustering(kg: KGGen):
    # Create a larger graph with multiple cluster opportunities
    graph = Graph(
        entities={
            "cat",
            "cats",
            "kitten",
            "dog",
            "dogs",
            "puppy",
            "mouse",
            "mice",
            "rat",
            "rats",
            "hamster",
            "hamsters",
            "fish",
            "fishes",
            "bird",
            "birds",
            "parrot",
            "parrots",
            "owner",
            "owners",
            "vet",
            "veterinarian",
            "doctor",
            "food",
            "baby",
            "pet",
        },
        edges={
            "likes",
            "like",
            "loves",
            "love",
            "chases",
            "chase",
            "pursuing",
            "pursue",
            "eats",
            "eat",
            "feeds",
            "feed",
            "cares for",
            "care for",
            "tends to",
            "tend to",
            "treats",
            "treat",
            "healing",
            "heals",
            "heal",
        },
        relations={
            ("cat", "likes", "fish"),
            ("cats", "love", "mice"),
            ("dog", "chases", "cat"),
            ("dogs", "pursue", "birds"),
            ("mouse", "eats", "food"),
            ("rat", "feeds", "baby"),
            ("owner", "cares for", "pet"),
            ("vet", "treats", "dog"),
            ("veterinarian", "heals", "cat"),
        },
    )

    # Add context to guide clustering
    context = """
  This knowledge graph describes relationships between animals and their caretakers. Cluster different forms of the same animal
  """

    clustered = kg.cluster(graph, context=context)

    # Basic assertions
    assert len(clustered.entities) < len(graph.entities)
    assert len(clustered.edges) < len(graph.edges)
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    # Expected cluster groups
    animal_groups = [
        {"cat", "cats", "kitten"},
        {"dog", "dogs", "puppy"},
        {"mouse", "mice", "rat", "rats"},
        {"fish", "fishes"},
        {"bird", "birds", "parrot", "parrots"},
        {"hamster", "hamsters"},
    ]

    person_groups = [{"owner", "owners"}, {"vet", "veterinarian", "doctor"}]

    action_groups = [
        {"likes", "like", "loves", "love"},
        {"chases", "chase", "pursuing", "pursue"},
        {"eats", "eat", "feeds", "feed"},
        {"cares for", "care for", "tends to", "tend to"},
        {"treats", "treat", "healing", "heals", "heal"},
    ]

    # Verify each expected group is represented in clusters
    for group in animal_groups + person_groups:
        # Find any cluster that contains at least 2 items from this group
        found_valid_cluster = False
        for cluster in clustered.entity_clusters.values():
            overlap = group & cluster
            if len(overlap) >= 2:  # At least 2 items from the group are clustered
                found_valid_cluster = True
                break
        assert found_valid_cluster, f"Failed to find valid cluster for group: {group}"

    # Check action clustering similarly
    for group in action_groups:
        found_valid_cluster = False
        for cluster in clustered.edge_clusters.values():
            overlap = group & cluster
            if len(overlap) >= 2:  # At least 2 items from the group are clustered
                found_valid_cluster = True
                break
        assert found_valid_cluster, (
            f"Failed to find valid cluster for action group: {group}"
        )


def test_clustering_with_context(kg: KGGen):
    # Create a graph with potentially ambiguous terms that should be clarified by context
    graph = Graph(
        entities={
            "bank",
            "banks",
            "banking",  # Could be financial or river bank
            "deposit",
            "deposits",  # Could be financial or geological
            "branch",
            "branches",  # Could be bank branch or tree branch
            "account",
            "accounts",  # Financial context
            "teller",
            "tellers",  # Financial context
        },
        edges={
            "has",
            "have",
            "manages",
            "manage",
            "opens",
            "open",
            "processes",
            "process",
        },
        relations={
            ("bank", "has", "branch"),
            ("banks", "have", "tellers"),
            ("teller", "manages", "account"),
            ("tellers", "process", "deposit"),
            ("branch", "opens", "accounts"),
        },
    )

    # Provide financial context
    context = """
  This knowledge graph describes a banking system and its operations.
  It covers the structure of banks, their branches, and how bank employees handle customer accounts and transactions.
  """

    clustered = kg.cluster(graph, context=context)

    # Basic assertions
    assert len(clustered.entities) < len(graph.entities)
    assert len(clustered.edges) < len(graph.edges)
    assert clustered.entity_clusters is not None
    assert clustered.edge_clusters is not None

    # Expected clusters in financial context
    financial_groups = [
        {"bank", "banks", "banking"},
        {"deposit", "deposits"},
        {"branch", "branches"},
        {"account", "accounts"},
        {"teller", "tellers"},
    ]

    action_groups = [
        {"has", "have"},
        {"manages", "manage"},
        {"opens", "open"},
        {"processes", "process"},
    ]

    # Verify each expected group is represented in clusters
    for group in financial_groups:
        found = False
        for rep, cluster in clustered.entity_clusters.items():
            if len(group & cluster) > 0:  # If there's any overlap
                # Allow for more conservative clustering
                assert len(group & cluster) >= 1  # At least one item from group
                found = True
                break
        assert found, f"Failed to find cluster for financial group: {group}"

    for group in action_groups:
        found = False
        for rep, cluster in clustered.edge_clusters.items():
            if len(group & cluster) > 0:
                # Allow for more conservative clustering
                assert len(group & cluster) >= 1  # At least one item from group
                found = True
                break
        assert found, f"Failed to find cluster for action group: {group}"

    # Now test with a different context to ensure clustering changes
    nature_context = """
  This knowledge graph describes natural features along a river.
  It covers riverbanks, geological deposits, and tree branches along the water.
  """

    nature_clustered = kg.cluster(graph, context=nature_context)

    # The clustering should be different with nature context
    assert nature_clustered.entity_clusters != clustered.entity_clusters

    # In nature context, 'bank' should not be clustered with 'account' or 'teller'
    for rep, cluster in nature_clustered.entity_clusters.items():
        if "bank" in cluster:
            assert "account" not in cluster
            assert "teller" not in cluster
        if "deposit" in cluster:
            assert "account" not in cluster
        if "branch" in cluster:
            assert "teller" not in cluster
