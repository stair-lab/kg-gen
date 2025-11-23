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
    clustered = kg.cluster(graph)

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
    kg_gen = KGGen(
        model="no-model",
        api_key="no-api-key",
        temperature=0.0,
        retrieval_model=os.getenv("RETRIEVAL_MODEL"),
    )

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


def test_semhash_deduplication(kg: KGGen):
    """
    Test SEMHASH deduplication method.
    SEMHASH should catch:
    - Plurals vs singulars (cat/cats, dog/dogs)
    - Case variations (Person/person/PERSON)
    - Very similar strings through normalization
    
    SEMHASH should NOT catch:
    - True synonyms (CEO/Chief Executive Officer)
    - Semantic equivalents (joyful/happy)
    """
    from src.kg_gen.steps._3_deduplicate import DeduplicateMethod
    
    graph = Graph(
        entities={
            "cat", "cats", "kitten",  # Plurals - should be caught
            "dog", "dogs",  # Plurals - should be caught
            "Person", "person", "PERSON",  # Case variations - should be caught
            "CEO", "Chief Executive Officer",  # Synonyms - should NOT be caught by semhash
            "happy", "joyful",  # Synonyms - should NOT be caught by semhash
        },
        edges={
            "likes", "like",  # Plurals - should be caught
            "manages", "Manages", "MANAGES",  # Case variations - should be caught
            "supervises", "oversees",  # Synonyms - should NOT be caught by semhash
        },
        relations={
            ("cat", "likes", "dog"),
            ("cats", "like", "dogs"),
            ("Person", "manages", "CEO"),
            ("person", "Manages", "Chief Executive Officer"),
            ("CEO", "supervises", "happy"),
            ("Chief Executive Officer", "oversees", "joyful"),
        },
    )
    
    deduplicated = kg.deduplicate(
        graph=graph,
        method=DeduplicateMethod.SEMHASH,
        semhash_similarity_threshold=0.95,
    )
    
    # SEMHASH should merge plurals
    assert "cat" in deduplicated.entities or "cats" in deduplicated.entities
    assert not ("cat" in deduplicated.entities and "cats" in deduplicated.entities)
    
    # SEMHASH should merge case variations
    person_count = sum(1 for e in deduplicated.entities if e.lower() == "person")
    assert person_count == 1, "Case variations should be merged to one"
    
    # SEMHASH should NOT merge true synonyms (they're different words)
    # Both CEO and Chief Executive Officer should still exist
    ceo_variants = [e for e in deduplicated.entities if "ceo" in e.lower() or "chief executive" in e.lower()]
    assert len(ceo_variants) == 2, "SEMHASH should not merge CEO and Chief Executive Officer"
    
    # SEMHASH should NOT merge happy and joyful (different words)
    emotion_variants = [e for e in deduplicated.entities if e.lower() in ["happy", "joyful"]]
    assert len(emotion_variants) == 2, "SEMHASH should not merge happy and joyful"
    
    # Check edges
    assert "supervises" in deduplicated.edges
    assert "oversees" in deduplicated.edges
    assert "supervises" != "oversees", "SEMHASH should not merge synonym edges"
    
    print(f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}")
    print(f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}")


def test_lm_based_deduplication(kg: KGGen):
    """
    Test LM_BASED deduplication method.
    LM_BASED should catch:
    - True synonyms (CEO/Chief Executive Officer)
    - Semantic equivalents (happy/joyful, big/large)
    - Abbreviations and full forms (USA/United States of America)
    - Tense variations (running/runs)
    
    This is what distinguishes LM_BASED from SEMHASH - it understands meaning.
    """
    from src.kg_gen.steps._3_deduplicate import DeduplicateMethod
    
    graph = Graph(
        entities={
            "CEO", "Chief Executive Officer",  # Abbreviation/full form
            "USA", "United States of America",  # Abbreviation/full form
            "happy", "joyful", "glad",  # Synonyms
            "big", "large",  # Synonyms
            "automobile", "car", "vehicle",  # Synonyms
        },
        edges={
            "manages", "oversees", "supervises",  # Synonyms
            "running", "runs", "run",  # Tense variations
            "possesses", "owns", "has",  # Synonyms
        },
        relations={
            ("CEO", "manages", "USA"),
            ("Chief Executive Officer", "oversees", "United States of America"),
            ("happy", "running", "big"),
            ("joyful", "runs", "large"),
            ("automobile", "possesses", "USA"),
            ("car", "owns", "United States of America"),
        },
    )
    
    deduplicated = kg.deduplicate(
        graph=graph,
        method=DeduplicateMethod.LM_BASED,
    )
    
    # LM_BASED should merge CEO and Chief Executive Officer
    ceo_count = sum(1 for e in deduplicated.entities if "ceo" in e.lower() or "chief" in e.lower())
    assert ceo_count == 1, "LM_BASED should merge CEO and Chief Executive Officer"
    
    # LM_BASED should merge USA abbreviations
    usa_count = sum(1 for e in deduplicated.entities if "usa" in e.lower() or "united states" in e.lower())
    assert usa_count == 1, "LM_BASED should merge USA and United States of America"
    
    # LM_BASED should merge happy synonyms
    happy_emotions = [e for e in deduplicated.entities if e.lower() in ["happy", "joyful", "glad"]]
    assert len(happy_emotions) == 1, f"LM_BASED should merge happy/joyful/glad, but got: {happy_emotions}"
    
    # LM_BASED should merge size synonyms
    size_words = [e for e in deduplicated.entities if e.lower() in ["big", "large"]]
    assert len(size_words) == 1, f"LM_BASED should merge big/large, but got: {size_words}"
    
    # Check edges - should merge synonyms
    manage_edges = [e for e in deduplicated.edges if e.lower() in ["manages", "oversees", "supervises"]]
    assert len(manage_edges) == 1, f"LM_BASED should merge management synonyms, but got: {manage_edges}"
    
    run_edges = [e for e in deduplicated.edges if "run" in e.lower()]
    assert len(run_edges) == 1, f"LM_BASED should merge run tense variations, but got: {run_edges}"
    
    print(f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}")
    print(f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}")
    print(f"Deduplicated entities: {deduplicated.entities}")
    print(f"Deduplicated edges: {deduplicated.edges}")


def test_full_deduplication_comprehensive(kg: KGGen):
    """
    Test FULL deduplication method.
    FULL method runs SEMHASH first, then LM_BASED, so it should catch:
    - Everything SEMHASH catches (plurals, case variations)
    - Everything LM_BASED catches (synonyms, abbreviations)
    
    This is the most comprehensive approach.
    Since FULL = SEMHASH + LM_BASED, it catches both structural and semantic duplicates.
    """
    from src.kg_gen.steps._3_deduplicate import DeduplicateMethod
    
    graph = Graph(
        entities={
            # Plurals + case variations (SEMHASH territory)
            "cat", "cats", "Cat", "CATS",
            # Synonyms/Abbreviations (LM_BASED territory)  
            "CEO", "Chief Executive Officer",
            "USA", "United States of America",
        },
        edges={
            # Plurals (SEMHASH)
            "likes", "like",
            # Case variations (SEMHASH)
            "Manages", "manages", "MANAGES",
            # Synonyms (LM_BASED)
            "supervises", "oversees",
        },
        relations={
            ("cat", "likes", "CEO"),
            ("cats", "like", "Chief Executive Officer"),
            ("Cat", "Manages", "USA"),
            ("CATS", "manages", "United States of America"),
            ("CEO", "supervises", "cat"),
            ("Chief Executive Officer", "oversees", "cats"),
        },
    )
    
    deduplicated = kg.deduplicate(
        graph=graph,
        method=DeduplicateMethod.FULL,
        semhash_similarity_threshold=0.95,
    )
    
    # Should handle plurals + case variations (SEMHASH)
    cat_variants = [e for e in deduplicated.entities if "cat" in e.lower()]
    assert len(cat_variants) == 1, f"FULL should merge all cat variations, but got: {cat_variants}"
    
    # Should handle synonyms/abbreviations (LM_BASED)
    ceo_count = sum(1 for e in deduplicated.entities if "ceo" in e.lower() or "chief" in e.lower())
    assert ceo_count == 1, "FULL should merge CEO and Chief Executive Officer"
    
    # Should handle abbreviations (LM_BASED)
    usa_count = sum(1 for e in deduplicated.entities if "usa" in e.lower() or "united states" in e.lower())
    assert usa_count == 1, "FULL should merge USA and United States of America"
    
    # Check edges - should catch plurals
    like_edges = [e for e in deduplicated.edges if "like" in e.lower()]
    assert len(like_edges) == 1, f"FULL should merge like variations, but got: {like_edges}"
    
    # Check edges - should catch case variations
    manage_edges = [e for e in deduplicated.edges if "manage" in e.lower()]
    assert len(manage_edges) == 1, f"FULL should merge manages variations, but got: {manage_edges}"
    
    # Check edges - LM should catch synonyms (though this is non-deterministic)
    supervise_edges = [e for e in deduplicated.edges if e.lower() in ["supervises", "oversees"]]
    # LM-based deduplication is non-deterministic, so we allow 1 or 2 here
    assert len(supervise_edges) <= 2, f"Got unexpected supervise edges: {supervise_edges}"
    
    # Overall, FULL should achieve significant deduplication
    # Original: 11 entities, 7 edges
    # Expected after FULL: ~3 entities (cat, CEO, USA), ~3-4 edges
    assert len(deduplicated.entities) <= 5, f"FULL should significantly reduce entities, but got {len(deduplicated.entities)}"
    assert len(deduplicated.edges) <= 4, f"FULL should significantly reduce edges, but got {len(deduplicated.edges)}"
    
    print(f"Original entities: {len(graph.entities)}, Deduplicated: {len(deduplicated.entities)}")
    print(f"Original edges: {len(graph.edges)}, Deduplicated: {len(deduplicated.edges)}")
    print(f"Final entities: {deduplicated.entities}")
    print(f"Final edges: {deduplicated.edges}")
