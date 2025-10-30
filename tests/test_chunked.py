from src.kg_gen import KGGen
import os
from fixtures import kg


def test_chunked(kg: KGGen):
    with open("tests/data/kingkiller_chapter_one.txt", "r", encoding="utf-8") as f:
        text = f.read()

    graph = kg.generate(
        input_data=text,
    )
    print("Without chunking:")
    print("Entities:", graph.entities)
    print("Edges:", graph.edges)
    print("Relations:", graph.relations)

    # Generate graph from wiki text with chunking
    graph_chunked = kg.generate(
        input_data=text,
        chunk_size=1000,
    )
    print("\nWith chunking:")
    print("Entities:", graph_chunked.entities)
    print("Edges:", graph_chunked.edges)
    print("Relations:", graph_chunked.relations)

    # Compare differences
    print("\nDifferences between chunked and non-chunked graph generation:")
    print(
        "Entities found only in chunked graph:", graph_chunked.entities - graph.entities
    )
    print(
        "Entities found only in non-chunked graph:",
        graph.entities - graph_chunked.entities,
    )
    print("Edge types found only in chunked graph:", graph_chunked.edges - graph.edges)
    print(
        "Edge types found only in non-chunked graph:", graph.edges - graph_chunked.edges
    )
    print(
        "Relationships found only in chunked graph:",
        graph_chunked.relations - graph.relations,
    )
    print(
        "Relationships found only in non-chunked graph:",
        graph.relations - graph_chunked.relations,
    )


def test_chunk_and_cluster(kg: KGGen):
    # Load fresh wiki content
    with open("tests/data/fresh_wiki_article.md", "r", encoding="utf-8") as f:
        text = f.read()

    # # Generate graph from wiki text with chunking
    graph = kg.generate(
        input_data=text,
        chunk_size=5000,
        cluster=True,
    )
    print("Entities:", graph.entities)
    print("Edges:", graph.edges)
    print("Relations:", graph.relations)
