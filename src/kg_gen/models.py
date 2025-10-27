import json
from pydantic import BaseModel, Field
from typing import Tuple, Optional


# ~~~ DATA STRUCTURES ~~~
class Graph(BaseModel):
    entities: set[str] = Field(
        ..., description="All entities including additional ones from response"
    )
    edges: set[str] = Field(..., description="All edges")
    relations: set[Tuple[str, str, str]] = Field(
        ..., description="List of (subject, predicate, object) triples"
    )
    entity_clusters: Optional[dict[str, set[str]]] = None
    edge_clusters: Optional[dict[str, set[str]]] = None

    @staticmethod
    def from_file(file_path: str) -> "Graph":
        """
        Load the graph from a file.
        Fix graph entities and edges for missing ones defined in relations.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            graph = Graph.model_validate(data)

        # Fix graph entities and edges
        for relation in graph.relations:
            if relation[0] not in graph.entities:
                graph.entities.add(relation[0])
            if relation[1] not in graph.edges:
                graph.edges.add(relation[1])
            if relation[2] not in graph.entities:
                graph.entities.add(relation[2])

        return graph

    def to_file(self, file_path: str):
        """
        Save the graph to a file.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))

    def stats(self, name: Optional[str] = None):
        """
        Print the stats of the graph.
        """
        print(f"{name or 'Graph'} with:\n\t{len(self.entities)} entities\n\t{len(self.edges)} edges\n\t{len(self.relations)} relations")
