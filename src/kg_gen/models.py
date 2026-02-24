import json
from pydantic import BaseModel, Field
from typing import Any, Tuple, Optional

# Delimiter for relation_scores dict key (subject, predicate, object) -> string.
# Used so keys are JSON-serializable and unambiguous.
_REL_KEY_SEP = "||"


def _relation_to_key(relation: Tuple[str, str, str]) -> str:
    """Encode a (subject, predicate, object) triple as a string key."""
    return _REL_KEY_SEP.join([relation[0], relation[1], relation[2]])


def _key_to_relation(key: str) -> Tuple[str, str, str]:
    """Decode a string key back to (subject, predicate, object)."""
    parts = key.split(_REL_KEY_SEP, 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid relation key: {key}")
    return (parts[0], parts[1], parts[2])


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

    entity_metadata: dict[str, set[str]] | None = None
    relation_scores: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional confidence score per relation. Keys are 'subject||predicate||object', values in [0.0, 1.0].",
    )

    @staticmethod
    def from_file(file_path: str) -> "Graph":
        """
        Load the graph from a file.
        Fix graph entities and edges for missing ones defined in relations.
        """
        with open(file_path, "r", encoding="utf-8") as f:
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
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    def get_relation_score(self, relation: Tuple[str, str, str]) -> Optional[float]:
        """Return confidence score for a relation triple, or None if not scored."""
        if not self.relation_scores:
            return None
        key = _relation_to_key(relation)
        return self.relation_scores.get(key)

    def stats(self, name: Optional[str] = None):
        """
        Print the stats of the graph.
        """
        print(
            f"{name or 'Graph'} with:\n\t{len(self.entities)} entities\n\t{len(self.edges)} edges\n\t{len(self.relations)} relations"
        )
