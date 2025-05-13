from pydantic import BaseModel, Field
from typing import Tuple, Optional, Dict, List

# ~~~ DATA STRUCTURES ~~~
class Graph(BaseModel):
  entities: set[str] = Field(..., description="All entities including additional ones from response")
  edges: set[str] = Field(..., description="All edges")
  relations: set[Tuple[str, str, str]] = Field(..., description="List of (subject, predicate, object) triples")
  entity_clusters: Optional[dict[str, set[str]]] = None
  edge_clusters: Optional[dict[str, set[str]]] = None
  entity_types: Optional[Dict[str, str]] = None  # Maps entity -> type
  edge_types: Optional[Dict[str, str]] = None    # Maps edge -> type
