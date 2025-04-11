from pydantic import BaseModel, model_validator, Field
from typing import Tuple, Optional

# ~~~ DATA STRUCTURES ~~~
class Graph(BaseModel):
  entities: set[str] = Field(..., description="All entities including additional ones from response")
  edges: set[str] = Field(..., description="All edges")
  relations: set[Tuple[str, str, str]] = Field(..., description="List of (subject, predicate, object) triples")
  entity_clusters: Optional[dict[str, set[str]]] = None
  edge_clusters: Optional[dict[str, set[str]]] = None

  @model_validator(mode='after')
  def validate_consistency(self) -> 'Graph':
    entities = set(self.entities)
    edges = set(self.edges)
    # Validate relations
    for subj, pred, obj in self.relations:
      if subj not in entities:
        raise ValueError(f"Relation subject '{subj}' not in entities")
      if obj not in entities:
        raise ValueError(f"Relation object '{obj}' not in entities") 
      if pred not in edges:
        raise ValueError(f"Relation pred '{pred}' not edges")
        
    # Validate entity clusters
    if self.entity_clusters:
      for key, values in self.entity_clusters.items():
        if key not in entities:
          raise ValueError(f"Entity cluster key '{key}' not in entities")
        for value in values:
          if value in entities and value != key:
            raise ValueError(f"Entity cluster value '{value}' appears in entities but is not the cluster key")
          
    # Validate edge clusters  
    if self.edge_clusters:
      for key, values in self.edge_clusters.items():
        if key not in edges:
          raise ValueError(f"Edge cluster key '{key}' not in edges")
        for value in values:
          if value in edges and value != key:
            raise ValueError(f"Edge cluster value '{value}' appears in edges but is not the cluster key")
    return self
