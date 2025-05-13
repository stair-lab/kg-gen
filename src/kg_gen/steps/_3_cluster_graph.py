from ..models import Graph
import dspy
from typing import Optional
from pydantic import BaseModel

LOOP_N = 8 
from typing import Literal

BATCH_SIZE = 10

ItemType = Literal["entities", "edges"]

class ChooseRepresentative(dspy.Signature):
  """Select the best item name to represent the cluster, ideally from the cluster.
  Prefer shorter names and generalizability across the cluster."""
  
  cluster: set[str] = dspy.InputField()
  context: str = dspy.InputField(desc="the larger context in which the items appear")
  representative: str = dspy.OutputField()

choose_rep = dspy.Predict(ChooseRepresentative)

class Cluster(BaseModel):
  representative: str
  members: set[str]

def cluster_items(dspy: dspy.dspy, items: set[str], item_type: ItemType = "entities", context: str = "") -> tuple[set[str], dict[str, set[str]]]:
  """Returns item set and cluster dict mapping representatives to sets of items"""
  
  # If there are fewer than 2 items, we can't cluster
  if len(items) < 2:
    return items, {item: {item} for item in items}
  
  # Enhanced context to ensure better clustering
  if context:
    context_message = f"{item_type} of a graph extracted from source text. {context} Focus on clustering items with similar forms, meanings, or semantic relationships."
  else:
    context_message = f"{item_type} of a graph extracted from source text. Focus on identifying items that should be clustered together because they represent the same concept, such as singular/plural forms, synonyms, abbreviations, or different spellings."
  
  remaining_items = items.copy()
  clusters: list[Cluster] = []
  no_progress_count = 0
  
  # Pre-process to identify obvious clusters (like singular/plural forms, capitalization differences)
  stemmed_items = {}
  
  # Simple stemming and normalization for obvious matches
  for item in items:
    # Create normalized form (lowercase)
    normalized = item.lower()
    
    # Simple plural handling (assuming English)
    if normalized.endswith('s') and normalized[:-1] in [i.lower() for i in items]:
      stem = normalized[:-1]
    elif normalized.endswith('es') and normalized[:-2] in [i.lower() for i in items]:
      stem = normalized[:-2]
    elif normalized + 's' in [i.lower() for i in items]:
      stem = normalized
    else:
      stem = normalized
      
    if stem not in stemmed_items:
      stemmed_items[stem] = set()
    stemmed_items[stem].add(item)
  
  # Create initial clusters from the stemming
  for stem, stem_items in stemmed_items.items():
    if len(stem_items) > 1:  # Only create clusters with more than one item
      # Find the best representative (prefer shorter items)
      representative = min(stem_items, key=len)
      clusters.append(Cluster(
        representative=representative,
        members=stem_items
      ))
      # Remove these items from remaining items
      remaining_items = remaining_items - stem_items
  
  # Main clustering loop for semantic matches
  while len(remaining_items) > 0:
    ItemsLiteral = Literal[tuple(items)]
    
    class ExtractCluster(dspy.Signature):
      """Find one cluster of related items from the list.
      A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, cases, or are synonyms. 
      Return populated list only if you find items that clearly belong together, else return empty list.
      Be thorough in identifying related items."""
      
      items: set[ItemsLiteral] = dspy.InputField()
      context: str = dspy.InputField(desc="The larger context in which the items appear")
      cluster: list[ItemsLiteral] = dspy.OutputField(desc="List of items that clearly belong together in a cluster")

    extract = dspy.Predict(ExtractCluster)
    
    suggested_cluster: set[ItemsLiteral] = set(extract(
      items=remaining_items, 
      context=context_message
    ).cluster)
    
    if len(suggested_cluster) > 0:
      ClusterLiteral = Literal[tuple(suggested_cluster)]
      
      class ValidateCluster(dspy.Signature):
        """Validate if these items belong in the same cluster.
        A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, cases, or are synonyms. 
        Return populated list only if you find items that clearly belong together, else return empty list.
        Be precise but also thorough in validation - don't miss valid clusters."""
        
        cluster: set[ClusterLiteral] = dspy.InputField()
        context: str = dspy.InputField(desc="The larger context in which the items appear")
        validated_items: list[ClusterLiteral] = dspy.OutputField(desc="All the items that belong together in the cluster")
      
      validate = dspy.Predict(ValidateCluster)
      
      validated_cluster = set(validate(
        cluster=suggested_cluster, 
        context=context_message
      ).validated_items)
      
      if len(validated_cluster) > 1:
        no_progress_count = 0
        
        representative = choose_rep(
          cluster=validated_cluster, 
          context=context_message
        ).representative
        
        clusters.append(Cluster(
          representative=representative,
          members=validated_cluster
        ))
        remaining_items = {item for item in remaining_items if item not in validated_cluster}
        continue
      
    no_progress_count += 1
    
    # Break if we've had too many attempts without progress or if we're done
    if no_progress_count >= LOOP_N or len(remaining_items) == 0:
      break
  
  # Assign any remaining items to singleton clusters
  for item in remaining_items:
    clusters.append(Cluster(
      representative=item,
      members={item}
    ))
  
  # Prepare the final output format expected by the calling function:
  # 1. A dictionary mapping representative -> set of members
  # 2. A set containing all unique representatives
  final_clusters_dict = {c.representative: c.members for c in clusters}
  new_items = set(final_clusters_dict.keys()) # The set of representatives
  
  return new_items, final_clusters_dict

def cluster_graph(dspy: dspy.dspy, graph: Graph, context: str = "") -> Graph:
  """Cluster entities and edges in a graph, updating relations accordingly.
  
  Args:
      dspy: The DSPy runtime
      graph: Input graph with entities, edges, and relations
      context: Additional context string for clustering
      
  Returns:
      Graph with clustered entities and edges, updated relations, and cluster mappings
  """
  entities, entity_clusters = cluster_items(dspy, graph.entities, "entities", context)
  edges, edge_clusters = cluster_items(dspy, graph.edges, "edges", context)
  
  # Update relations based on clusters
  relations: set[tuple[str, str, str]] = set()
  for s, p, o in graph.relations:
    # Look up subject in entity clusters
    if s not in entities:
      for rep, cluster in entity_clusters.items():
        if s in cluster:
          s = rep
          break
          
    # Look up predicate in edge clusters
    if p not in edges:
      for rep, cluster in edge_clusters.items():
        if p in cluster:
          p = rep
          break
          
    # Look up object in entity clusters
    if o not in entities:
      for rep, cluster in entity_clusters.items():
        if o in cluster:
          o = rep
          break
          
    relations.add((s, p, o))
  
  # Initialize type mappings for clustered graph
  entity_types = None
  edge_types = None
  
  # Update entity_types if present in original graph
  if hasattr(graph, 'entity_types') and graph.entity_types:
    entity_types = {}
    # Copy existing type mappings for representative entities
    for entity in entities:
      # If this entity is already directly typed in the original graph
      if entity in graph.entity_types:
        entity_types[entity] = graph.entity_types[entity]
      else:
        # Check if any members of this entity's cluster have types
        for rep, cluster in entity_clusters.items():
          if entity == rep:  # Found this entity's cluster
            # Find any typed members in the cluster
            for member in cluster:
              if member in graph.entity_types:
                # Assign the first found type to the representative
                entity_types[rep] = graph.entity_types[member]
                break
  
  # Update edge_types if present in original graph
  if hasattr(graph, 'edge_types') and graph.edge_types:
    edge_types = {}
    # Copy existing type mappings for representative edges
    for edge in edges:
      # If this edge is already directly typed in the original graph
      if edge in graph.edge_types:
        edge_types[edge] = graph.edge_types[edge]
      else:
        # Check if any members of this edge's cluster have types
        for rep, cluster in edge_clusters.items():
          if edge == rep:  # Found this edge's cluster
            # Find any typed members in the cluster
            for member in cluster:
              if member in graph.edge_types:
                # Assign the first found type to the representative
                edge_types[rep] = graph.edge_types[member]
                break
  
  # Create graph with all the updated information
  graph_args = {
    "entities": entities,
    "edges": edges,
    "relations": relations,
    "entity_clusters": entity_clusters,
    "edge_clusters": edge_clusters
  }
  
  # Add type information if available
  if entity_types:
    graph_args["entity_types"] = entity_types
  if edge_types:
    graph_args["edge_types"] = edge_types
      
  return Graph(**graph_args)

if __name__ == "__main__":
  import os
  from ..kg_gen import KGGen
  
  model = "openai/gpt-4o"
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    print("Please set OPENAI_API_KEY environment variable")
    exit(1)
    
  # Example with pets and animals
  # kg_gen = KGGen(
  #   model=model,
  #   temperature=0.0,
  #   api_key=api_key
  # )
  # graph = Graph(
  #   entities={
  #     "cat", "cats", "dog", "dogs", "mouse", "mice", "fish", "fishes",
  #     "bird", "birds", "hamster", "hamsters", "person", "people",
  #     "owner", "owners", "vet", "veterinarian", "food", "treats"
  #   },
  #   edges={
  #     "like", "likes", "love", "loves", "eat", "eats", 
  #     "chase", "chases", "feed", "feeds", "care for", "cares for",
  #     "visit", "visits", "play with", "plays with"
  #   },
  #   relations={
  #     ("cat", "likes", "fish"),
  #     ("cats", "love", "mice"),
  #     ("dog", "chases", "cat"),
  #     ("dogs", "chase", "birds"),
  #     ("mouse", "eats", "food"),
  #     ("mice", "eat", "treats"),
  #     ("person", "feeds", "cat"),
  #     ("people", "feed", "dogs"),
  #     ("owner", "cares for", "hamster"),
  #     ("owners", "care for", "hamsters"),
  #     ("vet", "visits", "dog"),
  #     ("veterinarian", "visit", "cats"),
  #     ("bird", "plays with", "fish"),
  #     ("birds", "play with", "fishes")
  #   }
  # )
  

  # Example with family relationships
  kg_gen = KGGen(
    model=model,
    temperature=0.0,
    api_key=api_key
  )
  graph = Graph(
    entities={
      "Linda", "Joshua", "Josh", "Ben", "Andrew", "Judy"
    },
    edges={
      "is mother of", "is brother of", "is father of",
      "is sister of", "is nephew of", "is aunt of",
      "is same as"
    },
    relations={
      ("Linda", "is mother of", "Joshua"),
      ("Ben", "is brother of", "Josh"),
      ("Andrew", "is father of", "Josh"),
      ("Judy", "is sister of", "Andrew"),
      ("Josh", "is nephew of", "Judy"),
      ("Judy", "is aunt of", "Josh"),
      ("Josh", "is same as", "Joshua")
    }
  )
  
  try: 
    clustered_graph = kg_gen.cluster(graph=graph)
    print('Clustered graph:', clustered_graph)
    
  except Exception as e:
    raise ValueError(e)