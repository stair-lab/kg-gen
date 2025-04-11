from ..models import Graph
import dspy
from typing import Optional

LOOP_N = 8 
BATCH_SIZE = 10

class ExtractCluster(dspy.Signature):
  """Find one cluster of related items from the list.
  A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases. 
  Return populated list only if you find items that clearly belong together, else return empty list."""
  
  items: set[str] = dspy.InputField()
  context: str = dspy.InputField(desc="the larger context in which the items appear")
  cluster: set[str] = dspy.OutputField()

class ValidateCluster(dspy.Signature):
  """Verify if these items belong in the same cluster.
  A cluster should contain items that are the same in meaning, with different tenses, plural forms, stem forms, or cases. 
  Return populated list only if you find items that clearly belong together, else return empty list."""
  
  cluster: set[str] = dspy.InputField()
  context: str = dspy.InputField(desc="the larger context in which the items appear")
  validated_items: set[str] = dspy.OutputField(desc="All the items that belong together in the cluster")

class ChooseRepresentative(dspy.Signature):
  """Select the best item name to represent the cluster, ideally from the cluster.
  Prefer shorter names and generalizability across the cluster."""
  
  cluster: set[str] = dspy.InputField()
  context: str = dspy.InputField(desc="the larger context in which the items appear")
  representative: str = dspy.OutputField()

class CheckExistingClusters(dspy.Signature):
  """Determine if the given items can be added to any of the existing clusters.
  Return representative of matching cluster for each item, or None if there is no match."""
  
  items: list[str] = dspy.InputField()
  clusters: dict[str, set[str]] = dspy.InputField(desc="Mapping of cluster representatives to their cluster members")
  context: str = dspy.InputField(desc="the larger context in which the items appear")
  cluster_reps_that_items_belong_to: list[Optional[str]] = dspy.OutputField(desc="ordered list of cluster representatives where each is the cluster where that item belongs to, or None if no match. THIS LIST LENGTH IS SAME AS ITEMS LIST LENGTH")


def cluster_items(dspyi: dspy.dspy, items: set[str], item_type: str = "entities", context: str = "") -> tuple[set[str], dict[str, set[str]]]:
  """Returns item set and cluster dict mapping representatives to sets of items"""
  
  context = f"{item_type} of a graph extracted from source text." + context
  remaining_items = items.copy()
  clusters = {} 
  no_progress_count = 0
  
  extract = dspyi.Predict(ExtractCluster)
  validate = dspyi.Predict(ValidateCluster)
  choose_rep = dspyi.Predict(ChooseRepresentative)
  check_existing = dspyi.ChainOfThought(CheckExistingClusters)
  
  while len(remaining_items) > 0:
    e_result = extract(items=remaining_items, context=context)
    suggested_cluster = e_result.cluster
    
    if len(suggested_cluster) > 0:
      v_result = validate(cluster=suggested_cluster, context=context)
      validated_cluster = v_result.validated_items
      
      if len(validated_cluster) > 1:
        no_progress_count = 0
        r_result = choose_rep(cluster=validated_cluster, context=context)
        representative = r_result.representative
        
        clusters[representative] = validated_cluster
        remaining_items = {item for item in remaining_items if item not in validated_cluster}
        continue
      
    no_progress_count += 1
    
    if no_progress_count >= LOOP_N or len(remaining_items) == 0:
      break
    
  if len(remaining_items) > 0:
    items_to_process = list(remaining_items) 
      
    for i in range(0, len(items_to_process), BATCH_SIZE):
      batch = items_to_process[i:min(i + BATCH_SIZE, len(items_to_process))]
      
      if not clusters:
        for item in batch:
          clusters[item] = {item}
        continue
      
      c_result = check_existing(
        items=batch,
        clusters=clusters,
        context=context
      )
      cluster_reps = c_result.cluster_reps_that_items_belong_to  
      
      # Process each item with its corresponding representative
      for i, item in enumerate(batch):
        rep = cluster_reps[i] if i < len(cluster_reps) else None
        if rep is not None and rep in clusters:
          new_cluster = clusters[rep] | {item}
          v_result = validate(cluster=new_cluster, context=context)
          validated_items = v_result.validated_items
          if len(validated_items) == len(clusters[rep]) + 1:
            clusters[rep].add(item)
          else:
            clusters[item] = {item}
        else:
          clusters[item] = {item}
  new_items = set(clusters.keys())
  
  return new_items, clusters

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

  return Graph(
    entities=entities,  
    edges=edges,  
    relations=relations,
    entity_clusters=entity_clusters,
    edge_clusters=edge_clusters
  )

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