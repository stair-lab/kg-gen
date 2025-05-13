from typing import Union, List, Dict, Optional
from openai import OpenAI

from .steps._1_get_entities import get_entities
from .steps._2_get_relations import get_relations
from .steps._3_cluster_graph import cluster_graph
from .utils.chunk_text import chunk_text
from .models import Graph
import dspy
import json
import os
from concurrent.futures import ThreadPoolExecutor

class KGGen:
  def __init__(
    self,
    model: str = "openai/gpt-4o",
    temperature: float = 0.0,
    api_key: str = None,
    api_base: str = None
  ):
    """Initialize KGGen with optional model configuration

    Args:
        model: Name of model to use (e.g. 'gpt-4')
        temperature: Temperature for model sampling
        api_key: API key for model access
        api_base: Specify the base URL endpoint for making API calls to a language model service
    """
    self.dspy = dspy
    self.model = model
    self.temperature = temperature
    self.api_key = api_key
    self.api_base = api_base
    self.init_model(model, temperature, api_key, api_base)

  def init_model(
    self,
    model: str = None,
    temperature: float = None,
    api_key: str = None,
    api_base: str = None
  ):
    """Initialize or reinitialize the model with new parameters

    Args:
        model: Name of model to use (e.g. 'gpt-4')
        temperature: Temperature for model sampling
        api_key: API key for model access
        api_base: API base for model access
    """
    # Update instance variables if new values provided
    if model is not None:
      self.model = model
    if temperature is not None:
      self.temperature = temperature
    if api_key is not None:
      self.api_key = api_key
    if api_base is not None:
      self.api_base = api_base

    # Initialize dspy LM with current settings
    if self.api_key:
      self.lm = dspy.LM(model=self.model, api_key=self.api_key, temperature=self.temperature, api_base=self.api_base)
    else:
      self.lm = dspy.LM(model=self.model, temperature=self.temperature, api_base=self.api_base)

    self.dspy.configure(lm=self.lm)

  def generate(
    self,
    input_data: Union[str, List[Dict]],
    model: str = None,
    api_key: str = None,
    api_base: str = None,
    context: str = "",
    # example_relations: Optional[Union[
    #   List[Tuple[str, str, str]],
    #   List[Tuple[Tuple[str, str], str, Tuple[str, str]]]
    # ]] = None,
    chunk_size: Optional[int] = None,
    cluster: bool = False,
    temperature: float = None,
    node_type: Optional[List[str]] = None,
    edge_type: Optional[List[str]] = None,
    require_node_type: bool = True,
    require_edge_type: bool = True,
    # ontology: Optional[List[Tuple[str, str, str]]] = None,
    output_folder: Optional[str] = None
  ) -> Graph:
    """Generate a knowledge graph from input text or messages.

    Args:
        input_data: Text string or list of message dicts
        model: Name of OpenAI model to use
        api_key (str): OpenAI API key for making model calls
        chunk_size: Max size of text chunks in characters to process
        context: Description of data context
        example_relations: Example relationship tuples
        node_type: List of allowed node types
        edge_type: List of allowed edge types
        require_node_type: Whether every node must have one of the specified types (default True)
        require_edge_type: Whether every edge must have one of the specified types (default True)
        ontology: Valid node-edge-node structure tuples
        output_folder: Path to save partial progress

    Returns:
        Generated knowledge graph
    """

    # Process input data
    is_conversation = isinstance(input_data, list)
    if is_conversation:
      # Extract text from messages
      text_content = []
      for message in input_data:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
          raise ValueError("Messages must be dicts with 'role' and 'content' keys")
        if message['role'] in ['user', 'assistant']:
          text_content.append(f"{message['role']}: {message['content']}")

      # Join with newlines to preserve message boundaries
      processed_input = "\n".join(text_content)
    else:
      processed_input = input_data

    # Reinitialize dspy with new parameters if any are provided
    if any([model, temperature, api_key, api_base]):
      self.init_model(
        model=model or self.model,
        temperature=temperature or self.temperature,
        api_key=api_key or self.api_key,
        api_base=api_base or self.api_base,
      )
    
    # Initialize type maps
    entity_types_map = None
    edge_types_map = None

    if not chunk_size:
      # Process without chunking
      entities_result = get_entities(
        self.dspy, 
        processed_input, 
        is_conversation=is_conversation,
        node_types=node_type,
        require_node_type=require_node_type
      )
      
      # Unpack results - entities_result is a tuple of (entities_list, entity_types_dict)
      if isinstance(entities_result, tuple) and len(entities_result) == 2:
        entities, entity_types_map = entities_result
      else:
        entities = entities_result
        entity_types_map = None
        
      # Get relations with edge types if specified
      relations_result = get_relations(
        self.dspy, 
        processed_input, 
        entities, 
        is_conversation=is_conversation,
        edge_types=edge_type,
        require_edge_type=require_edge_type
      )
      
      # Unpack relations results
      if isinstance(relations_result, tuple) and len(relations_result) == 2:
        relations, edge_types_map = relations_result
      else:
        relations = relations_result
        edge_types_map = None
        
    else:
      # Process with chunking
      chunks = chunk_text(processed_input, chunk_size)
      entities = set()
      relations = set()
      
      # Initialize type dictionaries
      all_entity_types = {}
      all_edge_types = {}

      def process_chunk(chunk):
        # Get entities with types
        entities_result = get_entities(
          self.dspy, 
          chunk, 
          is_conversation=is_conversation,
          node_types=node_type,
          require_node_type=require_node_type
        )
        
        # Unpack entity results
        if isinstance(entities_result, tuple) and len(entities_result) == 2:
          chunk_entities, chunk_entity_types = entities_result
        else:
          chunk_entities = entities_result
          chunk_entity_types = None
          
        # Get relations with types
        relations_result = get_relations(
          self.dspy, 
          chunk, 
          chunk_entities, 
          is_conversation=is_conversation,
          edge_types=edge_type,
          require_edge_type=require_edge_type
        )
        
        # Unpack relation results
        if isinstance(relations_result, tuple) and len(relations_result) == 2:
          chunk_relations, chunk_edge_types = relations_result
        else:
          chunk_relations = relations_result
          chunk_edge_types = None
          
        return chunk_entities, chunk_relations, chunk_entity_types, chunk_edge_types

      # Process chunks in parallel using ThreadPoolExecutor
      with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))

      # Combine results
      for chunk_entities, chunk_relations, chunk_entity_types, chunk_edge_types in results:
        entities.update(chunk_entities)
        relations.update(chunk_relations)
        
        # Update type dictionaries if they exist
        if chunk_entity_types:
          all_entity_types.update(chunk_entity_types)
        if chunk_edge_types:
          all_edge_types.update(chunk_edge_types)
          
      # Set the combined type maps
      entity_types_map = all_entity_types if all_entity_types else None
      edge_types_map = all_edge_types if all_edge_types else None
    
    # Create graph with type information if available
    graph_args = {
      "entities": entities,
      "relations": relations,
      "edges": {relation[1] for relation in relations}
    }
    
    # Add type information if available
    if entity_types_map:
      graph_args["entity_types"] = entity_types_map
    if edge_types_map:
      graph_args["edge_types"] = edge_types_map
      
    graph = Graph(**graph_args)
    
    if cluster:
      graph = self.cluster(graph, context)

    if output_folder:
      os.makedirs(output_folder, exist_ok=True)
      output_path = os.path.join(output_folder, 'graph.json')

      graph_dict = {
        'entities': list(entities),
        'relations': list(relations),
        'edges': list(graph.edges)
      }
      
      # Include entity and edge type information if available
      if hasattr(graph, 'entity_types') and graph.entity_types:
        graph_dict['entity_types'] = graph.entity_types
      if hasattr(graph, 'edge_types') and graph.edge_types:
        graph_dict['edge_types'] = graph.edge_types
      
      # Include clustering information if available
      if hasattr(graph, 'entity_clusters') and graph.entity_clusters:
        graph_dict['entity_clusters'] = {rep: list(cluster) for rep, cluster in graph.entity_clusters.items()}
      if hasattr(graph, 'edge_clusters') and graph.edge_clusters:
        graph_dict['edge_clusters'] = {rep: list(cluster) for rep, cluster in graph.edge_clusters.items()}
      
      with open(output_path, 'w') as f:
        json.dump(graph_dict, f, indent=2)

    return graph

  def cluster(
    self,
    graph: Graph,
    context: str = "",
    model: str = None,
    temperature: float = None,
    api_key: str = None,
    api_base: str = None
  ) -> Graph:
    # Reinitialize dspy with new parameters if any are provided
    if any([model, temperature, api_key, api_base]):
      self.init_model(
        model=model or self.model,
        temperature=temperature or self.temperature,
        api_key=api_key or self.api_key,
        api_base=api_base or self.api_base,
      )

    return cluster_graph(self.dspy, graph, context)

  def aggregate(self, graphs: list[Graph]) -> Graph:
    # Initialize empty sets for combined graph
    all_entities = set()
    all_relations = set()
    all_edges = set()

    # Combine all graphs
    for graph in graphs:
      all_entities.update(graph.entities)
      all_relations.update(graph.relations)
      all_edges.update(graph.edges)

    # Create and return aggregated graph
    return Graph(
      entities=all_entities,
      relations=all_relations,
      edges=all_edges
    )
