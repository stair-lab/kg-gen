import pytest
from src.kg_gen import KGGen
from src.kg_gen.models import Graph
import os

# Test configurations
TEST_MODEL = "openai/gpt-4o"
TEST_TEMP = 0.0
TEST_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")

def test_entities_with_node_types():
  """Test graph generation with node type constraints."""
  kg_gen = KGGen(
    model=TEST_MODEL,
    temperature=TEST_TEMP,
    api_key=TEST_API_KEY
  )
  
  # Input text with different entity types
  text = "Harry Potter is a student at Hogwarts School. Hermione Granger is a friend of Harry. " \
         "Professor Dumbledore is the headmaster of Hogwarts. The Hogwarts Express is a train " \
         "that takes students to Hogwarts."
  
  # Define node types to test
  node_types = ["Person", "Location", "Organization", "Vehicle"]
  
  # Generate graph with node types
  graph = kg_gen.generate(
    input_data=text,
    node_type=node_types,
    require_node_type=True
  )
  
  # Assert entity types are present
  assert hasattr(graph, 'entity_types')
  assert graph.entity_types is not None
  assert len(graph.entity_types) > 0
  
  # Check that each entity has a type from the allowed list
  for entity, entity_type in graph.entity_types.items():
    assert entity_type in node_types
    
  # Validate entity classification
  if "Harry Potter" in graph.entity_types:
    assert graph.entity_types["Harry Potter"] == "Person"
  if "Hogwarts" in graph.entity_types:
    assert graph.entity_types["Hogwarts"] in ["Location", "Organization"]
  if "Hogwarts Express" in graph.entity_types:
    assert graph.entity_types["Hogwarts Express"] == "Vehicle"

def test_relations_with_edge_types():
  """Test graph generation with edge type constraints."""
  kg_gen = KGGen(
    model=TEST_MODEL,
    temperature=TEST_TEMP,
    api_key=TEST_API_KEY
  )
  
  # Input text with different relation types
  text = "Alice works for Microsoft. Bob is married to Carol. " \
         "Dave lives in New York. Eve owns a laptop. " \
         "Frank travels to Paris frequently."
  
  # Define edge types to test
  edge_types = ["Employment", "Family", "Location", "Possession", "Action"]
  
  # Generate graph with edge types
  graph = kg_gen.generate(
    input_data=text,
    edge_type=edge_types,
    require_edge_type=True
  )
  
  # Assert edge types are present
  assert hasattr(graph, 'edge_types')
  assert graph.edge_types is not None
  assert len(graph.edge_types) > 0
  
  # Check that each edge has a type from the allowed list
  for edge, edge_type in graph.edge_types.items():
    assert edge_type in edge_types
    
  # Validate edge classification
  if "works for" in graph.edge_types:
    assert graph.edge_types["works for"] == "Employment"
  if "is married to" in graph.edge_types:
    assert graph.edge_types["is married to"] == "Family"
  if "lives in" in graph.edge_types:
    assert graph.edge_types["lives in"] == "Location"
  if "owns" in graph.edge_types:
    assert graph.edge_types["owns"] == "Possession"
  if "travels to" in graph.edge_types:
    assert graph.edge_types["travels to"] == "Action"

def test_both_node_and_edge_types():
  """Test graph generation with both node and edge types."""
  kg_gen = KGGen(
    model=TEST_MODEL,
    temperature=TEST_TEMP,
    api_key=TEST_API_KEY
  )
  
  # Input text with relationships
  text = "John, who is a researcher, works at MIT University. " \
         "MIT is located in Cambridge. Jane, a professor, teaches Computer Science at MIT. " \
         "Cambridge is a city in Massachusetts."
  
  # Define both node and edge types
  node_types = ["Person", "Organization", "Location", "Subject"]
  edge_types = ["Employment", "Location", "TeachingRole"]
  
  # Generate graph with both types
  graph = kg_gen.generate(
    input_data=text,
    node_type=node_types,
    edge_type=edge_types,
    require_node_type=True,
    require_edge_type=True
  )
  
  # Assert both types are present
  assert hasattr(graph, 'entity_types')
  assert graph.entity_types is not None
  assert hasattr(graph, 'edge_types')
  assert graph.edge_types is not None
  
  # Validate entity types
  if "John" in graph.entity_types:
    assert graph.entity_types["John"] == "Person"
  if "MIT" in graph.entity_types or "MIT University" in graph.entity_types:
    mit_entity = "MIT" if "MIT" in graph.entity_types else "MIT University"
    assert graph.entity_types[mit_entity] == "Organization"
  if "Cambridge" in graph.entity_types:
    assert graph.entity_types["Cambridge"] == "Location"
  if "Computer Science" in graph.entity_types:
    assert graph.entity_types["Computer Science"] == "Subject"
    
  # Validate edge types
  for edge, edge_type in graph.edge_types.items():
    assert edge_type in edge_types

def test_optional_type_assignment():
  """Test graph generation where entity/edge types are not required."""
  kg_gen = KGGen(
    model=TEST_MODEL,
    temperature=TEST_TEMP,
    api_key=TEST_API_KEY
  )
  
  # Input text with mixed relationships
  text = "The Earth orbits the Sun. Water consists of hydrogen and oxygen. " \
         "Photosynthesis occurs in plants. Jupiter has many moons."
  
  # Define limited types that won't cover all entities/edges
  node_types = ["Planet", "Star", "Element"]
  edge_types = ["Orbit", "Composition"]
  
  # Generate graph without requiring all entities/edges to have types
  graph = kg_gen.generate(
    input_data=text,
    node_type=node_types,
    edge_type=edge_types,
    require_node_type=False,  # Don't require all nodes to have types
    require_edge_type=False   # Don't require all edges to have types
  )
  
  # Check that we have both typed and untyped entities
  assert hasattr(graph, 'entity_types')
  assert graph.entity_types is not None
  assert len(graph.entity_types) > 0
  
  # Check that we have both typed and untyped edges
  assert hasattr(graph, 'edge_types')
  assert graph.edge_types is not None
  
  # Verify some type assignments
  if "Earth" in graph.entity_types:
    assert graph.entity_types["Earth"] == "Planet"
  if "Sun" in graph.entity_types:
    assert graph.entity_types["Sun"] == "Star"
  if "hydrogen" in graph.entity_types:
    assert graph.entity_types["hydrogen"] == "Element"
    
  # Verify some entities might not have types
  assert len(graph.entity_types) < len(graph.entities)
  
  # Verify edge types
  if "orbits" in graph.edge_types:
    assert graph.edge_types["orbits"] == "Orbit"
  if "consists of" in graph.edge_types:
    assert graph.edge_types["consists of"] == "Composition"
    
  # Verify some edges might not have types
  assert len(graph.edge_types) < len(graph.edges)

def test_chunked_processing_with_types():
  """Test chunked processing with entity and edge types."""
  kg_gen = KGGen(
    model=TEST_MODEL,
    temperature=TEST_TEMP,
    api_key=TEST_API_KEY
  )
  
  # Create a longer text that will be chunked
  chunk1 = "Mark is the CEO of Facebook. Facebook is headquartered in Menlo Park. "
  chunk2 = "Twitter was founded by Jack Dorsey. Sundar Pichai leads Google. "
  chunk3 = "Apple makes iPhones. Microsoft develops Windows operating system. "
  long_text = chunk1 + chunk2 + chunk3
  
  # Define types
  node_types = ["Person", "Company", "Product", "Location"]
  edge_types = ["Leadership", "Creation", "Location"]
  
  # Generate graph with chunking
  graph = kg_gen.generate(
    input_data=long_text,
    chunk_size=50,  # Small chunk size to force multiple chunks
    node_type=node_types,
    edge_type=edge_types,
    require_node_type=True,
    require_edge_type=True
  )
  
  # Verify entity and edge types are present across chunks
  assert hasattr(graph, 'entity_types')
  assert graph.entity_types is not None
  assert len(graph.entity_types) > 0
  
  assert hasattr(graph, 'edge_types')
  assert graph.edge_types is not None
  assert len(graph.edge_types) > 0
  
  # Check entity types from different chunks
  person_entities = [
    entity for entity, entity_type in graph.entity_types.items() 
    if entity_type == "Person"
  ]
  assert len(person_entities) >= 3  # Mark, Jack, Sundar
  
  company_entities = [
    entity for entity, entity_type in graph.entity_types.items() 
    if entity_type == "Company"
  ]
  assert len(company_entities) >= 4  # Facebook, Twitter, Google, Apple, Microsoft
  
  # Check edge types from different chunks
  leadership_edges = [
    edge for edge, edge_type in graph.edge_types.items() 
    if edge_type == "Leadership"
  ]
  assert len(leadership_edges) >= 2  # CEO of, leads
  
  # Verify that types are consistent across chunks
  if "Mark" in graph.entity_types:
    assert graph.entity_types["Mark"] == "Person"
  if "Apple" in graph.entity_types:
    assert graph.entity_types["Apple"] == "Company"

def test_clustering_with_types():
  """Test graph clustering with entity and edge types."""
  kg_gen = KGGen(
    model=TEST_MODEL,
    temperature=TEST_TEMP,
    api_key=TEST_API_KEY
  )
  
  # Input with similar entities
  text = "Dogs and cats are pets. Canines include dogs, wolves, and jackals. " \
         "Domesticated dogs are loyal to humans. Cats belong to the feline family. " \
         "Wild cats include lions and tigers."
  
  # Define types
  node_types = ["Animal", "AnimalGroup", "Person"]
  edge_types = ["Classification", "Characteristic", "Relationship"]
  
  # First generate a graph without clustering to verify types are assigned
  unclustered_graph = kg_gen.generate(
    input_data=text,
    node_type=node_types,
    edge_type=edge_types,
    cluster=False
  )
  
  # Verify unclustered graph has types
  assert hasattr(unclustered_graph, 'entity_types')
  assert unclustered_graph.entity_types is not None
  assert len(unclustered_graph.entity_types) > 0
  
  assert hasattr(unclustered_graph, 'edge_types')
  assert unclustered_graph.edge_types is not None
  assert len(unclustered_graph.edge_types) > 0
  
  # Now cluster the graph
  clustered_graph = kg_gen.cluster(unclustered_graph, context="Animal taxonomy")
  
  # Verify clustered graph has both clusters and types
  assert hasattr(clustered_graph, 'entity_types')
  assert clustered_graph.entity_types is not None
  assert len(clustered_graph.entity_types) > 0
  
  assert hasattr(clustered_graph, 'edge_types')
  assert clustered_graph.edge_types is not None
  assert len(clustered_graph.edge_types) > 0
  
  assert hasattr(clustered_graph, 'entity_clusters')
  assert clustered_graph.entity_clusters is not None
  
  assert hasattr(clustered_graph, 'edge_clusters')
  assert clustered_graph.edge_clusters is not None
  
  # Verify representative entities have types
  # Since clustering might combine entities, let's check a few key entities
  for entity, entity_type in clustered_graph.entity_types.items():
    assert entity_type in node_types
    assert entity in clustered_graph.entities
    
  # Verify representative edges have types
  for edge, edge_type in clustered_graph.edge_types.items():
    assert edge_type in edge_types
    assert edge in clustered_graph.edges
  
  # Check that some animal entity is typed correctly
  animal_entities = [entity for entity, entity_type in clustered_graph.entity_types.items() 
                     if entity_type == "Animal"]
  assert len(animal_entities) > 0
  
  # Check that some classification edge is typed correctly
  classification_edges = [edge for edge, edge_type in clustered_graph.edge_types.items() 
                         if edge_type == "Classification"]
  assert len(classification_edges) > 0

if __name__ == "__main__":
  # Setup - load environment variables if needed
  from dotenv import load_dotenv
  load_dotenv()
  
  # Run tests individually for development
  test_entities_with_node_types()
  test_relations_with_edge_types()
  test_both_node_and_edge_types()
  test_optional_type_assignment()
  test_chunked_processing_with_types()
  test_clustering_with_types()
  
  print("All tests completed successfully!")