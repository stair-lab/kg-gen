from src.kg_gen import KGGen 
import os

from tests.utils.visualize_kg import visualize

if __name__ == "__main__":
  # Load environment variables
  
  # Initialize KGGen
  kg = KGGen()
  
  # Test texts
  text1 = "Linda is Joshua's mother. Ben is Josh's brother. Andrew is Josh's father."
  text2 = "Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt. Josh also goes by Joshua."
  
  # Generate individual graphs
  graph1 = kg.generate(
    input_data=text1,
    model="openai/gpt-4o",
    context="Family relationships"
  )
  
  graph2 = kg.generate(
    input_data=text2, 
    model="openai/gpt-4o",
    context="Family relationships"
  )
  
  # # Aggregate the graphs
  combined_graph = kg.aggregate([graph1, graph2])
  
  # Cluster the combined graph
  clustered_graph = kg.cluster(
    combined_graph,
    context="Family relationships",
    model="openai/gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
  )
  
  # Print results
  print("\nGraph 1:")
  print("Entities:", graph1.entities)
  print("Relations:", graph1.relations)
  print("Edges:", graph1.edges)
  
  print("\nGraph 2:")
  print("Entities:", graph2.entities)
  print("Relations:", graph2.relations) 
  print("Edges:", graph2.edges)
  
  print("\nCombined Graph:")
  print("Entities:", combined_graph.entities)
  print("Relations:", combined_graph.relations)
  print("Edges:", combined_graph.edges)

  print("\nClustered Combined Graph:")
  print("Entities:", clustered_graph.entities)
  print("Relations:", clustered_graph.relations)
  print("Edges:", clustered_graph.edges)
  print("Entity Clusters:", clustered_graph.entity_clusters)
  print("Edge Clusters:", clustered_graph.edge_clusters)
  
  visualize(clustered_graph, "tests/test_basic_2.html")
  