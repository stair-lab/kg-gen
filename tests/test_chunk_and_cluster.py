from src.kg_gen import KGGen
import os
from dotenv import load_dotenv

from tests.utils.visualize_kg import visualize

if __name__ == "__main__":
  # Load environment variables
  load_dotenv()
  
  # Example usage
  kg = KGGen()
  
  # Load fresh wiki content
  with open('tests/data/fresh_wiki_article.md', 'r', encoding='utf-8') as f:
    text = f.read()
    
  # # Generate graph from wiki text with chunking
  graph = kg.generate(
    input_data=text,
    model="openai/gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    chunk_size=5000,
    cluster=True
  )
  print("Entities:", graph.entities)
  print("Edges:", graph.edges) 
  print("Relations:", graph.relations)
  
  visualize(graph, "tests/test_chunk_and_cluster.html")