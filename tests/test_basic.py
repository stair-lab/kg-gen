from src.kg_gen import KGGen
import os
from dotenv import load_dotenv

from tests.utils.visualize_kg import visualize

if __name__ == "__main__":
  # Load environment variables
  load_dotenv()
  
  # Example usage
  kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"))
  
  # Generate a simple graph
  text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."
  
  graph = kg.generate(
    input_data=text,
    model="openai/gpt-4o"
  )

  visualize(graph, "tests/test_basic.html")

  print(graph)