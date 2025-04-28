import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # uncomment this line if you clone the repo package instead of using pip install
exit(0)
from src.kg_gen import KGGen
from dotenv import load_dotenv


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("API_BASE")


if __name__ == "__main__":
  # Load environment variables
  load_dotenv()
  
  # Example usage
  kg = KGGen(api_key=OPENAI_API_KEY, api_base=API_BASE, model = "gpt-4o-mini")
  
  # Generate a simple graph
  text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."
  
  graph = kg.generate(
    input_data=text,
    api_key=OPENAI_API_KEY, 
    api_base=API_BASE,
    model="gpt-4o-mini",
  )
  print(graph)
