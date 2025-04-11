from typing import List
import dspy

class TextRelations(dspy.Signature):
  """Extract subject-predicate-object triples from the source text. Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
  This is for an extraction task, please be THOROUGH, accurate, and faithful to the reference text."""
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.InputField()
  relations: list[tuple[str, str, str]] = dspy.OutputField(desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. BE THOROUGH")

class ConversationRelations(dspy.Signature):
  """Extract subject-predicate-object triples from the conversation, including:
  1. Relations between concepts discussed
  2. Relations between speakers and concepts (e.g. user asks about X)
  3. Relations between speakers (e.g. assistant responds to user)
  Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
  This is for an extraction task, please be THOROUGH, accurate, and faithful to the reference text.
  """
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.InputField()
  relations: list[tuple[str, str, str]] = dspy.OutputField(desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. BE THOROUGH")

def get_relations(dspy: dspy.dspy, input_data: str, entities: list[str], is_conversation: bool = False) -> List[str]:
  if is_conversation:
    extract = dspy.Predict(ConversationRelations)
  else:
    extract = dspy.Predict(TextRelations)
    
  result = extract(source_text=input_data, entities=entities)
  filtered_relations = [
    (s, p, o) for s, p, o in result.relations 
    if s in entities and o in entities
  ]
  return filtered_relations