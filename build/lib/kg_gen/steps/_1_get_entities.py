from typing import List
import dspy 

class TextEntities(dspy.Signature):
  """Extract key entities from the source text. Extracted entities are subjects or objects.
  This is for an extraction task, please be THOROUGH and accurate to the reference text."""
  
  source_text: str = dspy.InputField()  
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

class ConversationEntities(dspy.Signature):
  """Extract key entities from the conversation Extracted entities are subjects or objects.
  Consider both explicit entities and participants in the conversation.
  This is for an extraction task, please be THOROUGH and accurate."""
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

def get_entities(dspy: dspy.dspy, input_data: str, is_conversation: bool = False) -> List[str]:
  if is_conversation:
    extract = dspy.Predict(ConversationEntities)
  else:
    extract = dspy.Predict(TextEntities)
    
  result = extract(source_text=input_data)
  return result.entities

