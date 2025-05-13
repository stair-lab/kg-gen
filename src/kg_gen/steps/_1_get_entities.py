from typing import List, Optional, Dict, Tuple
import dspy 
from pydantic import BaseModel

class EntityWithType(BaseModel):
  entity: str
  type: Optional[str] = None

class TextEntities(dspy.Signature):
  """Extract key entities from the source text. Extracted entities are subjects or objects.
  This is for an extraction task, please be THOROUGH and accurate to the reference text."""
  
  source_text: str = dspy.InputField()  
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

class TextEntitiesWithTypes(dspy.Signature):
  """Extract key entities from the source text and assign types to them.
  Extracted entities are subjects or objects.
  This is for an extraction task, please be THOROUGH and accurate to the reference text."""
  
  source_text: str = dspy.InputField()
  node_types: list[str] = dspy.InputField(desc="List of allowed entity types")
  require_type: bool = dspy.InputField(desc="Whether every entity must have a type from node_types")
  entities: list[dict] = dspy.OutputField(desc="THOROUGH list of entities with their types as {entity: str, type: str}")

class ConversationEntities(dspy.Signature):
  """Extract key entities from the conversation Extracted entities are subjects or objects.
  Consider both explicit entities and participants in the conversation.
  This is for an extraction task, please be THOROUGH and accurate."""
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

class ConversationEntitiesWithTypes(dspy.Signature):
  """Extract key entities from the conversation and assign types to them.
  Extracted entities are subjects or objects. Consider both explicit entities and participants.
  This is for an extraction task, please be THOROUGH and accurate."""
  
  source_text: str = dspy.InputField()
  node_types: list[str] = dspy.InputField(desc="List of allowed entity types")
  require_type: bool = dspy.InputField(desc="Whether every entity must have a type from node_types")
  entities: list[dict] = dspy.OutputField(desc="THOROUGH list of entities with their types as {entity: str, type: str}")

def get_entities(
  dspy: dspy.dspy,
  input_data: str,
  is_conversation: bool = False,
  node_types: Optional[List[str]] = None,
  require_node_type: bool = True
) -> Tuple[List[str], Optional[Dict[str, str]]]:
  """Extract entities from the input data.
  
  Args:
      dspy: DSPy module
      input_data: Text to process
      is_conversation: Whether the input is a conversation
      node_types: Optional list of allowed node types
      require_node_type: Whether every node must have a type from node_types
      
  Returns:
      Tuple of (list of entity strings, dictionary mapping entities to types)
  """
  if node_types:
    # Extract entities with types
    if is_conversation:
      extract = dspy.Predict(ConversationEntitiesWithTypes)
      result = extract(
        source_text=input_data,
        node_types=node_types,
        require_type=require_node_type
      )
    else:
      extract = dspy.Predict(TextEntitiesWithTypes)
      result = extract(
        source_text=input_data,
        node_types=node_types,
        require_type=require_node_type
      )
    
    # Extract entity strings and type mapping
    entities = []
    entity_types = {}
    
    for item in result.entities:
      entity = item["entity"]
      entities.append(entity)
      if "type" in item and item["type"]:
        entity_types[entity] = item["type"]
    
    return entities, entity_types
  else:
    # Standard entity extraction without types
    if is_conversation:
      extract = dspy.Predict(ConversationEntities)
    else:
      extract = dspy.Predict(TextEntities)
      
    result = extract(source_text=input_data)
    return result.entities, None

