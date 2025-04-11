from typing import List
import dspy

class TextRelations(dspy.Signature):
  """Extract precise subject-predicate-object triples from the source text.
  
  GUIDELINES:
  1. RELATIONSHIP TYPES:
     - ACTIONS: What entities do ("Person" -> "wrote" -> "Book")
     - ATTRIBUTES: Properties of entities ("Person" -> "has age" -> "35")
     - CONNECTIONS: How entities relate ("Person" -> "is employee of" -> "Company")
     - CLASSIFICATIONS: Types/categories ("Item" -> "is a type of" -> "Category")
     - TEMPORAL: Time-based relationships ("Event" -> "occurred in" -> "Year")
     - SPATIAL: Location-based relationships ("Entity" -> "is located in" -> "Place")
  
  2. PREDICATE QUALITY:
     - Use SPECIFIC, DESCRIPTIVE predicates - "is CEO of" not just "is"
     - Use DIRECTIONAL relationships - pay attention to which entity is subject vs object
     - Use CONSISTENT TENSE - prefer present tense unless clearly past
     - Use NATURAL LANGUAGE phrases that form readable sentences when combined with subject/object
  
  3. ACCURACY:
     - ONLY create relationships explicitly supported by the text
     - Use EXACT entity matches from the provided entities list
     - Ensure the relationship direction accurately reflects the text
  
  4. THOROUGHNESS:
     - Capture ALL meaningful relationships between entities
     - Include INDIRECT relationships implied by direct statements
  
  AVOID:
  - Vague predicates ("is related to", "has", "involves")
  - Speculative relationships not supported by the text
  - Redundant triples that convey the same information
  
  This is for an information extraction task - accuracy and completeness are critical."""
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.InputField()
  context: str = dspy.InputField(desc="Domain context to guide extraction (if any)")
  relations: list[tuple[str, str, str]] = dspy.OutputField(desc="Comprehensive list of subject-predicate-object triples where subject and object are from the entities list, and predicate is a specific, descriptive relationship")

class ConversationRelations(dspy.Signature):
  """Extract precise subject-predicate-object triples from the conversation.
  
  GUIDELINES:
  1. CONVERSATION-SPECIFIC RELATIONSHIPS:
     - DIALOGUE FLOW: ("Speaker1" -> "asked about" -> "Topic")
     - INFORMATION EXCHANGE: ("Speaker2" -> "provided information on" -> "Topic")
     - OPINIONS: ("Speaker1" -> "expressed preference for" -> "Option")
     - INTENT: ("User" -> "wanted to know" -> "Information")
     - SYSTEM ACTIONS: ("Assistant" -> "recommended" -> "Product")
  
  2. CONTENT RELATIONSHIPS:
     - Extract relationships between the TOPICS discussed
     - Capture how entities within the conversation relate to each other
     - Include relationships implied by the conversational context
  
  3. PREDICATE QUALITY:
     - Use SPECIFIC, DESCRIPTIVE predicates - "strongly disagreed with" not just "responded to"
     - Use DIRECTIONAL relationships - pay attention to which entity is subject vs object
     - Use NATURAL LANGUAGE phrases that form readable sentences when combined with subject/object
  
  4. ACCURACY & THOROUGHNESS:
     - ONLY create relationships supported by the conversation
     - Use EXACT entity matches from the provided entities list
     - Capture ALL meaningful relationships between entities
  
  AVOID:
  - Generic conversational predicates ("said", "replied")
  - Relationships not supported by the conversation content
  - Redundant triples that convey the same information
  
  This is for an information extraction task - accuracy and completeness are critical."""
  
  source_text: str = dspy.InputField()
  entities: list[str] = dspy.InputField()
  context: str = dspy.InputField(desc="Domain context to guide extraction (if any)")
  relations: list[tuple[str, str, str]] = dspy.OutputField(desc="Comprehensive list of subject-predicate-object triples where subject and object are from the entities list, and predicate is a specific, descriptive relationship")

def get_relations(dspy: dspy.dspy, input_data: str, entities: list[str], context: str = "", is_conversation: bool = False) -> List[tuple]:
  if is_conversation:
    extract = dspy.Predict(ConversationRelations)
  else:
    extract = dspy.Predict(TextRelations)
    
  result = extract(source_text=input_data, entities=entities, context=context)
  filtered_relations = [
    (s, p, o) for s, p, o in result.relations 
    if s in entities and o in entities
  ]
  return filtered_relations