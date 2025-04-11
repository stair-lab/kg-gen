from typing import List
import dspy 

class TextEntities(dspy.Signature):
  """Extract a comprehensive set of meaningful entities from the source text.
  
  GUIDELINES:
  1. ENTITY TYPES: Include people, organizations, locations, concepts, products, time periods, events, and technical terms
  2. GRANULARITY: Choose entities at the right level of granularity (e.g., "New York City" not just "city")
  3. NAMING: Use full, proper names where available ("United Nations" not "UN")
  4. FORMAT: Return clean entity strings without extraneous punctuation
  5. THOROUGHNESS: Be exhaustive - capture ALL relevant entities
  6. PRECISION: Each entity should represent a distinct, well-defined concept
  7. CONSISTENCY: Use consistent naming conventions for similar entities
  8.  NEO4J COMPATIBILITY: 
     - Use camelCase or snake_case for relationship types
     - Avoid spaces, special characters and starting with numbers
     - Use standardized relationship types like:
       * for people: "wasBornIn", "worksAt", "isMarriedTo", "discovered", "invented", "founded", etc.
       * for organizations: "isLocatedIn", "hasHeadquartersIn", "wasFoundedBy", "employs", etc.
       * for places: "isPartOf", "contains", "isBorderingWith", etc.
       * for concepts: "isTypeOf", "includes", "belongsTo", etc.
       * for events: "occurredIn", "involvedPerson", "tookPlaceAt", "happenedOn", etc.
  
  AVOID:
  - Generic terms that aren't meaningful on their own (e.g., "many", "some")
  - Overly vague concepts unless specifically mentioned as important
  - Duplicate entities with different spellings
  - Pronouns - resolve to their actual entities
  
  This is for an information extraction task - accuracy and completeness are critical."""
  
  source_text: str = dspy.InputField()  
  context: str = dspy.InputField(desc="Domain context to guide extraction (if any)")
  entities: list[str] = dspy.OutputField(desc="Comprehensive list of distinct, meaningful entities")

class ConversationEntities(dspy.Signature):
  """Extract a comprehensive set of meaningful entities from the conversation.
  
  GUIDELINES:
  1. ENTITY TYPES: Include:
     - PARTICIPANTS: All speakers/participants in the conversation
     - TOPICS: Main subjects discussed (people, organizations, products, concepts, etc.)
     - REFERENCES: Any entities referenced by participants
     - METADATA: Dates, locations, events mentioned
  
  2. CONVERSATION SPECIFICS:
     - Capture entities implied by conversational context
     - Include entities that represent key user intents or questions
     - Identify entities representing system capabilities or features mentioned
  
  3. NAMING & FORMAT:
     - Use full, proper names where available ("Artificial Intelligence" not just "AI")
     - For multi-turn conversations, ensure consistent entity naming across turns
     - Return clean entity strings without extraneous punctuation
  
  AVOID:
  - Generic conversational fillers ("hello", "thanks")
  - Overly abstract concepts unless specifically important to the conversation
  - Duplicate entities with different spellings
  
  This is for an information extraction task - accuracy and completeness are critical."""
  
  source_text: str = dspy.InputField()
  context: str = dspy.InputField(desc="Domain context to guide extraction (if any)")
  entities: list[str] = dspy.OutputField(desc="Comprehensive list of distinct, meaningful entities")

def get_entities(dspy: dspy.dspy, input_data: str, context: str = "", is_conversation: bool = False) -> List[str]:
  if is_conversation:
    extract = dspy.Predict(ConversationEntities)
  else:
    extract = dspy.Predict(TextEntities)
    
  result = extract(source_text=input_data, context=context)
  return result.entities

