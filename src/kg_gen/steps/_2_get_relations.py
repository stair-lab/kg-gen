from typing import List, Literal, Optional, Dict, Tuple
import dspy
from pydantic import BaseModel, create_model


def get_relations(
  dspy: dspy.dspy,
  input_data: str,
  entities: list[str],
  is_conversation: bool = False,
  edge_types: Optional[List[str]] = None,
  require_edge_type: bool = True
) -> Tuple[List[Tuple[str, str, str]], Optional[Dict[str, str]]]:
  """Extract relations from input data.
  
  Args:
      dspy: DSPy module
      input_data: Text to process
      entities: List of entities to relate
      is_conversation: Whether the input is a conversation
      edge_types: Optional list of allowed edge types
      require_edge_type: Whether every edge must have a type from edge_types
      
  Returns:
      Tuple of (list of relation tuples, dictionary mapping predicates to types)
  """
  class Relation(BaseModel):
    "Knowledge graph subject-predicate-object tuple"
    subject: Literal[tuple(entities)]
    predicate: str
    object: Literal[tuple(entities)]
    
  class RelationWithType(BaseModel):
    "Knowledge graph subject-predicate-object tuple with predicate type"
    subject: Literal[tuple(entities)]
    predicate: str
    predicate_type: Optional[str]
    object: Literal[tuple(entities)]
  
  if edge_types:
    # Define signatures for extracting relations with types
    class ExtractTextRelationsWithTypes(dspy.Signature):
      """Extract subject-predicate-object triples from the source text and assign types to predicates.
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      edge_types: list[str] = dspy.InputField(desc="List of allowed predicate types")
      require_type: bool = dspy.InputField(desc="Whether every predicate must have a type from edge_types")
      relations: list[RelationWithType] = dspy.OutputField(desc="List of subject-predicate-object tuples with predicate types")

    class ExtractConversationRelationsWithTypes(dspy.Signature):
      """Extract subject-predicate-object triples from the conversation and assign types to predicates.
      Include relations between concepts, speakers and concepts, and between speakers.
      Subject and object must be from entities list.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      edge_types: list[str] = dspy.InputField(desc="List of allowed predicate types")
      require_type: bool = dspy.InputField(desc="Whether every predicate must have a type from edge_types")
      relations: list[RelationWithType] = dspy.OutputField(desc="List of subject-predicate-object tuples with predicate types")

    # Extract relations with types
    if is_conversation:
      extract = dspy.Predict(ExtractConversationRelationsWithTypes)
      result = extract(
        source_text=input_data,
        entities=entities,
        edge_types=edge_types,
        require_type=require_edge_type
      )
    else:
      extract = dspy.Predict(ExtractTextRelationsWithTypes)
      result = extract(
        source_text=input_data,
        entities=entities,
        edge_types=edge_types,
        require_type=require_edge_type
      )

    # Process and filter relations with types
    filtered_relations = []
    edge_type_map = {}
    
    for rel in result.relations:
      if rel.subject in entities and rel.object in entities:
        filtered_relations.append((rel.subject, rel.predicate, rel.object))
        if hasattr(rel, 'predicate_type') and rel.predicate_type:
          edge_type_map[rel.predicate] = rel.predicate_type
    
    return filtered_relations, edge_type_map
    
  else:
    # Standard relation extraction without types
    class ExtractTextRelations(dspy.Signature):
      """Extract subject-predicate-object triples from the source text. Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples. Be thorough.")

    class ExtractConversationRelations(dspy.Signature):
      """Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text.
      """
      
      source_text: str = dspy.InputField()
      entities: list[str] = dspy.InputField()
      relations: list[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough")
      
    if is_conversation:
      extract = dspy.Predict(ExtractConversationRelations)
    else:
      extract = dspy.Predict(ExtractTextRelations)
      
    result = extract(source_text=input_data, entities=entities)
    
    # Filter relations
    filtered_relations = [
      (rel.subject, rel.predicate, rel.object) for rel in result.relations 
      if rel.subject in entities and rel.object in entities
    ]
    
    return filtered_relations, None