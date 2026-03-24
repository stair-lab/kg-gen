from typing import List, Tuple, Optional, Literal, Type
from pathlib import Path
import json
import dspy
import litellm
from pydantic import BaseModel, create_model, ValidationError


def parse_relations_response(
    raw_json: str,
    entities: List[str],
    response_model: Optional[Type[BaseModel]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Parse a relations JSON response with graceful fallback.

    First attempts strict Pydantic validation. If that fails (e.g., due to
    EntityLiteral validation), falls back to raw JSON parsing and filters
    out items with invalid subject/object.

    Args:
        raw_json: The raw JSON string from the LLM response
        entities: List of valid entity strings
        response_model: Optional Pydantic model for strict validation

    Returns:
        List of (subject, predicate, object) tuples with valid entities
    """
    entities_set = set(entities)

    # Try strict Pydantic validation first if model provided
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(raw_json)
            return [(r.subject, r.predicate, r.object) for r in parsed.relations]
        except ValidationError:
            pass  # Fall through to JSON parsing

    # Fallback: parse as raw JSON and filter
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    # Handle both {"relations": [...]} and direct list formats
    items = data.get("relations", data) if isinstance(data, dict) else data

    if not isinstance(items, list):
        return []

    relations = []
    for item in items:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")

        # Skip if missing required fields
        if not all([subject, predicate, obj]):
            continue

        # Skip if subject or object not in valid entities
        if subject not in entities_set or obj not in entities_set:
            continue

        relations.append((subject, predicate, obj))

    return relations


def _load_relations_prompt() -> str:
    """Load the relations prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "relations.txt"
    return prompt_path.read_text()


def _create_relations_model(entities: List[str]):
    """Dynamically create Pydantic models with entity literals for subject/object."""
    # Create a Literal type from the entities list
    EntityLiteral = Literal[tuple(entities)]  # type: ignore

    # Create RelationItem with constrained subject/object
    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),
        predicate=(str, ...),
        object=(EntityLiteral, ...),
    )

    # Create RelationsResponse containing list of RelationItem
    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )

    return RelationItem, RelationsResponse


def _get_relations_litellm(
    input_data: str,
    entities: List[str],
    model: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[Tuple[str, str, str]]:
    prompt_template = _load_relations_prompt()
    entities_str = "\n".join(f"- {e}" for e in entities)
    user_prompt = f"""
Here is the list of entities that were previously extracted from the source text:

<entities>
{entities_str}
</entities>

Here is the source text to analyze:

<text>
{input_data}
</text>
    """

    # Create dynamic model with entity constraints
    _, RelationsResponse = _create_relations_model(entities)

    # Build schema with additionalProperties: false (required by OpenAI)
    schema = RelationsResponse.model_json_schema()
    schema["additionalProperties"] = False
    # Also need to set additionalProperties on nested objects
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "relations_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    raw_json = response.output[-1].content[0].text
    return parse_relations_response(raw_json, entities, RelationsResponse)


def extraction_sig(
    Relation: BaseModel, is_conversation: bool, context: str = ""
) -> dspy.Signature:
    if not is_conversation:

        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the source text. 
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples. Be thorough."
            )

        return ExtractTextRelations
    else:

        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"""Extract subject-predicate-object triples from the conversation, including:
      1. Relations between concepts discussed
      2. Relations between speakers and concepts (e.g. user asks about X)
      3. Relations between speakers (e.g. assistant responds to user)
      Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
      This is for an extraction task, please be thorough, accurate, and faithful to the reference text. {context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough"
            )

        return ExtractConversationRelations


def fallback_extraction_sig(
    entities, is_conversation, context: str = ""
) -> dspy.Signature:
    """This fallback extraction does not strictly type the subject and object strings."""

    entities_str = "\n- ".join(entities)

    class Relation(BaseModel):
        # TODO: should use literal's here instead.
        __doc__ = f"""Knowledge graph subject-predicate-object tuple. Subject and object entities must be one of: {entities_str}"""

        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    return Relation, extraction_sig(Relation, is_conversation, context)


def _filter_entities(entities: List[str]) -> List[str]:
    """Filter out entities that contain backslashes."""
    return [e for e in entities if '"' not in e] # not received by oai api


def get_relations(
    input_data: str,
    entities: list[str],
    is_conversation: bool = False,
    context: str = "",
    use_litellm_prompt: bool = False,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[Tuple[str, str, str]]:
    # Filter out entities containing backslashes
    entities = _filter_entities(entities)

    if use_litellm_prompt and not is_conversation:
        return _get_relations_litellm(
            input_data,
            entities,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
        )

    class Relation(BaseModel):
        """Knowledge graph subject-predicate-object tuple."""

        subject: str = dspy.InputField(desc="Subject entity", examples=["Kevin"])
        predicate: str = dspy.InputField(desc="Predicate", examples=["is brother of"])
        object: str = dspy.InputField(desc="Object entity", examples=["Vicky"])

    ExtractRelations = extraction_sig(Relation, is_conversation, context)

    try:
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        return [(r.subject, r.predicate, r.object) for r in result.relations]

    except Exception as _:
        # print("get_relations: fallback extraction")
        Relation, ExtractRelations = fallback_extraction_sig(
            entities, is_conversation, context
        )
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)

        class FixedRelations(dspy.Signature):
            """Fix the relations so that every subject and object of the relations are exact matches to an entity. Keep the predicate the same. The meaning of every relation should stay faithful to the reference text. If you cannot maintain the meaning of the original relation relative to the source text, then do not return it."""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.InputField()
            fixed_relations: list[Relation] = dspy.OutputField()

        fix = dspy.ChainOfThought(FixedRelations)

        fix_res = fix(
            source_text=input_data, entities=entities, relations=result.relations
        )

        good_relations = []
        for rel in (fix_res.fixed_relations or []):
            if rel.subject in entities and rel.object in entities:
                good_relations.append(rel)
        return [(r.subject, r.predicate, r.object) for r in good_relations]
