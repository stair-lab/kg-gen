from typing import List, Literal
from pydantic import BaseModel

from ..langchain_runner import LangChainRunner


def extraction_prompt(is_conversation: bool, context: str = "") -> str:
    if not is_conversation:
        return (
            "Extract subject-predicate-object triples from the source text.\n"
            "Subject and object must be from entities list. Entities provided were previously extracted from the same source text.\n"
            "This is for an extraction task, please be thorough, accurate, and faithful to the reference text. " + context
        )
    return (
        "Extract subject-predicate-object triples from the conversation, including:\n"
        "1. Relations between concepts discussed\n"
        "2. Relations between speakers and concepts (e.g. user asks about X)\n"
        "3. Relations between speakers (e.g. assistant responds to user)\n"
        "Subject and object must be from entities list. Entities provided were previously extracted from the same source text.\n"
        "This is for an extraction task, please be thorough, accurate, and faithful to the reference text. " + context
    )


def get_relations(runner: LangChainRunner, input_data: str, entities: List[str], is_conversation: bool = False, context: str = "") -> List[tuple[str, str, str]]:
    instruction = extraction_prompt(is_conversation, context)
    result = runner.predict_json(
        instruction,
        {"source_text": input_data, "entities": entities},
        "relations",
    )
    if not result:
        return []
    triples: List[tuple[str, str, str]] = []
    for item in result:
        if isinstance(item, dict):
            triples.append((item.get("subject"), item.get("predicate"), item.get("object")))
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            triples.append(tuple(item))
    return triples
