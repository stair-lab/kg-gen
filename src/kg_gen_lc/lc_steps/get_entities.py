from typing import List

from ..langchain_runner import LangChainRunner


TEXT_ENTITIES_PROMPT = (
    "Extract key entities from the source text. Extracted entities are subjects or objects.\n"
    "This is for an extraction task, please be THOROUGH and accurate to the reference text."
)

CONVO_ENTITIES_PROMPT = (
    "Extract key entities from the conversation Extracted entities are subjects or objects.\n"
    "Consider both explicit entities and participants in the conversation.\n"
    "This is for an extraction task, please be THOROUGH and accurate."
)


def get_entities(runner: LangChainRunner, input_data: str, is_conversation: bool = False) -> List[str]:
    instruction = CONVO_ENTITIES_PROMPT if is_conversation else TEXT_ENTITIES_PROMPT
    result = runner.predict_json(instruction, {"source_text": input_data}, "entities")
    return result or []
