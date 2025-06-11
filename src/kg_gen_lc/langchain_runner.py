import json
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class LangChainRunner:
    """Simple wrapper around ChatOpenAI for structured prompts."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, api_key: str | None = None, api_base: str | None = None):
        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base=api_base,
        )

    def predict_json(self, instruction: str, inputs: Dict[str, Any], output_key: str):
        """Invoke LLM with instruction and return parsed JSON output."""
        messages = [
            SystemMessage(content=f"{instruction}\nReturn a JSON object with key '{output_key}'."),
            HumanMessage(content=json.dumps(inputs)),
        ]
        response = self.llm.invoke(messages)
        text = response.content
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            text = match.group()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return data.get(output_key)
