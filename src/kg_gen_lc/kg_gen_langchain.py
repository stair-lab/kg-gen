from typing import Union, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import os
import json

from .langchain_runner import LangChainRunner
from .lc_steps.get_entities import get_entities
from .lc_steps.get_relations import get_relations
from .lc_steps.cluster_graph import cluster_graph
from kg_gen.utils.chunk_text import chunk_text
from kg_gen.models import Graph


class KGGenLangChain:
    """KGGen implementation using LangChain instead of DSPy."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0, api_key: str | None = None, api_base: str | None = None):
        self.runner = LangChainRunner(model=model, temperature=temperature, api_key=api_key, api_base=api_base)
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base

    def init_model(self, model: str | None = None, temperature: float | None = None, api_key: str | None = None, api_base: str | None = None):
        if model is not None:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        self.runner = LangChainRunner(model=self.model, temperature=self.temperature, api_key=self.api_key, api_base=self.api_base)

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        cluster: bool = False,
        temperature: float | None = None,
        output_folder: Optional[str] = None,
    ) -> Graph:
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            text_content = []
            for message in input_data:
                if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                    raise ValueError("Messages must be dicts with 'role' and 'content' keys")
                if message['role'] in ['user', 'assistant']:
                    text_content.append(f"{message['role']}: {message['content']}")
            processed_input = "\n".join(text_content)
        else:
            processed_input = input_data

        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )

        if not chunk_size:
            entities = get_entities(self.runner, processed_input, is_conversation=is_conversation)
            relations = get_relations(self.runner, processed_input, entities, is_conversation=is_conversation, context=context)
        else:
            chunks = chunk_text(processed_input, chunk_size)
            entities = set()
            relations = set()

            def process_chunk(chunk: str):
                chunk_entities = get_entities(self.runner, chunk, is_conversation=is_conversation)
                chunk_relations = get_relations(self.runner, chunk, chunk_entities, is_conversation=is_conversation, context=context)
                return chunk_entities, chunk_relations

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_chunk, chunks))

            for chunk_entities, chunk_relations in results:
                entities.update(chunk_entities)
                relations.update(chunk_relations)

        graph = Graph(
            entities=entities,
            relations=relations,
            edges={rel[1] for rel in relations},
        )

        if cluster:
            graph = self.cluster(graph, context)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, 'graph.json')
            graph_dict = {
                'entities': list(graph.entities),
                'relations': list(graph.relations),
                'edges': list(graph.edges),
            }
            with open(output_path, 'w') as f:
                json.dump(graph_dict, f, indent=2)

        return graph

    def cluster(
        self,
        graph: Graph,
        context: str = "",
        model: str | None = None,
        temperature: float | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> Graph:
        if any([model, temperature, api_key, api_base]):
            self.init_model(
                model=model or self.model,
                temperature=temperature or self.temperature,
                api_key=api_key or self.api_key,
                api_base=api_base or self.api_base,
            )
        return cluster_graph(self.runner, graph, context)

    def aggregate(self, graphs: list[Graph]) -> Graph:
        all_entities = set()
        all_relations = set()
        all_edges = set()
        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)
        return Graph(entities=all_entities, relations=all_relations, edges=all_edges)
