"""
Relation confidence scoring for knowledge graphs.

Scores each (subject, predicate, object) triple with a value in [0.0, 1.0]
indicating how well the relation is supported by the source text. Useful for
RAG filtering, evaluation at different confidence thresholds, and uncertainty-aware
downstream use (e.g. human-in-the-loop review of low-confidence triples).
"""

from __future__ import annotations

import logging
import re
from typing import Tuple, List, Optional

import dspy

from kg_gen.models import Graph, _relation_to_key

logger = logging.getLogger(__name__)


def _parse_scores_from_response(s: str, expected_count: int) -> List[float]:
    """Parse a list of floats from LM response. Handles comma/space/newline separated."""
    scores: List[float] = []
    # Match numbers (optional minus, digits, optional decimal). Clamp to [0, 1] below.
    for m in re.finditer(r"-?\d*\.?\d+", s):
        try:
            v = float(m.group())
            scores.append(max(0.0, min(1.0, v)))
        except ValueError:
            continue
    while len(scores) < expected_count:
        scores.append(0.5)
    return scores[:expected_count]


def _score_relations_batch(
    lm: dspy.LM,
    source_text: str,
    relations: List[Tuple[str, str, str]],
    context: str = "",
) -> List[float]:
    """Score a batch of relations given source text. Returns list of floats in [0, 1]."""
    if not relations:
        return []

    rel_strs = [f"({s}, {p}, {o})" for s, p, o in relations]
    relations_block = "\n".join(f"- {r}" for r in rel_strs)

    class ScoreRelations(dspy.Signature):
        __doc__ = f"""Given the source text and a list of knowledge graph relations (subject, predicate, object),
rate your confidence that each relation is correctly extracted and supported by the source text.
Output exactly one confidence score from 0.0 to 1.0 for each relation, in the same order as the list.
Use only numbers. Format: comma-separated or one per line, e.g. 0.9, 0.7, 0.3.
Be strict: only relations clearly stated or strongly implied in the text should get high scores.
{context}"""

        source_text: str = dspy.InputField(
            desc="The original text the relations were extracted from"
        )
        relations_list: str = dspy.InputField(desc="List of relations, one per line")
        scores: str = dspy.OutputField(
            desc="One confidence score per relation, same order, values between 0.0 and 1.0, comma or newline separated"
        )

    with dspy.context(lm=lm):
        predictor = dspy.Predict(ScoreRelations)
        result = predictor(
            source_text=source_text[:8000],
            relations_list=relations_block,
        )

    raw = getattr(result, "scores", "") or ""
    return _parse_scores_from_response(raw, len(relations))


def run_relation_scoring(
    lm: dspy.LM,
    graph: Graph,
    source_text: str,
    batch_size: int = 15,
    context: str = "",
) -> Graph:
    """
    Attach confidence scores to each relation in the graph using the LM.

    Args:
        lm: DSPy language model.
        graph: Knowledge graph whose relations to score.
        source_text: Original text the graph was extracted from (used as evidence).
        batch_size: Number of relations to score per LM call.
        context: Optional context to guide scoring (e.g. domain).

    Returns:
        New Graph with relation_scores populated (same entities/relations/edges).
    """
    relations = list(graph.relations)
    if not relations:
        return Graph(
            entities=graph.entities,
            relations=graph.relations,
            edges=graph.edges,
            entity_clusters=graph.entity_clusters,
            edge_clusters=graph.edge_clusters,
            entity_metadata=graph.entity_metadata,
            relation_scores={},
        )

    relation_scores: dict[str, float] = {}
    for i in range(0, len(relations), batch_size):
        batch = relations[i : i + batch_size]
        try:
            batch_scores = _score_relations_batch(lm, source_text, batch, context=context)
            for rel, score in zip(batch, batch_scores):
                relation_scores[_relation_to_key(rel)] = score
        except Exception as e:
            logger.warning("Relation scoring batch failed: %s. Assigning 0.5 to batch.", e)
            for rel in batch:
                relation_scores[_relation_to_key(rel)] = 0.5

    return Graph(
        entities=graph.entities,
        relations=graph.relations,
        edges=graph.edges,
        entity_clusters=graph.entity_clusters,
        edge_clusters=graph.edge_clusters,
        entity_metadata=graph.entity_metadata,
        relation_scores=relation_scores,
    )
