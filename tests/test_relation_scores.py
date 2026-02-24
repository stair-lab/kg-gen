"""Tests for relation confidence scoring and Graph.relation_scores."""

import json
import os
import sys
from pathlib import Path

# Allow importing kg_gen from source when not installed (e.g. in CI)
_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest

from kg_gen import KGGen
from kg_gen.models import Graph, _relation_to_key, _key_to_relation
from kg_gen.utils.score_relations import _parse_scores_from_response


def test_relation_key_roundtrip():
    r = ("Alice", "is mother of", "Bob")
    assert _key_to_relation(_relation_to_key(r)) == r
    key = "Alice||is mother of||Bob"
    assert _relation_to_key(_key_to_relation(key)) == key


def test_graph_relation_scores_optional():
    g = Graph(
        entities={"A", "B"},
        edges={"rel"},
        relations={("A", "rel", "B")},
    )
    assert g.relation_scores is None
    assert g.get_relation_score(("A", "rel", "B")) is None


def test_graph_get_relation_score():
    g = Graph(
        entities={"A", "B", "C"},
        edges={"r1", "r2"},
        relations={("A", "r1", "B"), ("B", "r2", "C")},
        relation_scores={"A||r1||B": 0.9, "B||r2||C": 0.5},
    )
    assert g.get_relation_score(("A", "r1", "B")) == 0.9
    assert g.get_relation_score(("B", "r2", "C")) == 0.5
    assert g.get_relation_score(("X", "r1", "Y")) is None


def test_parse_scores_from_response():
    assert _parse_scores_from_response("0.9, 0.7, 0.3", 3) == [0.9, 0.7, 0.3]
    assert _parse_scores_from_response("0.9\n0.7\n0.3", 3) == [0.9, 0.7, 0.3]
    assert _parse_scores_from_response("1.0, 0.0", 2) == [1.0, 0.0]
    # Pad if fewer
    assert _parse_scores_from_response("0.8", 3) == [0.8, 0.5, 0.5]
    # Truncate if more
    assert _parse_scores_from_response("0.1, 0.2, 0.3, 0.4", 2) == [0.1, 0.2]
    # Clamp to [0, 1] (negative and >1)
    got = _parse_scores_from_response("1.5, -0.1", 2)
    assert got[0] == 1.0 and got[1] == 0.0


def test_aggregate_preserves_relation_scores():
    g1 = Graph(
        entities={"A", "B"},
        edges={"r"},
        relations={("A", "r", "B")},
        relation_scores={"A||r||B": 0.8},
    )
    g2 = Graph(
        entities={"C", "D"},
        edges={"s"},
        relations={("C", "s", "D")},
        relation_scores={"C||s||D": 0.6},
    )
    kg = KGGen(
        model="no-model",
        api_key="no-key",
        temperature=0.0,
    )
    combined = kg.aggregate([g1, g2])
    assert combined.relation_scores is not None
    assert combined.get_relation_score(("A", "r", "B")) == 0.8
    assert combined.get_relation_score(("C", "s", "D")) == 0.6


def test_aggregate_relation_scores_take_max():
    g1 = Graph(
        entities={"A", "B"},
        edges={"r"},
        relations={("A", "r", "B")},
        relation_scores={"A||r||B": 0.7},
    )
    g2 = Graph(
        entities={"A", "B"},
        edges={"r"},
        relations={("A", "r", "B")},
        relation_scores={"A||r||B": 0.9},
    )
    kg = KGGen(model="no-model", api_key="no-key", temperature=0.0)
    combined = kg.aggregate([g1, g2])
    assert combined.get_relation_score(("A", "r", "B")) == 0.9


def test_export_import_relation_scores(tmp_path):
    g = Graph(
        entities={"A", "B"},
        edges={"r"},
        relations={("A", "r", "B")},
        relation_scores={"A||r||B": 0.85},
    )
    path = tmp_path / "graph.json"
    KGGen.export_graph(g, str(path))
    with open(path) as f:
        loaded = Graph.model_validate(json.load(f))
    assert loaded.relation_scores == {"A||r||B": 0.85}
    assert loaded.get_relation_score(("A", "r", "B")) == 0.85


@pytest.mark.skipif(
    not os.getenv("LLM_API_KEY"),
    reason="LLM_API_KEY not set; skipping integration test",
)
def test_score_relations_integration():
    """Requires LLM_API_KEY. Scores a small graph and checks structure."""
    from fixtures import kg

    text = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
    graph = kg.generate(input_data=text)
    assert len(graph.relations) >= 1
    scored = kg.score_relations(graph, source_text=text, batch_size=5)
    assert scored.relation_scores is not None
    assert len(scored.relation_scores) == len(scored.relations)
    for rel in scored.relations:
        s = scored.get_relation_score(rel)
        assert s is not None
        assert 0.0 <= s <= 1.0
