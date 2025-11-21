"""
Integration tests for deduplication in the KGGen pipeline.
"""

import pytest
from src.kg_gen import KGGen
from src.kg_gen.models import Graph
from src.kg_gen.steps._3_deduplicate import (
    DeduplicationStrategy,
    _apply_deterministic_deduplication_to_graph,
)


class TestDeduplicationIntegration:
    """Test that deduplication strategies work in the pipeline."""
    
    def test_deterministic_deduplication_helper(self):
        """Test the _apply_deterministic_deduplication_to_graph helper function."""
        # Create a graph with obvious duplicates
        test_graph = Graph(
            entities={"Stanford", "stanford", "MIT", "  MIT  "},
            edges={"works_at", "WORKS_AT"},
            relations={
                ("Stanford", "works_at", "MIT"),
                ("stanford", "WORKS_AT", "  MIT  "),
            },
        )
        
        # Apply deterministic deduplication
        deduped = _apply_deterministic_deduplication_to_graph(test_graph)
        
        # Should reduce entities and edges
        assert len(deduped.entities) < len(test_graph.entities)
        assert len(deduped.edges) < len(test_graph.edges)
        
        # Relations should be updated to use canonical forms
        assert len(deduped.relations) <= len(test_graph.relations)
        
        # All entities in relations should be in the deduplicated entities set
        for subject, predicate, obj in deduped.relations:
            assert subject in deduped.entities
            assert predicate in deduped.edges
            assert obj in deduped.entities
    
    def test_deduplication_preserves_graph_structure(self):
        """Test that deduplication preserves graph structure."""
        # Create a graph with various duplicates
        test_graph = Graph(
            entities={"Alice", "alice", "Bob", "BOB", "Charlie"},
            edges={"knows", "KNOWS", "loves"},
            relations={
                ("Alice", "knows", "Bob"),
                ("alice", "KNOWS", "BOB"),
                ("Bob", "loves", "Charlie"),
            },
        )
        
        deduped = _apply_deterministic_deduplication_to_graph(test_graph)
        
        # Should have fewer entities/edges but same logical structure
        assert len(deduped.entities) >= 1  # At least one unique entity
        assert len(deduped.edges) >= 1  # At least one unique edge
        assert len(deduped.relations) >= 1  # At least one relation
        
        # Verify relations are valid
        for subject, predicate, obj in deduped.relations:
            assert subject in deduped.entities
            assert predicate in deduped.edges
            assert obj in deduped.entities
    
    def test_no_duplicates_unchanged(self):
        """Test that graphs without duplicates remain unchanged."""
        # Graph with no duplicates
        test_graph = Graph(
            entities={"Alice", "Bob", "Charlie"},
            edges={"knows", "loves"},
            relations={
                ("Alice", "knows", "Bob"),
                ("Bob", "loves", "Charlie"),
            },
        )
        
        deduped = _apply_deterministic_deduplication_to_graph(test_graph)
        
        # Should have same number of entities and edges
        assert len(deduped.entities) == len(test_graph.entities)
        assert len(deduped.edges) == len(test_graph.edges)
        assert len(deduped.relations) == len(test_graph.relations)
    
    def test_empty_graph(self):
        """Test handling of empty graph."""
        test_graph = Graph(
            entities=set(),
            edges=set(),
            relations=set(),
        )
        
        deduped = _apply_deterministic_deduplication_to_graph(test_graph)
        
        assert len(deduped.entities) == 0
        assert len(deduped.edges) == 0
        assert len(deduped.relations) == 0
    
    def test_deduplication_strategy_enum(self):
        """Test that DeduplicationStrategy enum works correctly."""
        assert DeduplicationStrategy.DETERMINISTIC.value == "deterministic"
        assert DeduplicationStrategy.SEMANTIC_HASH.value == "semantic_hash"
        assert DeduplicationStrategy.BOTH.value == "both"
