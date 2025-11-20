"""
Tests for deduplication module.
"""

import pytest
from src.kg_gen.deduplication import (
    EntityDeduplicator,
    RelationDeduplicator,
    DeduplicationMethod,
    deduplicate_entities,
)


class TestDeterministicDeduplication:
    
    def test_exact_duplicates_case_insensitive(self):
        entities = [
            "Stanford University",
            "stanford university",
            "STANFORD UNIVERSITY"
        ]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        # Should keep only one
        assert len(unique) == 1
        assert "Stanford University" in unique
        
        # Other two should map to the first
        assert len(dup_map) == 2
        assert dup_map["stanford university"] == "Stanford University"
        assert dup_map["STANFORD UNIVERSITY"] == "Stanford University"
    
    def test_whitespace_normalization(self):
        """Test that whitespace variations are treated as duplicates."""
        entities = [
            "MIT",
            "  MIT  ",
            "MIT   ",
            "   MIT"
        ]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        assert len(unique) == 1
        assert len(dup_map) == 3
    
    def test_different_entities_not_merged(self):
        """Test that semantically different entities stay separate."""
        entities = [
            "Winter Olympics",
            "Olympic Winter Games",
            "Summer Olympics"
        ]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        # All different - should not be merged
        assert len(unique) == 3
        assert len(dup_map) == 0
    
    def test_empty_list(self):
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate([])
        
        assert unique == []
        assert dup_map == {}
    
    def test_single_item(self):
        """Test handling of single item."""
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(["Test"])
        
        assert unique == ["Test"]
        assert dup_map == {}
    
    def test_preserves_first_occurrence(self):
        """Test that first occurrence is used as canonical form."""
        entities = ["First", "first", "FIRST"]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        # Should preserve the first one exactly as written
        assert unique[0] == "First"
        assert dup_map["first"] == "First"
        assert dup_map["FIRST"] == "First"
    
    def test_case_sensitive_mode(self):
        entities = ["Stanford", "stanford", "STANFORD"]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC,
            normalize_case=False  
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        # With case sensitivity, these are different
        assert len(unique) == 3
        assert len(dup_map) == 0


class TestSemanticDeduplication:
    """Tests for semantic (LLM-based) deduplication."""
    
    @pytest.mark.skip(reason="Semantic deduplication not yet implemented")
    def test_near_duplicates_merged(self):
        """Test that semantically similar entities are merged."""
        entities = [
            "Winter Olympics",
            "Olympic Winter Games",
            "winter Olympic games"
        ]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.SEMANTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        # Should merge to one canonical form
        assert len(unique) == 1
        assert len(dup_map) == 2


class TestRelationDeduplicator:
    
    def test_relation_deduplication(self):
        """Test that RelationDeduplicator works like EntityDeduplicator."""
        relations = ["works_at", "WORKS_AT", "works_at "]
        
        deduplicator = RelationDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(relations)
        
        assert len(unique) == 1
        assert len(dup_map) == 2


class TestConvenienceFunction:
    
    def test_convenience_function_deterministic(self):
        """With deterministic method."""
        entities = ["Entity", "entity", "ENTITY"]
        unique, dup_map = deduplicate_entities(
            entities, 
            method="deterministic"
        )
        
        assert len(unique) == 1
        assert len(dup_map) == 2
    
    def test_convenience_function_with_options(self):
        """With custom options."""
        entities = ["Entity", "entity"]
        unique, dup_map = deduplicate_entities(
            entities,
            method="deterministic",
            normalize_case=False
        )
        
        # With case sensitivity, should be different
        assert len(unique) == 2
        assert len(dup_map) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_unicode_entities(self):
        entities = ["Müller", "müller", "MÜLLER"]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        assert len(unique) == 1
        assert len(dup_map) == 2
    
    def test_special_characters(self):
        entities = ["Type-1", "type-1", "TYPE-1"]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        assert len(unique) == 1
    
    def test_very_long_entities(self):
        long_entity = "A" * 1000
        entities = [long_entity, long_entity.lower()]
        
        deduplicator = EntityDeduplicator(
            method=DeduplicationMethod.DETERMINISTIC
        )
        unique, dup_map = deduplicator.deduplicate(entities)
        
        assert len(unique) == 1
        assert len(dup_map) == 1

