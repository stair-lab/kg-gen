"""
Configurable deduplication methods for knowledge graph entities and relations.

Provides two strategies:
1. Deterministic: Fast hash-based exact matching (no LLM calls)
2. Semantic: Existing KGGen approach using embeddings + LLM verification
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class DeduplicationMethod(Enum):
    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class EntityDeduplicator:
    """
    Handles entity deduplication using configurable strategies.
    
    Args:
        method: Deduplication strategy to use
        hash_algorithm: Hash function for deterministic mode (default: md5)
        normalize_case: Whether to normalize case in deterministic mode
        normalize_whitespace: Whether to normalize whitespace
    """
    
    def __init__(
        self,
        method: DeduplicationMethod = DeduplicationMethod.SEMANTIC,
        hash_algorithm: str = "md5",
        normalize_case: bool = True,
        normalize_whitespace: bool = True,
    ):
        self.method = method
        self.hash_algorithm = hash_algorithm
        self.normalize_case = normalize_case
        self.normalize_whitespace = normalize_whitespace
    
    def deduplicate(
        self, 
        items: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Deduplicate a list of items (entities or relations)
        
        Args:
            items: List of strings to deduplicate
            
        Returns:
            Tuple of (unique_items, duplicate_map)
            - unique_items: List of deduplicated items
            - duplicate_map: Maps duplicate items to their canonical form
        """
        if self.method == DeduplicationMethod.DETERMINISTIC:
            return self._deduplicate_deterministic(items)
        elif self.method == DeduplicationMethod.SEMANTIC:
            return self._deduplicate_semantic(items)
        else:
            raise ValueError(f"Unknown deduplication method: {self.method}")
    
    def _normalize(self, text: str) -> str:
        if self.normalize_whitespace:
            text = " ".join(text.split())  # Normalize whitespace
        if self.normalize_case:
            text = text.lower()
        return text
    
    def _deduplicate_deterministic(
        self, 
        items: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Fast hash-based exact matching.
        
        This method:
        - Normalizes items (case, whitespace)
        - Computes hash for each normalized item
        - Maps duplicates to first occurrence
        
        Examples:
            "Stanford University" == "stanford university" (case)
            "  MIT  " == "MIT" (whitespace)
            "Winter Olympics" != "Olympic Winter Games" (different text)
        
        Returns:
            (unique_items, duplicate_map)
        """
        logger.info(
            f"Starting deterministic deduplication of {len(items)} items"
        )
        
        seen_hashes: Dict[str, str] = {}  # hash -> canonical form
        unique_items: List[str] = []
        duplicate_map: Dict[str, str] = {}
        
        for item in items:
            # Normalize
            normalized = self._normalize(item)
            
            # Compute hash
            hash_func = getattr(hashlib, self.hash_algorithm)
            item_hash = hash_func(normalized.encode('utf-8')).hexdigest()
            
            if item_hash in seen_hashes:
                # This is a duplicate
                canonical = seen_hashes[item_hash]
                duplicate_map[item] = canonical
                logger.debug(f"Found duplicate: '{item}' -> '{canonical}'")
            else:
                # First occurrence - use as canonical form
                seen_hashes[item_hash] = item
                unique_items.append(item)
        
        logger.info(
            f"Deterministic deduplication: {len(items)} -> {len(unique_items)} "
            f"({len(duplicate_map)} duplicates removed)"
        )
        
        return unique_items, duplicate_map
    
    def _deduplicate_semantic(
        self, 
        items: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Semantic similarity using embeddings + LLM verification.
        
        This wraps the existing KGGen entity resolution approach:
        1. Cluster using S-BERT embeddings + k-means
        2. Find top-k similar using BM25 + embeddings  
        3. LLM judges which are true duplicates
        
        Note: This is expensive but catches near-duplicates like:
            "Winter Olympics" == "Olympic Winter Games"
        
        TODO: Connect to existing KGGen resolution logic.
        For now, returns items unchanged to allow testing of deterministic method.
        """
        logger.info(
            f"Starting semantic deduplication of {len(items)} items"
        )
        
        # TODO: Import and call existing KGGen resolution
        # from kggen.resolution import resolve_entities
        # return resolve_entities(items)
        
        # Placeholder - return items as-is for now
        logger.warning(
            "Semantic deduplication not yet implemented - "
            "returning items unchanged"
        )
        return items, {}


class RelationDeduplicator(EntityDeduplicator):
    """
    Specialized deduplicator for relations.
    
    Inherits all functionality from EntityDeduplicator but can be
    extended with relation-specific logic if needed.
    """
    pass


def deduplicate_entities(
    entities: List[str],
    method: str = "semantic",
    **kwargs
) -> Tuple[List[str], Dict[str, str]]:
    """
    Convenience function for entity deduplication.
    
    Args:
        entities: List of entity strings
        method: "deterministic" or "semantic"
        **kwargs: Additional arguments for EntityDeduplicator
        
    Returns:
        (unique_entities, duplicate_map)
        
    Example:
        >>> entities = ["Stanford", "stanford", "MIT"]
        >>> unique, dups = deduplicate_entities(entities, method="deterministic")
        >>> print(unique)
        ['Stanford', 'MIT']
        >>> print(dups)
        {'stanford': 'Stanford'}
    """
    dedup_method = DeduplicationMethod(method)
    deduplicator = EntityDeduplicator(method=dedup_method, **kwargs)
    return deduplicator.deduplicate(entities)

