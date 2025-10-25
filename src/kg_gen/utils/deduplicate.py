import unicodedata
from kg_gen.models import Graph
from semhash import SemHash
import inflect


class DeduplicateList:
    threshold: float = 0.95
    inflect_engine: inflect.engine
    original_map: dict[str, str]
    items_map: dict[str, str]
    duplicates: dict[str, str]
    deduplicated: list[str]

    # Stats values
    total_items: int
    deduplicated_items: int
    duplicate_items: int
    reduction: float

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.inflect_engine = inflect.engine()
        self.original_map = {}
        self.items_map = {}
        self.duplicates = {}
        self.deduplicated = []

    def normalize(self, text: str) -> str:
        '''
        Normalize a text.
        '''
        return unicodedata.normalize("NFKC", text)

    def singularize(self, text: str) -> str:
        '''
        Singularize a text.
        '''
        # singularize each token when it looks like a plural noun
        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def deduplicate(self, items: list[str]) -> list[str]:
        '''
        Deduplicate a list of items using semantic hashing.
        Before deduplication, items are normalized and singularized.

        Args:
            items: List of items to deduplicate

        Returns:
            List of deduplicated items
        '''
        self.total_items = len(items)

        # Normalize and singularize each string
        normalized_items = set()
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            self.original_map[item] = singular
            self.items_map[singular] = item
            normalized_items.add(singular)

        # Deduplicate the normalized strings
        semhash = SemHash.from_records(records=list(normalized_items))
        deduplication_result = semhash.self_deduplicate(
            threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        self.duplicate_items = len(deduplication_result.duplicates)
        self.reduction = (self.duplicate_items / self.total_items) * 100

        # Map back to original strings
        duplicates = deduplication_result.duplicates
        for duplicate in duplicates:
            original = duplicate.record
            # Check if duplicates list is not empty before accessing
            if duplicate.duplicates and len(duplicate.duplicates) > 0 and len(duplicate.duplicates[0]) > 0:
                duplicate_value = duplicate.duplicates[0][0]
                self.items_map[original] = self.items_map[duplicate_value]
                if not original in self.duplicates:
                    self.duplicates[original] = duplicate_value

        self.deduplicated = deduplication_result.selected

    def stats(self) -> str:
        return f"Total items: {self.total_items}; Deduplicated items: {self.deduplicated_items}; Duplicate items: {self.duplicate_items}; Reduction: {self.reduction:.1f}"


def deduplicate_graph(graph: Graph) -> Graph:
    """
    Deduplicate the graph.
    """
    # Deduplicate each graph components
    entities_dedup = DeduplicateList()
    entities_dedup.deduplicate(graph.entities)
    edges_dedup = DeduplicateList()
    edges_dedup.deduplicate(graph.edges)

    def _get_relation(relation: list[str]) -> list[str]:
        """
        Get the transformed relation.
        """
        # Handle case where entity might not be in original_map due to normalization
        first_entity_original = relation[0]
        if first_entity_original in entities_dedup.original_map:
            first_entity = entities_dedup.items_map[entities_dedup.original_map[first_entity_original]]
        else:
            # If not found, use the original entity (it might have been normalized differently)
            first_entity = first_entity_original
            
        second_entity_original = relation[2]
        if second_entity_original in entities_dedup.original_map:
            second_entity = entities_dedup.items_map[entities_dedup.original_map[second_entity_original]]
        else:
            # If not found, use the original entity
            second_entity = second_entity_original
            
        edge_original = relation[1]
        if edge_original in edges_dedup.original_map:
            edge = edges_dedup.items_map[edges_dedup.original_map[edge_original]]
        else:
            # If not found, use the original edge
            edge = edge_original
            
        return [first_entity, edge, second_entity]

    # Deduplicate the graph
    new_entities = [entities_dedup.items_map[item]
                    for item in entities_dedup.deduplicated]
    new_edges = [edges_dedup.items_map[item]
                 for item in edges_dedup.deduplicated]
    new_relations = [_get_relation(relation) for relation in graph.relations]

    # Remove duplicate relations
    new_relations = list(set(tuple(relation) for relation in new_relations))

    return Graph(entities=new_entities,
                      edges=new_edges, relations=new_relations)
