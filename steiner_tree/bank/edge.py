from dataclasses import dataclass


@dataclass
class Edge:
    __slots__ = ("id", "source_id", "target_id", "edge_key", "weight", "n_edges")
    id: str
    source_id: str
    target_id: str
    edge_key: str
    weight: float
    n_edges: int
