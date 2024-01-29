import copy
from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter
from typing import Dict, FrozenSet, List, Set, Union

from graph.retworkx import BaseEdge, BaseNode, EdgeKey, RetworkXStrDiGraph

BankNode = BaseNode[str]


@dataclass
class BankEdge(BaseEdge[str, str]):
    __slots__ = ("id", "source", "target", "key", "weight", "n_edges")
    # put id here because can't initialize it with __slots__ before python 3.10 -- just set it to -1 when init the edge and it will be override by the graph
    id: int
    source: str
    target: str
    key: str
    weight: float
    # number of edges this edge represents
    n_edges: int

    def clone(self):
        return copy.copy(self)

    def __repr__(self) -> str:
        return f"{self.source} -> {self.key} -> {self.target} ({self.weight:.3f})"


class BankGraph(RetworkXStrDiGraph[str, BankNode, BankEdge]):
    pass


@dataclass
class Solution:
    id: FrozenSet
    graph: BankGraph
    weight: float

    @cached_property
    def num_edges(self):
        return sum(edge.n_edges for edge in self.graph.iter_edges())

    @staticmethod
    def from_graph(graph: BankGraph, weight: float):
        id = Solution.get_id(graph)
        return Solution(id, graph, weight)

    @staticmethod
    def get_id(graph: BankGraph):
        """Get the id of a graph, which is the set of all edges in the graph"""
        return frozenset(((e.source, e.target, e.key) for e in graph.iter_edges()))


class NoSingleRootException(Exception):
    pass


@dataclass
class UpwardPath:
    __slots__ = ("id", "visited_nodes", "path", "weight")

    # list of edge ids
    id: FrozenSet[str]
    # set of nodes that are visited in the path
    visited_nodes: Set[str]
    # edge in reversed order. for example, consider a path: [u0, u1, u2, u3], the path will be [(u2, u3), (u1, u2), (u0, u1)]
    path: List[BankEdge]
    weight: float

    @staticmethod
    def empty(start_node_id: str):
        return UpwardPath(frozenset(), {start_node_id}, [], 0.0)

    def push(self, edge: BankEdge) -> "UpwardPath":
        c = self.clone()
        c.id = c.id.union([str(edge.id)])
        c.path.append(edge)
        c.visited_nodes.add(edge.source)
        c.weight += edge.weight
        return c

    def clone(self):
        return UpwardPath(
            self.id, copy.copy(self.visited_nodes), copy.copy(self.path), self.weight
        )

    def __repr__(self):
        if len(self.path) == 0:
            return "UpwardPath(weight=0)"

        s = "UpwardPath("
        for edge in self.path[::-1]:
            s += edge.source + " -> "
        s += self.path[0].target + f", weight={self.weight:.3f})"

        return s


@dataclass
class UpwardTraversal:
    __slots__ = ("source_id", "paths")

    # TODO: change source id, paths to less confusing name
    # the node that we start traversing upward from
    source_id: str

    # storing that we can reach this node through those list of paths
    paths: Dict[str, List[UpwardPath]]

    @staticmethod
    def top_k_beamsearch(g: BankGraph, start_node_id: str, top_k_path: int):
        travel_hist = UpwardTraversal(start_node_id, dict())
        travel_hist.paths[start_node_id] = [UpwardPath.empty(start_node_id)]

        stack = [start_node_id]
        while len(stack) > 0:
            vid = stack.pop()

            for inedge in g.in_edges(vid):
                if inedge.source not in travel_hist.paths:
                    # we haven't visited this node yet
                    travel_hist.paths[inedge.source] = []

                prev_paths = {p.id for p in travel_hist.paths[inedge.source]}

                new_paths = []
                for path in travel_hist.paths[inedge.target]:
                    if inedge.source in path.visited_nodes:
                        # path will become loopy, which we don't want to have
                        continue
                    path = path.push(inedge)
                    path.visited_nodes

                    if path.id not in prev_paths:
                        new_paths.append(path)

                travel_hist.paths[inedge.source].extend(new_paths)
                if len(travel_hist.paths[inedge.source]) > top_k_path:
                    travel_hist.sort_paths(inedge.source)
                    travel_hist.paths[inedge.source] = travel_hist.paths[inedge.source][
                        :top_k_path
                    ]

                if {p.id for p in travel_hist.paths[inedge.source]} != prev_paths:
                    # the path changes, we propagate the changes to the parent nodes
                    stack.append(inedge.source)
                else:
                    # we must already explore the parent node
                    assert inedge.source in travel_hist.paths

        return travel_hist

    def sort_paths(self, node_id: str):
        self.paths[node_id] = sorted(self.paths[node_id], key=attrgetter("weight"))
