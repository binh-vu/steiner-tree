import copy
from dataclasses import dataclass
from operator import attrgetter
from typing import Dict, List, Set

import networkx as nx
from steiner_tree.bank.edge import Edge


@dataclass
class UpwardPath:
    __slots__ = ("visited_nodes", "path", "weight")

    # set of nodes that are visited in the path
    visited_nodes: Set[str]
    # edge in reversed order. for example, consider a path: [u0, u1, u2, u3], the path will be [(u2, u3), (u1, u2), (u0, u1)]
    path: List[Edge]
    weight: float

    @staticmethod
    def empty(start_node_id: str):
        return UpwardPath({start_node_id}, [], 0.0)

    def push(self, edge: Edge) -> "UpwardPath":
        c = self.clone()
        c.path.append(edge)
        c.visited_nodes.add(edge.source_id)
        c.weight += edge.weight
        return c

    def clone(self):
        return UpwardPath(
            copy.copy(self.visited_nodes), copy.copy(self.path), self.weight
        )


@dataclass
class UpwardTraversal:
    __slots__ = ("source_id", "paths")

    # TODO: change source id, paths to less confusing name
    # the node that we start traversing upward from
    source_id: str

    # storing that we can reach this node through those list of paths
    paths: Dict[str, List[UpwardPath]]

    @staticmethod
    def top_k_beamsearch(g: nx.MultiDiGraph, start_node_id: str, top_k_path: int):
        travel_hist = UpwardTraversal(start_node_id, dict())
        travel_hist.paths[start_node_id] = [UpwardPath.empty(start_node_id)]
        for source_id, target_id, edge_id, orientation in nx.edge_bfs(  # type: ignore
            g, start_node_id, orientation="reverse"
        ):
            if source_id not in travel_hist.paths:
                travel_hist.paths[source_id] = []

            edge: Edge = g.edges[source_id, target_id, edge_id]["data"]
            for path in travel_hist.paths[target_id]:
                if source_id in path.visited_nodes:
                    # path will become loopy, which we don't want to have
                    continue
                path = path.push(edge)
                travel_hist.paths[source_id].append(path)

            # we trim the number of paths in here
            if len(travel_hist.paths[source_id]) > top_k_path:
                # calculate the score of each path, and then select the best one
                travel_hist.sort_paths(source_id)
                travel_hist.paths[source_id] = travel_hist.paths[source_id][:top_k_path]
        return travel_hist

    def sort_paths(self, node_id: str):
        self.paths[node_id] = sorted(self.paths[node_id], key=attrgetter("weight"))
