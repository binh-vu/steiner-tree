"""Helpers to handle disconnected directed graphs"""


from typing import Callable, Optional, Set, TypeVar
from graph.interface import BaseNode, Edge, EdgeKey, IGraph, Node
from graph.retworkx import RetworkXStrDiGraph
from graph.retworkx.api import weakly_connected_components


PSEUDO_ROOT_ID = "pseudo_root"
PSEUDO_EDGE_KEY = "to_pseudo_root"
G = TypeVar("G", bound=IGraph)  # can't annotate generic type...


def add_pseudo_root(
    graph: G,
    create_node: Callable[[str], Node],
    create_edge: Callable[[str, str, str], Edge],
    connecting_nodes: Optional[Set[str]],
) -> G:
    """Add a pseudo node to the graph so that we always have a directed
    rooted tree

    Args:
        graph: the graph to impute (key must be string)
        create_node: a function to create a node
        create_edge: a function to create an edge
        connecting_nodes: a set of nodes that will have links from the pseudo root

    Returns:
        an imputed graph
    """
    global PSEUDO_ROOT_ID

    if graph.has_node(PSEUDO_ROOT_ID):
        raise Exception("Pseudo root already exists")

    newg = graph.copy()
    newg.add_node(create_node(PSEUDO_ROOT_ID))

    if connecting_nodes is None:
        for node in graph.iter_nodes():
            newg.add_edge(create_edge(PSEUDO_ROOT_ID, node.id, PSEUDO_EDGE_KEY))
    else:
        for node in graph.iter_nodes():
            if node.id in connecting_nodes:
                newg.add_edge(create_edge(PSEUDO_ROOT_ID, node.id, PSEUDO_EDGE_KEY))
    return newg
