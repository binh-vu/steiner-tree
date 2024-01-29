from __future__ import annotations

from collections import defaultdict
from copy import copy
from functools import cmp_to_key
from operator import attrgetter
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple

from graph.interface import Edge, IGraph, Node
from graph.retworkx.api import (
    has_cycle,
    is_weakly_connected,
    weakly_connected_components,
)
from steiner_tree.bank.struct import (
    BankEdge,
    BankGraph,
    BankNode,
    NoSingleRootException,
    Solution,
    UpwardTraversal,
)
from steiner_tree.disconnected import PSEUDO_ROOT_ID, add_pseudo_root

EdgeTriple = Tuple[str, str, str]


class BankSolver(Generic[Node, Edge]):
    """
    Args:
        original_graph: the original graph that we want to find steiner tree
        terminal_nodes: terminal nodes that the steiner tree should have
        weight_fn: function that extract weights from edges
        solution_cmp_fn: function to compare & sort solutions
        top_k_st: top K solutions to return
        top_k_path: top K paths to keep during backward search
        allow_shorten_graph: allow the graph to be shorten if possible
        invalid_roots: nodes that should not be considered as roots
    """

    def __init__(
        self,
        original_graph: IGraph[str, int, str, Node, Edge],
        terminal_nodes: Set[str],
        weight_fn: Callable[[Edge], float],
        solution_cmp_fn: Optional[Callable[[Solution, Solution], int]] = None,
        top_k_st: int = 10,
        top_k_path: int = 10,
        allow_shorten_graph: bool = True,
        invalid_roots: Optional[Set[str]] = None,
    ):
        # original graph
        self.original_graph = original_graph
        # function that extract weights
        self.weight_fn = weight_fn
        # function to compare & sort solutions
        self.solution_cmd_fn = solution_cmp_fn or (lambda x, y: x.weight - y.weight)
        self.solution_key_fn = (
            cmp_to_key(solution_cmp_fn)
            if solution_cmp_fn is not None
            else attrgetter("weight")
        )
        # graph that the bank algorithm will work with
        self.graph: BankGraph = BankGraph()
        # output graphs
        self.solutions: List[Solution] = []
        self.terminal_nodes = terminal_nodes
        self.top_k_st = top_k_st
        self.top_k_path = top_k_path
        self.allow_shorten_graph = allow_shorten_graph
        self.invalid_roots = invalid_roots or set()

    def run(self):
        self.graph, removed_nodes = self._preprocessing(
            self.original_graph, self.weight_fn
        )

        if is_weakly_connected(self.graph):
            graphs = [self.graph]
        else:
            graphs = self._split_connected_components(self.graph)

        final_solutions = None
        for g in graphs:
            terminal_nodes = self.terminal_nodes.intersection(
                [n.id for n in g.iter_nodes()]
            )

            newg = g
            try:
                solutions: List[Solution] = self._solve(
                    newg, terminal_nodes, self.top_k_st, self.top_k_path
                )
            except NoSingleRootException:
                # add pseudo root to ensure that we have a directed root tree
                newg = self._add_pseudo_root(g)
                solutions: List[Solution] = self._solve(
                    newg, terminal_nodes, self.top_k_st, self.top_k_path
                )

            # remove pseudo root from the solutions
            lst: List[Solution] = []
            has_pseudo_roots = False
            for i, sol in enumerate(solutions):
                if sol.graph.has_node(PSEUDO_ROOT_ID):
                    has_pseudo_roots = True
                    sol.graph.remove_node(PSEUDO_ROOT_ID)
                    if sol.graph.num_edges() == 0:
                        # can happen when all terminal nodes connected to the pseudo root
                        continue
                lst.append(
                    Solution.from_graph(sol.graph, self._get_solution_weight(sol.graph))
                )

            # comparing average != sum, and the order may change with one additional edge
            # e.g., before: (num_edges=5, weight=9.03) > (num_edges=4, weight=6.93)
            # after adding one edge of 10.8: (num_edges=6, weight=19.83) < (num_edges=5, weight=17.73)
            # however, for simplicity, we do not re-sort them, simply because one of the terminal can connect
            # to the pseudo root or to a normal node, after removing the pseudo root, the bad solution now
            # may have a slightly better weight (less than 1 edge) than the root solution.
            # TODO: address this issue as later we do sort, for now, select the top 1 so that sorting
            # won't affect the result
            if has_pseudo_roots:
                lst = lst[:1]
            solutions = lst

            if final_solutions is None:
                final_solutions = solutions
            else:
                next_solutions = []
                for current_sol in final_solutions:
                    for sol in solutions:
                        next_solutions.append(
                            self._merge_graph(current_sol.graph, sol.graph)
                        )

                final_solutions = self._sort_solutions(next_solutions)[: self.top_k_st]

        if final_solutions is not None:
            self.solutions = final_solutions

        # [self._get_roots(sol.graph) for sol in self.solutions]
        # [sol.weight for sol in self.solutions]
        return [
            self._postprocessing(self.original_graph, sol.graph, removed_nodes)
            for sol in self.solutions
        ], self.solutions

    def _preprocessing(
        self,
        g: IGraph[str, int, str, Node, Edge],
        weight_fn: Callable[[Edge], float],
    ) -> Tuple[BankGraph, Dict[EdgeTriple, List[Edge]]]:
        """Convert the input graph into a simplier form that it's easier and faster to work with.
        The output graph does not have parallel edges. Parallel edges are selected with
        """
        ng = BankGraph()

        # convert the input graph
        for u in g.nodes():
            ng.add_node(BankNode(u.id))

        # convert edges
        for edge in g.iter_edges():
            ng.add_edge(
                BankEdge(
                    id=-1,
                    source=edge.source,
                    target=edge.target,
                    key=edge.key,
                    weight=weight_fn(edge),
                    n_edges=1,
                ),
            )

        if self.allow_shorten_graph:
            # shorten path using the following heuristic
            # for a node that only connect two nodes (indegree & outdegree = 1) and not terminal nodes, we replace the node by one edge
            # map from the replaced edge to the removed nodes
            removed_nodes = {}
            for u in ng.nodes():
                if (
                    u.id not in self.terminal_nodes
                    and g.in_degree(u.id) == 1
                    and g.out_degree(u.id) == 1
                ):
                    inedge = ng.in_edges(u.id)[0]
                    outedge = ng.out_edges(u.id)[0]
                    newedge = BankEdge(
                        id=-1,
                        source=inedge.source,
                        target=outedge.target,
                        key=f"{inedge.key} -> {outedge.key}",
                        weight=inedge.weight + outedge.weight,
                        n_edges=inedge.n_edges + outedge.n_edges,
                    )
                    if not ng.has_edge_between_nodes(
                        newedge.source, newedge.target, newedge.key
                    ):
                        # replace it only if we don't have that link before
                        ng.remove_node(u.id)
                        ng.add_edge(newedge)

                        removed_nodes[
                            newedge.source, newedge.target, newedge.key
                        ] = removed_nodes.pop(
                            (inedge.source, inedge.target, inedge.key), [inedge]
                        ) + removed_nodes.pop(
                            (outedge.source, outedge.target, outedge.key), [outedge]
                        )

            return ng, removed_nodes
        return ng, {}

    def _postprocessing(
        self,
        origin_graph: IGraph[str, int, str, Node, Edge],
        out_graph: BankGraph,
        removed_nodes: Dict[EdgeTriple, List[Edge]],
    ):
        """Extract the solution from the output graph. Reserving the original node & edge ids"""
        g = origin_graph.copy()
        selected_edges = []
        for edge in out_graph.iter_edges():
            edge_triple = (edge.source, edge.target, edge.key)
            if edge_triple in removed_nodes:
                for subedge in removed_nodes[edge_triple]:
                    selected_edges.append((subedge.source, subedge.target, subedge.key))
            else:
                selected_edges.append(edge_triple)

        return g.subgraph_from_edge_triples(selected_edges)

    def _merge_graph(self, g1: BankGraph, g2: BankGraph) -> BankGraph:
        g = g1.copy()
        for edge in g2.iter_edges():
            if not g.has_node(edge.source):
                g.add_node(BankNode(edge.source))
            if not g.has_node(edge.target):
                g.add_node(BankNode(edge.target))
            g.add_edge(edge.clone())
        return g

    def _split_connected_components(self, g: BankGraph):
        return [
            g.subgraph_from_nodes(comp)
            for comp in weakly_connected_components(g)
            # must have at least two terminal nodes (to form a graph)
            if len(self.terminal_nodes.intersection(comp)) > 1
        ]

    def _solve(
        self,
        g: BankGraph,
        terminal_nodes: Set[str],
        top_k_st: int,
        top_k_path: int,
    ):
        """Despite the name, this is finding steiner tree. Assuming there is a root node that connects all
        terminal nodes together.
        """
        roots = {u.id for u in g.iter_nodes() if u.id not in self.invalid_roots}

        attr_visit_hists: List[Tuple[str, UpwardTraversal]] = []
        # to ensure the order
        for uid in list(sorted(terminal_nodes)):
            visit_hist = UpwardTraversal.top_k_beamsearch(g, uid, top_k_path)
            roots = roots.intersection(visit_hist.paths.keys())
            attr_visit_hists.append((uid, visit_hist))

        if len(roots) == 0:
            # there is no nodes that can connect to all terminal nodes either this are disconnected
            # components or you pass a directed graph with weakly connected components (a -> b <- c)
            if is_weakly_connected(g):
                # perhaps, we can break the weakly connected components by breaking one of the link (a -> b <- c)
                raise NoSingleRootException(
                    "You pass a weakly connected component and there are parts of the graph like this (a -> b <- c). Fix it before running this algorithm"
                )
            raise Exception(
                "Your graph is disconnected. Consider splitting them before calling bank solver"
            )

        # to ensure the order again & remove randomness
        roots = sorted(roots)

        # merge the paths using beam search
        results = []
        for root in roots:
            current_states = []
            uid, visit_hist = attr_visit_hists[0]
            for path in visit_hist.paths[root]:
                pg = BankGraph()
                if len(path.path) > 0:
                    assert uid == path.path[0].target
                pg.add_node(BankNode(uid))
                for e in path.path:
                    pg.add_node(BankNode(e.source))
                    pg.add_edge(e.clone())
                self._bank_graph_postconstruction(pg, 1)
                current_states.append(pg)

            if len(current_states) > top_k_st:
                current_states = [
                    _s.graph for _s in self._sort_solutions(current_states)[:top_k_st]
                ]

            for n_attrs, (uid, visit_hist) in enumerate(attr_visit_hists[1:], start=2):
                next_states = []
                for state in current_states:
                    for path in visit_hist.paths[root]:
                        pg = state.copy()
                        if len(path.path) > 0:
                            assert uid == path.path[0].target
                        if not pg.has_node(uid):
                            pg.add_node(BankNode(uid))
                        for e in path.path:
                            if not pg.has_node(e.source):
                                pg.add_node(BankNode(id=e.source))
                            # TODO: here we don't check by edge_key because we may create another edge of different key
                            # hope this new path has been exploited before.
                            if not pg.has_edges_between_nodes(e.source, e.target):
                                pg.add_edge(e.clone())

                        self._bank_graph_postconstruction(pg, n_attrs)
                        # after add a path to the graph, it can create new cycle
                        if not has_cycle(pg):
                            next_states.append(pg)
                        else:
                            # when the graph contains cycle, we have explored a subpath
                            # that do not create a cycle, so we can skip it.
                            # if we want to break cycle, try contracting and lift (like in edmonds algorithm)
                            pass

                        # the output graph should not have parallel edges
                        assert not pg.has_parallel_edges()
                if len(next_states) > top_k_st:
                    next_states = [
                        _s.graph for _s in self._sort_solutions(next_states)[:top_k_st]
                    ]
                # assert all(_.check_integrity() for _ in next_states)
                current_states = next_states
                # cgs = [g for g in next_states if len(list(nx.simple_cycles(g))) > 0]
                # nx.draw_networkx(cgs[0]); plt.show()
                # nx.draw(cgs[0]); plt.show()
            results += current_states

        return self._sort_solutions(results)

    def _sort_solutions(self, graphs: List[BankGraph]):
        """Sort the solutions, tree with the smaller weight is better (minimum steiner tree)"""
        solutions: Dict[Any, Solution] = {}
        for g in graphs:
            # id of the graph is the edge
            id = Solution.get_id(g)
            if id in solutions:
                continue

            weight = self._get_solution_weight(g)
            solutions[id] = Solution(id, g, weight)

        _solutions = sorted(solutions.values(), key=self.solution_key_fn)
        return _solutions

    def _get_solution_weight(self, graph: BankGraph):
        return sum(e.weight for e in graph.iter_edges())

    def _add_pseudo_root(self, g: BankGraph) -> BankGraph:
        default_weight = sum(e.weight for e in g.iter_edges()) + 1
        return add_pseudo_root(
            g,
            create_node=BankNode,
            create_edge=lambda uid, vid, eid: BankEdge(
                -1, uid, vid, eid, default_weight, n_edges=1
            ),
            connecting_nodes={
                n.id for n in g.iter_nodes() if n.id not in self.invalid_roots
            },
        )

    def _bank_graph_postconstruction(
        self, g: BankGraph, n_attrs: int, update_graph: bool = False
    ) -> bool:
        """Post-graph construction step. This mutates the graph and can be used to:

        (1) modify how the graph is created such as adding required edges if select a node.
        (2) remove multiple paths between two nodes as we are doing in this implementation
        """
        if n_attrs == 1:
            return False

        update_graph = self._remove_multiple_paths_within_two_hop(g) or update_graph
        if update_graph:
            self._remove_standalone_nodes(g)
        return update_graph

    def _remove_multiple_paths_within_two_hop(self, g: BankGraph) -> bool:
        """Modify the graph to remove multiple paths between two nodes within two hop. This function
        mutates the graph"""
        update_graph = False
        for n in g.iter_nodes():
            if g.in_degree(n.id) >= 2:
                # hop 1
                grand_parents: dict[str, list[tuple[BankEdge, ...]]] = defaultdict(list)
                for inedge in g.in_edges(n.id):
                    grand_parents[inedge.source].append((inedge,))

                for grand_parent, edges in grand_parents.items():
                    if len(edges) > 1:
                        # we need to select one path from this grand parent to the rest
                        # they have the same length, so we select the one has smaller weight
                        edges = sorted(
                            edges,
                            key=lambda x: x[0].weight + x[1].weight
                            if len(x) == 2
                            else x[0].weight * 2,
                        )
                        for lst in edges[1:]:
                            for edge in lst:
                                g.remove_edge(edge.id)
                        update_graph = True

                # hop 2
                grand_parents: dict[str, list[tuple[BankEdge, ...]]] = defaultdict(list)
                for inedge in g.in_edges(n.id):
                    for grand_inedge in g.in_edges(inedge.source):
                        grand_parents[grand_inedge.source].append(
                            (grand_inedge, inedge)
                        )

                for grand_parent, edges in grand_parents.items():
                    if len(edges) > 1:
                        # we need to select one path from this grand parent to the rest
                        # they have the same length, so we select the one has smaller weight
                        edges = sorted(
                            edges,
                            key=lambda x: x[0].weight + x[1].weight
                            if len(x) == 2
                            else x[0].weight * 2,
                        )
                        for lst in edges[1:]:
                            for edge in lst:
                                g.remove_edge(edge.id)

                        # make sure that all edges that we want to retain are still there
                        # so no duplicate edges are removed
                        for edge in edges[0]:
                            assert g.has_edge(edge.id)

                        update_graph = True
        return update_graph

    def _remove_standalone_nodes(self, g: BankGraph) -> bool:
        """Remove standalone nodes"""
        prev_n_nodes = g.num_nodes()
        for n in g.nodes():
            if g.degree(n.id) == 0:
                g.remove_node(n.id)
        return g.num_nodes() != prev_n_nodes
