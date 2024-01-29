from __future__ import annotations

from operator import attrgetter

import pytest

from steiner_tree.bank.solver import BankSolver
from tests.bank.conftest import TestGraph


@pytest.mark.parametrize("name", ["graph_1"])
def test_solver(name: str, graphs: dict[str, TestGraph]):
    graph = graphs[name]

    solver = BankSolver(
        original_graph=graph.input,
        terminal_nodes=graph.terminal_nodes,
        weight_fn=attrgetter("weight"),
        invalid_roots=graph.invalid_roots,
    )

    trees, solutions = solver.run()
    print(trees[0].edges())
