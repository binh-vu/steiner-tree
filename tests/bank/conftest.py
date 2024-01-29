from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from graph.interface import BaseNode
from pytest import fixture

from steiner_tree.bank.struct import BankEdge, BankGraph


@dataclass
class TestGraph:
    input: BankGraph
    output: BankGraph
    terminal_nodes: set[str]
    invalid_roots: Optional[set[str]]

    @staticmethod
    def from_files(infile: Path, outfile: Path) -> TestGraph:
        ingraph = parse_graph(infile)
        terminal_nodes = {u.id for u in ingraph.iter_nodes() if u.id.startswith("col:")}
        return TestGraph(
            input=ingraph,
            output=parse_graph(outfile),
            terminal_nodes=terminal_nodes,
            invalid_roots={
                u.id for u in ingraph.iter_nodes() if u.id not in terminal_nodes
            },
        )


@fixture
def graphs() -> dict[str, TestGraph]:
    name2graph = defaultdict(dict)
    for file in (Path(__file__).parent.parent / "resources").iterdir():
        if file.suffix == ".input":
            name2graph[file.stem]["input"] = file
        elif file.suffix == ".output":
            name2graph[file.stem]["output"] = file
    return {
        name: TestGraph.from_files(o["input"], o["output"])
        for name, o in name2graph.items()
    }


def parse_graph(file: Union[str, Path]):
    with open(file, "r") as f:
        g = BankGraph()

        for line in f:
            m = re.match(
                r"^(?P<source>((?!->).)+)->(?P<edge>((?!->).)+)->(?P<target>([^(])+)\((?P<weight>\d+(\.\d+)?)\)$",
                line.strip(),
            )
            assert m is not None, line
            source = m.group("source").strip()
            edge = m.group("edge").strip()
            target = m.group("target").strip()
            weight = float(m.group("weight"))

            if not g.has_node(source):
                g.add_node(BaseNode(source))
            if not g.has_node(target):
                g.add_node(BaseNode(target))

            g.add_edge(
                BankEdge(
                    id=-1,
                    source=source,
                    key=edge,
                    target=target,
                    weight=weight,
                    n_edges=1,
                )
            )

        return g
