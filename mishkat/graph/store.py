"""مخزن الرسم: عُقدٌ موسومة وحوافُ موجّهة، بفهرسةٍ في الاتجاهين.

لماذا فهرسان؟
لأن السؤال «ماذا يخرج من هذه العقدة؟» و«ما الذي يشير إليها؟» سؤالان مختلفان،
وكلاهما متكرر. فهرسةٌ واحدة تعني مسحًا كاملًا لنصف الاستعلامات.

التسمية
-------
كل عقدةٍ لها معرّفٌ مركّب من نوعها، مثل ``doc:opt-01`` و``person:al-khwarizmi``
و``domain:فلك``. النوع يجعل الاستعلام ممكنًا بلا جدول أنواعٍ منفصل.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Node:
    """عقدةٌ في الرسم."""

    id: str
    label: str
    type: str = ""
    data: dict | None = None

    @property
    def kind(self) -> str:
        """النوع المستخرج من البادئة (``doc``، ``person`` …)."""
        return self.type or (self.id.split(":", 1)[0] if ":" in self.id else "")

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "type": self.kind, "data": self.data or {}}


@dataclass(frozen=True, slots=True)
class EdgeRef:
    """حافةٌ مع اتجاهها."""

    source: str
    rel: str
    target: str
    note: str = ""

    def to_dict(self) -> dict:
        return {"s": self.source, "r": self.rel, "t": self.target, "n": self.note}

    def as_triple(self) -> tuple[str, str, str]:
        return self.source, self.rel, self.target


class GraphStore:
    """رسمٌ موجّه بسيط وفعّال."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self._out: dict[str, list[EdgeRef]] = defaultdict(list)
        self._in: dict[str, list[EdgeRef]] = defaultdict(list)
        self._rel_index: dict[str, list[EdgeRef]] = defaultdict(list)
        self._seen: set[tuple[str, str, str]] = set()

    # ------------------------------------------------------------------ بناء
    def add_node(
        self, node_id: str, label: str | None = None, type: str = "", data: dict | None = None
    ) -> Node:
        if node_id in self.nodes:
            return self.nodes[node_id]
        node = Node(id=node_id, label=label or node_id.split(":")[-1], type=type, data=data)
        self.nodes[node_id] = node
        return node

    def add_edge(self, source: str, rel: str, target: str, note: str = "") -> EdgeRef | None:
        """يضيف حافة؛ يرفض التكرار ويضمن وجود الطرفين."""
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
        key = (source, rel, target)
        if key in self._seen:
            return None
        self._seen.add(key)
        edge = EdgeRef(source, rel, target, note)
        self._out[source].append(edge)
        self._in[target].append(edge)
        self._rel_index[rel].append(edge)
        return edge

    # -------------------------------------------------------------- استعلام
    def __len__(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self._seen)

    def node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def out_edges(self, node_id: str, rel: str | None = None) -> list[EdgeRef]:
        edges = self._out.get(node_id, [])
        return [e for e in edges if e.rel == rel] if rel else list(edges)

    def in_edges(self, node_id: str, rel: str | None = None) -> list[EdgeRef]:
        edges = self._in.get(node_id, [])
        return [e for e in edges if e.rel == rel] if rel else list(edges)

    def neighbors(
        self, node_id: str, rel: str | None = None, *, direction: str = "out"
    ) -> list[str]:
        if direction == "out":
            return [e.target for e in self.out_edges(node_id, rel)]
        if direction == "in":
            return [e.source for e in self.in_edges(node_id, rel)]
        return list(
            dict.fromkeys(
                [e.target for e in self.out_edges(node_id, rel)]
                + [e.source for e in self.in_edges(node_id, rel)]
            )
        )

    def relations(self) -> dict[str, int]:
        return {r: len(v) for r, v in sorted(self._rel_index.items(), key=lambda kv: -len(kv[1]))}

    def types(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            out[n.kind] += 1
        return dict(out)

    # --------------------------------------------------------------- مسارات
    def shortest_path(
        self, start: str, goal: str, *, max_depth: int = 6, rel: str | None = None
    ) -> list[EdgeRef] | None:
        """أقصر مسارٍ موجّه بـ BFS (بلا أوزان)."""
        if start not in self.nodes or goal not in self.nodes:
            return None
        if start == goal:
            return []
        queue: deque[tuple[str, list[EdgeRef]]] = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for edge in self.out_edges(node, rel):
                if edge.target in visited:
                    continue
                new_path = path + [edge]
                if edge.target == goal:
                    return new_path
                visited.add(edge.target)
                queue.append((edge.target, new_path))
        return None

    def ego(self, node_id: str, depth: int = 1) -> dict:
        """شبكةٌ حول عقدة (للعرض المرئي): عقدٌ وحوافٌ حتى عمقٍ معيّن."""
        seen_nodes = {node_id}
        seen_edges: dict[tuple[str, str, str], EdgeRef] = {}
        frontier = {node_id}
        for _ in range(max(1, depth)):
            nxt: set[str] = set()
            for n in frontier:
                for e in self.out_edges(n) + self.in_edges(n):
                    seen_edges[e.as_triple()] = e
                    for x in (e.source, e.target):
                        if x not in seen_nodes:
                            seen_nodes.add(x)
                            nxt.add(x)
            frontier = nxt
            if not frontier:
                break
        return {
            "nodes": [self.nodes[n].to_dict() for n in seen_nodes if n in self.nodes],
            "edges": [e.to_dict() for e in seen_edges.values()],
        }

    def to_json(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for edges in self._out.values() for e in edges],
        }

    def stats(self) -> dict:
        deg = [len(self._out.get(n, [])) + len(self._in.get(n, [])) for n in self.nodes]
        return {
            "nodes": len(self.nodes),
            "edges": self.edge_count(),
            "types": self.types(),
            "relations": self.relations(),
            "avg_degree": round(sum(deg) / max(1, len(deg)), 2),
            "max_degree": max(deg) if deg else 0,
        }

    # ------------------------------------------------------------- بناء جاهز
    @classmethod
    def from_corpus(cls, corpus) -> "GraphStore":
        """يبني الرسم من مستندات المجموعة وكياناتها وعلاقاتها."""
        g = cls()
        for d in corpus.documents:
            g.add_node(
                f"doc:{d.id}",
                label=d.title,
                type="مستند",
                data={"domain": d.domain, "era": d.era, "summary": d.summary},
            )
        for e in corpus.entities:
            g.add_node(e.id, label=e.label, type=e.type, data={"note": e.note})
        for d in corpus.documents:
            dom = f"domain:{d.domain}"
            g.add_node(dom, label=d.domain, type="نطاق")
            g.add_edge(f"doc:{d.id}", "ينتمي_إلى", dom)
        for e in corpus.edges:
            g.add_edge(e.source, e.rel, e.target, e.note)
        return g
