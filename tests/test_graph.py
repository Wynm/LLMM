"""اختبارات الرسم المعرفي: المخزن، الاستعلام، والاستدلال."""

from __future__ import annotations

import pytest

from mishkat.graph.infer import DEFAULT_RULES, InferenceEngine, Rule
from mishkat.graph.query import GraphQueryError, parse_graph_query, run_query
from mishkat.graph.store import GraphStore


@pytest.fixture()
def graph() -> GraphStore:
    g = GraphStore()
    g.add_node("doc:a", "مقالة أ", "مستند")
    g.add_node("doc:b", "مقالة ب", "مستند")
    g.add_node("doc:c", "مقالة ج", "مستند")
    g.add_node("person:x", "الخوارزمي", "شخص")
    g.add_node("domain:فلك", "فلك", "نطاق")
    g.add_edge("person:x", "ألّف", "doc:a")
    g.add_edge("doc:a", "يستند_إلى", "doc:b")
    g.add_edge("doc:b", "يستند_إلى", "doc:c")
    g.add_edge("doc:a", "ينتمي_إلى", "domain:فلك")
    g.add_edge("doc:c", "يوازي", "doc:b")
    return g


# -------------------------------------------------------------------- المخزن
def test_store_basics(graph: GraphStore) -> None:
    assert len(graph) == 5
    assert graph.edge_count() == 5
    assert graph.neighbors("doc:a", "يستند_إلى") == ["doc:b"]
    assert graph.neighbors("doc:b", direction="in") == ["doc:a", "doc:c"]


def test_store_rejects_duplicate_edge(graph: GraphStore) -> None:
    assert graph.add_edge("doc:a", "يستند_إلى", "doc:b") is None
    assert graph.edge_count() == 5


def test_store_auto_creates_nodes() -> None:
    g = GraphStore()
    g.add_edge("x", "r", "y")
    assert "x" in g.nodes and "y" in g.nodes


def test_shortest_path(graph: GraphStore) -> None:
    path = graph.shortest_path("person:x", "doc:c")
    assert path is not None
    assert [e.rel for e in path] == ["ألّف", "يستند_إلى", "يستند_إلى"]


def test_no_path(graph: GraphStore) -> None:
    assert graph.shortest_path("doc:c", "person:x") is None


def test_ego_network(graph: GraphStore) -> None:
    ego = graph.ego("doc:a", depth=1)
    assert len(ego["nodes"]) >= 3
    assert ego["edges"]


# ------------------------------------------------------------------ الاستعلام
def test_parse_pattern(graph: GraphStore) -> None:
    q = parse_graph_query("MATCH (?x) -[ألّف]-> (?y) LIMIT 3")
    assert len(q.patterns) == 1
    assert q.patterns[0].rel == "ألّف"
    assert q.patterns[0].source.var == "x"
    assert q.limit == 3


def test_parse_reverse_pattern() -> None:
    q = parse_graph_query("MATCH (?x) <-[ألّف]- (?y)")
    assert q.patterns[0].direction == "in"


def test_parse_invalid() -> None:
    with pytest.raises(GraphQueryError):
        parse_graph_query("SELECT * FROM graph")
    with pytest.raises(GraphQueryError):
        parse_graph_query("MATCH (?x) (?y)")


def test_query_returns_bindings(graph: GraphStore) -> None:
    r = run_query(graph, "MATCH (?x) -[ألّف]-> (?y)")
    assert r.total == 1
    assert r.bindings[0] == {"x": "person:x", "y": "doc:a"}


def test_query_const_target(graph: GraphStore) -> None:
    r = run_query(graph, "MATCH (?x) -[يستند_إلى]-> (?y), (?y) -[يستند_إلى]-> (doc:c)")
    assert r.bindings == [{"x": "doc:a", "y": "doc:b"}]


def test_query_wildcard(graph: GraphStore) -> None:
    r = run_query(graph, "MATCH (doc:a) -[*]-> (?n)")
    assert r.total == 2  # يستند_إلى → doc:b، ينتمي_إلى → domain:فلك


def test_query_type_constraint(graph: GraphStore) -> None:
    r = run_query(graph, "MATCH (?p:شخص) -[*]-> (?y)")
    assert r.total == 1


def test_query_where(graph: GraphStore) -> None:
    r = run_query(graph, "MATCH (?x) -[ينتمي_إلى]-> (?d) WHERE ?d = فلك")
    assert r.total == 1


def test_query_limit(graph: GraphStore) -> None:
    r = run_query(graph, "MATCH (?x) -[*]-> (?y) LIMIT 2")
    assert len(r.bindings) <= 2


# ------------------------------------------------------------------ الاستدلال
def test_symmetric_derivation(graph: GraphStore) -> None:
    eng = InferenceEngine(graph, [Rule("sym", "symmetric", "يوازي")])
    derived = eng.derive()
    assert any(d.source == "doc:b" and d.target == "doc:c" for d in derived)


def test_transitive_derivation(graph: GraphStore) -> None:
    eng = InferenceEngine(graph, [Rule("tr", "transitive", "يستند_إلى", max_depth=3)])
    derived = eng.derive()
    triples = {(d.source, d.target) for d in derived}
    assert ("doc:a", "doc:c") in triples


def test_derived_edges_have_premises(graph: GraphStore) -> None:
    eng = InferenceEngine(graph, [Rule("tr", "transitive", "يستند_إلى", max_depth=3)])
    for d in eng.derive():
        assert d.premises
        assert "مستنتج" in d.as_edge().note


def test_apply_adds_edges(graph: GraphStore) -> None:
    before = graph.edge_count()
    InferenceEngine(graph, DEFAULT_RULES).derive(apply=True)
    assert graph.edge_count() > before


def test_inverse_rule() -> None:
    g = GraphStore()
    g.add_edge("a", "أثّر_في", "b")
    eng = InferenceEngine(g, [Rule("inv", "inverse", "أثّر_في", "تأثّر_ب")])
    derived = eng.derive()
    assert any(d.rel == "تأثّر_ب" and d.source == "b" for d in derived)
