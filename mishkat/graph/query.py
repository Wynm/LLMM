"""لغة استعلامٍ صغيرة للرسم (MQL — Mishkat Query Language).

الصيغة
------
::

    MATCH (?x) -[ألّف]-> (?y)
    WHERE ?y = doc:math-01
    LIMIT 5

* النمط: ``(طرف) -[علاقة]-> (طرف)``. الاتجاه يُقلب بـ ``<-``، ويُهمَل بـ ``-``.
* الأطراف: متغيّر ``?x``، أو ثابتٌ مثل ``doc:math-01``، أو ``?x:شخص`` (متغيّرٌ
  مقيّد بالنوع).
* العلاقة ``*`` تعني أيّ علاقة.
* أنماطٌ متعددة تُفصل بفاصلة وتُدمج بالمشاركة في المتغيّرات (join).

المحلّل يدويٌّ، والمنفّذ يعتمد على فهرسة الرسم في الاتجاهين فلا يمسح الرسم كله
إلّا عند وجود متغيّرٍ حرّ.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from ..text.normalize import SEARCH, normalize
from .store import EdgeRef, GraphStore


class GraphQueryError(ValueError):
    """استعلام رسمٍ غير صالح."""


@dataclass(frozen=True)
class NodeTerm:
    """طرفٌ في نمط: متغيّرٌ أو ثابت."""

    raw: str
    var: str | None = None
    const: str | None = None
    type: str | None = None

    def __str__(self) -> str:
        return self.raw


@dataclass(frozen=True)
class Pattern:
    source: NodeTerm
    rel: str
    target: NodeTerm
    direction: str = "out"  # out | in | both


@dataclass(frozen=True)
class Condition:
    var: str
    value: str
    op: str = "="


@dataclass
class GraphQuery:
    raw: str
    patterns: list[Pattern] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    limit: int = 25

    def variables(self) -> list[str]:
        out: list[str] = []
        for p in self.patterns:
            for t in (p.source, p.target):
                if t.var and t.var not in out:
                    out.append(t.var)
        return out


_BRACKET_RE = re.compile(r"\[(?P<rel>[^\]]+)\]")


def _term(text: str) -> NodeTerm:
    t = text.strip().strip("()").strip()
    if not t:
        raise GraphQueryError("طرفٌ فارغ في النمط")
    if t.startswith("?"):
        body = t[1:]
        if ":" in body:
            var, typ = body.split(":", 1)
            return NodeTerm(raw=t, var=var.strip(), type=typ.strip())
        return NodeTerm(raw=t, var=body)
    return NodeTerm(raw=t, const=t)


def _parse_pattern(part: str) -> Pattern:
    m = _BRACKET_RE.search(part)
    if not m:
        raise GraphQueryError(f"نمط بلا علاقةٍ بين قوسين: {part!r}")
    before = part[: m.start()]
    after = part[m.end() :]

    left_arrow = before.rstrip().endswith("<-")
    right_arrow = after.lstrip().startswith("->")

    src_raw = before.rstrip()
    if left_arrow:
        src_raw = src_raw[:-2]
    src_raw = src_raw.rstrip().rstrip("-").strip()

    dst_raw = after.lstrip()
    if right_arrow:
        dst_raw = dst_raw[2:]
    dst_raw = dst_raw.lstrip().lstrip("-").strip()

    if left_arrow and right_arrow:
        direction = "both"
    elif left_arrow:
        direction = "in"
    else:
        direction = "out"

    return Pattern(
        source=_term(src_raw),
        rel=m.group("rel").strip(),
        target=_term(dst_raw),
        direction=direction,
    )


def parse_graph_query(text: str) -> GraphQuery:
    """يحلّل نصّ الاستعلام إلى تمثيلٍ بنيويّ."""
    raw = text.strip()
    if not raw:
        raise GraphQueryError("استعلام فارغ")
    m = re.match(r"(?is)^MATCH\b(.*)$", raw)
    if not m:
        raise GraphQueryError("الاستعلام يجب أن يبدأ بـ MATCH")
    body = m.group(1).strip()

    limit = 25
    lm = re.search(r"(?is)\bLIMIT\s+(\d+)\s*$", body)
    if lm:
        limit = int(lm.group(1))
        body = body[: lm.start()].strip()

    conditions: list[Condition] = []
    wm = re.search(r"(?is)\bWHERE\b(.*)$", body)
    if wm:
        where_src = wm.group(1).strip()
        body = body[: wm.start()].strip()
        for part in re.split(r"(?is)\bAND\b", where_src):
            part = part.strip()
            if not part:
                continue
            cm = re.match(r"(?s)^(\?\w+)\s*(=|!=)\s*(.+)$", part)
            if not cm:
                raise GraphQueryError(f"شرط غير مفهوم: {part!r}")
            conditions.append(
                Condition(cm.group(1)[1:], cm.group(3).strip().strip("'\""), cm.group(2))
            )

    parts: list[str] = []
    depth = 0
    chunk = ""
    for ch in body:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(chunk)
            chunk = ""
            continue
        chunk += ch
    parts.append(chunk)

    patterns = [_parse_pattern(p) for p in parts if p.strip()]
    if not patterns:
        raise GraphQueryError("لا أنماط في الاستعلام")
    return GraphQuery(raw=raw, patterns=patterns, conditions=conditions, limit=limit)


def _resolve(store: GraphStore, term: NodeTerm) -> list[str]:
    """يحوّل طرفًا ثابتًا إلى معرّفات عقدٍ موجودة."""
    assert term.const is not None
    c = term.const.strip()
    if c in store.nodes:
        return [c]
    if ":" in c:
        kind, label = c.split(":", 1)
        norm_label = normalize(label, SEARCH)
        cands = [
            n.id
            for n in store.nodes.values()
            if n.kind == kind and normalize(n.label, SEARCH) == norm_label
        ]
        if cands:
            return cands
    norm = normalize(c, SEARCH)
    return [
        n.id
        for n in store.nodes.values()
        if norm in {normalize(n.label, SEARCH), normalize(n.id.split(":")[-1], SEARCH)}
    ]


def _candidates(store: GraphStore, term: NodeTerm, binding: dict[str, str]) -> list[str]:
    """المرشّحون لطرفٍ في ضوء ارتباطٍ حاليّ."""
    if term.var and term.var in binding:
        return [binding[term.var]]
    if term.const:
        return _resolve(store, term)
    return [
        n.id
        for n in store.nodes.values()
        if not term.type or n.kind == term.type or n.type == term.type
    ]


def _edges(store: GraphStore, node: str, rel: str, direction: str) -> Iterator[tuple[EdgeRef, str]]:
    """حوافٌّ من عقدةٍ مع الطرف الآخر."""
    rel_filter = None if rel in {"*", ""} else rel
    if direction in ("out", "both"):
        for e in store.out_edges(node, rel_filter):
            yield e, e.target
    if direction in ("in", "both"):
        for e in store.in_edges(node, rel_filter):
            yield e, e.source


@dataclass
class QueryResult:
    """نتيجة استعلام رسم."""

    bindings: list[dict[str, str]] = field(default_factory=list)
    triples: list[dict] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    query: str = ""
    total: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "bindings": self.bindings[:200],
            "triples": self.triples[:400],
            "nodes": self.nodes,
            "count": self.total,
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


def run_query(store: GraphStore, query: str | GraphQuery) -> QueryResult:
    """ينفّذ استعلامًا على الرسم."""
    q = query if isinstance(query, GraphQuery) else parse_graph_query(query)
    started = time.perf_counter()
    bindings: list[dict[str, str]] = [{}]
    triples: dict[tuple[str, str, str], dict] = {}

    for pat in q.patterns:
        next_bindings: list[dict[str, str]] = []
        for b in bindings:
            for s in _candidates(store, pat.source, b):
                for edge, other in _edges(store, s, pat.rel, pat.direction):
                    # هل يطابق الطرف الآخر؟
                    if pat.target.const:
                        if other not in _resolve(store, pat.target):
                            continue
                    elif pat.target.var and pat.target.var in b:
                        if b[pat.target.var] != other:
                            continue
                    elif pat.target.type:
                        node = store.nodes.get(other)
                        if node is None or (node.kind != pat.target.type and node.type != pat.target.type):
                            continue
                    nb = dict(b)
                    if pat.source.var:
                        nb[pat.source.var] = edge.source if pat.direction != "in" else edge.target
                    if pat.target.var:
                        nb[pat.target.var] = other
                    next_bindings.append(nb)
                    triples[(edge.source, edge.rel, edge.target)] = edge.to_dict()
        bindings = next_bindings
        if not bindings:
            break

    for cond in q.conditions:
        want = cond.value
        norm_want = normalize(want, SEARCH)

        def ok(b: dict[str, str]) -> bool:
            got = b.get(cond.var)
            if got is None:
                return False
            node = store.nodes.get(got)
            label = node.label if node else ""
            hit = (
                got == want
                or normalize(got, SEARCH) == norm_want
                or normalize(label, SEARCH) == norm_want
                or got.endswith(":" + want)
            )
            return hit if cond.op == "=" else not hit

        bindings = [b for b in bindings if ok(b)]

    seen: set[tuple] = set()
    uniq: list[dict[str, str]] = []
    for b in bindings:
        key = tuple(sorted(b.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)

    nodes = sorted({v for b in uniq for v in b.values()})
    return QueryResult(
        bindings=uniq[: q.limit],
        triples=list(triples.values()),
        nodes=nodes,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        query=q.raw,
        total=len(uniq),
    )
