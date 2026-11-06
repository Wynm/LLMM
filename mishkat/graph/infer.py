"""الاستدلال على الرسم: قواعدٌ صريحة تُنتج حوافًا جديدة *مع إثباتها*.

لا نضيف حافةً مستنتجةً بلا سند: كل حافةٍ تحمل قاعدةَ الاستنتاج ومقدّماتها
(الحواف التي بُنيت عليها). هذا هو الفرق بين استدلالٍ وتخمين، وهو ما يجعل
المحرّك قابلًا للمراجعة.

القواعد المدعومة
----------------
* ``symmetric``  : إن كان (أ يوازي ب) فإن (ب يوازي أ).
* ``inverse``    : (أ أثّر_في ب) ⇔ (ب تأثّر_ب أ).
* ``transitive`` : (أ يستند_إلى ب) و(ب يستند_إلى ج) ⇒ (أ يستند_إلى ج).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .store import EdgeRef, GraphStore


@dataclass(frozen=True)
class Rule:
    """قاعدة استدلالٍ واحدة."""

    name: str
    kind: str  # symmetric | inverse | transitive
    rel: str
    other: str | None = None
    max_depth: int = 3

    def describe(self) -> str:
        if self.kind == "symmetric":
            return f"({self.rel}) متماثلة"
        if self.kind == "inverse":
            return f"({self.rel}) عكس ({self.other})"
        return f"({self.rel}) متعدّية حتى عمق {self.max_depth}"


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("sym-يوازي", "symmetric", "يوازي"),
    Rule("sym-يرتبط", "symmetric", "يرتباط_ب"),
    Rule("sym-ربط", "symmetric", "يرتبط_ب"),
    Rule("inv-تأثير", "inverse", "أثّر_في", "تأثّر_ب"),
    Rule("inv-تأثر", "inverse", "تأثّر_ب", "أثّر_في"),
    Rule("trans-استناد", "transitive", "يستند_إلى", max_depth=3),
    Rule("trans-امتداد", "transitive", "امتداد_ل", max_depth=3),
)


@dataclass
class DerivedEdge:
    """حافةٌ مستنتجة مع سندها."""

    source: str
    rel: str
    target: str
    rule: str
    premises: tuple[tuple[str, str, str], ...] = ()

    def as_edge(self) -> EdgeRef:
        chain = " ← ".join(f"{s} -{r}-> {t}" for s, r, t in self.premises)
        return EdgeRef(self.source, self.rel, self.target, note=f"مستنتجٌ بالقاعدة {self.rule}: {chain}")

    def to_dict(self) -> dict:
        return {
            "s": self.source,
            "r": self.rel,
            "t": self.target,
            "rule": self.rule,
            "premises": [list(p) for p in self.premises],
        }


class InferenceEngine:
    """محرّك استدلالٍ بالقواعد على رسمٍ موسوم."""

    def __init__(self, store: GraphStore, rules: Iterable[Rule] | None = None) -> None:
        self.store = store
        self.rules = list(rules if rules is not None else DEFAULT_RULES)

    # ----------------------------------------------------------- الاستنتاج
    def derive(self, *, apply: bool = False, max_new: int = 300) -> list[DerivedEdge]:
        """يولّد الحواف المستنتجة (ويضيفها إن طُلب)."""
        existing = {
            (e.source, e.rel, e.target)
            for edges in self.store._out.values()
            for e in edges
        }
        out: list[DerivedEdge] = []
        added: set[tuple[str, str, str]] = set()

        def push(d: DerivedEdge) -> bool:
            key = (d.source, d.rel, d.target)
            if key in existing or key in added or d.source == d.target:
                return False
            added.add(key)
            out.append(d)
            return len(out) < max_new

        for rule in self.rules:
            if rule.kind == "symmetric":
                for e in list(self.store._out.values()):
                    for edge in e:
                        if edge.rel != rule.rel:
                            continue
                        if not push(DerivedEdge(edge.target, rule.rel, edge.source, rule.name, (edge.as_triple(),))):
                            return self._finish(out, apply)
            elif rule.kind == "inverse":
                for edges in list(self.store._out.values()):
                    for edge in edges:
                        if edge.rel != rule.rel:
                            continue
                        if not push(
                            DerivedEdge(
                                edge.target,
                                rule.other or rule.rel,
                                edge.source,
                                rule.name,
                                (edge.as_triple(),),
                            )
                        ):
                            return self._finish(out, apply)
            elif rule.kind == "transitive":
                for edges in list(self.store._out.values()):
                    for edge in edges:
                        if edge.rel != rule.rel:
                            continue
                        # مساراتٌ بطولٍ حتى max_depth
                        frontier: list[tuple[str, list[tuple[str, str, str]]]] = [
                            (edge.target, [edge.as_triple()])
                        ]
                        for _ in range(rule.max_depth - 1):
                            nxt: list[tuple[str, list[tuple[str, str, str]]]] = []
                            for node, path in frontier:
                                for nxt_edge in self.store.out_edges(node, rule.rel):
                                    new_path = path + [nxt_edge.as_triple()]
                                    if not push(
                                        DerivedEdge(
                                            edge.source,
                                            rule.rel,
                                            nxt_edge.target,
                                            rule.name,
                                            tuple(new_path),
                                        )
                                    ):
                                        return self._finish(out, apply)
                                    nxt.append((nxt_edge.target, new_path))
                            frontier = nxt
                            if not frontier:
                                break
        return self._finish(out, apply)

    def _finish(self, derived: list[DerivedEdge], apply: bool) -> list[DerivedEdge]:
        if apply:
            for d in derived:
                e = d.as_edge()
                self.store.add_edge(e.source, e.rel, e.target, e.note)
        return derived

    # ------------------------------------------------------------- الشروح
    def explain(self, source: str, rel: str, target: str) -> list[DerivedEdge]:
        """يبحث في الحواف المستنتجة عن سندٍ لهذه الحافة."""
        return [
            d
            for d in self.derive(apply=False, max_new=5000)
            if (d.source, d.rel, d.target) == (source, rel, target)
        ]

    def summary(self) -> dict:
        derived = self.derive(apply=False)
        by_rule: dict[str, int] = {}
        for d in derived:
            by_rule[d.rule] = by_rule.get(d.rule, 0) + 1
        return {
            "rules": [r.describe() for r in self.rules],
            "derivable": len(derived),
            "by_rule": by_rule,
            "samples": [d.to_dict() for d in derived[:8]],
        }
