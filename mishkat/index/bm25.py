"""BM25: ترجيحٌ يجمع تكرار المصطلح ونُدرته وطول المستند.

المعادلة
--------
.. math::

    score(q,d) = \\sum_{t \\in q} idf(t) \\cdot
      \\frac{f(t,d)\\,(k_1+1)}{f(t,d) + k_1\\left(1-b+b\\frac{|d|}{avgdl}\\right)}

حيث :math:`f(t,d)` التكرار الموزون بالحقول، و:math:`|d|` طول المستند الموزون،
و:math:`k_1` يضبط تشبّع التكرار، و:math:`b` يضبط شدّة تعويض الطول.

نضيف فوق BM25 ثلاث إضافاتٍ عملية:

1. **توسيع الاستعلام** بالساق والجذر (من طبقة النصّ)، فتُطابق «المدارس» سؤالَ
   «الدراسة».
2. **مكافأة العبارة**: إن ظهرت عبارةٌ استعلامية متتالية، تُضاف مكافأة.
3. **مقاطع الإبراز**: نُرجع أفضل جملةٍ في المستند، لا المستند كله.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..text.normalize import SEARCH, normalize
from ..text.tokenize import split_sentences, tokenize
from .inverted import InvertedIndex
from .query import ParsedQuery, build_terms, parse_query


@dataclass(frozen=True, slots=True)
class BM25Params:
    """معاملات BM25."""

    k1: float = 1.5
    b: float = 0.75
    phrase_bonus: float = 1.8
    require_all: bool = False


@dataclass(frozen=True, slots=True)
class SearchHit:
    """نتيجة استرجاعٍ واحدة."""

    doc_id: str
    score: float
    matched: tuple[str, ...] = ()
    snippet: str = ""
    why: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "matched": list(self.matched),
            "snippet": self.snippet,
            "why": list(self.why),
        }


class BM25:
    """مُسجّل BM25 فوق فهرسٍ مقلوب."""

    def __init__(self, index: InvertedIndex, params: BM25Params | None = None) -> None:
        self.index = index
        self.params = params or BM25Params()

    # -------------------------------------------------------------- تقييم
    def term_score(self, term: str, doc_id: str) -> float:
        p = self.index.postings.get(term, {}).get(doc_id)
        if p is None:
            return 0.0
        k1, b = self.params.k1, self.params.b
        dl = self.index.doc_len.get(doc_id, 0.0)
        avgdl = self.index.avg_doc_len or 1.0
        denom = p.tf + k1 * (1 - b + b * dl / avgdl)
        if denom <= 0:
            return 0.0
        return self.index.idf(term) * p.tf * (k1 + 1) / denom

    def score_document(
        self, doc_id: str, mapping: dict[str, list[str]]
    ) -> tuple[float, list[str]]:
        """درجة مستندٍ واحد مقابل مصطلحاتٍ موسّعة."""
        score = 0.0
        matched: list[str] = []
        for form, originals in mapping.items():
            s = self.term_score(form, doc_id)
            if s > 0:
                score += s
                matched.extend(originals)
        return score, list(dict.fromkeys(matched))

    # ------------------------------------------------------------ البحث
    def search(
        self,
        query: str | ParsedQuery,
        *,
        k: int = 10,
        mode: str | None = None,
        with_snippets: bool = True,
        expand: Sequence[str] = ("stem", "root"),
    ) -> list[SearchHit]:
        """يبحث ويعيد أفضل k نتيجة، مرتّبةً تنازليًّا."""
        q = query if isinstance(query, ParsedQuery) else parse_query(query)
        if q.is_empty:
            return []

        mapping = build_terms(q, expand=tuple(expand))
        # مصطلحات المستبعدة تُستخدم للتصفية فقط
        excl_forms = {
            f for f in build_terms(ParsedQuery(raw="", excluded=q.excluded), expand=tuple(expand))
        } if q.excluded else set()

        mode = mode or ("and" if (self.params.require_all or q.required) else "or")
        cands = self.index.candidates(mapping.keys(), mode="or")
        if not cands:
            return []

        # تصفية الإلزام والمستبعد
        if q.required:
            req_forms = {
                f
                for f in build_terms(
                    ParsedQuery(raw="", required=q.required), expand=tuple(expand)
                )
            }
            cands = {d for d in cands if any(f in self.index.postings and d in self.index.postings[f] for f in req_forms)}
            if mode == "and":
                cands = {
                    d
                    for d in cands
                    if all(
                        d in self.index.postings.get(f, {})
                        for f in req_forms
                        if f in self.index.postings
                    )
                }
        if excl_forms:
            cands = {
                d
                for d in cands
                if not any(d in self.index.postings.get(f, {}) for f in excl_forms)
            }
        if not cands:
            return []

        # مكافأة العبارات
        phrase_docs: dict[str, float] = {}
        for phrase in q.phrases:
            forms = [normalize(t, SEARCH) for t in phrase]
            # نجرّب العبارة بصيغتها السطحية أولًا، ثم بالساق
            for variant in (forms, [self._stem(t) for t in forms]):
                hit = self.index.phrase(variant, field_name="body")
                for d in hit:
                    phrase_docs[d] = max(phrase_docs.get(d, 0.0), self.params.phrase_bonus)
                if hit:
                    break

        hits: list[SearchHit] = []
        for doc_id in cands:
            score, matched = self.score_document(doc_id, mapping)
            score += phrase_docs.get(doc_id, 0.0)
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    score=score,
                    matched=tuple(matched),
                    why=tuple(sorted(set(mapping[f][0] for f in mapping if f in matched))[:6])
                    if matched
                    else (),
                )
            )
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        top = hits[:k]
        if with_snippets:
            top = [
                SearchHit(
                    doc_id=h.doc_id,
                    score=h.score,
                    matched=h.matched,
                    snippet="",
                    why=h.why,
                )
                for h in top
            ]
        return top

    @staticmethod
    def _stem(term: str) -> str:
        from ..text.stem import light_stem

        return light_stem(term)


def best_snippet(text: str, terms: Iterable[str], *, width: int = 180) -> str:
    """يختار أفضل جملةٍ تحتوي مصطلحات الاستعلام."""
    wanted = {t for t in terms if t}
    sents = split_sentences(text)
    if not sents:
        return text[:width]
    best, best_score = sents[0], -1
    for s in sents:
        norm = normalize(s.text, SEARCH)
        score = sum(1 for w in tokenize(s.text) if w.norm in wanted)
        if score > best_score:
            best, best_score = s, score
    out = best.text.strip()
    if len(out) > width:
        cut = out.rfind(" ", 0, width)
        out = out[: cut if cut > 40 else width].rstrip() + "…"
    return out


def highlight(text: str, terms: Iterable[str]) -> list[dict]:
    """يُرجع فترات (start,end) للكلمات المطابقة، لإبرازها في الواجهة."""
    wanted = {t for t in terms if t}
    spans: list[dict] = []
    for tok in tokenize(text):
        if tok.norm in wanted:
            spans.append({"start": tok.start, "end": tok.end, "text": tok.raw})
    return spans
