"""الدمج الهجين: البحث المعجمي + البحث الدلالي في قائمةٍ واحدة.

لماذا الدمج؟
------------
البحث المعجمي (BM25) دقيقٌ في المصطلح لكنه أعمى عن المعنى: يجد «فلك» ولا يجد
«مجرة» إن لم تُذكر. والبحث الدلالي (متجهات) يفهم التقارب لكنه قد يُفلت من
المصطلح الصريح. لذلك نُشغّل الاثنين ثم ندمج.

طريقة الدمج: Reciprocal Rank Fusion
-----------------------------------
.. math::

    RRF(d) = \\sum_{r \\in rankings} \\frac{w_r}{k + rank_r(d)}

الميزة أنها تعمل على *الرتب* لا على الدرجات، فلا حاجة لمقارنة درجة BM25
(بلا حدٍّ أعلى) بالتشابه الجيبيّ (بين -1 و1). وهذا يجعلها مستقرّةً بلا معايرة.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass
class FusionHit:
    """نتيجةٌ بعد الدمج مع تفصيل مساهمة كل مسار."""

    doc_id: str
    score: float
    sources: dict[str, int] = field(default_factory=dict)  # مسار ← رتبة
    contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 5),
            "sources": self.sources,
        }


def rrf_fuse(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusionHit]:
    """يدمج ترتيباتٍ متعدّدة بـ RRF.

    Parameters
    ----------
    rankings:
        قاموس: اسم المسار ← قائمة معرّفات المستندات مرتّبة (الأفضل أولًا).
    k:
        ثابت RRF (60 قيمةٌ معتادة: تُخفض تأثير الرتب العليا بحيث لا يهيمن مسارٌ واحد).
    weights:
        وزنٌ لكل مسار (الافتراضي 1.0).
    limit:
        عدد النتائج القصوى.
    """
    weights = dict(weights or {})
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    contribs: dict[str, dict[str, float]] = {}

    for name, docs in rankings.items():
        w = float(weights.get(name, 1.0))
        for rank, doc_id in enumerate(docs, start=1):
            add = w / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + add
            sources.setdefault(doc_id, {})[name] = rank
            contribs.setdefault(doc_id, {})[name] = add

    hits = [
        FusionHit(
            doc_id=doc_id,
            score=score,
            sources=sources[doc_id],
            contributions=contribs[doc_id],
        )
        for doc_id, score in scores.items()
    ]
    hits.sort(key=lambda h: (-h.score, h.doc_id))
    return hits[:limit] if limit else hits


def weighted_score_fuse(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusionHit]:
    """دمجٌ بديل: يُطبّع كل مسار إلى [0,1] ثم يجمع بالوزن.

    أدقّ من RRF عندما تكون الدرجات ذات معنى، لكنه حسّاسٌ للتطبيع؛ نُبقيه
    متاحًا للمقارنة في تقرير الأداء.
    """
    weights = dict(weights or {})
    normed: dict[str, dict[str, float]] = {}
    for name, pairs in rankings.items():
        if not pairs:
            continue
        lo = min(s for _, s in pairs)
        hi = max(s for _, s in pairs)
        span = (hi - lo) or 1.0
        w = float(weights.get(name, 1.0))
        normed[name] = {doc: w * (s - lo) / span for doc, s in pairs}

    totals: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    for name, pairs in rankings.items():
        for rank, (doc, _) in enumerate(pairs, start=1):
            sources.setdefault(doc, {})[name] = rank
    for name, d in normed.items():
        for doc, s in d.items():
            totals[doc] = totals.get(doc, 0.0) + s

    hits = [
        FusionHit(doc_id=doc, score=s, sources=sources.get(doc, {}), contributions={})
        for doc, s in totals.items()
    ]
    hits.sort(key=lambda h: (-h.score, h.doc_id))
    return hits[:limit] if limit else hits
