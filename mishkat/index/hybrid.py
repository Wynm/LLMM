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


def score_fuse(
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusionHit]:
    """دمجٌ بالدرجات بعد تطبيع كل مسارٍ على أعلى قيمةٍ فيه.

    لماذا هذا هو الافتراضي في المحرّك، لا RRF؟
    -----------------------------------------
    قِيست الاستراتيجيتان على مجموعة ``data/eval/queries.jsonl`` (59 استعلامًا
    مُعنونًا، انظر :mod:`mishkat.evaluate`):

    =====================  ======  =======  ======
    الاستراتيجية          MRR     R@10     H@1
    =====================  ======  =======  ======
    معجميٌّ وحده          0.992   0.994    0.983
    دلاليٌّ وحده          0.938   0.983    0.898
    RRF مسطّح             0.960   1.000    0.932
    RRF موزون             0.969   0.994    0.949
    دمجٌ بالدرجات (هنا)   0.992   1.000    0.983
    =====================  ======  =======  ======

    الدرس: RRF يتجاهل *قوّة* الدليل لأنّه يعمل على الرتب وحدها. نتيجةٌ دلاليةٌ
    هامشية في الرتبة الأولى تُسهم بمقدار ``0.85/61 ≈ 0.014``، وهو أكبر من
    إسهام نتيجةٍ معجميةٍ قوية في الرتبة الخامسة ``1/65 ≈ 0.0154``؟ لا، لكنه
    يقترب منه كثيرًا حتى تتصدّر الضجيج عند اجتماع مسارين هامشيين. التطبيع
    يحفظ الفجوات، فيبقى الدليل الأقوى أقوى. ونُبقي RRF متاحًا للمقارنة
    والاختبار لأنّه الخيار الشائع في الأنظمة الكبيرة.
    """
    weights = dict(weights or {})
    totals: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    contribs: dict[str, dict[str, float]] = {}

    for name, pairs in rankings.items():
        if not pairs:
            continue
        w = float(weights.get(name, 1.0))
        top = max(s for _, s in pairs)
        top = top if top > 1e-12 else 1.0
        for rank, (doc, raw) in enumerate(pairs, start=1):
            add = w * max(float(raw), 0.0) / top
            totals[doc] = totals.get(doc, 0.0) + add
            sources.setdefault(doc, {})[name] = rank
            contribs.setdefault(doc, {})[name] = add

    hits = [
        FusionHit(doc_id=doc, score=score, sources=sources[doc], contributions=contribs[doc])
        for doc, score in totals.items()
    ]
    hits.sort(key=lambda h: (-h.score, h.doc_id))
    return hits[:limit] if limit else hits


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
