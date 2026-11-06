"""تقييم جودة الاسترجاع: قياسٌ لا انطباع.

المشكلة
-------
«البحث يبدو جيدًا» ليست عبارةً تُشحن. نحتاج أرقامًا على مجموعة استعلاماتٍ
مُعنونة مسبقًا (qrels مصغّرة)، ثم نقارن بها كل استراتيجية دمج.

المقاييس
--------
- **MRR@k**: متوسّط مقلوب رتبة أوّل نتيجةٍ صحيحة. يجيب: «كم يبعد أوّل جواب؟»
- **Recall@k**: نسبة الاستعلامات التي ظهر فيها جوابٌ صحيح في أعلى k.
- **nDCG@k**: مكافأة الترتيب مع خصمٍ لوغاريتميّ للرتبة (المستويات هنا ثنائية).
- **Hit@1**: نسبة الاستعلامات التي كان أوّل جوابٍ فيها صحيحًا.

الملفّ: ``data/eval/queries.jsonl`` وفيه ``query`` و``expect`` (معرّفاتٌ
مرتّبة بالأهمية) و``kind`` (lexical/semantic/mixed/entity) لتحليل الأخطاء.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = ROOT / "data" / "eval" / "queries.jsonl"


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expect: tuple[str, ...]
    kind: str = "mixed"

    def to_dict(self) -> dict:
        return {"query": self.query, "expect": list(self.expect), "kind": self.kind}


def load_queries(path: Path | str = QUERIES_PATH) -> list[EvalQuery]:
    """يقرأ مجموعة الاستعلامات المُعنونة، ويرفض الملفّ المشوّه بوضوح."""
    out: list[EvalQuery] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - حمايةٌ فقط
            raise ValueError(f"سطر {lineno}: JSON غير صالح ({exc.msg})") from exc
        q = str(row.get("query", "")).strip()
        expect = [str(x) for x in row.get("expect", [])]
        if not q or not expect:
            raise ValueError(f"سطر {lineno}: استعلامٌ بلا نصّ أو بلا نتائج متوقّعة")
        if q in seen:
            raise ValueError(f"سطر {lineno}: استعلامٌ مكرّر ({q})")
        seen.add(q)
        out.append(EvalQuery(q, tuple(expect), str(row.get("kind", "mixed"))))
    if not out:
        raise ValueError("ملفّ الاستعلامات فارغ")
    return out


# --------------------------------------------------------------- المقاييس


def reciprocal_rank(ranked: Sequence[str], expect: Sequence[str], k: int = 10) -> float:
    """مقلوب رتبة أوّل نتيجةٍ صحيحة داخل أعلى k (صفرٌ إن لم توجد)."""
    wanted = set(expect)
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in wanted:
            return 1.0 / i
    return 0.0


def recall_at(ranked: Sequence[str], expect: Sequence[str], k: int = 10) -> float:
    """نسبة النتائج المتوقّعة التي ظهرت في أعلى k."""
    wanted = set(expect)
    if not wanted:
        return 0.0
    return len(wanted & set(ranked[:k])) / len(wanted)


def ndcg_at(ranked: Sequence[str], expect: Sequence[str], k: int = 10) -> float:
    """nDCG ثنائيٌّ: 1/log2(r+1) لكل نتيجةٍ صحيحة، مُطبَّعًا على الأفضل ممكن."""
    wanted = set(expect)
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in wanted:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class EvalReport:
    strategy: str
    n: int
    mrr: float
    recall: float
    ndcg: float
    hit1: float
    misses: list[dict]
    latency_ms: float = 0.0
    groups: dict[str, "EvalReport"] = None  # ← التوزيع بحسب الفئة (يُملأ لاحقًا)

    def group_rows(self) -> list[str]:
        out = []
        for name, g in sorted((self.groups or {}).items()):
            out.append(f"   └ {name:<16} n={g.n:<3} MRR {g.mrr:.3f}  R@10 {g.recall:.3f}  H@1 {g.hit1:.3f}")
        return out

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n": self.n,
            "mrr": round(self.mrr, 4),
            "recall": round(self.recall, 4),
            "ndcg": round(self.ndcg, 4),
            "hit1": round(self.hit1, 4),
            "latency_ms": round(self.latency_ms, 3),
            "misses": self.misses[:8],
        }

    def row(self) -> str:
        return (
            f"{self.strategy:<22} MRR {self.mrr:.3f}  R@10 {self.recall:.3f}  "
            f"nDCG {self.ndcg:.3f}  H@1 {self.hit1:.3f}  {self.latency_ms:6.1f} ms"
        )


def evaluate(
    searcher: Callable[[str, int], Sequence[str]],
    queries: Iterable[EvalQuery],
    *,
    strategy: str = "",
    k: int = 10,
    latency_ms: float = 0.0,
    bucket_of: Callable[[EvalQuery], str] | None = None,
) -> EvalReport:
    """يشغّل أي دالة بحث ويعيد المقاييس، مع تسجيل الإخفاقات الكاملة.

    ``bucket_of`` تُصنّف الاستعلامات (مثلًا: «مصطلحٌ مشترك» مقابل «إعادة صياغة»)
    فنرى أين تتفوّق كل استراتيجية بدل رقمٍ واحدٍ يخفي الفروق.
    """
    qs = list(queries)
    acc: dict[str, list[float]] = {}

    def bump(bucket: str, r: float, nd: float, rc: float, h1v: float) -> None:
        row = acc.setdefault(bucket, [0.0, 0.0, 0.0, 0.0, 0.0])
        row[0] += r
        row[1] += nd
        row[2] += rc
        row[3] += h1v
        row[4] += 1.0

    misses: list[dict] = []
    per_bucket_misses: dict[str, list[dict]] = {}
    for q in qs:
        ranked = list(searcher(q.query, k))
        r = reciprocal_rank(ranked, q.expect, k)
        nd = ndcg_at(ranked, q.expect, k)
        rc = recall_at(ranked, q.expect, k)
        h1v = 1.0 if r == 1.0 else 0.0
        bucket = bucket_of(q) if bucket_of else q.kind
        bump(bucket, r, nd, rc, h1v)
        bump("__all__", r, nd, rc, h1v)
        if r == 0.0:
            miss = {"query": q.query, "kind": q.kind, "expect": list(q.expect), "got": ranked[:5]}
            misses.append(miss)
            per_bucket_misses.setdefault(bucket, []).append(miss)

    total = acc.get("__all__", [0, 0, 0, 0, 1])
    n = max(int(total[4]), 1)
    groups: dict[str, EvalReport] = {}
    for name, row in acc.items():
        if name == "__all__":
            continue
        gn = max(int(row[4]), 1)
        groups[name] = EvalReport(name, int(row[4]), row[0] / gn, row[2] / gn, row[1] / gn, row[3] / gn, per_bucket_misses.get(name, []))
    return EvalReport(
        strategy, len(qs), total[0] / n, total[2] / n, total[1] / n, total[3] / n, misses, latency_ms, groups
    )


# ---------------------------------------------------- تصنيفٌ لا يعتمد على الحدس
def term_overlap(engine, q: EvalQuery) -> float:
    """نسبة جذوع كلمات الاستعلام التي تظهر فعلًا في المستندات المتوقّعة.

    الغرض: لا نثق بتسميتنا اليدوية (lexical/semantic) لأنها انطباع؛ نحسب
    التطابق الحقيقي فنعرف أيّ استعلامٍ «معجميّ» وأيّها «إعادة صياغة».
    """
    from .text.analyze import analyze

    stems = {analyze(w).stem for w in q.query.split() if len(w) > 2}
    stems = {s for s in stems if s}
    if not stems:
        return 0.0
    hits = 0
    for doc_id in q.expect:
        doc = engine.corpus.doc(doc_id)
        forms = {analyze(t).stem for t in doc.body.split()}
        hits = max(hits, len(stems & forms) / len(stems))
    return hits


def bucket_of(engine):
    """مُصنِّفٌ ثلاثيّ: مصطلحٌ صريح، صريحٌ جزئيًّا، إعادة صياغة."""

    def f(q: EvalQuery) -> str:
        ov = term_overlap(engine, q)
        if ov >= 0.75:
            return "term-explicit"
        if ov >= 0.34:
            return "partial-term"
        return "paraphrase"

    return f


# ------------------------------------------------------------ تشغيلٌ على المحرّك
def run_eval(
    engine,
    queries: Sequence[EvalQuery] | None = None,
    *,
    k: int = 10,
    include_rrf: bool = True,
) -> list[EvalReport]:
    """يقارن مسارات البحث الفعلية على مجموعةٍ مُعنونة.

    يُشغّل: المعجمي، الدلالي، الهجين المُشحون (كما يستخدمه المحرّك)، ثم RRF
    كخطّ أساسٍ للمقارنة — فالهدف معرفة ما إذا كان خيار الدمج مُبرَّرًا بالبيانات.
    """
    import time as _time

    from .index.hybrid import rrf_fuse
    from .index.query import parse_query

    qs = list(queries or load_queries())
    bucket = bucket_of(engine)
    reports: list[EvalReport] = []

    def timed(fn):
        def wrapped(q: str, kk: int):
            t0 = _time.perf_counter()
            out = fn(q, kk)
            wrapped.elapsed += _time.perf_counter() - t0  # type: ignore[attr-defined]
            return out

        wrapped.elapsed = 0.0  # type: ignore[attr-defined]
        return wrapped

    lex = timed(lambda q, kk: [h.doc_id for h in engine.bm25.search(parse_query(q), k=kk)])
    sem = timed(lambda q, kk: [d for d, _ in engine.space.search_docs(q, k=kk)])
    hyb = timed(lambda q, kk: [h.doc_id for h in engine.search(q, k=kk, mode="hybrid")])

    for name, fn in (("معجميّ", lex), ("دلاليّ", sem), ("هجين (المُشحون)", hyb)):
        rep = evaluate(fn, qs, strategy=name, k=k, bucket_of=bucket)
        rep.latency_ms = 1000 * fn.elapsed / max(len(qs), 1)
        reports.append(rep)

    if include_rrf:
        def rrf(q: str, kk: int):
            lex_hits = engine.bm25.search(parse_query(q), k=max(kk * 3, 24))
            sem_hits = engine.space.search_docs(q, k=max(kk * 3, 24))
            fus = rrf_fuse(
                {"lexical": [h.doc_id for h in lex_hits], "semantic": [d for d, _ in sem_hits]},
                k=60,
                weights={"lexical": 1.0, "semantic": 0.85},
                limit=kk,
            )
            return [h.doc_id for h in fus]

        rrf_t = timed(rrf)
        rep = evaluate(rrf_t, qs, strategy="RRF (مقارنة)", k=k, bucket_of=bucket)
        rep.latency_ms = 1000 * rrf_t.elapsed / max(len(qs), 1)
        reports.append(rep)

    return reports
