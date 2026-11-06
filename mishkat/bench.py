"""تقرير أداءٍ قابل للتكرار: كل رقمٍ هنا مقيس، لا مُدَّعى.

يقيس: زمن بناء كل طبقة، زمن الاستعلام في الأنماط الثلاثة، استدعاء HNSW مقابل
البحث الشامل، ومعدّل التوليد في المحوّل إن وُجد.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from .corpus import load_corpus
from .engine import Engine
from .index.hnsw import HNSWParams, build_from_matrix, recall_at_k

QUERIES = [
    "كيف قاس البيروني الأرض؟",
    "ابن الهيثم والضوء",
    "الوقف وتمويل المدارس",
    "مشكلة المياه في المدن",
    "تعلم الآلة والانتباه",
    "الخوارزمي والجبر",
    "الدورة الدموية الصغرى",
    "علم العروض عند الخليل",
    "التشفير والمفاتيح",
    "الطاقة الشمسية والتخزين",
]


def _timeit(fn, repeat: int = 5) -> dict:
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.mean(times), 2),
        "p95_ms": round(sorted(times)[max(0, int(len(times) * 0.95) - 1)], 2),
        "n": len(out) if hasattr(out, "__len__") else None,
    }


def run_benchmark(*, queries: int = 40) -> dict:
    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    t0 = time.perf_counter()
    corpus = load_corpus()
    report["corpus_load_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    report["corpus"] = {"documents": len(corpus), "tokens": corpus.stats()["tokens"]}

    t0 = time.perf_counter()
    engine = Engine.build(corpus)
    report["engine_build_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    report["build_breakdown"] = {
        "index": engine.index.stats(),
        "space": engine.space.stats(),
        "hnsw": {k: v for k, v in engine.hnsw.stats().items() if k != "params"},
        "graph": {k: v for k, v in engine.graph.stats().items() if k != "relations"},
    }

    qs = QUERIES[: max(1, min(queries, len(QUERIES)))]
    report["search"] = {
        mode: _timeit(lambda m=mode: [engine.search(q, k=8, mode=m) for q in qs], repeat=3)
        for mode in ("lexical", "semantic", "hybrid")
    }

    if engine.space.doc_vectors is not None and engine.space.doc_vectors.shape[0] > 1:
        queries_mat = engine.space.doc_vectors[:10]
        t0 = time.perf_counter()
        approx = [[engine.hnsw.search(v, 5) for v in queries_mat]]
        ann_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        for v in queries_mat:
            engine.hnsw.search(v, 5, exact=True)
        exact_ms = (time.perf_counter() - t0) * 1000
        report["hnsw"] = {
            "recall_at_5": round(recall_at_k(engine.hnsw, queries_mat, 5), 3),
            "ann_ms": round(ann_ms, 2),
            "exact_ms": round(exact_ms, 2),
            "speedup": round(exact_ms / max(ann_ms, 1e-9), 2),
        }

    # قياس HNSW على أحجامٍ متعددة: الفرق يظهر حين تكبر البيانات
    rng = np.random.default_rng(0)
    scaling = []
    for n, dim in ((200, 32), (1000, 32), (3000, 32)):
        centers = rng.normal(size=(24, dim))
        X = np.vstack([centers[i % 24] + 0.4 * rng.normal(size=dim) for i in range(n)]).astype(np.float32)
        t0 = time.perf_counter()
        idx = build_from_matrix(X, [f"v{i}" for i in range(n)], HNSWParams(M=16, ef_construction=120, ef_search=48))
        build_ms = (time.perf_counter() - t0) * 1000
        qs_mat = X[:20]
        t0 = time.perf_counter()
        for v in qs_mat:
            idx.search(v, 10)
        ann = (time.perf_counter() - t0) * 1000 / 20
        t0 = time.perf_counter()
        for v in qs_mat:
            idx.search(v, 10, exact=True)
        exact = (time.perf_counter() - t0) * 1000 / 20
        scaling.append(
            {
                "n": n,
                "build_ms": round(build_ms, 1),
                "ann_ms": round(ann, 3),
                "exact_ms": round(exact, 3),
                "speedup": round(exact / max(ann, 1e-9), 2),
                "recall_at_10": round(recall_at_k(idx, qs_mat, 10), 3),
            }
        )
    report["hnsw_scaling"] = scaling

    # تقييم الجودة: الأداء بلا جودة رقمٌ ناقص
    from .engine import Engine as _Engine
    from .evaluate import run_eval

    try:
        eng = _Engine.load()
        evals = run_eval(eng, k=10, include_rrf=True)
        report["retrieval_eval"] = [r.to_dict() | {"groups": {k: v.to_dict() for k, v in (r.groups or {}).items()}} for r in evals]
    except Exception as exc:  # pragma: no cover - التقييم اختياريٌّ في التقرير
        report["retrieval_eval"] = {"error": str(exc)}

    report["graph_query_ms"] = _timeit(
        lambda: engine.graph_query("MATCH (?x) -[ألّف]-> (?y)"), repeat=5
    )

    try:
        from .lm.sample import generate, lm_status

        st = lm_status()
        if st.get("trained"):
            t0 = time.perf_counter()
            generate("العلم", length=120)
            report["lm"] = {
                "chars_per_s": round(120 / max(1e-9, time.perf_counter() - t0), 1),
                "params": st["params"],
            }
    except Exception:
        pass

    return report
