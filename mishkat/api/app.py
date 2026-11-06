"""خادم مِشكاة: JSON بسيط + استوديو ثابت.

كل نقطةٍ تُرجع ما يكفي للواجهة لتعرض *سببها*: لا نتيجة بلا تفسير. ولهذا فإن
``/api/search`` يُرجع الأثر (trace) مع النتائج، و``/api/doc`` يُرجع الجذور
والعلاقات والجيران.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..engine import Engine
from ..graph.query import GraphQueryError
from ..text.analyze import analyze
from ..text.roots import root_candidates
from ..text.stem import light_stem, strict_stem
from ..text.tokenize import tokenize

STUDIO_DIR = Path(__file__).resolve().parents[2] / "studio"
_STARTED = time.time()


@lru_cache(maxsize=1)
def engine() -> Engine:
    """محرّكٌ واحد مشترك (يُبنى مرة واحدة عند أول طلب)."""
    return Engine.load_or_build(verbose=False)


class GraphQueryBody(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class AnalyzeBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


def create_app() -> FastAPI:
    app = FastAPI(
        title="مِشكاة · Mishkat",
        description="محرّك معرفةٍ عربيٍّ يعمل دون إنترنت: بحث هجين، فضاء دلالي، رسم معرفي.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------- الأساس
    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "engine_loaded": engine.cache_info().currsize > 0,
            "uptime_s": round(time.time() - _STARTED, 1),
        }

    @app.get("/api/stats")
    def stats() -> dict:
        return engine().stats()

    @app.get("/api/domains")
    def domains() -> dict:
        eng = engine()
        counts: dict[str, int] = {}
        for d in eng.corpus.documents:
            counts[d.domain] = counts.get(d.domain, 0) + 1
        icons = eng.corpus.stats().get("icons", {})
        return {
            "domains": [
                {"name": k, "count": v, "icon": icons.get(k, "◆")}
                for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
        }

    # -------------------------------------------------------------- البحث
    @app.get("/api/search")
    def search(
        q: str = Query("", max_length=400),
        k: int = Query(8, ge=1, le=50),
        mode: str = Query("hybrid", pattern="^(hybrid|lexical|semantic)$"),
        domain: str | None = Query(None, max_length=40),
    ) -> dict:
        eng = engine()
        hits, trace = eng.search(q, k=k, mode=mode, domain=domain, with_trace=True)
        return {
            "query": q,
            "mode": mode,
            "trace": trace.to_dict(),
            "results": [h.to_dict() for h in hits],
        }

    @app.get("/api/doc/{doc_id}")
    def doc(doc_id: str) -> dict:
        eng = engine()
        try:
            detail = eng.doc_detail(doc_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"مستند غير موجود: {doc_id}") from exc
        detail["highlight"] = []
        return detail

    @app.get("/api/docs")
    def docs(limit: int = Query(200, ge=1, le=500)) -> dict:
        eng = engine()
        return {
            "documents": [
                {
                    "id": d.id,
                    "title": d.title,
                    "domain": d.domain,
                    "era": d.era,
                    "summary": d.summary,
                }
                for d in eng.corpus.documents[:limit]
            ]
        }

    # --------------------------------------------------------------- الرسم
    @app.get("/api/graph/overview")
    def graph_overview(limit: int = Query(140, ge=10, le=400)) -> dict:
        eng = engine()
        g = eng.graph
        nodes = []
        for n in list(g.nodes.values())[:limit]:
            nodes.append(
                {
                    "id": n.id,
                    "label": n.label,
                    "type": n.kind,
                    "degree": len(g.out_edges(n.id)) + len(g.in_edges(n.id)),
                }
            )
        ids = {n["id"] for n in nodes}
        edges = [
            e.to_dict()
            for edges in g._out.values()
            for e in edges
            if e.source in ids and e.target in ids
        ]
        return {"nodes": nodes, "edges": edges, "stats": g.stats()}

    @app.get("/api/graph/ego/{node_id:path}")
    def graph_ego(node_id: str, depth: int = Query(1, ge=1, le=3)) -> dict:
        eng = engine()
        if node_id not in eng.graph.nodes:
            raise HTTPException(status_code=404, detail="عقدة غير موجودة")
        return eng.ego(node_id, depth)

    @app.post("/api/graph/query")
    def graph_query(body: GraphQueryBody) -> dict:
        eng = engine()
        try:
            result = eng.graph_query(body.query)
        except GraphQueryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        labels: dict[str, str] = {}
        for node_id in result["nodes"]:
            node = eng.graph.nodes.get(node_id)
            labels[node_id] = node.label if node else node_id
        result["labels"] = labels
        return result

    @app.get("/api/graph/examples")
    def graph_examples() -> dict:
        return {
            "examples": [
                "MATCH (?x) -[ألّف]-> (?y)",
                "MATCH (?x:شخص) -[أثّر_في]-> (?y)",
                "MATCH (?x) -[ينتمي_إلى]-> (domain:فلك)",
                "MATCH (?a) -[يستند_إلى]-> (?b), (?b) -[يستند_إلى]-> (?c)",
                "MATCH (doc:opt-01) -[*]-> (?n)",
                "MATCH (?x) <-[أثّر_في]- (?y)",
            ]
        }

    # ------------------------------------------------------- الفضاء الدلالي
    @app.get("/api/words/{word}/neighbors")
    def word_neighbors(word: str, k: int = Query(10, ge=1, le=40)) -> dict:
        eng = engine()
        vec = eng.space.vector(word)
        return {
            "word": word,
            "found": vec is not None,
            "vector_dim": eng.space.dim,
            "neighbors": eng.word_neighbors(word, k),
        }

    @app.get("/api/clusters")
    def clusters(k: int = Query(6, ge=2, le=12)) -> dict:
        eng = engine()
        return eng.clusters(k)

    @app.get("/api/space/map")
    def space_map() -> dict:
        """إسقاط المستندات إلى بُعدين (لرسم خريطة المواضيع)."""
        eng = engine()
        if eng.space.doc_vectors is None:
            return {"points": []}
        import numpy as np

        X = eng.space.doc_vectors.astype(np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        # إسقاط على أول مُتجهين أساسيَّين (قيمٍ مفردة)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = U[:, :2] * S[:2]
        # تطبيع إلى [-1,1]
        span = np.abs(coords).max(axis=0)
        span[span == 0] = 1.0
        coords = coords / span
        points = []
        for i, doc_id in enumerate(eng.space.doc_ids):
            doc = eng.corpus._by_id.get(doc_id)
            points.append(
                {
                    "doc_id": doc_id,
                    "title": doc.title if doc else doc_id,
                    "domain": doc.domain if doc else "",
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                }
            )
        return {"points": points, "explained": round(float((S[:2] ** 2).sum() / (S**2).sum()), 3)}

    # --------------------------------------------------------- مختبر النصّ
    @app.post("/api/analyze")
    def analyze_text(body: AnalyzeBody) -> dict:
        text = body.text
        toks = tokenize(text)
        rows = []
        roots: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for t in toks:
            a = analyze(t.raw)
            roots[a.root] = roots.get(a.root, 0) + 1
            kinds[a.kind] = kinds.get(a.kind, 0) + 1
            rows.append(
                {
                    "raw": t.raw,
                    "norm": t.norm,
                    "stem": a.stem,
                    "root": a.root,
                    "kind": a.kind,
                    "stop": a.is_stop,
                    "confidence": a.confidence,
                    "start": t.start,
                    "end": t.end,
                }
            )
        return {
            "tokens": rows,
            "counts": {
                "tokens": len(toks),
                "types": len({t.norm for t in toks}),
                "roots": len(roots),
                "kinds": kinds,
            },
            "top_roots": [
                {"root": r, "count": c}
                for r, c in sorted(roots.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
            ],
        }

    @app.get("/api/word/{word}")
    def word_card(word: str) -> dict:
        return {
            "word": word,
            "analysis": analyze(word).to_dict(),
            "candidates": [
                {"root": c.root, "score": round(c.score, 2), "source": c.source}
                for c in root_candidates(word, limit=6)
            ],
            "stems": {"light": light_stem(word), "strict": strict_stem(word)},
        }

    # ------------------------------------------------------------ المحوّل
    @app.get("/api/lm/status")
    def lm_status() -> dict:
        """حالة المحوّل العصبي (يُبنى ويدرَّب في وحدةٍ منفصلة)."""
        from ..lm import lm_status as _status

        return _status()

    @app.get("/api/lm/generate")
    def lm_generate(
        prompt: str = Query("", max_length=120),
        length: int = Query(120, ge=10, le=600),
        temperature: float = Query(0.9, ge=0.1, le=2.0),
        top_k: int = Query(12, ge=1, le=60),
    ) -> dict:
        from ..lm import generate_text

        try:
            return generate_text(prompt, length=length, temperature=temperature, top_k=top_k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # ------------------------------------------------------------- الاستوديو
    if STUDIO_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STUDIO_DIR / "assets"), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STUDIO_DIR / "index.html")

        @app.get("/{path:path}")
        def static_or_index(path: str) -> Any:
            candidate = (STUDIO_DIR / path).resolve()
            if STUDIO_DIR.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return JSONResponse({"detail": "not found"}, status_code=404)

    return app


app = create_app()
