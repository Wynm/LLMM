"""المحرّك: طبقةٌ واحدة تربط النواة اللغوية بالاسترجاع والفضاء والرسم.

الواجهة العليا لكل شيء:

    >>> from mishkat.engine import Engine
    >>> eng = Engine.build()
    >>> eng.search("كيف قاس البيروني الأرض؟", k=3)[0].title

التصميم
-------
المحرّك **لا يعرف** كيف يعمل BM25 ولا HNSW؛ يستدعي واجهاتهما فقط. هذا الفصل
يسمح بتبديل أي طبقةٍ دون لمس الباقي، ويجعل الاختبار ممكنًا: يمكن حقن فضاءٍ
صغيرٍ بدل الفضاء الحقيقي.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .corpus import Corpus, Document, load_corpus
from .embed.space import WordSpace, build_space, kmeans
from .graph.infer import InferenceEngine
from .graph.query import run_query
from .graph.store import GraphStore
from .index.bm25 import BM25, BM25Params, best_snippet
from .index.hybrid import rrf_fuse
from .index.inverted import InvertedIndex
from .index.hnsw import HNSWParams, build_from_matrix
from .index.query import parse_query
from .text.analyze import analyze
from .text.normalize import SEARCH, normalize

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class Hit:
    """نتيجة بحثٍ موحّدة، تحمل سببها."""

    doc_id: str
    title: str
    domain: str
    score: float
    lexical: float = 0.0
    semantic: float = 0.0
    snippet: str = ""
    matched: tuple[str, ...] = ()
    rank_lexical: int | None = None
    rank_semantic: int | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "domain": self.domain,
            "score": round(self.score, 5),
            "lexical": round(self.lexical, 4),
            "semantic": round(self.semantic, 4),
            "snippet": self.snippet,
            "matched": list(self.matched),
            "ranks": {"lexical": self.rank_lexical, "semantic": self.rank_semantic},
        }


@dataclass
class Trace:
    """أثرٌ تفصيليّ لخطوات البحث (شفافية المحرّك)."""

    query: str = ""
    normalized: str = ""
    terms: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    lexical_ms: float = 0.0
    semantic_ms: float = 0.0
    fuse_ms: float = 0.0
    lexical_candidates: int = 0
    semantic_candidates: int = 0
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "normalized": self.normalized,
            "terms": self.terms,
            "roots": self.roots,
            "timing_ms": {
                "lexical": round(self.lexical_ms, 3),
                "semantic": round(self.semantic_ms, 3),
                "fuse": round(self.fuse_ms, 3),
                "total": round(self.lexical_ms + self.semantic_ms + self.fuse_ms, 3),
            },
            "candidates": {
                "lexical": self.lexical_candidates,
                "semantic": self.semantic_candidates,
            },
            "weights": self.weights,
        }


class Engine:
    """محرّك مِشكاة."""

    def __init__(
        self,
        corpus: Corpus,
        index: InvertedIndex,
        bm25: BM25,
        space: WordSpace,
        hnsw,
        graph: GraphStore,
        inference: InferenceEngine,
        *,
        built_at: float | None = None,
        build_seconds: float = 0.0,
    ) -> None:
        self.corpus = corpus
        self.index = index
        self.bm25 = bm25
        self.space = space
        self.hnsw = hnsw
        self.graph = graph
        self.inference = inference
        self.built_at = built_at or time.time()
        self.build_seconds = build_seconds
        self._doc_sim_cache: dict[str, list[tuple[str, float]]] = {}

    # ---------------------------------------------------------------- بناء
    @classmethod
    def build(
        cls,
        corpus: Corpus | None = None,
        *,
        dim: int = 96,
        window: int = 5,
        max_vocab: int = 2500,
        corpus_root: str | Path | None = None,
        verbose: bool = False,
    ) -> "Engine":
        """يبني كل شيء من المجموعة (فهرس، فضاء، HNSW، رسم)."""
        t0 = time.perf_counter()
        c = corpus or load_corpus(corpus_root)

        index = InvertedIndex()
        for d in c.documents:
            index.add_document(d.id, d.fields)
        index.finalize()
        if verbose:
            print(f"  فهرس مقلوب: {index.vocab_size} مصطلحًا، {index.size} مستندًا")

        bm25 = BM25(index, BM25Params())

        space = build_space(c.documents, dim=dim, window=window, max_vocab=max_vocab)
        if verbose:
            print(f"  فضاء دلالي: {space.size} متجهًا في {space.dim} بعدًا")

        hnsw = build_from_matrix(
            space.doc_vectors, space.doc_ids, HNSWParams(M=12, ef_construction=120, ef_search=48)
        )
        if verbose:
            print(f"  HNSW: {hnsw.size} عنصرًا، عمق {hnsw.max_level}")

        graph = GraphStore.from_corpus(c)
        inference = InferenceEngine(graph)
        inference.derive(apply=True)
        if verbose:
            print(f"  رسم: {len(graph)} عقدة، {graph.edge_count()} حافة (بعد الاستدلال)")

        return cls(
            corpus=c,
            index=index,
            bm25=bm25,
            space=space,
            hnsw=hnsw,
            graph=graph,
            inference=inference,
            build_seconds=time.perf_counter() - t0,
        )

    # -------------------------------------------------------------- البحث
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        domain: str | None = None,
        mode: str = "hybrid",
        with_trace: bool = False,
    ) -> list[Hit] | tuple[list[Hit], Trace]:
        """بحثٌ هجين (معجمي + دلالي) أو أحد المسارين."""
        if not query or not query.strip():
            return ([], Trace()) if with_trace else []

        trace = Trace(query=query, normalized=normalize(query, SEARCH))
        pq = parse_query(query)
        for t in pq.all_positive():
            a = analyze(t)
            trace.terms.append(a.stem)
            trace.roots.append(a.root)

        lexical: list = []
        semantic: list[tuple[str, float]] = []

        pool = max(k * 4, 20)

        if mode in {"hybrid", "lexical"}:
            t0 = time.perf_counter()
            lexical = self.bm25.search(pq, k=pool)
            trace.lexical_ms = (time.perf_counter() - t0) * 1000
            trace.lexical_candidates = len(lexical)

        if mode in {"hybrid", "semantic"}:
            t0 = time.perf_counter()
            semantic = self.space.search_docs(query, k=pool)
            trace.semantic_ms = (time.perf_counter() - t0) * 1000
            trace.semantic_candidates = len(semantic)

        t0 = time.perf_counter()
        if mode == "lexical":
            fused = [
                {"doc_id": h.doc_id, "score": h.score, "lex": h.score, "sem": 0.0,
                 "matched": h.matched, "rl": i + 1, "rs": None}
                for i, h in enumerate(lexical)
            ]
        elif mode == "semantic":
            fused = [
                {"doc_id": d, "score": s, "lex": 0.0, "sem": s, "matched": (),
                 "rl": None, "rs": i + 1}
                for i, (d, s) in enumerate(semantic)
            ]
        else:
            rankings = {
                "lexical": [h.doc_id for h in lexical],
                "semantic": [d for d, _ in semantic],
            }
            weights = {"lexical": 1.0, "semantic": 0.85}
            trace.weights = weights
            fus = rrf_fuse(rankings, k=60, weights=weights)
            lex_map = {h.doc_id: h for h in lexical}
            sem_map = dict(semantic)
            fused = [
                {
                    "doc_id": f.doc_id,
                    "score": f.score,
                    "lex": lex_map[f.doc_id].score if f.doc_id in lex_map else 0.0,
                    "sem": sem_map.get(f.doc_id, 0.0),
                    "matched": lex_map[f.doc_id].matched if f.doc_id in lex_map else (),
                    "rl": f.sources.get("lexical"),
                    "rs": f.sources.get("semantic"),
                }
                for f in fus
            ]
        trace.fuse_ms = (time.perf_counter() - t0) * 1000

        hits: list[Hit] = []
        for row in fused:
            doc = self.corpus._by_id.get(row["doc_id"])
            if doc is None:
                continue
            if domain and doc.domain != domain:
                continue
            hits.append(
                Hit(
                    doc_id=doc.id,
                    title=doc.title,
                    domain=doc.domain,
                    score=float(row["score"]),
                    lexical=float(row["lex"]),
                    semantic=float(row["sem"]),
                    matched=tuple(row["matched"]),
                    rank_lexical=row["rl"],
                    rank_semantic=row["rs"],
                )
            )
        hits = hits[:k]

        for h in hits:
            doc = self.corpus.doc(h.doc_id)
            terms = set(h.matched) | {
                t for t in trace.terms if t in {t2.norm for t2 in _doc_tokens(doc)}
            }
            h.snippet = best_snippet(doc.body, terms or set(trace.terms))

        return (hits, trace) if with_trace else hits

    # ------------------------------------------------------ تشابه المستندات
    def similar_docs(self, doc_id: str, k: int = 6) -> list[tuple[str, float]]:
        """مستنداتٌ مشابهة: مزيجٌ من التشابه الدلالي والتشابه البنيوي في الرسم."""
        if doc_id in self._doc_sim_cache:
            return self._doc_sim_cache[doc_id][:k]
        if self.space.doc_vectors is None or doc_id not in self.space.doc_ids:
            return []
        i = self.space.doc_ids.index(doc_id)
        sims = self.space.doc_vectors @ self.space.doc_vectors[i]

        # تشابه بنيوي: تقاطع الجيران في الرسم (معامل جاكار)
        gsrc = f"doc:{doc_id}"
        my_nb = set(self.graph.neighbors(gsrc)) | set(
            self.graph.neighbors(gsrc, direction="in")
        )
        blended: list[tuple[str, float]] = []
        for j, d_id in enumerate(self.space.doc_ids):
            if d_id == doc_id:
                continue
            other = f"doc:{d_id}"
            other_nb = set(self.graph.neighbors(other)) | set(
                self.graph.neighbors(other, direction="in")
            )
            union = my_nb | other_nb
            jac = len(my_nb & other_nb) / len(union) if union else 0.0
            score = 0.75 * float(sims[j]) + 0.25 * jac
            blended.append((d_id, score))
        blended.sort(key=lambda x: -x[1])
        self._doc_sim_cache[doc_id] = blended
        return blended[:k]

    # -------------------------------------------------------------- الكيانات
    def doc_detail(self, doc_id: str) -> dict:
        """ملفٌّ كامل عن مستند: تحليل، جيران، علاقات، كيانات."""
        doc = self.corpus.doc(doc_id)
        tokens = _doc_tokens(doc)
        roots: dict[str, int] = {}
        for t in tokens:
            a = analyze(t.raw)
            if a.root:
                roots[a.root] = roots.get(a.root, 0) + 1
        top_roots = sorted(roots.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

        gnode = f"doc:{doc_id}"
        out_edges = [e.to_dict() for e in self.graph.out_edges(gnode)]
        in_edges = [e.to_dict() for e in self.graph.in_edges(gnode)]

        related = [
            {
                "doc_id": d,
                "title": self.corpus._by_id[d].title if d in self.corpus._by_id else d,
                "domain": self.corpus._by_id[d].domain if d in self.corpus._by_id else "",
                "score": round(s, 4),
            }
            for d, s in self.similar_docs(doc_id, 6)
        ]
        return {
            "document": doc.to_dict(),
            "analysis": {
                "tokens": len(tokens),
                "types": len({t.norm for t in tokens}),
                "top_roots": [{"root": r, "count": c} for r, c in top_roots],
            },
            "graph": {"out": out_edges, "in": in_edges},
            "related": related,
        }

    def entities(self) -> list[dict]:
        return [e.to_dict() for e in self.corpus.entities]

    def graph_query(self, query: str) -> dict:
        return run_query(self.graph, query).to_dict()

    def ego(self, node_id: str, depth: int = 1) -> dict:
        return self.graph.ego(node_id, depth)

    def word_neighbors(self, word: str, k: int = 10) -> list[dict]:
        return [{"term": t, "score": round(s, 4)} for t, s in self.space.similar(word, k)]

    def clusters(self, k: int = 6) -> dict:
        """تجميع المستندات موضوعيًّا (k-means على متجهات المستندات)."""
        if self.space.doc_vectors is None:
            return {"clusters": []}
        centers, labels = kmeans(self.space.doc_vectors, k=k)
        groups: dict[int, list[str]] = {}
        for doc_id, lab in zip(self.space.doc_ids, labels):
            groups.setdefault(int(lab), []).append(doc_id)
        out = []
        inv_vocab = {i: w for w, i in self.space.vocab.items()}
        for lab, docs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            center = centers[lab]
            top_terms = [
                inv_vocab[i]
                for i in np.argsort(-(self.space.vectors @ center))[:6]
                if int(i) in inv_vocab
            ]
            out.append(
                {
                    "id": lab,
                    "size": len(docs),
                    "docs": [
                        {"doc_id": d, "title": self.corpus._by_id[d].title if d in self.corpus._by_id else d}
                        for d in docs
                    ],
                    "top_terms": top_terms,
                }
            )
        return {"clusters": out}

    # -------------------------------------------------------------- إحصاء
    def stats(self) -> dict:
        corpus_stats = self.corpus.stats()
        return {
            "corpus": corpus_stats,
            "index": self.index.stats(),
            "space": self.space.stats(),
            "hnsw": self.hnsw.stats(),
            "graph": self.graph.stats(),
            "inference": self.inference.summary(),
            "build_seconds": round(self.build_seconds, 2),
            "built_at": self.built_at,
            "version": "0.1.0",
        }

    # ---------------------------------------------------------------- حفظ
    def save(self, directory: Path | str = ARTIFACT_DIR) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with (d / "engine.pkl").open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (d / "stats.json").write_text(
            json.dumps(self.stats(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str = ARTIFACT_DIR) -> "Engine":
        d = Path(directory)
        with (d / "engine.pkl").open("rb") as fh:
            return pickle.load(fh)

    @classmethod
    def load_or_build(cls, directory: Path | str = ARTIFACT_DIR, **kwargs) -> "Engine":
        d = Path(directory)
        if (d / "engine.pkl").exists():
            try:
                return cls.load(d)
            except Exception:
                pass
        eng = cls.build(**kwargs)
        eng.save(d)
        return eng


def _doc_tokens(doc: Document):
    from .text.tokenize import tokenize

    return tokenize(doc.text)
