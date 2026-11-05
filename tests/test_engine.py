"""اختبارات المجموعة والمحرّك والفضاء الدلالي."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mishkat.corpus import CorpusError, load_corpus
from mishkat.embed.cooccur import build_cooccurrence, context_term
from mishkat.embed.ppmi import ppmi_matrix
from mishkat.embed.space import build_space, kmeans
from mishkat.embed.svd import explained_energy, randomized_svd
from mishkat.text import extract_root


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


# ------------------------------------------------------------------ المجموعة
def test_corpus_loads(corpus) -> None:
    assert len(corpus) >= 50
    assert corpus.entities
    assert corpus.edges


def test_corpus_ids_unique(corpus) -> None:
    ids = [d.id for d in corpus.documents]
    assert len(ids) == len(set(ids))


def test_corpus_doc_lookup(corpus) -> None:
    d = corpus.doc(corpus.documents[0].id)
    assert d.body and d.title


def test_corpus_missing_doc(corpus) -> None:
    with pytest.raises(CorpusError):
        corpus.doc("nope")


def test_corpus_edges_resolve(corpus) -> None:
    known = {f"doc:{d.id}" for d in corpus.documents} | {e.id for e in corpus.entities}
    for e in corpus.edges:
        assert e.source in known and e.target in known


def test_corpus_stats(corpus) -> None:
    st = corpus.stats()
    assert st["documents"] == len(corpus)
    assert st["tokens"] > 1000
    assert st["domains"]


def test_corpus_rejects_bad_schema(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "title": "بلا متن"}, ensure_ascii=False) + "\n")
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_corpus_rejects_short_body(tmp_path: Path) -> None:
    row = {"id": "x", "title": "ت", "domain": "علم", "summary": "س", "body": "كلمتان فقط"}
    (tmp_path / "a.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


def test_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = {
        "id": "x",
        "title": "ت",
        "domain": "علم",
        "summary": "س",
        "body": " ".join(["كلمة"] * 25),
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"
    (tmp_path / "a.jsonl").write_text(line)
    (tmp_path / "b.jsonl").write_text(line)
    with pytest.raises(CorpusError):
        load_corpus(tmp_path)


# --------------------------------------------------------------- الفضاء الدلالي
def test_context_term_uses_confident_root() -> None:
    assert context_term("المدرسة") == "درس"
    assert len(context_term("xyz")) >= 3


def test_cooccurrence_shape() -> None:
    docs = ["النجم ساطع في السماء", "السماء صافية والنجم بعيد", "الماء بارد"]
    vocab, counts = build_cooccurrence(docs, window=3)
    assert vocab and counts
    assert all(isinstance(k, tuple) and len(k) == 2 for k in counts)


def test_ppmi_non_negative() -> None:
    docs = ["علم الفلك رصد", "الفلك رصد النجوم", "الطب علاج المرض"]
    vocab, counts = build_cooccurrence(docs, window=2)
    m = ppmi_matrix(vocab, counts)
    assert m.shape == (len(vocab), len(vocab))
    assert (m >= 0).all()


def test_randomized_svd_approximates() -> None:
    rng = np.random.default_rng(0)
    # مصفوفةٌ رتبتها 5 فعلًا: يجب أن يُعيد SVD المضغوط تمثيلها بدقّة
    low = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 40))
    U, S, Vt = randomized_svd(low, n_components=8)
    approx = U @ np.diag(S) @ Vt
    assert approx.shape == low.shape
    assert np.linalg.norm(low - approx) / np.linalg.norm(low) < 0.05
    assert explained_energy(S) > 0.9


def test_space_semantics(corpus) -> None:
    sp = build_space(corpus.documents, dim=48, max_vocab=1500)
    assert sp.size > 100
    assert sp.dim == 48
    # البحث الدلالي عن النحو يجب أن يعيد مستندات لسانية
    hits = sp.search_docs("النحو العربي وقواعده", k=3)
    assert hits
    assert hits[0][1] > 0


def test_space_embed_text_zero_for_empty(corpus) -> None:
    sp = build_space(corpus.documents, dim=32)
    assert not np.any(sp.embed_text(""))


def test_kmeans_labels(corpus) -> None:
    sp = build_space(corpus.documents, dim=32)
    centers, labels = kmeans(sp.doc_vectors, 4)
    assert centers.shape[0] == 4
    assert len(set(labels.tolist())) >= 2


# ------------------------------------------------------------------ المحرّك
@pytest.fixture(scope="module")
def engine(corpus):
    from mishkat.engine import Engine

    return Engine.build(corpus, dim=48, max_vocab=2000)


def test_engine_search_finds_expert(engine) -> None:
    hits = engine.search("ابن الهيثم والضوء والبصريات", k=5)
    assert hits
    assert hits[0].doc_id in {"opt-01", "opt-02"}


def test_engine_search_empty(engine) -> None:
    assert engine.search("") == []


def test_engine_search_modes(engine) -> None:
    lex = engine.search("الفلك", k=3, mode="lexical")
    sem = engine.search("الفلك", k=3, mode="semantic")
    hyb = engine.search("الفلك", k=3, mode="hybrid")
    assert lex and sem is not None and hyb


def test_engine_domain_filter(engine) -> None:
    hits = engine.search("العلم", k=8, domain="فلك")
    assert all(h.domain == "فلك" for h in hits)


def test_engine_trace(engine) -> None:
    hits, trace = engine.search("النجوم", k=3, with_trace=True)
    t = trace.to_dict()
    assert t["terms"] and t["timing_ms"]["total"] >= 0


def test_engine_similar_docs(engine) -> None:
    sims = engine.similar_docs("opt-01", 3)
    assert sims and sims[0][0] != "opt-01"
    assert sims[0][1] >= sims[-1][1]


def test_engine_doc_detail(engine) -> None:
    d = engine.doc_detail("med-01")
    assert d["document"]["id"] == "med-01"
    assert d["analysis"]["top_roots"]
    assert d["related"]


def test_engine_graph_query(engine) -> None:
    r = engine.graph_query("MATCH (?x) -[ألّف]-> (doc:math-01)")
    assert r["count"] >= 1


def test_engine_word_neighbors(engine) -> None:
    js = engine.word_neighbors("مناعة", 5)
    assert js and all("term" in j for j in js)


def test_engine_clusters(engine) -> None:
    c = engine.clusters(4)
    assert c["clusters"]
    assert sum(x["size"] for x in c["clusters"]) == len(engine.corpus)


def test_engine_stats(engine) -> None:
    st = engine.stats()
    for key in ("corpus", "index", "space", "hnsw", "graph"):
        assert key in st


def test_engine_roots_are_used(engine) -> None:
    # «المياه» و«الماء» يجب أن يتقاربا في البحث
    hits = engine.search("المياه", k=5)
    assert any(h.doc_id in {"water-01", "eng-02"} for h in hits)
