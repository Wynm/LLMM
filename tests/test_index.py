"""اختبارات الاسترجاع: الفهرس المقلوب، BM25، HNSW، والدمج."""

from __future__ import annotations

import numpy as np
import pytest

from mishkat.index.bm25 import BM25, BM25Params
from mishkat.index.hnsw import HNSW, HNSWParams, build_from_matrix, recall_at_k
from mishkat.index.hybrid import rrf_fuse, weighted_score_fuse
from mishkat.index.inverted import InvertedIndex, index_text, query_terms
from mishkat.index.query import QueryError, parse_query


@pytest.fixture(scope="module")
def index() -> InvertedIndex:
    docs = {
        "a": {
            "title": "علم الفلك عند العرب",
            "summary": "رصد النجوم والكواكب",
            "body": "طوّر الفلكيون العرب أدوات الرصد وحسبوا مواضع الكواكب بدقة عالية.",
            "tags": "فلك",
        },
        "b": {
            "title": "الطب العربي",
            "summary": "التشخيص والعلاج",
            "body": "وصف الأطباء الأمراض وطوّروا أدوات الجراحة وطرق العلاج.",
            "tags": "طب",
        },
        "c": {
            "title": "المياه والري",
            "summary": "القنوات وإدارة المياه",
            "body": "أدارت المدن المياه بالقنوات وقواعد الحصص، والمياه أساس الزراعة.",
            "tags": "بيئة",
        },
    }
    idx = InvertedIndex()
    for doc_id, fields in docs.items():
        idx.add_document(doc_id, fields)
    return idx.finalize()


# ------------------------------------------------------------------ الاستعلام
def test_parse_query_basics() -> None:
    q = parse_query("فلك نجوم")
    assert q.terms == ["فلك", "نجوم"]


def test_parse_query_operators() -> None:
    q = parse_query('فلك +رصد -طب "علم النجوم"')
    assert "رصد" in q.required
    assert "طب" in q.excluded
    assert q.phrases and q.phrases[0] == ["علم", "النجوم"]


def test_parse_query_unclosed_quote() -> None:
    with pytest.raises(QueryError):
        parse_query('"عبارة غير مغلقة')


def test_parse_query_prefix() -> None:
    q = parse_query("فلك*")
    assert q.prefixes == ["فلك"]


def test_parse_query_drops_stopwords() -> None:
    q = parse_query("من في على")
    assert q.is_empty


# ------------------------------------------------------------------- الفهرس
def test_index_sizes(index: InvertedIndex) -> None:
    assert index.size == 3
    assert index.vocab_size > 20
    assert index.avg_doc_len > 0


def test_index_has_roots_and_stems(index: InvertedIndex) -> None:
    # «الفلك» و«فلكي» و«الفلكيون» تتقارب في الجذر
    assert index.df("فلك") >= 1
    assert index.idf("فلك") > 0


def test_index_prefix(index: InvertedIndex) -> None:
    assert any(t.startswith("فلك") for t in index.prefix("فلك"))


def test_index_rejects_duplicates(index: InvertedIndex) -> None:
    with pytest.raises(ValueError):
        index.add_document("a", {"body": "تكرار"})


def test_index_phrase(index: InvertedIndex) -> None:
    # «أدوات الرصد» متتاليتان في متن المستند a
    assert "a" in index.phrase(["أدوات", "الرصد"])
    # وترتيبٌ معكوس لا يُطابق
    assert "b" not in index.phrase(["الرصد", "أدوات"])


def test_index_terms_helpers() -> None:
    terms = index_text("النجوم والكواكب")
    assert terms and all(isinstance(t, str) for t in terms)
    assert "في" not in query_terms("في")


# ---------------------------------------------------------------------- BM25
def test_bm25_ranks_relevant_first(index: InvertedIndex) -> None:
    bm = BM25(index)
    hits = bm.search("الفلك والنجوم", k=3)
    assert hits and hits[0].doc_id == "a"


def test_bm25_exclusion(index: InvertedIndex) -> None:
    bm = BM25(index)
    hits = bm.search("المياه -الزراعة", k=3)
    assert all(h.doc_id != "c" for h in hits)


def test_bm25_required_term(index: InvertedIndex) -> None:
    bm = BM25(index)
    hits = bm.search("الطب +الجراحة", k=3)
    assert [h.doc_id for h in hits] == ["b"]


def test_bm25_empty_query(index: InvertedIndex) -> None:
    assert BM25(index).search("من في", k=3) == []


def test_bm25_scores_positive(index: InvertedIndex) -> None:
    for h in BM25(index, BM25Params(k1=1.2, b=0.6)).search("العلاج", k=3):
        assert h.score > 0


# ---------------------------------------------------------------------- HNSW
@pytest.fixture(scope="module")
def hnsw_data():
    rng = np.random.default_rng(5)
    centers = rng.normal(size=(8, 16))
    X = np.vstack([centers[i % 8] + 0.3 * rng.normal(size=16) for i in range(400)]).astype(np.float32)
    keys = [f"k{i}" for i in range(400)]
    return X, keys


def test_hnsw_recall(hnsw_data) -> None:
    X, keys = hnsw_data
    idx = build_from_matrix(X, keys, HNSWParams(M=16, ef_construction=100, ef_search=48))
    assert recall_at_k(idx, X[:30], 10) >= 0.9


def test_hnsw_search_returns_similar_first(hnsw_data) -> None:
    X, keys = hnsw_data
    idx = build_from_matrix(X, keys)
    got = idx.search(X[0], 1)
    assert got[0][0] == "k0"
    assert got[0][1] > 0.9


def test_hnsw_small_index_is_exact(hnsw_data) -> None:
    X, keys = hnsw_data
    idx = HNSW(16)
    for i in range(5):
        idx.add(X[i], keys[i])
    got = idx.search(X[2], 3)
    assert got[0][0] == "k2"


def test_hnsw_duplicate_key(hnsw_data) -> None:
    X, keys = hnsw_data
    idx = HNSW(16)
    idx.add(X[0], "z")
    with pytest.raises(ValueError):
        idx.add(X[1], "z")


def test_hnsw_dim_mismatch(hnsw_data) -> None:
    idx = HNSW(16)
    with pytest.raises(ValueError):
        idx.add(np.zeros(8, dtype=np.float32), "bad")


def test_hnsw_stats(hnsw_data) -> None:
    X, keys = hnsw_data
    idx = build_from_matrix(X, keys)
    st = idx.stats()
    assert st["size"] == 400
    assert st["edges"] > 400
    assert st["max_level"] >= 0


# --------------------------------------------------------------------- الدمج
def test_rrf_prefers_agreement() -> None:
    fused = rrf_fuse({"a": ["x", "y", "z"], "b": ["y", "x", "w"]}, limit=3)
    assert fused[0].doc_id in {"x", "y"}
    assert set(fused[0].sources) == {"a", "b"}


def test_rrf_weights() -> None:
    fused = rrf_fuse({"a": ["x"], "b": ["y"]}, weights={"a": 1.0, "b": 0.01})
    assert fused[0].doc_id == "x"


def test_weighted_fuse_normalizes() -> None:
    fused = weighted_score_fuse({"a": [("x", 10.0), ("y", 0.0)], "b": [("y", 5.0)]}, limit=2)
    assert fused[0].doc_id in {"x", "y"}
    assert all(0 <= h.score <= 2.0 for h in fused)
