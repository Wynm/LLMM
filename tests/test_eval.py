"""اختبارات التقييم والدمج: هل الأرقام صحيحة، وهل الدمج يتصرّف بعقل؟"""

from __future__ import annotations

import json

import pytest

from mishkat.evaluate import (
    EvalQuery,
    evaluate,
    load_queries,
    ndcg_at,
    recall_at,
    reciprocal_rank,
)
from mishkat.index.hybrid import FusionHit, rrf_fuse, score_fuse, weighted_score_fuse


# ----------------------------------------------------------------- المقاييس
def test_reciprocal_rank_basic():
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0
    assert reciprocal_rank(["a"], ["a", "b"]) == 1.0


def test_reciprocal_rank_respects_k():
    assert reciprocal_rank(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_is_over_all_expected():
    assert recall_at(["x", "a"], ["a", "b"]) == 0.5
    assert recall_at([], ["a"]) == 0.0
    assert recall_at(["a", "b"], ["a", "b"]) == 1.0


def test_ndcg_perfect_and_degraded():
    assert ndcg_at(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    # ترتيبٌ معكوس يُنقص القيمة لكنه لا يُصفّرها
    got = ndcg_at(["b", "a"], ["a"])
    assert 0.0 < got < 1.0


def test_evaluate_aggregates_and_records_misses():
    qs = [EvalQuery("س", ("a",)), EvalQuery("ع", ("z",))]
    ranks = {"س": ["a"], "ع": ["b"]}
    rep = evaluate(lambda q, k: ranks[q], qs, strategy="t")
    assert rep.n == 2
    assert rep.mrr == pytest.approx(0.5)
    assert rep.hit1 == pytest.approx(0.5)
    assert len(rep.misses) == 1


def test_evaluate_groups_by_bucket():
    qs = [EvalQuery("س", ("a",)), EvalQuery("ع", ("z",))]
    ranks = {"س": ["a"], "ع": ["b"]}
    rep = evaluate(lambda q, k: ranks[q], qs, bucket_of=lambda q: "صعب" if q.query == "ع" else "سهل")
    assert rep.groups["سهل"].mrr == 1.0
    assert rep.groups["صعب"].mrr == 0.0
    assert rep.group_rows()


# ------------------------------------------------------------ ملفّ الاستعلامات
def test_load_queries_real_file():
    qs = load_queries()
    assert len(qs) >= 40
    assert all(q.expect for q in qs)
    kinds = {q.kind for q in qs}
    assert {"lexical", "semantic"} <= kinds


def test_load_queries_rejects_empty_expect(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"query": "س", "expect": []}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(p)


def test_load_queries_rejects_duplicates(tmp_path):
    p = tmp_path / "q.jsonl"
    line = json.dumps({"query": "س", "expect": ["a"]}, ensure_ascii=False)
    p.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_queries(p)


# ------------------------------------------------------------------- الدمج
def test_score_fuse_prefers_strong_lexical_evidence():
    """الدرجة المعجمية القوية يجب أن تتفوّق على دلاليٍّ هامشيّ."""
    lex = [("a", 12.0), ("b", 2.0)]
    sem = [("c", 0.99), ("a", 0.4)]
    hits = score_fuse({"lexical": lex, "semantic": sem}, weights={"lexical": 1.0, "semantic": 0.85})
    assert isinstance(hits[0], FusionHit)
    assert hits[0].doc_id == "a"


def test_score_fuse_keeps_semantic_only_candidate():
    lex = [("a", 5.0)]
    sem = [("z", 0.8)]
    docs = [h.doc_id for h in score_fuse({"lexical": lex, "semantic": sem})]
    assert "z" in docs


def test_score_fuse_handles_empty_and_zero_scores():
    assert score_fuse({"lexical": []}) == []
    hits = score_fuse({"lexical": [("a", 0.0)]})
    assert hits[0].doc_id == "a" and hits[0].score == 0.0


def test_score_fuse_limit_and_order():
    lex = [(f"d{i}", float(10 - i)) for i in range(10)]
    hits = score_fuse({"lexical": lex}, limit=3)
    assert len(hits) == 3
    assert [h.doc_id for h in hits] == ["d0", "d1", "d2"]


def test_rrf_still_available_and_normalized_by_rank():
    hits = rrf_fuse({"a": ["x", "y"], "b": ["y"]}, k=60)
    # y يظهر في مسارين فيتفوّق على x رغم رتبته الثانية
    assert hits[0].doc_id == "y"
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_weighted_score_fuse_normalizes():
    hits = weighted_score_fuse({"a": [("x", 100.0), ("y", 50.0)]}, limit=2)
    assert hits[0].doc_id == "x"
    assert 0.0 <= hits[1].score <= hits[0].score


def test_score_fuse_beats_rrf_on_golden_queries():
    """حارسٌ ضدّ الانحدار: الدمج بالدرجات لا يهبط تحت RRF على مجموعة التقييم."""
    from mishkat.engine import Engine

    engine = Engine.load()
    qs = load_queries()
    from mishkat.index.query import parse_query

    def ranked(strategy: str):
        def f(q: str, k: int):
            lex = engine.bm25.search(parse_query(q), k=max(k * 3, 24))
            sem = engine.space.search_docs(q, k=max(k * 3, 24))
            lex_pairs = [(h.doc_id, h.score) for h in lex]
            sem_pairs = [(d, s) for d, s in sem]
            if strategy == "score":
                out = score_fuse({"lexical": lex_pairs, "semantic": sem_pairs},
                                 weights={"lexical": 1.0, "semantic": 0.85}, limit=k)
            else:
                out = rrf_fuse({"lexical": [h.doc_id for h in lex], "semantic": [d for d, _ in sem]},
                               weights={"lexical": 1.0, "semantic": 0.85}, limit=k)
            return [h.doc_id for h in out]

        return f

    score_rep = evaluate(ranked("score"), qs, k=10)
    rrf_rep = evaluate(ranked("rrf"), qs, k=10)
    assert score_rep.mrr >= rrf_rep.mrr
