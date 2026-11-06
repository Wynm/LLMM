"""الفضاء الدلالي: متجهات كلماتٍ ومستندات، جيران، وبحثٌ بالمعنى.

بعد SVD نحصل على متجهٍ لكل مصطلح. لتقييم التشابه نُطبّع المتجهات ونستخدم جيب
التمام، ولتمثيل مستندٍ نجمع متجهات كلماته موزونةً بـ TF-IDF ثم نُطبّع.

السؤال الذي يجيب عنه هذا الملف: «أيّ مستندٍ يشبه هذا السؤال حتى لو لم يستخدم
كلماته؟» — وهذا ما يعجز عنه BM25.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..text.normalize import SEARCH, normalize
from ..text.stem import light_stem
from ..text.stopwords import is_stopword
from ..text.tokenize import tokenize
from .cooccur import build_cooccurrence, context_term
from .ppmi import ppmi_matrix
from .svd import explained_energy, randomized_svd


@dataclass
class WordSpace:
    """فضاءٌ دلاليٌّ محفوظ: مفرداتٌ + متجهاتٌ + تمثيلات مستندات."""

    vocab: dict[str, int]
    vectors: np.ndarray
    doc_ids: list[str] = field(default_factory=list)
    doc_vectors: np.ndarray | None = None
    idf: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        v = np.asarray(self.vectors, dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = v / norms
        if self.doc_vectors is not None:
            d = np.asarray(self.doc_vectors, dtype=np.float32)
            dn = np.linalg.norm(d, axis=1, keepdims=True)
            dn[dn == 0] = 1.0
            self.doc_vectors = d / dn

    # ------------------------------------------------------------- خصائص
    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    @property
    def size(self) -> int:
        return len(self.vocab)

    def lookup(self, word: str) -> str | None:
        """يجد أفضل مفتاحٍ في المفردات لكلمة: مطابقةٌ ثم سابقة."""
        key = context_term(word)
        if key in self.vocab:
            return key
        base = normalize(word, SEARCH)
        for cand in (light_stem(base), base):
            if cand in self.vocab:
                return cand
        best: str | None = None
        best_idf = float("inf")
        for term in self.vocab:
            if term.startswith(base) and len(term) - len(base) <= 3:
                idf = self.idf.get(term, 1.0)
                if idf < best_idf:
                    best, best_idf = term, idf
        return best

    def vector(self, word: str) -> np.ndarray | None:
        key = self.lookup(word)
        return None if key is None else self.vectors[self.vocab[key]]

    # ------------------------------------------------------------ الجيران
    def similar(self, word: str, k: int = 8) -> list[tuple[str, float]]:
        """أقرب الكلمات دلاليًّا إلى كلمة (بحسب متجهاتها)."""
        v = self.vector(word)
        if v is None:
            return []
        sims = self.vectors @ v
        order = np.argsort(-sims)[: k + 1]
        inv = {i: w for w, i in self.vocab.items()}
        out: list[tuple[str, float]] = []
        target = self.vocab.get(self.lookup(word) or "")
        for i in order:
            if int(i) == target:
                continue
            out.append((inv[int(i)], float(sims[int(i)])))
        return out[:k]

    def nearest_words(self, vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if self.vectors.size == 0:
            return []
        q = vec / (np.linalg.norm(vec) or 1.0)
        sims = self.vectors @ q
        order = np.argsort(-sims)[:k]
        inv = {i: w for w, i in self.vocab.items()}
        return [(inv[int(i)], float(sims[int(i)])) for i in order]

    # --------------------------------------------------------- تمثيل نصّ
    def embed_text(self, text: str) -> np.ndarray:
        """متجهٌ لنصّ: متوسّط موزون بـ IDF لمتجهات كلماته المعروفة."""
        acc = np.zeros(self.dim, dtype=np.float32)
        weight_sum = 0.0
        for tok in tokenize(text, level=SEARCH, sentences=False):
            w = tok.norm
            if not w or len(w) < 2 or is_stopword(w):
                continue
            term = context_term(tok.raw)
            v = self.vector(term)
            if v is None:
                continue
            weight = self.idf.get(term, 1.0)
            acc += weight * v
            weight_sum += weight
        if weight_sum > 0:
            acc /= weight_sum
        norm = np.linalg.norm(acc)
        if norm > 0:
            acc /= norm
        return acc

    # --------------------------------------------------------- بحث مستند
    def search_docs(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """أقرب المستندات دلاليًّا إلى نصّ."""
        if self.doc_vectors is None or not self.doc_ids:
            return []
        q = self.embed_text(query)
        if not np.any(q):
            return []
        sims = self.doc_vectors @ q
        order = np.argsort(-sims)[:k]
        return [(self.doc_ids[int(i)], float(sims[int(i)])) for i in order]

    # ------------------------------------------------------------- الحفظ
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            vectors=self.vectors,
            doc_vectors=self.doc_vectors if self.doc_vectors is not None else np.zeros((0, 0)),
            doc_ids=np.array(self.doc_ids, dtype=object),
            vocab=np.array(list(self.vocab.items()), dtype=object),
        )
        path.with_suffix(".json").write_text(
            json.dumps({"meta": self.meta, "idf": self.idf}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: Path | str) -> "WordSpace":
        path = Path(path)
        data = np.load(path.with_suffix(".npz"), allow_pickle=True)
        meta = json.loads(path.with_suffix(".json").read_text())
        vocab = {str(k): int(v) for k, v in data["vocab"]}
        docs = [str(x) for x in data["doc_ids"]] if data["doc_ids"].size else []
        dv = data["doc_vectors"]
        return cls(
            vocab=vocab,
            vectors=data["vectors"],
            doc_ids=docs,
            doc_vectors=dv if dv.size else None,
            idf={str(k): float(v) for k, v in meta.get("idf", {}).items()},
            meta=meta.get("meta", {}),
        )

    def stats(self) -> dict:
        return {
            "vocab": self.size,
            "dim": self.dim,
            "documents": len(self.doc_ids),
            **{k: v for k, v in self.meta.items() if k in {"energy", "window", "pairs"}},
        }


def build_space(
    documents: Iterable,
    *,
    dim: int = 96,
    window: int = 5,
    max_vocab: int = 2500,
    min_count: int = 1,
    seed: int = 7,
) -> WordSpace:
    """يبني فضاءً دلاليًّا من مستندات المجموعة (كائنات فيها ``.id`` و``.text``)."""
    docs = list(documents)
    texts = [d.text for d in docs]
    vocab, counts = build_cooccurrence(
        texts, window=window, min_count=min_count, max_vocab=max_vocab
    )
    if not vocab:
        raise ValueError("مفردات فارغة: لا يمكن بناء فضاء")

    mat = ppmi_matrix(vocab, counts)
    U, S, _ = randomized_svd(mat, n_components=dim, seed=seed)
    # متجه الكلمة = U * sqrt(S) (ترجيحٌ معتاد لتوازن المحاور)
    word_vecs = U * np.sqrt(np.maximum(S, 1e-9))

    # IDF لكل مصطلح
    n_docs = len(docs)
    df: dict[str, int] = {}
    doc_forms: list[list[str]] = []
    for d in docs:
        forms: list[str] = []
        for tok in tokenize(d.text, level=SEARCH, sentences=False):
            w = tok.norm
            if not w or len(w) < 2 or is_stopword(w):
                continue
            forms.append(context_term(tok.raw))
        doc_forms.append(forms)
        for f in set(forms):
            df[f] = df.get(f, 0) + 1
    idf = {f: math.log((n_docs + 1) / (c + 0.5)) for f, c in df.items()}

    space = WordSpace(vocab=vocab, vectors=word_vecs, idf=idf)

    # تمثيلات المستندات
    mat_docs = np.zeros((n_docs, word_vecs.shape[1]), dtype=np.float32)
    for i, forms in enumerate(doc_forms):
        acc = np.zeros(word_vecs.shape[1], dtype=np.float32)
        wsum = 0.0
        tf: dict[str, int] = {}
        for f in forms:
            tf[f] = tf.get(f, 0) + 1
        for f, c in tf.items():
            v = space.vector(f)
            if v is None:
                continue
            weight = (1 + math.log(c)) * idf.get(f, 1.0)
            acc += weight * v
            wsum += weight
        if wsum > 0:
            mat_docs[i] = acc / wsum
    space.doc_vectors = mat_docs
    space.doc_ids = [d.id for d in docs]
    space.meta = {
        "window": window,
        "pairs": len(counts),
        "energy": round(explained_energy(S, float(np.sum(mat**2)) or 1.0), 4),
        "built_from": n_docs,
    }
    return space


# ------------------------------------------------------------------ تجميع
def kmeans(
    vectors: np.ndarray,
    k: int = 5,
    *,
    iters: int = 40,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """تجميعٌ k-means++ بسيط: يعيد (مراكز، تسمية كل عنصر).

    يُستخدم في الاستوديو لرسم «خريطة المواضيع»: كل نقطةٍ مستند، ولونها عنقودها.
    """
    X = np.asarray(vectors, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0)), np.zeros(0, dtype=int)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # k-means++ لاختيار مراكز ابتدائية متباعدة
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            np.array([np.sum((X - c) ** 2, axis=1) for c in centers]), axis=0
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1 / n)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
    return C, labels
