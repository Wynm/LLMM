"""HNSW — البحث التقريبيّ عن أقرب الجيران، مبنيًّا من الصفر.

الفكرة
------
بناء رسمٍ متعدّد الطبقات: الطبقة العليا قليلة العُقد وروابطها طويلة (تقرّب
سريع)، والطبقة السفلى كثيفة (تدقيق). البحث ينزل من القمّة: في كل طبقة يمشي
بجشعٍ نحو الجار الأقرب، ثم ينزل، حتى يصل إلى الطبقة صفر فيُجري بحثًا محليًّا
واسعًا (:math:`ef`) ويعيد أفضل :math:`k`.

الثمن
-----
البحث :math:`O(\\log n)` تقريبًا بدل :math:`O(n)` في البحث الشامل، مقابل دقّةٍ
أقلّ قليلًا (تنازلٌ يقتضيه الحجم). لذلك نُبقي مسارًا دقيقًا (brute force) عندما
يكون عدد العناصر صغيرًا: الخوّارزم «الأفضل» ليس الأسرع حين تصغر البيانات.

التنفيذ
-------
- التوزيع الطبقي أُسّيّ: :math:`\\ell = \\lfloor -\\ln(U)\\, m_L \\rfloor` حيث
  :math:`m_L = 1/\\ln M`.
- عند الإدخال نبحث في كل طبقة عن مرشّحين بحجم :math:`ef_{construction}`، ثم
  نختار منهم :math:`M` جارًا ونربط الرابطة في الاتجاهين مع تقليم المتبقّي.
- المسافة جيب التمام: نُطبّع المتجهات عند الإدخال فيصبح الضرب النقطيّ مسافةً.

هذا تنفيذٌ تعليميّ صريح: كل خطوةٍ قابلة للقراءة والتعديل، بلا مكتبةٍ خارجية.
"""

from __future__ import annotations

import heapq
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class HNSWParams:
    """معاملات بناء الرسم."""

    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    seed: int = 13
    metric: str = "cosine"

    def to_dict(self) -> dict:
        return asdict(self)


class HNSW:
    """فهرسٌ تقريبيّ لأقرب الجيران."""

    def __init__(self, dim: int, params: HNSWParams | None = None) -> None:
        if dim <= 0:
            raise ValueError("البُعد يجب أن يكون موجبًا")
        self.dim = dim
        self.params = params or HNSWParams()
        self._rng = random.Random(self.params.seed)
        self._ml = 1.0 / math.log(max(2, self.params.M))
        self.vectors: list[np.ndarray] = []
        self.keys: list[str] = []
        self._index: dict[str, int] = {}
        self.links: list[dict[int, list[int]]] = []
        self.entry: int | None = None
        self.max_level: int = -1
        self._norms: list[float] = []

    # ------------------------------------------------------------ أدوات
    @property
    def size(self) -> int:
        return len(self.vectors)

    def _normalize(self, v: np.ndarray) -> tuple[np.ndarray, float]:
        v = np.asarray(v, dtype=np.float32)
        if v.shape[0] != self.dim:
            raise ValueError(f"بُعد غير مطابق: {v.shape[0]} != {self.dim}")
        norm = float(np.linalg.norm(v))
        if norm == 0:
            return v, 0.0
        return v / norm, norm

    def _dist(self, idx: int, q: np.ndarray) -> float:
        """مسافة جيب التمام (1 - التشابه) لعنصرٍ مُطبَّع."""
        return 1.0 - float(np.dot(self.vectors[idx], q))

    @staticmethod
    def _random_level(rng: random.Random, ml: float) -> int:
        u = max(1e-12, rng.random())
        return int(-math.log(u) * ml)

    # ------------------------------------------------------------ الإدخال
    def add(self, vector: np.ndarray, key: str) -> int:
        """يضيف عنصرًا واحدًا ويعيد فهرسه الداخلي."""
        if key in self._index:
            raise ValueError(f"مفتاح مكرر: {key!r}")
        v, norm = self._normalize(vector)
        idx = len(self.vectors)
        self.vectors.append(v)
        self._norms.append(norm)
        self.keys.append(key)
        self._index[key] = idx
        self.links.append({})

        level = self._random_level(self._rng, self._ml)
        for lv in range(level + 1):
            self.links[idx][lv] = []

        if self.entry is None:
            self.entry = idx
            self.max_level = level
            return idx

        ep = self.entry
        # النزول الجشع في الطبقات الأعلى من مستوى العنصر الجديد
        for lv in range(self.max_level, level, -1):
            ep = self._greedy(v, ep, lv)

        for lv in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(v, [ep], self.params.ef_construction, lv)
            neighbors = self._select(candidates, self.params.M)
            self.links[idx][lv] = [n for _, n in neighbors]
            for _, n in neighbors:
                lst = self.links[n].setdefault(lv, [])
                lst.append(idx)
                if len(lst) > self.params.M:
                    self._prune(n, lv)
            if candidates:
                ep = candidates[0][1]  # العنصر الأقرب في هذه الطبقة

        if level > self.max_level:
            self.entry = idx
            self.max_level = level
        return idx

    def add_batch(self, vectors: np.ndarray, keys: Sequence[str]) -> None:
        """إضافة دفعةٍ كاملة (ترتيبٌ عشوائيّ يُحسّن بنية الرسم)."""
        order = list(range(len(keys)))
        self._rng.shuffle(order)
        for i in order:
            self.add(vectors[i], keys[i])

    def _prune(self, node: int, level: int) -> None:
        """يُبقي أقرب M جارًا فقط (تبسيطٌ مقصود للتنفيذ التعليمي)."""
        lst = self.links[node].get(level, [])
        if len(lst) <= self.params.M:
            return
        vec = self.vectors[node]
        lst.sort(key=lambda n: self._dist(n, vec))
        self.links[node][level] = lst[: self.params.M]

    def _greedy(self, q: np.ndarray, ep: int, level: int) -> int:
        """بحثٌ جشعٌ بنقطة انطلاقٍ واحدة في طبقةٍ معيّنة."""
        best, best_d = ep, self._dist(ep, q)
        improved = True
        while improved:
            improved = False
            for n in self.links[best].get(level, []):
                d = self._dist(n, q)
                if d < best_d:
                    best, best_d, improved = n, d, True
        return best

    def _search_layer(
        self, q: np.ndarray, eps: Iterable[int] | Sequence[int], ef: int, level: int
    ) -> list[tuple[float, int]]:
        """البحث المحلّي الواسع في طبقة: يعيد قائمة (مسافة، فهرس) مرتّبة."""
        visited: set[int] = set()
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        for ep in eps:
            if ep in visited:
                continue
            visited.add(ep)
            d = self._dist(ep, q)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
        if not candidates:
            return []

        while candidates:
            d, c = heapq.heappop(candidates)
            worst = -results[0][0]
            if d > worst and len(results) >= ef:
                break
            for n in self.links[c].get(level, []):
                if n in visited:
                    continue
                visited.add(n)
                dn = self._dist(n, q)
                if len(results) < ef or dn < -results[0][0]:
                    heapq.heappush(candidates, (dn, n))
                    heapq.heappush(results, (-dn, n))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted((-d, i) for d, i in results)

    def _select(
        self, candidates: list[tuple[float, int]], m: int
    ) -> list[tuple[float, int]]:
        """اختيار الجيران باستدلال التنوّع (Algorithm 4 في ورقة HNSW).

        القاعدة: لا تُقبل عقدةٌ إن كانت أقرب إلى جارٍ مختارٍ منها إلى الاستعلام،
        لأن الرابط عندئذ «زائد» — الجار المختار يغطّي تلك المنطقة. هذا يُنتج
        رسمًا أكثر ترابطًا واستدعاءً أعلى من مجرّد اختيار أقرب m.
        """
        cand = sorted(candidates)
        if len(cand) <= m:
            return cand
        selected: list[tuple[float, int]] = []
        for d, i in cand:
            keep = True
            for _, j in selected:
                if self._dist(i, self.vectors[j]) < d:
                    keep = False
                    break
            if keep:
                selected.append((d, i))
                if len(selected) >= m:
                    return selected
        # إكمالٌ بالأقرب إن لم يكتمل العدد
        chosen = {i for _, i in selected}
        for d, i in cand:
            if i not in chosen:
                selected.append((d, i))
                if len(selected) >= m:
                    break
        return selected

    # ------------------------------------------------------------- البحث
    def search(
        self, vector: np.ndarray, k: int = 10, *, ef: int | None = None, exact: bool = False
    ) -> list[tuple[str, float]]:
        """يعيد [(مفتاح، تشابه)] مرتّبةً تنازليًّا بالتشابه.

        Parameters
        ----------
        exact:
            إن كان صحيحًا يُستخدم البحث الشامل (للمقارنة والقياس).
        """
        q, _ = self._normalize(vector)
        if not self.vectors:
            return []
        if exact or self.size <= max(2 * self.params.M, 32):
            sims = [1.0 - self._dist(i, q) for i in range(self.size)]
            order = np.argsort(sims)[::-1][:k]
            return [(self.keys[int(i)], float(sims[int(i)])) for i in order]

        ep = self.entry
        assert ep is not None
        for lv in range(self.max_level, 0, -1):
            ep = self._greedy(q, ep, lv)
        found = self._search_layer(q, [ep], max(ef or self.params.ef_search, k), 0)
        best = sorted(found)[:k]
        return [(self.keys[i], 1.0 - d) for d, i in best]

    def neighbors(self, key: str) -> dict[int, list[str]]:
        """جيران مفتاحٍ في كل طبقة (للعرض في الواجهة)."""
        idx = self._index.get(key)
        if idx is None:
            return {}
        return {
            lv: [self.keys[n] for n in nodes]
            for lv, nodes in self.links[idx].items()
        }

    # ------------------------------------------------------------ الحفظ
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            vectors=np.vstack(self.vectors),
            keys=np.array(self.keys, dtype=object),
        )
        meta = {
            "dim": self.dim,
            "params": self.params.to_dict(),
            "entry": self.entry,
            "max_level": self.max_level,
            "links": {str(k): {str(lv): v for lv, v in d.items()} for k, d in enumerate(self.links)},
            "keys": self.keys,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | str) -> "HNSW":
        """يستعيد فهرسًا محفوظًا (متجهات + روابط + معاملات)."""
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        data = np.load(path.with_suffix(".npz"), allow_pickle=True)
        idx = cls(meta["dim"], HNSWParams(**meta["params"]))
        idx.keys = list(meta["keys"])
        idx.vectors = [np.asarray(v, dtype=np.float32) for v in data["vectors"]]
        idx._norms = [float(np.linalg.norm(v)) for v in idx.vectors]
        idx._index = {k: i for i, k in enumerate(idx.keys)}
        idx.entry = meta["entry"]
        idx.max_level = meta["max_level"]
        idx.links = [
            {int(lv): [int(n) for n in nodes] for lv, nodes in meta["links"][str(i)].items()}
            for i in range(len(idx.keys))
        ]
        return idx

    # ------------------------------------------------------------ إحصاء
    def stats(self) -> dict:
        edges = sum(len(v) for d in self.links for v in d.values())
        levels = [len(d) for d in self.links]
        return {
            "size": self.size,
            "dim": self.dim,
            "max_level": self.max_level,
            "edges": edges,
            "avg_degree": round(edges / max(1, self.size), 2),
            "avg_levels": round(sum(levels) / max(1, len(levels)), 2),
            "params": self.params.to_dict(),
        }


def build_from_matrix(
    vectors: np.ndarray, keys: Sequence[str], params: HNSWParams | None = None
) -> HNSW:
    """يبني فهرسًا من مصفوفةٍ كاملة."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(keys) != vectors.shape[0]:
        raise ValueError("عدد المفاتيح لا يطابق عدد المتجهات")
    idx = HNSW(vectors.shape[1], params)
    idx.add_batch(vectors, keys)
    return idx


def recall_at_k(
    index: HNSW, queries: np.ndarray, k: int = 10, *, ef: int | None = None
) -> float:
    """متوسّط الاستدعاء@k مقابل البحث الشامل (مقياس جودة HNSW)."""
    if index.size == 0:
        return 0.0
    total = 0.0
    for q in queries:
        approx = {key for key, _ in index.search(q, k, ef=ef)}
        exact = {key for key, _ in index.search(q, k, exact=True)}
        total += len(approx & exact) / max(1, len(exact))
    return total / len(queries)
