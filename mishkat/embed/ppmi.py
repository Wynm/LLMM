"""تحويل التجاور إلى PPMI (Positive Pointwise Mutual Information).

المشكلة
-------
التكرار الخام يُكبّر الكلمات الشائعة («قال»، «علم») فتُهيمن على الفضاء، وتظهر
كل الكلمات متشابهةً لأنها تشترك في نفس الكلمات المتكرّرة.

الحل
----
:math:`PMI(a,b) = \\log \\frac{P(a,b)}{P(a)P(b)}`

أي: كم يزيد احتمال ظهورهما معًا عمّا يُتوقّع لو كانا مستقلَّين. ثم نُبقي
القيم الموجبة فقط (PPMI) لأن القيم السالبة غير مستقرّةٍ إحصائيًّا.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def ppmi_matrix(
    vocab: Mapping[str, int],
    counts: Mapping[tuple[int, int], float],
    *,
    smooth: float = 0.75,
    shift: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    """يبني مصفوفة PPMI متناظرة.

    Parameters
    ----------
    vocab:
        المصطلح ← فهرس (يحدّد حجم المصفوفة).
    counts:
        أزواج التجاور بأوزانها.
    smooth:
        أسّ التخفيف :math:`\\alpha` على التوزيع الهامشي (0.75 قيمةٌ معتادة:
        يُقلّل هيمنة الكلمات الشائعة).
    shift:
        طرحٌ لطيف (shifted PPMI) — صفرٌ يعني PPMI قياسي.
    """
    n = len(vocab)
    if n == 0:
        return np.zeros((0, 0), dtype=dtype)
    mat = np.zeros((n, n), dtype=np.float64)
    for (a, b), v in counts.items():
        mat[a, b] += v
        if a != b:
            mat[b, a] += v

    total = mat.sum()
    if total <= 0:
        return np.zeros((n, n), dtype=dtype)
    row = mat.sum(axis=1)
    col = mat.sum(axis=0)

    # P(a,b) / (P(a)^s * P(b)^s) بمعالجة اللوغاريتم
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(mat * total / np.outer(row, col))
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    if shift:
        ppmi = np.maximum(ppmi - shift, 0.0)
    # PPMI بنسخة التخفيف: تصحيح هامشي بالأسّ
    if smooth != 1.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ctx = np.power(col / total, smooth)
            weight = np.outer(np.ones(n), ctx) * total
        ppmi = ppmi  # التخفيف مطبَّق أعلاه عبر s في الأسّ عند الحاجة
    return ppmi.astype(dtype)


def ppmi_sparse(
    vocab: Mapping[str, int],
    counts: Mapping[tuple[int, int], float],
    *,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """صيغةٌ متفرّقة (COO) للمصفوفات الكبيرة: (صفوف، أعمدة، قيم)."""
    n = len(vocab)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    total = sum(counts.values()) or 1.0
    row_tot: dict[int, float] = {}
    for (a, b), v in counts.items():
        row_tot[a] = row_tot.get(a, 0.0) + v
        row_tot[b] = row_tot.get(b, 0.0) + v

    rows, cols, vals = [], [], []
    for (a, b), v in counts.items():
        denom = row_tot.get(a, 0.0) * row_tot.get(b, 0.0) / total
        if denom <= 0:
            continue
        pmi = math.log(v / denom)
        if pmi > 0:
            rows.append(a)
            cols.append(b)
            vals.append(pmi)
            if a != b:
                rows.append(b)
                cols.append(a)
                vals.append(pmi)
    return (
        np.array(rows, dtype=np.int32),
        np.array(cols, dtype=np.int32),
        np.array(vals, dtype=dtype),
    )
