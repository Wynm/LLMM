"""SVD مضغوط عشوائيًّا: تقليل الأبعاد بلا حسابٍ كامل.

لماذا لا نُجري SVD كاملًا؟
لأن تكلفته :math:`O(n^3)`. طريقة Halko–Martinsson–Tropp تحسب تقريبًا جيّدًا
بثمن :math:`O(n\\,k^2 \\log k)` عبر:

1. إسقاط المصفوفة على فضاءٍ عشوائيٍّ منخفض :math:`Q = A\\,\\Omega`.
2. تعامدٍ (QR) لتحصل على أساسٍ متعامد.
3. SVD صغير على :math:`Q^T A` (بحجم :math:`k \\times n`).
4. إعادة التركيب.

النتيجة: متجهاتٌ دلاليةٌ دقيقة بما يكفي، تُبنى في ثوانٍ على حاسوبٍ عادي.
"""

from __future__ import annotations

import numpy as np


def randomized_svd(
    matrix: np.ndarray,
    n_components: int = 100,
    *,
    n_oversamples: int = 10,
    n_iter: int = 4,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """يحسب SVD تقريبيًّا لمصفوفةٍ كثيفة.

    Returns
    -------
    U, S, Vt
        بحيث ``U @ diag(S) @ Vt`` يُقارب المصفوفة.
    """
    A = np.asarray(matrix, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("المصفوفة يجب أن تكون ثنائية الأبعاد")
    n = min(A.shape)
    k = max(1, min(n_components, n))
    rng = np.random.default_rng(seed)

    omega = rng.normal(size=(A.shape[1], k + n_oversamples))
    Y = A @ omega
    # تكرارات القوّة: تُحسّن دقّة الفضاء الجزئي المهيمن
    for _ in range(max(0, n_iter)):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], S[:k], Vt[:k]


def truncated_svd(matrix: np.ndarray, n_components: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """SVD كامل/مقتطع عبر NumPy (للمقارنة في القياس)."""
    U, S, _ = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    k = min(n_components, S.shape[0])
    return U[:, :k], S[:k]


def explained_energy(S: np.ndarray, total: float | None = None) -> float:
    """نسبة الطاقة (التباين) المحفوظة في القيم المفردة."""
    denom = total if total is not None else float(np.sum(S**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(S**2) / denom)
