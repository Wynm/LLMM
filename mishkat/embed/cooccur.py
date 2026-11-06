"""مصفوفة التجاور: كم مرةٍ يظهر مصطلحان قريبين في النص.

الفكرة
------
المعنى يُستدلّ عليه من السياق: الكلمات التي تظهر في بيئاتٍ متشابهة تتقارب
دلاليًّا. نبني هذه البيئات بعدًّا: لكل موضعٍ في النصّ، نعدّ أزواج (كلمة، سياق)
داخل نافذةٍ منزلقة.

لماذا وزنٌ عكسيّ بالمسافة؟
لأن الجار المباشر أقوى دلالةً من الجار البعيد؛ نستخدم وزنًا
:math:`1/|\\Delta|` أوّليًّا لتخفيف أثر أطراف النافذة.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from ..text.analyze import analyze
from ..text.normalize import SEARCH
from ..text.stem import light_stem
from ..text.stopwords import is_stopword
from ..text.tokenize import tokenize


def context_term(raw: str) -> str:
    """المصطلح الذي يُستخدم في السياق: الجذر إن أمكن، وإلا الساق.

    لماذا الجذر؟ لأن «مدرسة» و«مدارس» و«دراسة» سياقٌ واحد؛ واستخدام السيقان
    يُفرّقها في ثلاثة أبعادٍ متباعدة فيضعف الفضاء الدلالي. الجذر يوحّدها.
    """
    a = analyze(raw)
    # الجذر يُقبل فقط إن كانت الثقة عالية (جذرٌ في المُعجم أو مطابقٌ لوزن)؛
    # وإلا فالجذر المُخمَّن يزيد الضجيج أكثر مما يوحّد الدلالة.
    if a.root and len(a.root) >= 3 and a.confidence >= 0.6:
        return a.root
    term = a.stem or a.norm
    return term


def term_frequencies(text: str) -> Counter:
    """تكرار السيقان في نصّ (بلا كلمات إيقاف)."""
    c: Counter = Counter()
    for tok in tokenize(text, level=SEARCH, sentences=False):
        w = tok.norm
        if not w or len(w) < 2 or is_stopword(w):
            continue
        c[context_term(tok.raw)] += 1
    return c


def build_cooccurrence(
    documents: Sequence[str],
    *,
    window: int = 5,
    weighted: bool = True,
    min_count: int = 1,
    max_vocab: int | None = None,
) -> tuple[dict[str, int], dict[tuple[int, int], float]]:
    """يبني التجاور من قائمة نصوص.

    Returns
    -------
    vocab:
        قاموس المصطلح ← فهرسه.
    counts:
        قاموس (فهرس، فهرس) ← وزن.
    """
    if window < 1:
        raise ValueError("حجم النافذة يجب أن يكون 1 على الأقل")

    tf: Counter = Counter()
    tokenized: list[list[str]] = []
    for doc in documents:
        toks: list[str] = []
        for t in tokenize(doc, level=SEARCH, sentences=False):
            w = t.norm
            if not w or len(w) < 2 or is_stopword(w):
                continue
            toks.append(context_term(t.raw))
        tokenized.append(toks)
        tf.update(toks)

    if max_vocab:
        keep = {w for w, _ in tf.most_common(max_vocab)}
    else:
        keep = {w for w, c in tf.items() if c >= min_count}
    vocab = {w: i for i, w in enumerate(sorted(keep))}

    counts: dict[tuple[int, int], float] = {}
    for toks in tokenized:
        ids = [vocab[t] for t in toks if t in vocab]
        n = len(ids)
        for i, a in enumerate(ids):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                b = ids[j]
                w = 1.0 / abs(i - j) if weighted else 1.0
                key = (a, b) if a <= b else (b, a)
                counts[key] = counts.get(key, 0.0) + w
    return vocab, counts


def cooccurrence_stats(vocab: dict[str, int], counts: dict[tuple[int, int], float]) -> dict:
    """إحصاءاتٌ سريعة عن المصفوفة."""
    if not counts:
        return {"vocab": len(vocab), "pairs": 0}
    vals = sorted(counts.values())
    return {
        "vocab": len(vocab),
        "pairs": len(counts),
        "density": round(len(counts) / (len(vocab) ** 2), 4),
        "max": max(vals),
        "median": vals[len(vals) // 2],
    }
