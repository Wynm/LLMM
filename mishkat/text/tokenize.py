"""تقطيع النص العربي إلى كلماتٍ بمواضع، وجُمل، وn-grams.

المواضع (offsets) ضروريةٌ لإبراز نتائج البحث في الواجهة دون إعادة حساب التطبيع:
نقطّع النصّ الأصلي، ثم نطبّع كلَّ كلمةٍ على حدة.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from .normalize import SEARCH, normalize

# كلمة = سلسلة من الحروف العربية (مع التشكيل) أو أرقام أو لاتيني، مع علاماتٍ داخلية
_WORD_RE = re.compile(
    r"[0-9\u0660-\u0669\u06F0-\u06F9]+(?:[.,][0-9]+)*"          # أعداد
    r"|[A-Za-z][A-Za-z0-9_+\-.]*"                                # لاتيني/تقني
    r"|[\u0621-\u064A\u066E-\u06D3\u0671-\u06D5\u06FA-\u06FF]"
    r"[\u0621-\u064A\u064B-\u065F\u0670\u0640\u066E-\u06D3\u06FA-\u06FF]*"
)

_SENTENCE_BREAKS = set(".!?؟؛:\n")
_ABBREVIATIONS = {"د.", "م.", "ق.م", "هـ", "ص.", "ج.", "ر.", "ت."}


@dataclass(frozen=True, slots=True)
class Token:
    """كلمةٌ واحدة مع موضعها في النصّ الأصلي."""

    raw: str
    norm: str
    start: int
    end: int
    index: int
    sentence: int = 0

    @property
    def is_word(self) -> bool:
        return bool(self.norm)

    def __len__(self) -> int:
        return len(self.raw)

    def __str__(self) -> str:  # pragma: no cover - عرض فقط
        return self.raw


@dataclass(frozen=True, slots=True)
class Sentence:
    """جملةٌ مع موضعها ونصّها المطبّع."""

    text: str
    norm: str
    start: int
    end: int
    index: int
    tokens: tuple[Token, ...] = field(default_factory=tuple)


def tokenize(
    text: str,
    *,
    level: str = SEARCH,
    sentences: bool = True,
) -> list[Token]:
    """يقطّع النصّ إلى كلماتٍ مع مواضعها ورقم جملتها.

    >>> [t.norm for t in tokenize("النجم ساطع، والسماء صافية.")]
    ['النجم', 'ساطع', 'والسماء', 'صافيه']
    """
    sents = split_sentences(text) if sentences else []
    tokens: list[Token] = []
    for m in _WORD_RE.finditer(text):
        raw = m.group(0)
        norm = normalize(raw, level)
        if not norm:
            continue
        sidx = _sentence_index(sents, m.start()) if sents else 0
        tokens.append(
            Token(
                raw=raw,
                norm=norm,
                start=m.start(),
                end=m.end(),
                index=len(tokens),
                sentence=sidx,
            )
        )
    return tokens


def _sentence_index(sents: Sequence[Sentence], pos: int) -> int:
    for s in sents:
        if s.start <= pos < s.end:
            return s.index
    return max(0, len(sents) - 1)


def split_sentences(text: str, *, level: str = SEARCH) -> list[Sentence]:
    """يقطّع النصّ إلى جُملٍ بعلامات الترقيم العربية والفواصل والأسطر."""
    if not text or not text.strip():
        return []
    boundaries: list[int] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _SENTENCE_BREAKS:
            # لا نقطع عند نقطة اختصار مثل «د.»
            tail = text[max(0, i - 3) : i + 1]
            if ch == "." and any(tail.endswith(a) for a in _ABBREVIATIONS):
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in "\"'»)] ":
                j += 1
            boundaries.append(j)
            i = j
            continue
        i += 1
    if not boundaries or boundaries[-1] < n:
        boundaries.append(n)

    sentences: list[Sentence] = []
    start = 0
    for end in boundaries:
        chunk = text[start:end]
        if chunk.strip():
            norm = normalize(chunk, level)
            if norm:
                sentences.append(
                    Sentence(
                        text=chunk.strip(),
                        norm=norm,
                        start=start,
                        end=end,
                        index=len(sentences),
                    )
                )
        start = end
    return sentences


def iter_ngrams(tokens: Sequence[Token], n: int) -> Iterator[tuple[Token, ...]]:
    """يولّد n-grams متتالية من الكلمات."""
    if n <= 0:
        raise ValueError("n يجب أن يكون أكبر من صفر")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def words(text: str, *, level: str = SEARCH) -> list[str]:
    """كلماتٌ مطبّعة فقط (بلا مواضع)."""
    return [t.norm for t in tokenize(text, level=level, sentences=False)]
