"""المحلّل الموحّد: كلمةٌ داخلة، تحليلٌ صرفيّ كامل خارج.

    >>> a = analyze("والكتابة")
    >>> a.norm, a.stem, a.root, a.kind
    ('والكتابه', 'كتابه', 'كتب', 'اسم')
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

from .normalize import DISPLAY, SEARCH, STRICT, normalize
from .roots import extract_root, root_candidates, root_confidence
from .stem import light_stem, strict_stem
from .stopwords import is_stopword
from .tokenize import Token, tokenize


class Kind(str, Enum):
    """نوعٌ تقريبيّ للكلمة (لا إعراب)."""

    NOUN = "اسم"
    VERB = "فعل"
    PARTICLE = "حرف"
    UNKNOWN = "مجهول"


_PARTICLES = frozenset(
    "من في الى عن على مع ان انه لكن بل ثم او ام لا لم لن قد هل يا ما اذا اذ".split()
)
_VERB_PREFIXES = ("ي", "ت", "ن", "ا", "س", "سي", "ست")


@dataclass(frozen=True, slots=True)
class Analysis:
    """تحليلٌ صرفيّ لكلمةٍ واحدة."""

    raw: str
    display: str
    norm: str
    stem: str
    deeper: str
    root: str
    kind: str
    confidence: float
    is_stop: bool
    alternatives: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alternatives"] = list(self.alternatives)
        return d


def _guess_kind(word: str, stem: str) -> Kind:
    if word in _PARTICLES:
        return Kind.PARTICLE
    if word.startswith("ال") or word.startswith("وال") or word.startswith("بال"):
        return Kind.NOUN
    if word.startswith(_VERB_PREFIXES) and len(word) >= 4:
        return Kind.VERB
    if word.endswith(("ون", "ين", "ات", "ان")):
        return Kind.NOUN
    if stem.endswith("ه"):
        return Kind.NOUN
    return Kind.NOUN if len(word) >= 3 else Kind.UNKNOWN


def analyze(word: str, *, with_alternatives: bool = False) -> Analysis:
    """يحلّل كلمةً واحدة تحليلًا كاملًا."""
    raw = word.strip()
    disp = normalize(raw, DISPLAY)
    norm = normalize(raw, SEARCH)
    stem = light_stem(raw)
    deeper = strict_stem(raw)
    root = extract_root(raw)
    kind = _guess_kind(norm, stem)
    alts: tuple[str, ...] = ()
    if with_alternatives:
        alts = tuple(c.root for c in root_candidates(raw, limit=4)[1:])
    return Analysis(
        raw=raw,
        display=disp,
        norm=norm,
        stem=stem,
        deeper=deeper,
        root=root,
        kind=kind.value,
        confidence=round(root_confidence(raw), 3),
        is_stop=is_stopword(norm),
        alternatives=alts,
    )


def analyze_text(text: str) -> list[tuple[Token, Analysis]]:
    """يحلّل نصًّا كاملًا، كلمةً كلمة، مع الحفاظ على المواضع."""
    return [(t, analyze(t.raw)) for t in tokenize(text)]


def content_words(
    text: str, *, drop_stop: bool = True, use_root: bool = False
) -> Iterable[str]:
    """كلماتٌ محتوى (بلا إيقاف)، إمّا مادّية أو جذور."""
    for t in tokenize(text):
        if drop_stop and is_stopword(t.norm):
            continue
        if len(t.norm) < 2:
            continue
        yield extract_root(t.raw) if use_root else t.norm


def analyze_summary(text: str) -> dict:
    """ملخّصٌ إحصائيٌّ سريع لنصّ (يُستخدم في لوحة الإحصاءات)."""
    toks = tokenize(text)
    an = [analyze(t.raw) for t in toks]
    roots = [a.root for a in an if a.root]
    return {
        "tokens": len(toks),
        "types": len({t.norm for t in toks}),
        "stopwords": sum(1 for a in an if a.is_stop),
        "roots": len(set(roots)),
        "avg_token_len": round(sum(len(t.norm) for t in toks) / max(1, len(toks)), 2),
        "top_roots": _top(roots, 8),
    }


def _top(items: list[str], n: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
