"""الفهرس المقلوب: من مصطلحٍ إلى قائمة المستندات التي تحويه.

نخزّن لكل (مصطلح، مستند) تكراره داخل كل حقلٍ ومواضعه، لأن المواضع تسمح
بالبحث العبارتيّ («علم الفلك» متتاليةً لا مبعثرة) ولأن الحقول تسمح بوزن
العنوان أعلى من المتن.

مصطلحات الفهرسة
---------------
لكل كلمةٍ نُدخل صيغتين: **الساق** (light stem) و**الجذر** (root). البحث يوسّع
الاستعلام بالصيغتين معًا، فيجد «المدارس» حين يُسأل عن «الدراسة»، دون أن
يخلط بين كل مشتقّات الجذر. هذا حلٌّ وسط بين الدقة والاستدعاء.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from ..text.analyze import Kind, analyze
from ..text.normalize import SEARCH, normalize
from ..text.stopwords import is_stopword
from ..text.tokenize import tokenize

# أوزان الحقول عند حساب التكرار الموزون
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "summary": 2.0,
    "body": 1.0,
    "tags": 2.0,
}


@dataclass(slots=True)
class Posting:
    """تمثيل مصطلحٍ في مستندٍ واحد."""

    tf: float = 0.0
    positions: dict[str, list[int]] = field(default_factory=dict)

    def add(self, field_name: str, position: int, weight: float) -> None:
        self.tf += weight
        self.positions.setdefault(field_name, []).append(position)


def query_terms(text: str, *, expand_root: bool = True) -> list[str]:
    """يحوّل نصًّا إلى مصطلحات استعلام (بلا إيقاف، مع الساق والجذر)."""
    out: list[str] = []
    for tok in tokenize(text, level=SEARCH, sentences=False):
        w = tok.norm
        if not w or len(w) < 2 or is_stopword(w):
            continue
        a = analyze(tok.raw)
        out.append(a.stem)
        if expand_root and a.root and len(a.root) >= 3:
            out.append(a.root)
        if a.kind == Kind.VERB and a.deeper:
            out.append(a.deeper)
    # إزالة التكرار مع الحفاظ على الترتيب
    return list(dict.fromkeys(x for x in out if x))


def index_text(text: str) -> list[str]:
    """مصطلحات الفهرسة لمقطعٍ نصّي (ساق + جذر، بلا إيقاف)."""
    out: list[str] = []
    for tok in tokenize(text, level=SEARCH, sentences=False):
        w = tok.norm
        if not w or len(w) < 2 or is_stopword(w):
            continue
        a = analyze(tok.raw)
        out.append(a.stem)
        if a.root and len(a.root) >= 3 and a.root != a.stem:
            out.append(a.root)
    return out


class InvertedIndex:
    """فهرسٌ مقلوب قابل للبناء تدريجيًّا."""

    def __init__(self, field_weights: dict[str, float] | None = None) -> None:
        self.field_weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)
        self.postings: dict[str, dict[str, Posting]] = defaultdict(dict)
        self.doc_len: dict[str, float] = {}
        self.doc_field_len: dict[str, dict[str, int]] = {}
        self.doc_ids: list[str] = []
        self._finalized = False

    # ------------------------------------------------------------------ بناء
    def add_document(self, doc_id: str, fields: dict[str, str]) -> None:
        """يضيف مستندًا بحقولٍ متعددة."""
        if doc_id in self.doc_len:
            raise ValueError(f"مستند مكرر: {doc_id}")
        total = 0.0
        field_lengths: dict[str, int] = {}
        for field_name, text in fields.items():
            weight = self.field_weights.get(field_name, 1.0)
            toks = tokenize(text, level=SEARCH, sentences=False)
            field_lengths[field_name] = len(toks)
            for pos, tok in enumerate(toks):
                if not tok.norm or len(tok.norm) < 2 or is_stopword(tok.norm):
                    continue
                a = analyze(tok.raw)
                forms = {a.stem}
                if a.root and len(a.root) >= 3:
                    forms.add(a.root)
                for form in forms:
                    if not form:
                        continue
                    p = self.postings[form].get(doc_id)
                    if p is None:
                        p = Posting()
                        self.postings[form][doc_id] = p
                    p.add(field_name, pos, weight)
                total += weight
        self.doc_len[doc_id] = total
        self.doc_field_len[doc_id] = field_lengths
        self.doc_ids.append(doc_id)

    def finalize(self) -> "InvertedIndex":
        self._finalized = True
        return self

    # --------------------------------------------------------------- استعلام
    @property
    def size(self) -> int:
        return len(self.doc_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    @property
    def avg_doc_len(self) -> float:
        if not self.doc_len:
            return 0.0
        return sum(self.doc_len.values()) / len(self.doc_len)

    def df(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        """IDF لوكالة BM25: موجبٌ دائمًا حتى لا تُصفَّر المصطلحات الشائعة."""
        import math

        n = self.size
        df = self.df(term)
        if df == 0 or n == 0:
            return 0.0
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def candidates(self, terms: Iterable[str], mode: str = "or") -> set[str]:
        """مجموعة المستندات المرشّحة لمصطلحاتٍ (or/and)."""
        sets = [set(self.postings.get(t, {})) for t in terms]
        sets = [s for s in sets if s]
        if not sets:
            return set()
        if mode == "and":
            out = sets[0]
            for s in sets[1:]:
                out &= s
            return out
        out: set[str] = set()
        for s in sets:
            out |= s
        return out

    def _form(self, term: str) -> str:
        """يحوّل مصطلحًا إلى صيغته المخزّنة (ساق) إن لزم."""
        if term in self.postings:
            return term
        from ..text.stem import light_stem

        stem = light_stem(term)
        return stem if stem in self.postings else term

    def phrase(self, terms: list[str], *, field_name: str | None = None) -> set[str]:
        """مستنداتٌ تحتوي المصطلحات متتاليةً (بحثٌ عبارتيّ)."""
        if not terms:
            return set()
        terms = [self._form(t) for t in terms]
        first = self.postings.get(terms[0], {})
        out: set[str] = set()
        for doc_id, p in first.items():
            for fname, positions in p.positions.items():
                if field_name and fname != field_name:
                    continue
                pos_set = set(positions)
                for start in positions:
                    ok = True
                    for offset, term in enumerate(terms[1:], 1):
                        other = self.postings.get(term, {}).get(doc_id)
                        if not other or start + offset not in set(
                            other.positions.get(fname, ())
                        ):
                            ok = False
                            break
                    if ok:
                        out.add(doc_id)
                        break
                if doc_id in out:
                    break
        return out

    def prefix(self, prefix: str, *, limit: int = 50) -> list[str]:
        """مصطلحاتٌ تبدأ بسابقة (للإكمال التلقائي)."""
        if not prefix:
            return []
        return sorted(t for t in self.postings if t.startswith(prefix))[:limit]

    def terms(self) -> Iterator[str]:
        return iter(self.postings)

    def expand_synonyms(self, terms: list[str], *, max_extra: int = 12) -> list[str]:
        """يوسّع مصطلحات الاستعلام بجذورها المشتركة (مرادفاتٌ صرفية)."""
        extra: list[str] = []
        for t in terms:
            for term in self.postings.get(t, {}) and [t]:
                _ = term
            # كل مصطلحٍ آخر يشترك معه في الجذر ليس متاحًا هنا مباشرة،
            # فيُترك التوسيع الدلالي لطبقة المتجهات.
        return list(dict.fromkeys(terms + extra))[: len(terms) + max_extra]

    # -------------------------------------------------------------- إحصاءات
    def stats(self) -> dict:
        return {
            "documents": self.size,
            "vocabulary": self.vocab_size,
            "avg_doc_len": round(self.avg_doc_len, 2),
            "postings": sum(len(p) for p in self.postings.values()),
            "fields": list(self.field_weights),
        }

    def to_dict(self) -> dict:
        """تصديرٌ قابل للتسلسل (للتصحيح والواجهة)."""
        return {
            "stats": self.stats(),
            "top_terms": sorted(
                ((t, len(p)) for t, p in self.postings.items()),
                key=lambda kv: (-kv[1], kv[0]),
            )[:40],
        }
