"""لغة استعلامٍ صغيرة للبحث المعجمي.

الصيغة
------
* كلماتٌ عادية: تُعالَج بـ OR (أيّ مطابقةٍ تُرجّح).
* ``"عبارة بين علامتَي تنصيص"``: مطابقةٌ متتالية إلزامية.
* ``+كلمة``: إلزامية.
* ``-كلمة``: مستبعدة.
* ``كلمة*``: سابقة (prefix).

مثال::

    علم الفلك +ابن -الحديث "كتاب المناظر"

المحلّل يدويٌّ (recursive descent مبسّط) لا regex، لأنه يجب أن يعرف الفرق بين
علامة تنصيصٍ مفتوحةٍ مغلقة، وعلامةٍ داخل كلمة، وأن يرفض الاستعلام المشوّه
بدل أن يُخمّن.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..text.normalize import SEARCH, normalize
from ..text.stopwords import is_stopword


class QueryError(ValueError):
    """استعلام غير صالح."""


@dataclass
class ParsedQuery:
    """تمثيلٌ بنيويّ لاستعلامٍ معجمي."""

    raw: str
    terms: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    phrases: list[list[str]] = field(default_factory=list)
    prefixes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.terms or self.required or self.phrases or self.prefixes)

    def all_positive(self) -> list[str]:
        """كل المصطلحات الموجبة (بلا المستبعدة)."""
        out = list(self.required) + list(self.terms)
        for ph in self.phrases:
            out.extend(ph)
        return list(dict.fromkeys(out))


def _clean(word: str) -> str:
    return normalize(word.strip("،؛؟,;:!?()[]{}"), SEARCH)


def parse_query(text: str, *, drop_stopwords: bool = True) -> ParsedQuery:
    """يحلّل نصّ استعلام إلى بنيته."""
    if text is None:
        raise QueryError("استعلام فارغ")
    q = ParsedQuery(raw=text)
    i, n = 0, len(text)
    word = ""
    sign = ""

    def flush() -> None:
        nonlocal word, sign
        if not word:
            sign = ""
            return
        w = _clean(word)
        if not w or (drop_stopwords and len(w) < 2) or (drop_stopwords and is_stopword(w)):
            word, sign = "", ""
            return
        if w.endswith("*") and len(w) > 1:
            q.prefixes.append(w[:-1])
        elif sign == "+":
            q.required.append(w)
        elif sign == "-":
            q.excluded.append(w)
        else:
            q.terms.append(w)
        word, sign = "", ""

    while i < n:
        ch = text[i]
        if ch in "\"«»":
            flush()
            close = {"\"": "\"", "«": "»"}[ch]
            j = text.find(close, i + 1)
            if j == -1:
                raise QueryError("علامة تنصيص غير مغلقة")
            phrase = text[i + 1 : j].strip()
            if phrase:
                toks = [
                    normalize(t, SEARCH)
                    for t in phrase.replace("،", " ").split()
                ]
                toks = [t for t in toks if t and not (drop_stopwords and is_stopword(t))]
                if toks:
                    q.phrases.append(toks)
            i = j + 1
            continue
        if ch in "+-":
            # العلامة تُعتبر تعديلًا فقط إن جاءت في بداية كلمة
            if not word:
                sign = ch
                i += 1
                continue
        if ch.isspace():
            flush()
            i += 1
            continue
        word += ch
        i += 1
    flush()

    if q.is_empty and q.excluded:
        raise QueryError("استعلام لا يحتوي إلا كلمات مستبعدة")
    return q


def build_terms(q: ParsedQuery, *, expand=("stem", "root")) -> dict[str, list[str]]:
    """يحوّل مصطلحات الاستعلام إلى صيغٍ مفهرسة (ساق/جذر).

    يعيد قاموسًا: صيغة الفهرس ← قائمة مصطلحات الاستعلام التي ولّدتها.
    """
    from ..text.analyze import analyze

    mapping: dict[str, list[str]] = {}

    def add(surface: str) -> None:
        a = analyze(surface)
        forms: list[str] = []
        if "surface" in expand:
            forms.append(a.norm)
        if "stem" in expand:
            forms.append(a.stem)
        if "root" in expand and a.root and len(a.root) >= 3:
            forms.append(a.root)
        for f in dict.fromkeys(forms):
            if f:
                mapping.setdefault(f, []).append(surface)

    for t in q.all_positive():
        add(t)
    for t in q.excluded:
        add(t)
    for p in q.prefixes:
        mapping.setdefault(p, []).append(p)
    return mapping
