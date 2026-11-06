"""محمّل المجموعة: ملفّات JSONL في ``data/corpus``.

المخطّط
-------
المستندات (``0*.jsonl``)::

    {"id": "math-01", "title": "...", "domain": "رياضيات", "era": "...",
     "summary": "...", "body": "...", "tags": ["...", "..."]}

الكيانات (``entities.jsonl``)::

    {"id": "person:al-khwarizmi", "type": "شخص", "label": "الخوارزمي", "note": "..."}

العلاقات (``edges.jsonl``)::

    {"s": "person:al-khwarizmi", "r": "ألّف", "t": "doc:math-01", "n": "..."}

لماذا JSONL؟ لأن كل سطرٍ مستقلّ: `git diff` يظهر ما أُضيف بالضبط، وسطرٌ تالف
لا يُفسد الملفّ كله، والبشر يقرأونه بلا أدوات.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..text.normalize import SEARCH, normalize

_REQUIRED_DOC_FIELDS = ("id", "title", "domain", "summary", "body")

# رمزٌ بصريّ لكل نطاق (يستخدمه الاستوديو)
DOMAIN_ICONS: dict[str, str] = {
    "فلك": "✦", "رياضيات": "∑", "طب": "⚕", "فلسفة": "∴", "لسانيات": "ء",
    "هندسة": "△", "بيئة": "❧", "اقتصاد": "◇", "فنون": "✧", "تاريخ": "⌛",
    "كيمياء": "⚗", "حوسبة": "⌘", "أحياء": "❀", "جغرافيا": "◈",
    "ملاحة": "⚓", "بصريات": "◎", "قانون": "§", "تعليم": "✎",
    "مدينة": "▦", "طاقة": "☀",
}
_MIN_BODY_WORDS = 20


class CorpusError(ValueError):
    """خطأ في مخطّط المجموعة."""


@dataclass(frozen=True, slots=True)
class Document:
    """مستندٌ واحد في المجموعة."""

    id: str
    title: str
    domain: str
    summary: str
    body: str
    era: str = ""
    tags: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """النصّ الكامل للبحث والتحليل."""
        return f"{self.title}. {self.summary} {self.body}"

    @property
    def fields(self) -> dict[str, str]:
        """حقولٌ موزونة يُبنى عليها الفهرس."""
        return {
            "title": self.title,
            "summary": self.summary,
            "body": self.body,
            "tags": " ".join(self.tags),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "domain": self.domain,
            "era": self.era,
            "summary": self.summary,
            "body": self.body,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class Entity:
    """كيانٌ في الرسم المعرفي (شخص، مكان، مفهوم، أداة)."""

    id: str
    type: str
    label: str
    note: str = ""
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "note": self.note,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class Edge:
    """حافةٌ في الرسم: (مصدر، علاقة، هدف)."""

    source: str
    rel: str
    target: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"s": self.source, "r": self.rel, "t": self.target, "n": self.note}


@dataclass
class Corpus:
    """المجموعة كاملة: مستندات + كيانات + علاقات، مع استعلاماتٍ سريعة."""

    documents: list[Document] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    root: Path | None = None

    # فهارس داخلية
    _by_id: dict[str, Document] = field(default_factory=dict, repr=False)
    _entity_by_id: dict[str, Entity] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {d.id: d for d in self.documents}
        self._entity_by_id = {e.id: e for e in self.entities}

    # ---------------------------------------------------------------- وصول
    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def doc(self, doc_id: str) -> Document:
        try:
            return self._by_id[doc_id]
        except KeyError:
            raise CorpusError(f"مستند غير موجود: {doc_id!r}") from None

    def entity(self, entity_id: str) -> Entity | None:
        return self._entity_by_id.get(entity_id)

    @property
    def domains(self) -> list[str]:
        seen: dict[str, int] = {}
        for d in self.documents:
            seen[d.domain] = seen.get(d.domain, 0) + 1
        return sorted(seen, key=lambda k: (-seen[k], k))

    @property
    def all_edges(self) -> list[Edge]:
        """كل الحواف: الصريحة + حافة «ينتمي_إلى» بين كل مستندٍ ونطاقه."""
        out = list(self.edges)
        for d in self.documents:
            out.append(Edge(f"doc:{d.id}", "ينتمي_إلى", f"domain:{d.domain}"))
        return out

    # ---------------------------------------------------------------- إحصاء
    def stats(self) -> dict[str, Any]:
        from collections import Counter

        words = Counter()
        for d in self.documents:
            words.update(normalize(d.text, SEARCH).split())
        domain_counts = Counter(d.domain for d in self.documents)
        era_counts = Counter(d.era for d in self.documents if d.era)
        return {
            "documents": len(self.documents),
            "entities": len(self.entities),
            "edges": len(self.edges),
            "domains": dict(domain_counts.most_common()),
            "eras": dict(era_counts.most_common(8)),
            "tokens": sum(words.values()),
            "types": len(words),
            "avg_doc_tokens": round(sum(words.values()) / max(1, len(self.documents)), 1),
            "icons": self._icon_map(),
        }

    def _icon_map(self) -> dict[str, str]:
        """رمزٌ بصريّ لكل نطاق (يستخدمه الاستوديو)."""
        return {d: DOMAIN_ICONS.get(d, "◆") for d in self.domains}

    def searchable(self, level: str = SEARCH) -> Iterable[tuple[str, str]]:
        """(معرّف المستند، نصّه المطبّع)."""
        for d in self.documents:
            yield d.id, normalize(d.text, level)


def _default_root() -> Path:
    env = os.environ.get("MISHKAT_CORPUS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "corpus"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path.name}:{lineno}: JSON غير صالح — {exc}") from exc
    return rows


def _validate_doc(row: dict[str, Any], where: str) -> Document:
    missing = [f for f in _REQUIRED_DOC_FIELDS if not row.get(f)]
    if missing:
        raise CorpusError(f"{where}: حقول ناقصة {missing}")
    body = str(row["body"])
    if len(body.split()) < _MIN_BODY_WORDS:
        raise CorpusError(
            f"{where}: متن قصير جدًّا ({len(body.split())} كلمة، الحدّ {_MIN_BODY_WORDS})"
        )
    tags = row.get("tags") or []
    if not isinstance(tags, list):
        raise CorpusError(f"{where}: tags يجب أن تكون قائمة")
    return Document(
        id=str(row["id"]),
        title=str(row["title"]).strip(),
        domain=str(row["domain"]).strip(),
        summary=str(row["summary"]).strip(),
        body=body.strip(),
        era=str(row.get("era", "")).strip(),
        tags=tuple(str(t) for t in tags),
    )


def load_corpus(root: Path | str | None = None, *, strict: bool = True) -> Corpus:
    """يقرأ المجموعة من مجلّدٍ يحتوي ``*.jsonl``.

    Parameters
    ----------
    root:
        مجلّد المجموعة؛ الافتراضي ``data/corpus`` في جذر المشروع، أو
        متغيّر البيئة ``MISHKAT_CORPUS``.
    strict:
        إن كان صحيحًا يرفع خطأً عند أيّ سطرٍ غير صالح.
    """
    base = Path(root) if root else _default_root()
    if not base.exists():
        raise CorpusError(f"مجلّد المجموعة غير موجود: {base}")

    documents: list[Document] = []
    entities: list[Entity] = []
    edges: list[Edge] = []
    seen_ids: set[str] = set()

    for path in sorted(base.glob("*.jsonl")):
        rows = _read_jsonl(path)
        name = path.name
        for i, row in enumerate(rows, 1):
            where = f"{name}#{i}"
            if "s" in row and "r" in row and "t" in row:
                edges.append(
                    Edge(str(row["s"]), str(row["r"]), str(row["t"]), str(row.get("n", "")))
                )
            elif "type" in row and "label" in row:
                entities.append(
                    Entity(
                        id=str(row["id"]),
                        type=str(row["type"]),
                        label=str(row["label"]),
                        note=str(row.get("note", "")),
                        aliases=tuple(str(a) for a in row.get("aliases", [])),
                    )
                )
            elif "id" in row and "title" in row:
                doc = _validate_doc(row, where)
                if doc.id in seen_ids:
                    raise CorpusError(f"{where}: معرّف مكرر {doc.id!r}")
                seen_ids.add(doc.id)
                documents.append(doc)
            elif strict:
                raise CorpusError(f"{where}: سطر لا يُعرف نوعه: {list(row)}")

    if not documents:
        raise CorpusError(f"لا مستندات في {base}")

    # تحقّق من سلامة العلاقات (لا حافة تشير إلى عدم)
    known = {f"doc:{d.id}" for d in documents} | {e.id for e in entities}
    known |= {f"domain:{d.domain}" for d in documents}
    broken = [e for e in edges if e.source not in known or e.target not in known]
    if broken and strict:
        b = broken[0]
        raise CorpusError(f"حافة معلّقة: {b.source} --{b.rel}--> {b.target}")

    return Corpus(documents=documents, entities=entities, edges=edges, root=base)
