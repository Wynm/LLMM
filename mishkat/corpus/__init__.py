"""المجموعة النصّية: قراءة JSONL، تحقّق، وفهرسة."""

from .loader import Corpus, Document, CorpusError, load_corpus

__all__ = ["Corpus", "Document", "CorpusError", "load_corpus"]
