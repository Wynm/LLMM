"""النواة اللغوية العربية — Arabic linguistic core (stdlib only).

لا تعتمد هذه الحزمة على أي شيء خارج المكتبة القياسية، حتى تعمل في أي بيئة.
"""

from .normalize import (
    DISPLAY,
    LIGHT,
    SEARCH,
    STRICT,
    LEVELS,
    normalize,
    normalize_digits,
    skeleton,
    strip_diacritics,
    strip_tatweel,
    unify_alef,
    unify_hamza,
    unify_ta,
    unify_ya,
)
from .tokenize import (
    Sentence,
    Token,
    iter_ngrams,
    split_sentences,
    tokenize,
)
from .stopwords import is_stopword, stopword_count, stopwords
from .stem import light_stem, stem_pair, strict_stem, varies
from .roots import root_candidates, extract_root
from .analyze import Analysis, analyze, analyze_summary, analyze_text, content_words

__all__ = [
    "DISPLAY", "LIGHT", "SEARCH", "STRICT", "LEVELS",
    "normalize", "normalize_digits", "skeleton", "strip_diacritics", "strip_tatweel",
    "unify_alef", "unify_hamza", "unify_ta", "unify_ya",
    "Token", "Sentence", "tokenize", "split_sentences", "iter_ngrams",
    "stopwords", "is_stopword", "stopword_count",
    "light_stem", "strict_stem", "stem_pair", "varies",
    "root_candidates", "extract_root",
    "Analysis", "analyze", "analyze_text", "analyze_summary", "content_words",
]
