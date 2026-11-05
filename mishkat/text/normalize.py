"""تطبيع النص العربي على أربعة مستويات.

مستويات التطبيع
---------------
``display``  للنشر والعرض: يحافظ على الهمزات والألف بأشكالها، يزيل التشكيل والتطويل
             ومحارف الاتجاه، ويوحّد المسافات. نصٌّ نظيفٌ يُقرأ.
``light``    للفهرسة الوسطى: كل ما سبق + توحيد الألف (أإآٱ → ا) والياء (ى → ي)
             والهمزة على الكرسي (ؤ → و، ئ → ي). يحافظ على التاء المربوطة.
``search``   للبحث (الافتراضي): كل ما سبق + التاء المربوطة (ة → ه) + توحيد الأرقام
             الهندية إلى ASCII + تحويل علامات الترقيم العربية إلى نظيراتها.
``strict``   للتجذيع وتحليل الصرف: حروف عربية وأرقام لاتينية فقط، وإسقاط كل ما
             عداها (ترقيم، رموز، لاتيني) — «هيكل» الكلمة.

لماذا أربعة مستويات؟ لأن «مدرسة» و«مدرسه» يجب أن تتطابقا في البحث ولا تتطابقا في
العرض، ولأن «قال» تُجذَّر من هيكلها لا من شكلها المعروض.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

DISPLAY: Final = "display"
LIGHT: Final = "light"
SEARCH: Final = "search"
STRICT: Final = "strict"
LEVELS: Final = (DISPLAY, LIGHT, SEARCH, STRICT)

# التشكيل وعلامات القرآن والعلامات الصرفية العليا
_DIACRITICS: Final = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u08D3-\u08FF\uFE70-\uFE7F]"
)
# الشدّة وحدها (نحتاجها لاستعادة التضعيف قبل حذف التشكيل)
_SHADDA: Final = "\u0651"
_TATWEEL: Final = re.compile("\u0640")
_BIDI_CONTROLS: Final = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2066-\u2069\u061c\ufeff\u200e\u200f]"
)
_SPACES: Final = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")

_ALEF_FORMS: Final = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا"})
_HAMZA_CARRIERS: Final = str.maketrans({"ؤ": "و", "ئ": "ي", "ٮ": "ت"})
_YA_FORMS: Final = str.maketrans({"ى": "ي", "ی": "ي", "ے": "ي"})
_TA_FORMS: Final = str.maketrans({"ة": "ه"})
_ARABIC_PUNCT: Final = str.maketrans({
    "،": ",", "؛": ";", "؟": "?", "٪": "%", "۔": ".", "٫": ".", "٬": ",",
    "«": '"', "»": '"', "﴿": "(", "﴾": ")", "ـ": "",
})

_ARABIC_LETTERS: Final = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"


def _to_ascii_digits(text: str) -> str:
    """يحوّل الأرقام الهندية (٠-٩) والفارسية (۰-۹) إلى ASCII."""
    out = []
    for ch in text:
        code = ord(ch)
        if 0x0660 <= code <= 0x0669:
            out.append(chr(code - 0x0660 + 0x30))
        elif 0x06F0 <= code <= 0x06F9:
            out.append(chr(code - 0x06F0 + 0x30))
        else:
            out.append(ch)
    return "".join(out)


def normalize_digits(text: str) -> str:
    """واجهة عامة لتحويل الأرقام الهندية إلى لاتينية."""
    return _to_ascii_digits(text)


def _expand_shadda(text: str) -> str:
    """يستعيد الحرف المضعّف: ``شدّ`` → ``شدد`` (قبل حذف التشكيل)."""
    out: list[str] = []
    for ch in text:
        if ch == _SHADDA and out:
            out.append(out[-1])
        else:
            out.append(ch)
    return "".join(out)


def strip_diacritics(text: str) -> str:
    """يزيل التشكيل وعلامات الضبط وعلامات القرآن."""
    return _DIACRITICS.sub("", text)


def strip_tatweel(text: str) -> str:
    """يزيل التطويل (الكشيدة)."""
    return _TATWEEL.sub("", text)


def unify_alef(text: str) -> str:
    """يوحّد أشكال الألف: أ إ آ ٱ → ا."""
    return text.translate(_ALEF_FORMS)


def unify_hamza(text: str) -> str:
    """يوحّد الهمزة على الكرسي: ؤ → و، ئ → ي."""
    return text.translate(_HAMZA_CARRIERS)


def unify_ya(text: str) -> str:
    """يوحّد الياء: ى → ي."""
    return text.translate(_YA_FORMS)


def unify_ta(text: str) -> str:
    """يوحّد التاء المربوطة: ة → ه."""
    return text.translate(_TA_FORMS)


def normalize(text: str, level: str = SEARCH, *, expand_shadda: bool = False) -> str:
    """يطبّع نصًّا عربيًّا إلى المستوى المطلوب.

    Parameters
    ----------
    text:
        النصّ الأصلي بأي حالة.
    level:
        أحد ``display`` / ``light`` / ``search`` / ``strict``.
    expand_shadda:
        إن كان صحيحًا تُستعاد الحروف المضعّفة (مفيد قبل التجذيع).

    Examples
    --------
    >>> normalize("المَدْرَسَةُ")
    'المدرسه'
    >>> normalize("أَحْمَد", level="display")
    'أحمد'
    """
    if level not in LEVELS:
        raise ValueError(f"مستوى تطبيع غير معروف: {level!r}؛ المتاح: {LEVELS}")
    if not text:
        return ""

    s = unicodedata.normalize("NFKC", text)
    s = _BIDI_CONTROLS.sub("", s)
    if expand_shadda:
        s = _expand_shadda(s)
    s = strip_tatweel(s)
    s = strip_diacritics(s)
    s = _SPACES.sub(" ", s)

    if level == DISPLAY:
        return s.strip()

    s = unify_alef(s)
    s = unify_hamza(s)
    s = unify_ya(s)

    if level == LIGHT:
        return s.strip()

    s = unify_ta(s)
    s = _to_ascii_digits(s)
    s = s.translate(_ARABIC_PUNCT)

    if level == SEARCH:
        return s.strip()

    # STRICT: حروف عربية + أرقام لاتينية + مسافة فقط
    kept = []
    for ch in s:
        if ch in _ARABIC_LETTERS or ch.isdigit() or ch == " ":
            kept.append(ch)
    return "".join(kept).strip()


def expand_shadda(text: str) -> str:
    """واجهة عامة: يستعيد الحرف المضعّف قبل حذف التشكيل."""
    return _expand_shadda(text)


def skeleton(text: str) -> str:
    """هيكل الكلمة: حروف عربية فقط بعد تطبيع صارم (بلا أرقام ولا مسافات)."""
    return "".join(ch for ch in normalize(text, STRICT) if ch in _ARABIC_LETTERS)


def is_arabic_word(text: str) -> bool:
    """هل النصّ كلمةٌ عربية (حروف عربية فقط)؟"""
    return bool(text) and all(ch in _ARABIC_LETTERS for ch in text)
