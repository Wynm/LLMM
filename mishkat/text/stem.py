"""مُجذِّع عربي خفيف (Light-10) ومُجذِّع قاسٍ.

المُجذِّع الخفيف يحذف السوابق واللواحق الشائعة دون المساس ببنية الجذر؛ وهو ما
يحتاجه البحث (لا يحتاج «مدرسة» أن تصير «درس» بل أن تتوحّد مع «المدارس»).
أما المُجذِّع القاسٍ فيدفع إلى الجذر الثلاثي ويُستخدم للتجميع الدلالي.

كل عمليات الحذف محكومةٌ بحدودٍ دنيا للطول، حتى لا تُشوَّه الكلمات القصيرة:

* ``MIN_STEM = 3`` — لا يُحذف شيءٌ إن كان الناتج أقصر من ثلاثة أحرف.
* لا تُحذف سابقة ``و`` وحدها إن كانت الكلمة من أصلٍ ثلاثيّ (مثل «وعد»).
"""

from __future__ import annotations

from typing import Final

from .normalize import SEARCH, STRICT, normalize, skeleton

MIN_STEM: Final = 3

# سوابق مرتّبة من الأطول إلى الأقصر (تُجرَّب بالترتيب، أولُ مطابقةٍ تفوز)
PREFIXES: Final = (
    "وال", "بال", "كال", "فال", "ولل", "ال", "لل", "و",
)

# لواحق مرتّبة من الأطول إلى الأقصر
SUFFIXES: Final = (
    "كما", "هما", "تين", "تان", "اته", "ات", "ون", "ين", "ان", "ها",
    "هم", "هن", "كم", "كن", "نا", "ني", "ية", "ته", "ه", "ك", "ي", "ا",
    "ت", "ن", "و", "ة",
)

# سوابق تحمل دلالةً صرفية قوية (فعل/مصدر) — تُحذف في التجذيع القاسي فقط
VERBAL_PREFIXES: Final = ("است", "مست", "ان", "افت", "ت", "ي", "ن", "م")

# حروف لا تظهر في الجذر الثلاثي في الموضع الأول (زوائد)
_DEFECTIVE: Final = "اوي"


def _strip_one_prefix(word: str, min_len: int = MIN_STEM) -> str:
    for p in PREFIXES:
        if not word.startswith(p) or len(word) - len(p) < min_len:
            continue
        rest = word[len(p) :]
        # قاعدة عربية: لا تبدأ الكلمة (بعد السابقة) بألفٍ إلا في «ال»
        if rest.startswith("ا"):
            continue
        # لا نحذف سابقةً من كلمةٍ ثلاثيةٍ صحيحة (وعد، فم)
        if len(p) == 1 and len(word) <= 3:
            continue
        return rest
    return word


def _strip_one_suffix(word: str, min_len: int = MIN_STEM) -> str:
    for s in SUFFIXES:
        # اللواحق المفردة (ه، ك، ي، ت، ن، ا، و) لا تُحذف إن قصُر الباقي عن أربعة
        need = max(min_len, 4) if len(s) == 1 else min_len
        if word.endswith(s) and len(word) - len(s) >= need:
            return word[: -len(s)]
    return word


def light_stem(word: str) -> str:
    """تجذيع خفيف: إزالة سابقةٍ ولواحقَ شائعةٍ مع حفظ بنية الكلمة.

    >>> light_stem("والمدارس")
    'مدارس'
    >>> light_stem("الكتاب")
    'كتاب'
    """
    w = normalize(word, SEARCH)
    if not w or " " in w.strip():
        return w
    w = _strip_one_prefix(w)
    # اللواحق تُحذف تكرارًا (بحدٍّ أقصى اثنتين) مثل «كتابهم» → «كتاب»
    for _ in range(2):
        new = _strip_one_suffix(w)
        if new == w:
            break
        w = new
    return w


def strict_stem(word: str) -> str:
    """تجذيع قاسٍ: يحذف السوابق الصرفية واللواحق حتى يبقى ما يشبه الجذر.

    >>> strict_stem("والاستخدامات")
    'استخدام'
    """
    w = skeleton(word)
    if len(w) <= 3:
        return w
    for p in sorted(set(PREFIXES) | set(VERBAL_PREFIXES), key=len, reverse=True):
        if w.startswith(p) and len(w) - len(p) >= MIN_STEM:
            w = w[len(p) :]
            break
    for _ in range(3):
        new = _strip_one_suffix(w, min_len=MIN_STEM)
        if new == w:
            break
        w = new
    # إزالة الزوائد المضعّفة في النهاية (مثل «شدّد» بعد استعادة التضعيف)
    while len(w) > 3 and w[-1] == w[-2]:
        w = w[:-1]
    # إسقاط حروف العلّة الزائدة لتقارب العائلة الصرفية (كاتب/كتاب/مكتوب → كتب)
    i = 1
    while i < len(w) and len(w) > 3:
        if w[i] in _DEFECTIVE:
            w = w[:i] + w[i + 1 :]
        else:
            i += 1
    while len(w) > 3 and w[-1] == w[-2]:
        w = w[:-1]
    return w


def stem_pair(word: str) -> tuple[str, str]:
    """(خفيف، قاسٍ) في استدعاءٍ واحد."""
    return light_stem(word), strict_stem(word)


def varies(word: str) -> list[str]:
    """صيغ الكلمة المحتملة (الكلمة نفسها + الجذعين) لإثراء مطابقة الاستعلام."""
    w = normalize(word, STRICT)
    out = [w, normalize(word, SEARCH), light_stem(word), strict_stem(word)]
    seen, uniq = set(), []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq
