"""استخراج الجذر العربي بالأنماط الصرفية مع معالجة حروف العلّة.

المشكلة
-------
الجذر ليس عملية نصّية بسيطة: «المدارس» → درس، لكن «مدارس» كذلك، و«دارس» →
درس أو دور؟ و«قال» جذرها ق-و-ل لا ق-ا-ل. أي أداةٍ تحذف سوابق ولواحق فقط
تُخطئ في نصف الحالات.

الطريقة
-------
1. **توليد قواعد (bases) متعدّدة** من الكلمة: كما هي، وبلا سابقةٍ اسمية، وبلا
   سابقةٍ فعلية، وبلا لواحق — كلُّ تبادلٍ يولّد قاعدةً مرشّحة، ولا نحذف شيئًا
   نهائيًّا (لأن الحذف الخاطئ لا رجعة فيه).
2. **توليد الجذور** من كل قاعدة: مطابقة أوزان (فاعل، مفعول، استفعال …)، أو
   قبولٍ مباشر للثلاثي، أو مطابقة جذورٍ رباعية معروفة، أو إسقاط حرفٍ زائد.
3. **التصحيح الإعلالي**: بدائل (ا↔و↔ي) لأن حرف العلّة يتغيّر سطحًا ويبقى جذرًا.
4. **الترجيح** بمُعجم جذورٍ مضمَّن + قواعدٍ إحصائية؛ يفوز المرشّح المعروف.

نُعلن حدود الدقّة بدل ادّعاء الكمال: النظام يعمل بلا مُعجمٍ صرفيّ ضخم،
ويعطي درجة ثقةٍ مع كل جذر حتى يعرف المستخدم متى يشكّ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Iterator

from .normalize import STRICT, normalize

WEAK: Final = "اوي"

# أوزان صرفية أساسية. الحروف ف/ع/ل مواضع الجذر، وما عداها حرفٌ ثابت.
PATTERNS: Final[tuple[str, ...]] = (
    "فعل", "فعال", "فعول", "فعيل", "فعله", "فعلان", "فعلي", "فعولان",
    "فاعل", "فاعله", "فاعول", "فعاله", "فعاليل", "فاعلين", "فاعلات",
    "مفعل", "مفعله", "مفعول", "مفعال", "مفاعل", "مفاعله", "مفاعيل",
    "تفعل", "تفعله", "تفاعل", "تفعال", "تفعيل", "تفعله", "متفاعل",
    "افعل", "افعله", "افتعل", "افتعال", "انفعل", "انفعال",
    "استفعل", "استفعال", "مستفعل", "مستفعلين",
    "مفعلين", "مفعلات", "مفعولين",
)

# سوابق اسمية
_NOMINAL_PREFIXES: Final[tuple[str, ...]] = (
    "وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل",
)

# سوابق فعلية (تُجرَّب كمسارٍ مستقلّ، ولا تُحذف نهائيًّا)
_VERBAL_PREFIXES: Final[tuple[str, ...]] = (
    "يست", "تست", "نست", "مست", "است", "افت", "ان", "سي",
    "ي", "ت", "ن", "م", "س",
)

# لواحق (تُجرَّب كمسارٍ مستقلّ)
_SUFFIXES: Final[tuple[str, ...]] = (
    "كما", "هما", "تين", "تان", "اته", "اتهم", "اتها", "اتنا",
    "ات", "ون", "ين", "ان", "ها", "هم", "هن", "كم", "كن", "نا", "ني",
    "يه", "ته", "ه", "ك", "ي", "ا", "ت", "ن", "و", "ة",
)

# جذور رباعية شائعة (وزن «فعلل» وما أُلحق به من محدثات)
QUAD_ROOTS: Final[frozenset[str]] = frozenset(
    "برمج هندس زلزل وسوس دحرج ترجم برهن قهقر حوسب شفرن رقمن عربن "
    "عسكر جندل شنطر طمأن سلسل زخرف كرمل".split()
)

_RAW_ROOTS: Final = """
كتب علم درس قرا قول فعل صنع خلق جعل وضع رفع نزل دخل خرج سال سير طير طار
نظر بصر سمع حسس لمس شمم ذوق كلم حدث خبر عرف جهل فهم عقل فكر ذكر نسا حيي
كون كان صار صير اصب ظلل بات بيت نام نوم يقظ صحو فتح غلق ربط شدد مدد جمد
سيل جري عدو مشي سقط وقع طلع هبط صعد علا سمو كبر صغر طال طول قصر عرض
كثف رقق لطف عنف سهل صعب جمع فرق قسم وزع عدد حسب قاس قيس زيد نقص كمل تمم
بدا بدء نها اول اخر وسط حلل رتب نظم نسق صفف
ملك حكم قضي قضا عدل ظلم رحم عذب الم وجع شفي دوا دوي طبب مرض صحح سقم
اكل شرب هضم طعم ذاق جوع عطش نفس تنفس قلب دمي دمو عرق جلل جلد عظم عضم
شعر رءس يدي يدو رجل عين اذن انف فوه سنن لسن
ابو ابا اخو اخا امم بنو بني ولد والد نسل سبط قرب بعد وصل فصل ضمم
صهر نسب زوج طلق رضع حضن طفل شيخ كهل شاب
طريق سلك عبر جسر ركب ركو سفن بحر نهر مطر غيث سحب ريح هوي سما
شمس قمر نجم فلك كوب مدار برج خسف كسف غرب شرق ليل صبح فجر ظهر عصر
نار نور حرر حرق برد ثلج ذاب بخخ دخن لهب شعل وقو وقد
موه ارض ترا ترب حجر صخر جبل ودي واد سهب صحر غاب غيب نبت زرع حصد
شجر ورق جذر ثمر زهر عشب حشر نحل سمك سبع
مدن قري سكن عمر بناء شكل صمم خطط رسم رقم جبر معي منطق برهن ثبت فرض
لغو صرف نحو بلغ حرف صوت نطق سمي اسم
فلس قيم خير شر حقق صدق كذب جمل حسن قبح جود سوء خلق ادب قلم صفح ورق
نفع ضرر عون بيع خوف عيش موت حيو صيد رعي سعي رمو رمي قيو
حفظ درس درس فهم وعي شكر صبر امل رجو رجا خاف رهب حزن فرح ضحك بكي
فتح قفل سجن اطلق حرر عبء حمل رفع خفض مدد بسط قبض عقد حلل
بحث درس قرا كتب نشر طبع صحف اعلن بلغ وصل شبك حوسب رقمن برمج
بيت سكن عمر هدم بني شيد صمم هندس مسح قاس رصد
حسب عدد ضرب قسم كسر جمع طرح ضاع ضعف قوي كبر صغر
""".split()

_ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
KNOWN_ROOTS: Final[frozenset[str]] = frozenset(
    r for r in _RAW_ROOTS if len(r) == 3 and all(ch in _ALPHABET for ch in r)
)


@dataclass(frozen=True, slots=True)
class RootCandidate:
    """مرشّح جذرٍ مع درجته وسبب اختياره."""

    root: str
    score: float
    source: str

    def __repr__(self) -> str:  # pragma: no cover
        return f"RootCandidate({self.root!r}, {self.score:.1f}, {self.source})"


# ---------------------------------------------------------------- توليد القواعد
def _bases(word: str) -> list[str]:
    """يولّد قواعد (bases) متعدّدة: بلا حذفٍ نهائي، بل مساراتٍ تُقيَّم."""
    seeds: list[str] = [word]

    b = word
    for p in _NOMINAL_PREFIXES:
        if b.startswith(p) and len(b) - len(p) >= 3 and not b[len(p) :].startswith("ا"):
            seeds.append(b[len(p) :])
            break

    for x in list(seeds):
        v = x
        for p in _VERBAL_PREFIXES:
            if v.startswith(p) and len(v) - len(p) >= 3:
                seeds.append(v[len(p) :])
                break

    out: list[str] = []
    for x in seeds:
        out.append(x)
        y = x
        for _ in range(2):
            hit = None
            for s in _SUFFIXES:
                if y.endswith(s) and len(y) - len(s) >= 3:
                    hit = s
                    break
            if not hit:
                break
            y = y[: -len(hit)]
            out.append(y)
    return list(dict.fromkeys(out))


def _match_pattern(word: str, pattern: str) -> str | None:
    """يطابق كلمةً مع وزنٍ؛ يعيد الجذر إن طابق."""
    if len(word) != len(pattern):
        return None
    root_chars: list[str] = []
    for wc, pc in zip(word, pattern):
        if pc in "فعل":
            root_chars.append(wc)
        elif wc != pc:
            return None
    return "".join(root_chars)


def _weak_variants(root: str) -> Iterable[str]:
    """يولّد بدائل الإعلال لجذرٍ فيه حرف علّة."""
    yield root
    if len(root) != 3:
        return
    subs = {"ا": ("و", "ي"), "و": ("ا", "ي"), "ي": ("ا", "و")}
    for i, ch in enumerate(root):
        for alt in subs.get(ch, ()):
            yield root[:i] + alt + root[i + 1 :]


def _score(root: str, source: str) -> float:
    if len(root) == 4:
        return 7.0 if root in QUAD_ROOTS else -2.5
    if len(root) != 3:
        return -10.0 + len(root)
    s = 0.0
    if root in KNOWN_ROOTS:
        s += 6.0
    if root[0] in WEAK:
        s -= 1.5
    if root[0] == "ا":
        s -= 2.0
    if root.count(root[0]) == 3:
        s -= 3.0
    if root[1] == "ا" and root not in KNOWN_ROOTS:
        s -= 2.0
    if root[-1] in WEAK and root in KNOWN_ROOTS:
        s += 0.5
    s += {"pattern": 1.5, "strip": 1.0, "quad": 1.0, "asis": 0.5}.get(source, -1.0)
    return s


def _candidates(word: str) -> list[RootCandidate]:
    cands: list[RootCandidate] = []

    for base in _bases(word):
        if len(base) == 3:
            cands.append(RootCandidate(base, _score(base, "strip"), "strip"))
            continue
        if len(base) == 4 and base in QUAD_ROOTS:
            cands.append(RootCandidate(base, _score(base, "quad"), "quad"))
        for pat in PATTERNS:
            r = _match_pattern(base, pat)
            if r:
                cands.append(RootCandidate(r, _score(r, "pattern"), "pattern"))
        # إسقاط حرفٍ أو حرفين حتى الوصول إلى ثلاثة
        _reduce(base, cands, depth=0)

    # التصحيح الإعلالي
    expanded = list(cands)
    for c in cands:
        for alt in _weak_variants(c.root):
            if alt != c.root:
                expanded.append(RootCandidate(alt, c.score - 0.5, c.source + "+weak"))
    return expanded


def _reduce(
    w: str, out: list[RootCandidate], *, depth: int, max_depth: int = 2
) -> None:
    if len(w) == 3:
        out.append(RootCandidate(w, _score(w, "delete"), "delete"))
        return
    if len(w) < 3 or depth >= max_depth:
        return
    for i in range(len(w)):
        _reduce(w[:i] + w[i + 1 :], out, depth=depth + 1, max_depth=max_depth)


def root_candidates(word: str, *, limit: int = 5) -> list[RootCandidate]:
    """كل مرشّحي الجذر مرتّبين بالدرجة تنازليًّا (بلا تكرار)."""
    w = normalize(word, STRICT, expand_shadda=True)
    if not w:
        return []
    if len(w) < 3:
        return [RootCandidate(w, 0.2, "asis")]
    if len(w) == 3:
        best = RootCandidate(w, _score(w, "asis"), "asis")
        for alt in _weak_variants(w):
            c = RootCandidate(alt, _score(alt, "asis") - 0.2, "asis+weak")
            if c.score > best.score:
                best = c
        return [best]

    best: dict[str, RootCandidate] = {}
    for c in _candidates(w):
        cur = best.get(c.root)
        if cur is None or c.score > cur.score:
            best[c.root] = c
    out = sorted(
        best.values(),
        key=lambda c: (-c.score, c.root not in KNOWN_ROOTS, c.root),
    )
    return out[:limit]


def extract_root(word: str) -> str:
    """الجذر الأرجح للكلمة، أو الكلمة المطبّعة إن تعذّر.

    >>> extract_root("والمدرسة")
    'درس'
    >>> extract_root("استخدام")
    'خدم'
    >>> extract_root("قال")
    'قول'
    >>> extract_root("برمجة")
    'برمج'
    """
    w = normalize(word, STRICT, expand_shadda=True)
    if not w:
        return ""
    cands = root_candidates(word, limit=1)
    return cands[0].root if cands else w


def root_confidence(word: str) -> float:
    """ثقةٌ تقريبية [0..1] في الجذر المستخرج."""
    cands = root_candidates(word, limit=1)
    if not cands:
        return 0.0
    s = cands[0].score
    return round(max(0.0, min(1.0, (s + 3.0) / 10.0)), 3)
