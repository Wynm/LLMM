"""اختبارات النواة اللغوية — لا تعتمد على أي شيء خارج المكتبة القياسية."""

from __future__ import annotations

import pytest

from mishkat.text import (
    analyze,
    analyze_summary,
    content_words,
    extract_root,
    is_stopword,
    iter_ngrams,
    light_stem,
    normalize,
    normalize_digits,
    root_candidates,
    skeleton,
    split_sentences,
    stopword_count,
    strict_stem,
    tokenize,
)


# ------------------------------------------------------------------ التطبيع
@pytest.mark.parametrize(
    "raw,level,expected",
    [
        ("المَدْرَسَةُ", "search", "المدرسه"),
        ("المَدْرَسَةُ", "display", "المدرسة"),
        ("أَحْمَد", "display", "أحمد"),
        ("أحمد", "light", "احمد"),
        ("إنسان", "search", "انسان"),
        ("سماءْ", "search", "سماء"),
        ("مُؤَسَّسَة", "search", "موسسه"),
        ("قـــلم", "search", "قلم"),
        ("على", "light", "علي"),
        ("٠١٢٣٤٥", "search", "012345"),
        ("۱۲۳", "search", "123"),
        ("ما؟", "strict", "ما"),
        ("مرحبا 123!", "strict", "مرحبا 123"),
        ("", "search", ""),
        ("  متباعد   جدًا  ", "search", "متباعد جدا"),
    ],
)
def test_normalize_levels(raw: str, level: str, expected: str) -> None:
    assert normalize(raw, level) == expected


def test_normalize_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        normalize("نص", "nonexistent")


def test_normalize_digits_helper() -> None:
    assert normalize_digits("٣٤٥") == "345"


def test_shadda_expansion() -> None:
    assert normalize("شدّ", "strict", expand_shadda=True) == "شدد"


def test_skeleton_drops_everything_but_letters() -> None:
    assert skeleton("(١٢٣) كِتَاب!") == "كتاب"
    assert skeleton("GPT-4 و مِشكاة") == "ومشكاه"


# ------------------------------------------------------------------ التقطيع
def test_tokenize_offsets_are_exact() -> None:
    text = "النجمُ ساطع، والسماء صافية."
    for tok in tokenize(text):
        assert text[tok.start : tok.end] == tok.raw


def test_tokenize_normalized_forms() -> None:
    assert [t.norm for t in tokenize("النجم ساطع، والسماء صافية.")] == [
        "النجم",
        "ساطع",
        "والسماء",
        "صافيه",
    ]


def test_tokenize_numbers_and_latin() -> None:
    norms = [t.norm for t in tokenize("عام 1450 م و GPT-4 نموذج")]
    assert "1450" in norms
    assert any(n.lower().startswith("gpt") for n in norms)


def test_sentence_split() -> None:
    sents = split_sentences("النجم ساطع. والسماء صافية! هل ترى؟")
    assert len(sents) == 3
    assert sents[0].text == "النجم ساطع."


def test_sentence_split_keeps_abbreviation() -> None:
    # نقطة الاختصار «م.» ليست نهاية جملة
    sents = split_sentences("وُلد ابن خلدون سنة 1332 م. ثم رحل إلى مصر بعد عامين.")
    assert len(sents) == 1
    assert "م." in sents[0].text


def test_empty_text() -> None:
    assert tokenize("") == []
    assert split_sentences("   ") == []


def test_ngrams() -> None:
    toks = tokenize("أ ب ج د", sentences=False)
    grams = list(iter_ngrams(toks, 2))
    assert len(grams) == 3
    assert [t.norm for t in grams[0]] == ["ا", "ب"]


def test_ngrams_rejects_zero() -> None:
    with pytest.raises(ValueError):
        list(iter_ngrams([], 0))


# ------------------------------------------------------------- كلمات الإيقاف
def test_stopwords_are_many() -> None:
    assert stopword_count() >= 200


@pytest.mark.parametrize("w", ["من", "في", "هذا", "لكن", "التي"])
def test_stopword_true(w: str) -> None:
    assert is_stopword(w)


@pytest.mark.parametrize("w", ["نجم", "فلسفة", "خوارزم", "مدرسة"])
def test_stopword_false(w: str) -> None:
    assert not is_stopword(w)


def test_soft_stopwords_excludable() -> None:
    assert is_stopword("كان")
    assert not is_stopword("كان", soft=False)


# ------------------------------------------------------------------ التجذيع
@pytest.mark.parametrize(
    "word,expected",
    [
        ("والمدرسة", "مدرس"),
        ("الكتاب", "كتاب"),
        ("كتابهم", "كتاب"),
        ("استخدامات", "استخدام"),
        ("سماء", "سماء"),
    ],
)
def test_light_stem(word: str, expected: str) -> None:
    assert light_stem(word) == expected


def test_strict_stem_collapses_family() -> None:
    family = {"الكتابة", "كاتب", "مكتوب", "كتّاب"}
    stems = {strict_stem(w) for w in family}
    assert len(stems) <= 2  # تتقارب كلها حول «كتب»


# ------------------------------------------------------------------- الجذور
@pytest.mark.parametrize(
    "word,expected",
    [
        ("والمدرسة", "درس"),
        ("استخدام", "خدم"),
        ("قال", "قول"),
        ("مكتوب", "كتب"),
        ("تعاون", "عون"),
        ("النجوم", "نجم"),
        ("مدارس", "درس"),
        ("يستخدم", "خدم"),
        ("برمجة", "برمج"),
        ("هندسة", "هندس"),
        ("باع", "بيع"),
        ("يتعلمون", "علم"),
    ],
)
def test_extract_root(word: str, expected: str) -> None:
    assert extract_root(word) == expected


def test_root_candidates_are_ranked() -> None:
    cands = root_candidates("الكتابة")
    assert cands[0].root == "كتب"
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_short_words_do_not_crash() -> None:
    for w in ["و", "من", "لا", ""]:
        assert isinstance(extract_root(w), str)


# ------------------------------------------------------------------ المحلّل
def test_analyze_full_record() -> None:
    a = analyze("والكتابة")
    assert a.norm == "والكتابه"
    assert a.root == "كتب"
    assert a.display == "والكتابة"
    assert a.kind in {"اسم", "فعل", "حرف", "مجهول"}
    assert 0.0 <= a.confidence <= 1.0


def test_analyze_to_dict_is_serializable() -> None:
    d = analyze("مدرسة", with_alternatives=True).to_dict()
    assert isinstance(d["alternatives"], list)
    assert d["root"] == "درس"


def test_content_words_drop_stopwords() -> None:
    got = list(content_words("النجم في السماء والسماء صافية"))
    assert "في" not in got
    assert "النجم" in got


def test_content_words_roots() -> None:
    got = list(content_words("المدارس والمدرسة", use_root=True))
    assert got.count("درس") == 2


def test_analyze_summary_shape() -> None:
    s = analyze_summary("العلم نور في الأمم، والجهل ظلام في القلوب.")
    assert s["tokens"] > 5
    assert s["types"] <= s["tokens"]
    assert s["stopwords"] >= 1
    assert isinstance(s["top_roots"], list)
