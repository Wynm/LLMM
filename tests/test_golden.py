"""اختبارٌ ذهبيّ: دقّة استخراج الجذر على مجموعةٍ يدويّة مُعنونة.

المجموعة في ``data/eval/roots.jsonl``، وهي مرجعٌ خارجيّ عن الشيفرة: أي تعديلٍ
يُخفّض الدقّة يظهر هنا فورًا. الحدّ الأدنى المطلوب 0.90 (الهدف في خارطة الطريق).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mishkat.text import extract_root

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "eval" / "roots.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_set_exists() -> None:
    rows = _rows()
    assert len(rows) >= 50
    assert all("word" in r and "root" in r for r in rows)


def test_root_accuracy_above_threshold() -> None:
    rows = _rows()
    correct = sum(1 for r in rows if extract_root(r["word"]) == r["root"])
    accuracy = correct / len(rows)
    assert accuracy >= 0.90, f"دقّة الجذور {accuracy:.3f} أقل من الحدّ الأدنى"


@pytest.mark.parametrize("row", _rows())
def test_root_individual(row: dict) -> None:
    """الجذر المتوقّع، أو أحد البدائل المقبولة لغويًّا (يُصرَّح به في الملف)."""
    accepted = {row["root"], *row.get("alt", [])}
    assert extract_root(row["word"]) in accepted
