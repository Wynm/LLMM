"""مُقطِّعٌ حرفيّ للنصّ العربي.

لماذا حرفيّ لا كلميّ؟ لأن العربية صرفيّةٌ غنيّة: كل كلمةٍ قد تكون صيغةً جديدة لم
يراها النموذج. الحرف وحدةٌ ثابتة العدد، فيتعلّم النموذج *كيف تُبنى* الكلمة لا
كيف تُكرَّر. وحجم المفردات يصغر من عشرات الآلاف إلى عشرات الحروف.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


class CharTokenizer:
    """يحوّل نصًّا إلى أرقامٍ حرفية والعكس."""

    def __init__(self, chars: Sequence[str] | None = None) -> None:
        self.pad_token = "\u0000"
        self.unk_token = "\u0001"
        if chars is None:
            chars = []
        alphabet = list(dict.fromkeys([self.pad_token, self.unk_token] + list(chars)))
        self.itos: list[str] = alphabet
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(alphabet)}

    # ------------------------------------------------------------------ بناء
    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    # --------------------------------------------------------------- تحويل
    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            ch = self.itos[int(i)]
            if skip_special and ch in (self.pad_token, self.unk_token):
                continue
            out.append(ch)
        return "".join(out)

    # --------------------------------------------------------------- الحفظ
    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.itos = list(data["itos"])
        tok.stoi = {c: i for i, c in enumerate(tok.itos)}
        return tok

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover
        return f"CharTokenizer(vocab={self.vocab_size})"
