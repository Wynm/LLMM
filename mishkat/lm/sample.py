"""توليد النصّ من المحوّل: temperature وtop-k، مع حالةٍ جاهزة للواجهة."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .model import Transformer
from .tokenizer import CharTokenizer

MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "lm"


def _require_model() -> tuple[Transformer, CharTokenizer]:
    if not (MODEL_DIR / "model.npz").exists():
        raise FileNotFoundError(
            "لا يوجد نموذجٌ مدرَّب. شغّل: make train  (أو python -m mishkat.cli train)"
        )
    return load_model(MODEL_DIR)


@lru_cache(maxsize=1)
def load_model(directory: Path | str = MODEL_DIR) -> tuple[Transformer, CharTokenizer]:
    """يحمّل النموذج والمقطّع (مع تخزينٍ مؤقّت)."""
    d = Path(directory)
    model = Transformer.load(d / "model")
    tok = CharTokenizer.load(d / "tokenizer.json")
    return model, tok


def generate(
    prompt: str,
    *,
    length: int = 160,
    temperature: float = 0.9,
    top_k: int = 12,
    seed: int | None = None,
) -> str:
    """يولّد نصًّا يبدأ من ``prompt``."""
    model, tok = _require_model()
    cfg = model.cfg
    rng = np.random.default_rng(seed)

    ids = tok.encode(prompt) if prompt else [tok.encode(" ")[0]]
    ids = ids[-cfg.block_size :]
    out = list(ids)

    for _ in range(max(0, length)):
        x = np.array([out[-cfg.block_size :]], dtype=np.int32)
        logits, _, _ = model.forward(x)
        last = logits[0, -1, :].astype(np.float64) / max(1e-6, temperature)
        if top_k and top_k < last.shape[0]:
            kth = np.sort(last)[-top_k]
            last = np.where(last < kth, -np.inf, last)
        z = last - last.max()
        probs = np.exp(z)
        probs /= probs.sum()
        nxt = int(rng.choice(len(probs), p=probs))
        out.append(nxt)
        if len(out) > cfg.block_size * 8:
            break
    return tok.decode(out)


def lm_status() -> dict:
    """حالةٌ يستهلكها الاستوديو."""
    report_path = MODEL_DIR / "report.json"
    if not (MODEL_DIR / "model.npz").exists():
        return {
            "trained": False,
            "hint": "شغّل: python -m mishkat.cli train --steps 1200",
        }
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    model, tok = load_model(MODEL_DIR)
    return {
        "trained": True,
        "params": model.num_params(),
        "vocab": tok.vocab_size,
        "context": model.cfg.block_size,
        "layers": model.cfg.n_layer,
        "heads": model.cfg.n_head,
        "final_loss": report.get("final_loss"),
        "val_loss": report.get("final_val_loss"),
        "elapsed_s": report.get("elapsed_s"),
        "steps": report.get("train", {}).get("steps"),
        "chars": report.get("chars"),
        "history": report.get("history", [])[-8:],
    }


def generate_text(
    prompt: str,
    *,
    length: int = 160,
    temperature: float = 0.9,
    top_k: int = 12,
) -> dict:
    """واجهة API: توليد + بيانات الحالة."""
    text = generate(prompt, length=length, temperature=temperature, top_k=top_k)
    return {
        "prompt": prompt,
        "text": text,
        "length": len(text),
        "params": {"temperature": temperature, "top_k": top_k},
    }
