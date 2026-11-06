"""تدريب المحوّل: Adam، جدولة معدّل تعلّمٍ جيبية، وقصّ التدرّج.

كل شيء يدويّ: لا مُحسِّن جاهز، ولا حلقة تدريبٍ مخفيّة. الهدف أن يكون التدريب
قابلًا للقراءة بأسطرٍ معدودة، ومُقاسًا بتقريرٍ مكتوب (خسارة، حجز، زمن).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .model import Config, Transformer
from .tokenizer import CharTokenizer


@dataclass
class TrainConfig:
    """معاملات التدريب."""

    steps: int = 1200
    batch_size: int = 12
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup: int = 80
    clip: float = 1.0
    weight_decay: float = 0.01
    eval_interval: int = 100
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _get_batch(data: np.ndarray, block: int, batch: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[i : i + block] for i in ix])
    y = np.stack([data[i + 1 : i + block + 1] for i in ix])
    return x, y


class Adam:
    """AdamW مبسّط: زخمٌ + تدرّجٌ تربيعيّ + تحلّل أوزانٍ منفصل."""

    def __init__(self, params: dict[str, np.ndarray], lr: float, *, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(
        self,
        params: dict[str, np.ndarray],
        grads: dict[str, np.ndarray],
        lr: float,
        weight_decay: float = 0.0,
    ) -> None:
        self.t += 1
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            update = lr * mhat / (np.sqrt(vhat) + self.eps)
            if weight_decay and params[k].ndim == 2:
                update = update + lr * weight_decay * params[k]
            params[k] -= update


def clip_gradients(grads: dict[str, np.ndarray], max_norm: float) -> float:
    """يقصّ التدرّج الكلّي ويعيد قاعدته قبل القصّ."""
    total = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
    if total > max_norm and total > 0:
        scale = max_norm / (total + 1e-12)
        for g in grads.values():
            g *= scale
    return total


def lr_at(step: int, cfg: TrainConfig) -> float:
    """جدولة: تسخينٌ خطّيّ ثم انحدارٌ جيبيّ إلى min_lr."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(1, cfg.warmup)
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    progress = min(1.0, progress)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def train(
    text: str,
    out_dir: Path | str,
    *,
    cfg: Config | None = None,
    tcfg: TrainConfig | None = None,
    tokenizer: CharTokenizer | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """يدرّب نموذجًا على نصّ ويحفظ الحزمة (نموذج + مقطّع + تقرير)."""
    tcfg = tcfg or TrainConfig()
    tokenizer = tokenizer or CharTokenizer.from_text(text)
    cfg = cfg or Config(vocab_size=tokenizer.vocab_size, block_size=48, n_layer=2, n_head=4, n_embd=96)
    if cfg.vocab_size != tokenizer.vocab_size:
        # المفردات تحدّدها البيانات لا الطلب: نُصحّح بدل أن نخدع
        cfg = Config(
            vocab_size=tokenizer.vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
        )

    data = np.array(tokenizer.encode(text), dtype=np.int32)
    if len(data) < cfg.block_size + 2:
        raise ValueError("النصّ أقصر من أن يُدرَّب عليه")

    split = int(len(data) * 0.95)
    train_data, val_data = data[:split], data[split:]
    if len(val_data) < cfg.block_size + 2:
        val_data = data[-cfg.block_size - 2 :]

    rng = np.random.default_rng(tcfg.seed)
    model = Transformer(cfg, seed=tcfg.seed)
    opt = Adam(model.params, tcfg.lr)

    history: list[dict] = []
    started = time.perf_counter()
    best_val = float("inf")

    for step in range(tcfg.steps):
        x, y = _get_batch(train_data, cfg.block_size, tcfg.batch_size, rng)
        _, loss, cache = model.forward(x, y)
        grads = model.backward(cache)
        gnorm = clip_gradients(grads, tcfg.clip)
        lr = lr_at(step, tcfg)
        opt.step(model.params, grads, lr, tcfg.weight_decay)

        if step % tcfg.eval_interval == 0 or step == tcfg.steps - 1:
            vx, vy = _get_batch(val_data, cfg.block_size, max(4, tcfg.batch_size // 2), rng)
            _, val_loss, _ = model.forward(vx, vy)
            best_val = min(best_val, float(val_loss))
            row = {
                "step": step,
                "loss": round(float(loss), 4),
                "val_loss": round(float(val_loss), 4),
                "lr": round(lr, 6),
                "grad_norm": round(float(gnorm), 3),
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
            history.append(row)
            if progress:
                progress(row)

    elapsed = time.perf_counter() - started
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")
    tokenizer.save(out / "tokenizer.json")
    report = {
        "config": cfg.to_dict(),
        "train": tcfg.to_dict(),
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
        "final_val_loss": float(best_val),
        "elapsed_s": round(elapsed, 2),
        "params": model.num_params(),
        "chars": len(data),
        "vocab": tokenizer.vocab_size,
        "steps_per_second": round(tcfg.steps / max(elapsed, 1e-9), 2),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
