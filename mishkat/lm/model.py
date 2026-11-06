"""محوّلٌ عصبيّ حرفيّ (decoder-only) مُنفَّذٌ بـ NumPy، بتدرّجاتٍ يدويّة.

المعمار
-------
::

    idx ──► Emb(token) + Emb(position)
          ──► [ LayerNorm → Self-Attention(4 رؤوس, سببي) → + ] × 2
          ──► LayerNorm ──► Linear(مشترك مع Emb) ──► logits

كل تدرّجٍ في هذا الملف مشتقٌّ بالورقة والقلم ثم مُتحقَّقٌ منه عدديًّا
(``tests/test_lm.py::test_gradients_match_numerical``). لا autograd هنا: إن
أخطأنا في اشتقاق، يظهر الخطأ فورًا في الاختبار.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EPS = 1e-5


@dataclass(frozen=True)
class Config:
    """معاملات النموذج."""

    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 96

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd يجب أن يقبل القسمة على n_head")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────── دوالّ مساعدة ───────────────────────────────
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _softmax_backward(dy: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y * (dy - (dy * y).sum(axis=-1, keepdims=True))


def _ln_forward(x: np.ndarray, g: np.ndarray, b: np.ndarray):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + EPS)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv, g)


def _ln_backward(dy: np.ndarray, cache, n_shape: tuple):
    xh, inv, g = cache
    dg = (dy * xh).reshape(-1, xh.shape[-1]).sum(axis=0)
    db = dy.reshape(-1, dy.shape[-1]).sum(axis=0)
    dxh = dy * g
    dx = inv * (
        dxh
        - dxh.mean(axis=-1, keepdims=True)
        - xh * (dxh * xh).mean(axis=-1, keepdims=True)
    )
    return dx, dg, db


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU بتقريب الظلّ الزائدي (كما في GPT-2)."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def _gelu_backward(x: np.ndarray) -> np.ndarray:
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x**3)
    t = np.tanh(u)
    du = c * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * du


class Transformer:
    """نموذجٌ لغويّ عصبيّ صغير يعمل على NumPy فقط."""

    def __init__(self, cfg: Config, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.params = self._init_params(rng)

    # ─────────────────────────────── التهيئة ───────────────────────────────
    def _init_params(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        c = self.cfg
        d = c.n_embd
        p: dict[str, np.ndarray] = {
            "wte": rng.normal(0, 0.02, size=(c.vocab_size, d)),
            "wpe": rng.normal(0, 0.02, size=(c.block_size, d)),
        }
        for i in range(c.n_layer):
            p[f"b{i}.ln1.g"] = np.ones(d)
            p[f"b{i}.ln1.b"] = np.zeros(d)
            p[f"b{i}.wq"] = rng.normal(0, 0.02, size=(d, d))
            p[f"b{i}.bq"] = np.zeros(d)
            p[f"b{i}.wk"] = rng.normal(0, 0.02, size=(d, d))
            p[f"b{i}.bk"] = np.zeros(d)
            p[f"b{i}.wv"] = rng.normal(0, 0.02, size=(d, d))
            p[f"b{i}.bv"] = np.zeros(d)
            p[f"b{i}.wo"] = rng.normal(0, 0.02, size=(d, d))
            p[f"b{i}.bo"] = np.zeros(d)
            p[f"b{i}.ln2.g"] = np.ones(d)
            p[f"b{i}.ln2.b"] = np.zeros(d)
            p[f"b{i}.w1"] = rng.normal(0, 0.02, size=(d, 4 * d))
            p[f"b{i}.b1"] = np.zeros(4 * d)
            p[f"b{i}.w2"] = rng.normal(0, 0.02, size=(4 * d, d))
            p[f"b{i}.b2"] = np.zeros(d)
        p["lnf.g"] = np.ones(d)
        p["lnf.b"] = np.zeros(d)
        return p

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ──────────────────────────────── الأمام ────────────────────────────────
    def forward(self, idx: np.ndarray, targets: np.ndarray | None = None):
        """تمريرٌ أماميّ كامل. يعيد (logits, loss, cache)."""
        c = self.cfg
        B, T = idx.shape
        if T > c.block_size:
            raise ValueError(f"طول المتتالية {T} يتجاوز block_size={c.block_size}")
        p = self.params
        H, Dh = c.n_head, c.head_dim

        tok = p["wte"][idx]                     # (B,T,d)
        pos = p["wpe"][:T][None, :, :]          # (1,T,d)
        x = tok + pos

        cache: dict = {"idx": idx, "T": T, "B": B, "blocks": [], "x0": x}
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)

        for i in range(c.n_layer):
            # ---- LayerNorm 1 ----
            h, ln1_cache = _ln_forward(x, p[f"b{i}.ln1.g"], p[f"b{i}.ln1.b"])

            # ---- الانتباه متعدّد الرؤوس ----
            q = h @ p[f"b{i}.wq"] + p[f"b{i}.bq"]
            k = h @ p[f"b{i}.wk"] + p[f"b{i}.bk"]
            v = h @ p[f"b{i}.wv"] + p[f"b{i}.bv"]

            qh = q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
            kh = k.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
            vh = v.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

            scale = 1.0 / math.sqrt(Dh)
            scores = (qh @ kh.transpose(0, 1, 3, 2)) * scale
            scores = np.where(mask[None, None, :, :], -1e9, scores)
            attn = _softmax(scores, axis=-1)
            ctx = attn @ vh                               # (B,H,T,Dh)
            ctx_t = ctx.transpose(0, 2, 1, 3).reshape(B, T, c.n_embd)
            attn_out = ctx_t @ p[f"b{i}.wo"] + p[f"b{i}.bo"]

            x = x + attn_out

            # ---- LayerNorm 2 + شبكة أمامية ----
            h2, ln2_cache = _ln_forward(x, p[f"b{i}.ln2.g"], p[f"b{i}.ln2.b"])
            pre = h2 @ p[f"b{i}.w1"] + p[f"b{i}.b1"]
            act = _gelu(pre)
            ffn_out = act @ p[f"b{i}.w2"] + p[f"b{i}.b2"]
            x = x + ffn_out

            cache["blocks"].append(
                {
                    "ln1": ln1_cache, "h1": h, "q": q, "k": k, "v": v,
                    "qh": qh, "kh": kh, "vh": vh, "attn": attn, "ctx_t": ctx_t,
                    "ln2": ln2_cache, "h2": h2, "pre": pre, "act": act,
                    "x_in": cache["x0"] if i == 0 else cache["blocks"][-1]["x_out"],
                }
            )
            cache["blocks"][-1]["x_out"] = x
            cache["x_in_next"] = x

        # ---- طبقةٌ نهائية ----
        xf, lnf_cache = _ln_forward(x, p["lnf.g"], p["lnf.b"])
        logits = xf @ p["wte"].T                     # ربط الأوزان (weight tying)
        cache.update({"xf": xf, "lnf": lnf_cache, "logits": logits, "x_final": x})

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(-1, c.vocab_size)
            tgt = targets.reshape(-1)
            m = logits_flat.max(axis=1, keepdims=True)
            z = logits_flat - m
            logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
            logp = z - logsumexp
            n = tgt.shape[0]
            loss = float(-logp[np.arange(n), tgt].mean())
            probs = np.exp(logp)
            dlogits = probs
            dlogits[np.arange(n), tgt] -= 1.0
            dlogits /= n
            cache["dlogits"] = dlogits.reshape(B, T, c.vocab_size)
        return logits, loss, cache

    # ─────────────────────────────── الخلف ───────────────────────────────
    def backward(self, cache: dict) -> dict[str, np.ndarray]:
        """الانتشار الخلفيّ اليدويّ: يعيد تدرّج كل وسيط."""
        c = self.cfg
        p = self.params
        B, T = cache["B"], cache["T"]
        H, Dh = c.n_head, c.head_dim
        grads: dict[str, np.ndarray] = {}

        dlogits = cache["dlogits"]                      # (B,T,V)
        xf, lnf_cache = cache["xf"], cache["lnf"]
        grads["wte"] = dlogits.reshape(-1, c.vocab_size).T @ xf.reshape(-1, c.n_embd)
        dxf = dlogits @ p["wte"]
        dx, grads["lnf.g"], grads["lnf.b"] = _ln_backward(dxf, lnf_cache, xf.shape)

        for i in reversed(range(c.n_layer)):
            blk = cache["blocks"][i]
            x_in = cache["x0"] if i == 0 else cache["blocks"][i - 1]["x_out"]

            # ---- مسار الشبكة الأمامية ----
            # x_out = x_mid + act @ w2 + b2
            grads[f"b{i}.w2"] = blk["act"].reshape(-1, 4 * c.n_embd).T @ dx.reshape(-1, c.n_embd)
            grads[f"b{i}.b2"] = dx.reshape(-1, c.n_embd).sum(axis=0)
            dact = dx @ p[f"b{i}.w2"].T
            dpre = dact * _gelu_backward(blk["pre"])
            grads[f"b{i}.w1"] = blk["h2"].reshape(-1, c.n_embd).T @ dpre.reshape(-1, 4 * c.n_embd)
            grads[f"b{i}.b1"] = dpre.reshape(-1, 4 * c.n_embd).sum(axis=0)
            dh2 = dpre @ p[f"b{i}.w1"].T
            dln2, grads[f"b{i}.ln2.g"], grads[f"b{i}.ln2.b"] = _ln_backward(
                dh2, blk["ln2"], blk["h2"].shape
            )
            dx = dx + dln2      # المتبقّي الأول: x_mid

            # ---- مسار الانتباه ----
            grads[f"b{i}.wo"] = blk["ctx_t"].reshape(-1, c.n_embd).T @ dx.reshape(-1, c.n_embd)
            grads[f"b{i}.bo"] = dx.reshape(-1, c.n_embd).sum(axis=0)
            dctx = dx @ p[f"b{i}.wo"].T
            dctx_h = dctx.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)     # (B,H,T,Dh)

            attn = blk["attn"]
            dvh = attn.transpose(0, 1, 3, 2) @ dctx_h
            dattn = dctx_h @ blk["vh"].transpose(0, 1, 3, 2)
            dscores = _softmax_backward(dattn, attn) * (1.0 / math.sqrt(Dh))
            dqh = dscores @ blk["kh"]
            dkh = dscores.transpose(0, 1, 3, 2) @ blk["qh"]

            dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, c.n_embd)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, c.n_embd)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, c.n_embd)

            h1 = blk["h1"].reshape(-1, c.n_embd)
            grads[f"b{i}.wq"] = h1.T @ dq.reshape(-1, c.n_embd)
            grads[f"b{i}.wk"] = h1.T @ dk.reshape(-1, c.n_embd)
            grads[f"b{i}.wv"] = h1.T @ dv.reshape(-1, c.n_embd)
            grads[f"b{i}.bq"] = dq.reshape(-1, c.n_embd).sum(axis=0)
            grads[f"b{i}.bk"] = dk.reshape(-1, c.n_embd).sum(axis=0)
            grads[f"b{i}.bv"] = dv.reshape(-1, c.n_embd).sum(axis=0)

            dh1 = (
                dq.reshape(-1, c.n_embd) @ p[f"b{i}.wq"].T
                + dk.reshape(-1, c.n_embd) @ p[f"b{i}.wk"].T
                + dv.reshape(-1, c.n_embd) @ p[f"b{i}.wv"].T
            ).reshape(B, T, c.n_embd)
            dln1, grads[f"b{i}.ln1.g"], grads[f"b{i}.ln1.b"] = _ln_backward(
                dh1, blk["ln1"], blk["h1"].shape
            )
            dx = dx + dln1      # المتبقّي الثاني: x_in

        # تدرّج التضمين
        dx0 = dx
        dpos = dx0.sum(axis=0)                         # (T,d)
        grads["wpe"] = np.zeros_like(p["wpe"])
        grads["wpe"][:T] = dpos
        dte = np.zeros_like(p["wte"])
        np.add.at(dte, cache["idx"].reshape(-1), dx0.reshape(-1, c.n_embd))
        grads["wte"] = grads["wte"] + dte
        return grads

    # ─────────────────────────────── الحفظ ───────────────────────────────
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), **self.params)
        path.with_suffix(".json").write_text(json.dumps(self.cfg.to_dict(), ensure_ascii=False))

    @classmethod
    def load(cls, path: Path | str) -> "Transformer":
        path = Path(path)
        cfg = Config(**json.loads(path.with_suffix(".json").read_text()))
        model = cls(cfg)
        data = np.load(path.with_suffix(".npz"))
        model.params = {k: data[k] for k in data.files}
        return model

    def summary(self) -> dict:
        return {
            "params": self.num_params(),
            "config": self.cfg.to_dict(),
            "layers": self.cfg.n_layer,
            "heads": self.cfg.n_head,
            "context": self.cfg.block_size,
        }
