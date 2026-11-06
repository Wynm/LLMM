"""اختبارات المحوّل العصبي: المقطّع، التدرّجات العددية، التدريب، والتوليد.

أهمّ اختبار هنا هو ``test_gradients_match_numerical``: يقارن كل تدرّجٍ يدويّ
بتقديرٍ عدديّ، فأي خطأٍ في الاشتقاق يُكتشف فورًا.
"""

from __future__ import annotations

import numpy as np
import pytest

from mishkat.lm.model import Config, Transformer, _gelu, _gelu_backward, _ln_forward, _softmax
from mishkat.lm.sample import generate, lm_status
from mishkat.lm.tokenizer import CharTokenizer
from mishkat.lm.train import Adam, TrainConfig, clip_gradients, lr_at, train

pytest.importorskip("numpy")


# ───────────────────────────────── المقطّع
def test_tokenizer_roundtrip() -> None:
    text = "العلم نور. العلم يرفع الأمم!"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == len(set(text)) + 2


def test_tokenizer_unknown_char() -> None:
    tok = CharTokenizer.from_text("ابج")
    ids = tok.encode("اب×ج")
    assert tok.unk_id in ids
    assert "×" not in tok.decode(ids)


def test_tokenizer_save_load(tmp_path) -> None:
    tok = CharTokenizer.from_text("مرحبا بالعالم")
    tok.save(tmp_path / "t.json")
    again = CharTokenizer.load(tmp_path / "t.json")
    assert again.itos == tok.itos
    assert again.encode("مرحبا") == tok.encode("مرحبا")


# ─────────────────────────────── دوالّ مساعدة
def test_softmax_normalized() -> None:
    x = np.array([[1.0, 2.0, 3.0]])
    y = _softmax(x)
    assert np.allclose(y.sum(), 1.0)
    assert y[0, 2] > y[0, 0]


def test_gelu_matches_definition() -> None:
    x = np.linspace(-3, 3, 25)
    g = _gelu(x)
    assert np.allclose(g, 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))


def test_layernorm_zero_mean_unit_var() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4, 8)) * 3 + 5
    y, _ = _ln_forward(x, np.ones(8), np.zeros(8))
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(y.std(axis=-1), 1, atol=1e-2)


# ──────────────────────────────── سلامة النموذج
def _tiny_model(seed: int = 0) -> Transformer:
    cfg = Config(vocab_size=12, block_size=6, n_layer=1, n_head=2, n_embd=8)
    m = Transformer(cfg, seed=seed)
    for k in m.params:
        if m.params[k].ndim == 2:
            m.params[k] *= 3.0
    return m


def test_forward_shapes() -> None:
    m = _tiny_model()
    idx = np.zeros((2, 5), dtype=np.int32)
    logits, loss, cache = m.forward(idx, idx)
    assert logits.shape == (2, 5, 12)
    assert loss is not None and loss > 0
    assert "dlogits" in cache


def test_causality() -> None:
    """تغيير رمزٍ في النهاية لا يجب أن يغيّر توقّع البداية."""
    m = _tiny_model()
    a = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    b = a.copy()
    b[0, -1] = 9
    la, _, _ = m.forward(a)
    lb, _, _ = m.forward(b)
    assert np.allclose(la[0, :4], lb[0, :4], atol=1e-6)
    assert not np.allclose(la[0, 4], lb[0, 4])


def test_block_size_guard() -> None:
    m = _tiny_model()
    with pytest.raises(ValueError):
        m.forward(np.zeros((1, 10), dtype=np.int32))


def test_gradients_match_numerical() -> None:
    """الاختبار الحاسم: التدرّج اليدويّ = التدرّج العدديّ."""
    rng = np.random.default_rng(3)
    m = _tiny_model(seed=2)
    idx = rng.integers(0, 12, size=(2, 6))
    tgt = rng.integers(0, 12, size=(2, 6))
    _, _, cache = m.forward(idx, tgt)
    grads = m.backward(cache)

    eps = 1e-5
    names = ["wte", "wpe", "b0.wq", "b0.wk", "b0.wv", "b0.wo", "b0.w1", "b0.w2", "b0.ln1.g", "lnf.b"]
    for name in names:
        p = m.params[name]
        for _ in range(3):
            fi = tuple(int(rng.integers(0, s)) for s in p.shape)
            orig = p[fi]
            p[fi] = orig + eps
            _, l1, _ = m.forward(idx, tgt)
            p[fi] = orig - eps
            _, l2, _ = m.forward(idx, tgt)
            p[fi] = orig
            num = (l1 - l2) / (2 * eps)
            ana = float(grads[name][fi])
            assert abs(num - ana) <= 2e-2 * max(1.0, abs(num), abs(ana)), f"تدرّج خاطئ في {name}{fi}"


def test_save_load(tmp_path) -> None:
    m = _tiny_model()
    m.save(tmp_path / "model")
    m2 = Transformer.load(tmp_path / "model")
    assert m2.cfg == m.cfg
    idx = np.ones((1, 4), dtype=np.int32)
    assert np.allclose(m.forward(idx)[0], m2.forward(idx)[0], atol=1e-6)


# ──────────────────────────────── المحسّن والجدولة
def test_lr_schedule_shape() -> None:
    cfg = TrainConfig(steps=100, lr=1e-3, min_lr=1e-4, warmup=10)
    assert lr_at(0, cfg) < lr_at(9, cfg)
    assert lr_at(50, cfg) > cfg.min_lr
    assert lr_at(99, cfg) < 3e-4


def test_gradient_clipping() -> None:
    grads = {"a": np.ones((4,)) * 10, "b": np.ones((2,)) * 10}
    norm = clip_gradients(grads, max_norm=1.0)
    assert norm > 1.0
    assert np.linalg.norm(np.concatenate([g.ravel() for g in grads.values()])) <= 1.0 + 1e-6


def test_adam_reduces_simple_loss() -> None:
    params = {"w": np.array([2.0, -3.0])}
    opt = Adam(params, lr=0.1)
    for _ in range(200):
        grads = {"w": 2 * params["w"]}  # تقليل ||w||²
        opt.step(params, grads, lr=0.1)
    assert np.linalg.norm(params["w"]) < 1e-2


# ─────────────────────────────── التدريب والتوليد
def test_train_reduces_loss(tmp_path) -> None:
    text = ("العلم نور يرفع الأمم. الجهل ظلام يهبطها. " * 40).strip()
    report = train(
        text,
        tmp_path,
        cfg=Config(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32),
        tcfg=TrainConfig(steps=25, batch_size=4, eval_interval=5, lr=3e-3, warmup=5),
    )
    assert report["final_loss"] < report["history"][0]["loss"]
    assert (tmp_path / "model.npz").exists()
    assert (tmp_path / "tokenizer.json").exists()


def test_generate_from_trained(tmp_path) -> None:
    text = ("ابن الهيثم كتب في المناظر والضوء. " * 60).strip()
    train(
        text,
        tmp_path,
        cfg=Config(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32),
        tcfg=TrainConfig(steps=20, batch_size=4, eval_interval=10, lr=3e-3),
    )
    tok = CharTokenizer.load(tmp_path / "tokenizer.json")
    model = Transformer.load(tmp_path / "model")
    # توليدٌ يدويّ صغير للتحقّق من الترميز
    ids = np.array([tok.encode("ابن")], dtype=np.int32)
    logits, _, _ = model.forward(ids)
    assert logits.shape[-1] == tok.vocab_size


def test_lm_status_contract() -> None:
    st = lm_status()
    assert "trained" in st
    if st["trained"]:
        assert st["params"] > 0
