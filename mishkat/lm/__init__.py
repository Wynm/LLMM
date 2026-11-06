"""المحوّل العصبي الحرفي — Transformer صغير مكتوب بـ NumPy من الصفر.

لا PyTorch، لا TensorFlow، لا autograd: كل تدرّجٍ مشتقٌّ يدويًّا ومُختبَر
عدديًّا (انظر ``tests/test_lm.py``).
"""

from .tokenizer import CharTokenizer
from .model import Config, Transformer
from .train import TrainConfig, train
from .sample import generate_text, lm_status, load_model

__all__ = [
    "CharTokenizer",
    "Config",
    "Transformer",
    "TrainConfig",
    "train",
    "generate_text",
    "lm_status",
    "load_model",
]
