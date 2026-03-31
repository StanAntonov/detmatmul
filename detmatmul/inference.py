"""
detmatmul.inference
===================
Lightweight deterministic transformer for API demo mode.

Used as a fallback when the full GPT-2 model cannot be loaded
(e.g. transformers / tiktoken not installed).

This is a small randomly-initialised transformer that demonstrates
the deterministic inference pipeline without requiring any downloads.
It is NOT a pretrained model — outputs are meaningless — but every
SHA-256 hash it produces is bit-exact across any hardware.
"""

import hashlib
import math

import numpy as np
from numba import njit

from detmatmul.core import _matmul_raw, SCALE_FACTOR


# ── Minimal deterministic ops ─────────────────────────────────────────────────

def _det_mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C, _, _ = _matmul_raw(
        A.astype(np.float32),
        B.astype(np.float32),
        use_relu=False,
        skip_overflow_check=True,
    )
    return C


@njit
def _layer_norm(x, gamma, beta, eps=1e-5):
    n    = len(x)
    mean = np.float32(0.0)
    for i in range(n):
        mean += x[i]
    mean /= np.float32(n)
    var = np.float32(0.0)
    for i in range(n):
        d    = x[i] - mean
        var += d * d
    var /= np.float32(n)
    std  = np.float32(math.sqrt(float(var) + eps))
    out  = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = gamma[i] * ((x[i] - mean) / std) + beta[i]
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  TINY DETERMINISTIC TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════

class DeterministicTransformer:
    """
    A tiny randomly-initialised transformer for API demo/fallback mode.

    Not a pretrained model. Used only to demonstrate that the API's
    /generate endpoint returns a valid SHA-256 hash even when the full
    GPT-2 weights are unavailable.

    Configuration: 2 layers, d_model=64, 2 heads, vocab=256 (byte-level).
    """

    N_LAYER  = 2
    D_MODEL  = 64
    N_HEAD   = 2
    D_HEAD   = 32
    VOCAB    = 256     # byte-level: maps each UTF-8 byte to a token

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)

        def _w(*shape):
            return (rng.standard_normal(shape) * 0.02).astype(np.float32)

        self.tok_emb = _w(self.VOCAB, self.D_MODEL)
        self.pos_emb = _w(1024, self.D_MODEL)
        self.layers  = []
        for _ in range(self.N_LAYER):
            self.layers.append({
                "ln1_g": np.ones(self.D_MODEL,  np.float32),
                "ln1_b": np.zeros(self.D_MODEL, np.float32),
                "ln2_g": np.ones(self.D_MODEL,  np.float32),
                "ln2_b": np.zeros(self.D_MODEL, np.float32),
                "qkv_w": _w(self.D_MODEL, 3 * self.D_MODEL),
                "o_w"  : _w(self.D_MODEL, self.D_MODEL),
                "ff_w1": _w(self.D_MODEL, 4 * self.D_MODEL),
                "ff_w2": _w(4 * self.D_MODEL, self.D_MODEL),
            })
        self.ln_f_g = np.ones(self.D_MODEL,  np.float32)
        self.ln_f_b = np.zeros(self.D_MODEL, np.float32)

    # ── tokenization ──────────────────────────────────────────────────────────

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list) -> str:
        return bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, token_ids: list) -> np.ndarray:
        T   = min(len(token_ids), 512)
        ids = np.array(token_ids[-T:], dtype=np.int32) % self.VOCAB
        x   = self.tok_emb[ids] + self.pos_emb[:T]

        for layer in self.layers:
            # Layer norm + attention (simplified: no masking for demo)
            xn = np.stack([
                _layer_norm(x[i], layer["ln1_g"], layer["ln1_b"])
                for i in range(T)
            ])
            qkv  = _det_mm(xn, layer["qkv_w"])
            attn = _det_mm(qkv[:, :self.D_MODEL], layer["o_w"])
            x    = x + attn

            # Layer norm + FFN
            xn  = np.stack([
                _layer_norm(x[i], layer["ln2_g"], layer["ln2_b"])
                for i in range(T)
            ])
            ff  = np.maximum(0, _det_mm(xn, layer["ff_w1"]))
            x   = x + _det_mm(ff, layer["ff_w2"])

        xn = np.stack([
            _layer_norm(x[i], self.ln_f_g, self.ln_f_b) for i in range(T)
        ])
        return _det_mm(xn, self.tok_emb.T)   # (T, VOCAB)

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float  = 0.8,
        top_k: int          = 40,
        seed: int           = 0,
    ) -> tuple:
        """
        Returns (generated_text, sha256_hash).

        The hash is bit-exact across any hardware running this spec.
        Note: outputs are not meaningful — this is a demo model only.
        """
        token_ids   = self.encode(prompt)
        prompt_seed = int(hashlib.sha256(
            (prompt + str(seed)).encode()
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(prompt_seed)

        for _ in range(max_new_tokens):
            logits    = self.forward(token_ids)
            next_logits = logits[-1].astype(np.float64) / max(temperature, 1e-6)
            if top_k > 0:
                kth = np.partition(next_logits, -top_k)[-top_k]
                next_logits[next_logits < kth] = -1e10
            next_logits -= next_logits.max()
            probs        = np.exp(next_logits)
            probs       /= probs.sum()
            next_token   = int(rng.choice(len(probs), p=probs))
            token_ids.append(next_token)

        text = self.decode(token_ids)
        h    = hashlib.sha256(
            np.array(token_ids, dtype=np.int32).tobytes()
        ).hexdigest()
        return text, h
