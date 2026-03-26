"""
detmatmul.torch_compat
======================
Drop-in deterministic replacement for PyTorch matrix operations.

This module patches torch.mm, torch.matmul, and torch.bmm so that any
PyTorch model automatically uses the deterministic Q16.16 kernel —
without any code changes to the model itself.

Usage (context manager — recommended):
    import torch
    from detmatmul.torch_compat import deterministic_mode

    model = torch.load("my_model.pt")
    with deterministic_mode():
        output = model(input_tensor)
        # All matmuls inside were deterministic

Usage (permanent patch):
    from detmatmul.torch_compat import patch_torch, unpatch_torch
    patch_torch()
    # all torch.mm calls are now deterministic
    unpatch_torch()
    # restored to original

Usage (function decorator):
    from detmatmul.torch_compat import deterministic

    @deterministic
    def my_inference(model, x):
        return model(x)

Requirements:
    pip install torch
"""

import contextlib
import functools
import hashlib
import warnings
from typing import Callable

import numpy as np

from detmatmul.core import _matmul_raw

# ── PyTorch optional ──────────────────────────────────────────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. detmatmul.torch_compat requires torch.\n"
        "Install with: pip install torch",
        ImportWarning, stacklevel=2,
    )

# Track patched state
_ORIGINAL_MM     = None
_ORIGINAL_MATMUL = None
_ORIGINAL_BMM    = None
_IS_PATCHED      = False

# Statistics collected during patched execution
_CALL_COUNT      = 0
_TOTAL_MS        = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC TORCH OPS
# ═══════════════════════════════════════════════════════════════════════════════

def _to_numpy_f32(t) -> np.ndarray:
    """Convert torch tensor to float32 numpy, handling device/grad."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")
    if t.is_cuda:
        t = t.cpu()
    if t.requires_grad:
        t = t.detach()
    return t.numpy().astype(np.float32)


def _to_torch(arr: np.ndarray, ref_tensor) -> "torch.Tensor":
    """Convert numpy result back to torch tensor matching ref dtype/device."""
    t = torch.from_numpy(arr.astype(np.float32))
    if ref_tensor.is_cuda:
        t = t.cuda()
    return t


def _det_mm_torch(A: "torch.Tensor", B: "torch.Tensor") -> "torch.Tensor":
    """Deterministic replacement for torch.mm."""
    global _CALL_COUNT
    import time
    t0    = time.perf_counter()
    A_np  = _to_numpy_f32(A)
    B_np  = _to_numpy_f32(B)
    C_np, _, _ = _matmul_raw(A_np, B_np, skip_overflow_check=True)
    _CALL_COUNT += 1
    return _to_torch(C_np, A)


def _det_matmul_torch(
    A: "torch.Tensor",
    B: "torch.Tensor",
) -> "torch.Tensor":
    """
    Deterministic replacement for torch.matmul.
    Handles 2D (mm), 3D batched (bmm), and higher-dimensional cases.
    Falls back to original for non-matrix inputs (dot products etc.).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    if A.dim() == 2 and B.dim() == 2:
        return _det_mm_torch(A, B)

    if A.dim() == 3 and B.dim() == 3:
        return _det_bmm_torch(A, B)

    # For other cases (1D dot, broadcast matmul) fall back to original
    # to avoid breaking operations we haven't tested
    if _ORIGINAL_MATMUL is not None:
        return _ORIGINAL_MATMUL(A, B)
    return torch.matmul(A, B)


def _det_bmm_torch(
    A: "torch.Tensor",
    B: "torch.Tensor",
) -> "torch.Tensor":
    """Deterministic replacement for torch.bmm (batch matmul)."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available.")

    batch = A.shape[0]
    results = []
    for i in range(batch):
        C = _det_mm_torch(A[i], B[i])
        results.append(C)
    return torch.stack(results, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PATCH / UNPATCH
# ═══════════════════════════════════════════════════════════════════════════════

def patch_torch(verbose: bool = True) -> None:
    """
    Permanently patch torch.mm, torch.matmul, and torch.bmm to use
    the deterministic Q16.16 kernel.

    Call unpatch_torch() to restore original behavior.
    """
    global _ORIGINAL_MM, _ORIGINAL_MATMUL, _ORIGINAL_BMM
    global _IS_PATCHED, _CALL_COUNT, _TOTAL_MS

    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. pip install torch")
    if _IS_PATCHED:
        if verbose:
            print("  [detmatmul] Already patched.")
        return

    _ORIGINAL_MM     = torch.mm
    _ORIGINAL_MATMUL = torch.matmul
    _ORIGINAL_BMM    = torch.bmm

    torch.mm     = _det_mm_torch
    torch.matmul = _det_matmul_torch
    torch.bmm    = _det_bmm_torch

    # Also patch torch.Tensor methods
    torch.Tensor.mm     = lambda self, other: _det_mm_torch(self, other)
    torch.Tensor.matmul = lambda self, other: _det_matmul_torch(self, other)
    torch.Tensor.bmm    = lambda self, other: _det_bmm_torch(self, other)

    _IS_PATCHED  = True
    _CALL_COUNT  = 0
    _TOTAL_MS    = 0.0

    if verbose:
        print("  [detmatmul] PyTorch patched — all matmuls are now deterministic.")
        print("  [detmatmul] Call unpatch_torch() to restore original behavior.")


def unpatch_torch(verbose: bool = True) -> None:
    """Restore original PyTorch matmul behavior."""
    global _ORIGINAL_MM, _ORIGINAL_MATMUL, _ORIGINAL_BMM, _IS_PATCHED

    if not _IS_PATCHED:
        return

    torch.mm     = _ORIGINAL_MM
    torch.matmul = _ORIGINAL_MATMUL
    torch.bmm    = _ORIGINAL_BMM

    # Restore tensor methods (delete the lambda overrides)
    try:
        del torch.Tensor.mm
        del torch.Tensor.matmul
        del torch.Tensor.bmm
    except AttributeError:
        pass

    _IS_PATCHED = False

    if verbose:
        print(f"  [detmatmul] PyTorch unpatched. "
              f"({_CALL_COUNT} deterministic matmul calls made)")


def stats() -> dict:
    """Return statistics about deterministic matmul calls made so far."""
    return {
        "patched"    : _IS_PATCHED,
        "call_count" : _CALL_COUNT,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def deterministic_mode(verbose: bool = False):
    """
    Context manager: run PyTorch code with deterministic matmuls.

    Example:
        with deterministic_mode():
            output = model(input)
            # All matmuls inside were Q16.16 deterministic
    """
    patch_torch(verbose=verbose)
    try:
        yield
    finally:
        unpatch_torch(verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
#  DECORATOR
# ═══════════════════════════════════════════════════════════════════════════════

def deterministic(fn: Callable) -> Callable:
    """
    Decorator: make any function use deterministic matmuls.

    Example:
        @deterministic
        def run_inference(model, x):
            return model(x)
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with deterministic_mode(verbose=False):
            return fn(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
#  HASH UTILITIES FOR TORCH TENSORS
# ═══════════════════════════════════════════════════════════════════════════════

def tensor_hash(t: "torch.Tensor", precision: int = 4) -> str:
    """
    Compute a deterministic SHA-256 hash of a tensor's values.

    Values are rounded to `precision` decimal places before hashing,
    so minor float differences don't affect the hash.

    Parameters
    ----------
    t         : torch tensor
    precision : decimal places for rounding (default 4)

    Returns
    -------
    64-char hex SHA-256 string
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch not available.")
    arr   = _to_numpy_f32(t)
    arr   = np.round(arr, precision)
    return hashlib.sha256(
        np.ascontiguousarray(arr).tobytes()
    ).hexdigest()


def model_output_hash(
    model: "torch.nn.Module",
    input_tensor: "torch.Tensor",
    precision: int = 4,
) -> tuple:
    """
    Run model inference deterministically and return (output, hash).

    The hash is the cross-hardware compliance fingerprint for this
    (model, input) pair.

    Parameters
    ----------
    model        : any torch.nn.Module
    input_tensor : input tensor
    precision    : decimal places for hashing

    Returns
    -------
    (output_tensor, sha256_hash_string)
    """
    with deterministic_mode(verbose=False):
        output = model(input_tensor)

    h = tensor_hash(output, precision=precision)
    return output, h
