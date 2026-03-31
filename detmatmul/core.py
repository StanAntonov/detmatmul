"""
detmatmul.core
==============
The deterministic fixed-point matmul engine.

This module is the specification.  The cpu_reference() function defines
correct behaviour.  Any GPU implementation that produces a matching
SHA-256 is compliant with spec v1.0.
"""

import math
import time
import hashlib
import warnings

import numpy as np
from numba import njit, prange, int64, int32
import numba

# ── CUDA optional ─────────────────────────────────────────────────────────────
_CUDA_AVAILABLE = False
try:
    from numba import cuda as _cuda
    _dev = _cuda.get_current_device()
    _dev.name
    _CUDA_AVAILABLE = True
except Exception:
    pass

warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  SPEC CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SPEC_VERSION  = "1.0"
SCALE_FACTOR  = 65536           # Q16.16
INT64_MAX     = 9_223_372_036_854_775_807

# Turbo kernel tile parameters
_TILE_M = 64;  _TILE_N = 64;  _TILE_K = 16
_TM     = 4;   _TN     = 4
_S_I_ROWS = _TILE_M;  _S_I_COLS = _TILE_K + 1
_S_W_ROWS = _TILE_K;  _S_W_COLS = _TILE_N + 1


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU KERNEL
# ═══════════════════════════════════════════════════════════════════════════════
if _CUDA_AVAILABLE:
    @_cuda.jit
    def _gpu_kernel(In, Weights, Out, M, N, K, use_relu):
        """
        4x4 register-blocked tiled fixed-point matmul.
        int32 shared memory (values fit after scaling), int64 accumulators.
        Shared memory padded +1 to eliminate bank conflicts.
        No floating point inside the kernel.
        """
        sI = _cuda.shared.array((64, 17), int32)
        sW = _cuda.shared.array((16, 65), int32)

        tx = _cuda.threadIdx.x
        ty = _cuda.threadIdx.y
        row_base = _cuda.blockIdx.y * _TILE_M + ty * _TM
        col_base = _cuda.blockIdx.x * _TILE_N + tx * _TN

        acc = _cuda.local.array(16, int64)
        for i in range(16):
            acc[i] = int64(0)

        for k_tile in range(0, K, _TILE_K):
            for i in range(_TM):
                r = ty * _TM + i
                g_row = _cuda.blockIdx.y * _TILE_M + r
                g_col = k_tile + tx
                sI[r, tx] = int32(In[g_row, g_col]) if (g_row < M and g_col < K) else int32(0)
            for j in range(_TN):
                c = tx * _TN + j
                g_row = k_tile + ty
                g_col = _cuda.blockIdx.x * _TILE_N + c
                sW[ty, c] = int32(Weights[g_row, g_col]) if (g_row < K and g_col < N) else int32(0)

            _cuda.syncthreads()

            for k in range(_TILE_K):
                vW0 = int64(sW[k, tx * _TN    ])
                vW1 = int64(sW[k, tx * _TN + 1])
                vW2 = int64(sW[k, tx * _TN + 2])
                vW3 = int64(sW[k, tx * _TN + 3])
                vI0 = int64(sI[ty * _TM    , k])
                vI1 = int64(sI[ty * _TM + 1, k])
                vI2 = int64(sI[ty * _TM + 2, k])
                vI3 = int64(sI[ty * _TM + 3, k])
                acc[0]  += vI0 * vW0;  acc[1]  += vI0 * vW1
                acc[2]  += vI0 * vW2;  acc[3]  += vI0 * vW3
                acc[4]  += vI1 * vW0;  acc[5]  += vI1 * vW1
                acc[6]  += vI1 * vW2;  acc[7]  += vI1 * vW3
                acc[8]  += vI2 * vW0;  acc[9]  += vI2 * vW1
                acc[10] += vI2 * vW2;  acc[11] += vI2 * vW3
                acc[12] += vI3 * vW0;  acc[13] += vI3 * vW1
                acc[14] += vI3 * vW2;  acc[15] += vI3 * vW3
            _cuda.syncthreads()

        half = int64(32768)
        for i in range(_TM):
            for j in range(_TN):
                r = row_base + i;  c = col_base + j
                if r < M and c < N:
                    val = (acc[i * _TN + j] + half) >> 16
                    Out[r, c] = int64(0) if (use_relu == 1 and val < 0) else val


# ═══════════════════════════════════════════════════════════════════════════════
#  CPU REFERENCE  — the specification
# ═══════════════════════════════════════════════════════════════════════════════
@njit
def cpu_reference(In, Weights, use_relu=1):
    """
    The specification.  Row-major sequential accumulation.
    Any compliant GPU implementation must produce a matching SHA-256.
    """
    M, K = In.shape
    _, N = Weights.shape
    Out  = np.zeros((M, N), dtype=np.int64)
    half = int64(32768)
    for r in range(M):
        for c in range(N):
            acc = int64(0)
            for k in range(K):
                acc += In[r, k] * Weights[k, c]
            val = (acc + half) >> 16
            Out[r, c] = int64(0) if (use_relu == 1 and val < 0) else val
    return Out


@njit(parallel=True)
def cpu_reference_parallel(In, Weights, use_relu=1):
    """Parallel version — same result as cpu_reference, faster on multi-core."""
    M, K = In.shape
    _, N = Weights.shape
    Out  = np.zeros((M, N), dtype=np.int64)
    half = int64(32768)
    for r in prange(M):
        for c in range(N):
            acc = int64(0)
            for k in range(K):
                acc += In[r, k] * Weights[k, c]
            val = (acc + half) >> 16
            Out[r, c] = int64(0) if (use_relu == 1 and val < 0) else val
    return Out


# ═══════════════════════════════════════════════════════════════════════════════
#  OVERFLOW UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def safe_input_range(K: int) -> float:
    """
    Maximum safe absolute input value for a K-length dot product.

    Derivation: K * max_val^2 * SCALE_FACTOR^2 < INT64_MAX / 2
    Solving:    max_val = sqrt(INT64_MAX / (2 * K * SCALE_FACTOR^2))

    Examples (SCALE_FACTOR=65536):
      K=32    →  5792   K=256  →  2048
      K=1024  →  1024   K=4096 →   512
    """
    return math.sqrt(INT64_MAX / (2.0 * K * (SCALE_FACTOR ** 2)))


def check_overflow(A: np.ndarray, B: np.ndarray,
                   label_a: str = "A", label_b: str = "B",
                   verbose: bool = True) -> bool:
    """
    Check whether the int64 accumulator will overflow for this matmul.

    Parameters
    ----------
    A, B     : input matrices (float32)
    label_a  : name for A in warning messages
    label_b  : name for B in warning messages
    verbose  : if True, print detailed diagnostics

    Returns
    -------
    True if safe, False if overflow risk detected.
    """
    max_a = float(np.abs(A).max())
    max_b = float(np.abs(B).max())
    K     = A.shape[1]

    max_accum  = max_a * SCALE_FACTOR * max_b * SCALE_FACTOR * K
    safe_limit = INT64_MAX / 2.0
    safe_val   = safe_input_range(K)
    safe       = max_accum <= safe_limit

    if verbose:
        headroom = safe_limit / max(max_accum, 1.0)
        status   = "Safe" if safe else "OVERFLOW RISK"
        print(f"  [{status}] max({label_a})={max_a:.4f}  "
              f"max({label_b})={max_b:.4f}  K={K}")
        print(f"    accumulator: {max_accum:.3e}  "
              f"limit: {safe_limit:.3e}  "
              f"safe|val|<{safe_val:.2f}")
        if safe:
            print(f"    headroom: {headroom:.0f}x")
        else:
            print(f"    Results will silently wrap — "
                  f"reduce inputs to |val| < {safe_val:.2f} or reduce K.")
    return safe


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════
def matmul(
    A: np.ndarray,
    B: np.ndarray,
    use_relu: bool = False,
    force_cpu: bool = False,
    parallel_cpu: bool = False,
    skip_overflow_check: bool = False,
) -> np.ndarray:
    """
    Deterministic fixed-point matrix multiplication.

    Returns a float32 array with the same shape as a standard A @ B,
    but computed via Q16.16 integer arithmetic so the result is
    bit-exact across any compliant hardware.

    Parameters
    ----------
    A, B               : float32 input matrices
    use_relu           : apply ReLU to output (default False)
    force_cpu          : skip GPU even if available
    parallel_cpu       : use multi-threaded CPU path (still deterministic)
    skip_overflow_check: disable pre-flight overflow check (faster in loops)

    Returns
    -------
    C : np.ndarray (float32) — deterministic result
    """
    C, _, _ = _matmul_raw(A, B, use_relu, force_cpu, parallel_cpu,
                           skip_overflow_check)
    return C


def spec_hash(
    A: np.ndarray,
    B: np.ndarray,
    use_relu: bool = False,
    force_cpu: bool = False,
) -> str:
    """
    Compute the SHA-256 fingerprint of the deterministic matmul result.

    This hash is guaranteed to be identical on any compliant hardware.
    Use it to verify reproducibility across machines.

    Returns
    -------
    64-character hex SHA-256 string
    """
    _, _, C_raw = _matmul_raw(A, B, use_relu, force_cpu,
                               skip_overflow_check=True)
    return hashlib.sha256(np.ascontiguousarray(C_raw).tobytes()).hexdigest()


def verify_hash(
    A: np.ndarray,
    B: np.ndarray,
    expected_hash: str,
    use_relu: bool = False,
    force_cpu: bool = False,
) -> bool:
    """
    Verify this machine produces the expected hash for A @ B.

    Returns True if compliant, False if there is a mismatch.
    """
    h = spec_hash(A, B, use_relu=use_relu, force_cpu=force_cpu)
    return h == expected_hash


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNAL
# ═══════════════════════════════════════════════════════════════════════════════
def _matmul_raw(A, B, use_relu=False, force_cpu=False, parallel_cpu=False,
                skip_overflow_check=False):
    """Returns (float32_C, elapsed_ms, int64_C_raw)."""
    M, K = A.shape;  K2, N = B.shape
    if K != K2:
        raise ValueError(f"Dimension mismatch: {A.shape} × {B.shape}")
    if not skip_overflow_check:
        check_overflow(A, B, verbose=False)

    A_f = (A * SCALE_FACTOR).astype(np.int64)
    B_f = (B * SCALE_FACTOR).astype(np.int64)
    relu_flag = 1 if use_relu else 0

    if _CUDA_AVAILABLE and not force_cpu:
        d_A = _cuda.to_device(A_f);  d_B = _cuda.to_device(B_f)
        d_C = _cuda.device_array((M, N), dtype=np.int64)
        grid  = (math.ceil(N / _TILE_N), math.ceil(M / _TILE_M))
        block = (16, 16)
        e0 = _cuda.event();  e1 = _cuda.event()
        e0.record()
        _gpu_kernel[grid, block](d_A, d_B, d_C, M, N, K, relu_flag)
        e1.record();  e1.synchronize()
        elapsed = _cuda.event_elapsed_time(e0, e1)
        C_raw   = d_C.copy_to_host()
    else:
        t0 = time.perf_counter()
        if parallel_cpu:
            C_raw = cpu_reference_parallel(A_f, B_f, use_relu=relu_flag)
        else:
            C_raw = cpu_reference(A_f, B_f, use_relu=relu_flag)
        elapsed = (time.perf_counter() - t0) * 1000

    C = C_raw.astype(np.float32) / SCALE_FACTOR
    return C, elapsed, C_raw
