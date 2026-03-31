"""
detmatmul — Deterministic Cross-Hardware Matrix Multiplication
==============================================================

Core guarantee: identical SHA-256 hashes for identical inputs,
regardless of GPU brand, generation, or operating system.

Achieved via Q16.16 fixed-point integer arithmetic (int64).
Integer ops are fully associative — no hardware-dependent rounding.

Basic usage
-----------
    import numpy as np
    from detmatmul import matmul, verify_hash, spec_hash

    A = np.random.randn(512, 512).astype(np.float32)
    B = np.random.randn(512, 512).astype(np.float32)

    # Deterministic matmul — same result on any hardware
    C = matmul(A, B)

    # Get the SHA-256 fingerprint of the raw integer result
    h = spec_hash(A, B)
    print(h)   # e.g. "bc071e9c..."

    # Verify this machine matches a known-good hash
    ok = verify_hash(A, B, expected_hash=h)
    assert ok

Spec
----
    SPEC_VERSION : str   — "1.0"
    SCALE_FACTOR : int   — 65536  (Q16.16)

    The specification is the cpu_reference() function in detmatmul.core.
    Any compliant implementation must produce a SHA-256 that matches it.
"""

from detmatmul._version import __version__
from detmatmul.core import (
    SPEC_VERSION,
    SCALE_FACTOR,
    matmul,
    spec_hash,
    verify_hash,
    safe_input_range,
    check_overflow,
)
from detmatmul.manifest import (
    load_manifest,
    save_manifest,
    build_manifest,
    compare_manifests,
)

__all__ = [
    "__version__",
    "SPEC_VERSION",
    "SCALE_FACTOR",
    "matmul",
    "spec_hash",
    "verify_hash",
    "safe_input_range",
    "check_overflow",
    "load_manifest",
    "save_manifest",
    "build_manifest",
    "compare_manifests",
]
