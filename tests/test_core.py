"""
Basic tests for detmatmul.core
Run with: pytest tests/
"""
import numpy as np
import pytest
from detmatmul import matmul, spec_hash, verify_hash, check_overflow, safe_input_range
from detmatmul.core import cpu_reference, SCALE_FACTOR


def test_matmul_shape():
    A = np.random.randn(64, 32).astype(np.float32)
    B = np.random.randn(32, 48).astype(np.float32)
    C = matmul(A, B, force_cpu=True)
    assert C.shape == (64, 48)


def test_matmul_zeros():
    A = np.zeros((32, 32), np.float32)
    B = np.zeros((32, 32), np.float32)
    C = matmul(A, B, force_cpu=True)
    assert np.all(C == 0)


def test_matmul_ones():
    """A=ones, B=ones -> C[i,j] = K (number of columns of A)."""
    K = 16
    A = np.ones((8, K), np.float32)
    B = np.ones((K, 8), np.float32)
    C = matmul(A, B, force_cpu=True)
    assert np.allclose(C, K, atol=1e-3)


def test_hash_reproducible():
    """Same inputs must always produce same hash."""
    rng = np.random.default_rng(42)
    A   = rng.standard_normal((64, 64)).astype(np.float32)
    B   = rng.standard_normal((64, 64)).astype(np.float32)
    h1  = spec_hash(A, B, force_cpu=True)
    h2  = spec_hash(A, B, force_cpu=True)
    assert h1 == h2


def test_verify_hash_pass():
    rng = np.random.default_rng(99)
    A   = rng.standard_normal((64, 64)).astype(np.float32)
    B   = rng.standard_normal((64, 64)).astype(np.float32)
    h   = spec_hash(A, B, force_cpu=True)
    assert verify_hash(A, B, expected_hash=h, force_cpu=True)


def test_verify_hash_fail():
    rng = np.random.default_rng(99)
    A   = rng.standard_normal((64, 64)).astype(np.float32)
    B   = rng.standard_normal((64, 64)).astype(np.float32)
    assert not verify_hash(A, B, expected_hash="0" * 64, force_cpu=True)


def test_relu():
    """With relu=True all negative values should become 0."""
    A = np.full((32, 32), -1.0, np.float32)
    B = np.full((32, 32),  1.0, np.float32)
    C = matmul(A, B, use_relu=True, force_cpu=True)
    assert np.all(C == 0)


def test_dimension_mismatch():
    A = np.random.randn(10, 5).astype(np.float32)
    B = np.random.randn(6,  3).astype(np.float32)
    with pytest.raises(ValueError):
        matmul(A, B, force_cpu=True)


def test_quantization_error_within_bounds():
    """Max error vs float64 should be within 5x theoretical bound."""
    import math
    rng    = np.random.default_rng(7)
    M,K,N  = 256, 256, 256
    A      = rng.standard_normal((M, K)).astype(np.float32)
    B      = rng.standard_normal((K, N)).astype(np.float32)
    C_det  = matmul(A, B, force_cpu=True)
    C_f64  = A.astype(np.float64) @ B.astype(np.float64)
    max_err = np.abs(C_det.astype(np.float64) - C_f64).max()
    bound   = K * (0.5 / SCALE_FACTOR) * math.sqrt(2 / math.pi) * 5
    assert max_err < bound, f"max_err={max_err:.6f} > bound={bound:.6f}"


def test_safe_input_range():
    for K in [32, 64, 256, 1024]:
        sr = safe_input_range(K)
        assert sr > 0
        # Values well within range should pass overflow check
        rng = np.random.default_rng(0)
        A   = rng.uniform(-sr*0.5, sr*0.5, (8, K)).astype(np.float32)
        B   = rng.uniform(-sr*0.5, sr*0.5, (K, 8)).astype(np.float32)
        assert check_overflow(A, B, verbose=False)


def test_overflow_detected():
    """Values above safe range should trigger overflow warning."""
    K   = 4096
    sr  = safe_input_range(K)
    rng = np.random.default_rng(0)
    A   = rng.uniform(-sr*2, sr*2, (8, K)).astype(np.float32)
    B   = rng.uniform(-sr*2, sr*2, (K, 8)).astype(np.float32)
    assert not check_overflow(A, B, verbose=False)
