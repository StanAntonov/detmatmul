# Deterministic Inference Standard — v1.0

**Project:** detmatmul  
**Status:** Stable  
**Date:** 2026  

---

## Abstract

This document defines the **Deterministic Inference Standard (DIS) v1.0** - a specification for matrix multiplication that produces bit-exact results across any hardware platform, operating system, or software environment.

The core mechanism is replacing IEEE-754 floating-point arithmetic with **Q16.16 fixed-point integer arithmetic**. Integer arithmetic is fully associative and produces identical results regardless of hardware, eliminating the non-determinism inherent in floating-point computation.

Any implementation that passes the canonical test suite in this document is certified as **DIS v1.0 compliant**.

---

## 1. Problem Statement

### 1.1 The Non-Determinism Problem

IEEE-754 floating-point arithmetic, while standardised at the operation level, does not guarantee identical results across hardware platforms. The following sources of non-determinism exist in current ML inference systems:

**Accumulation order.** The result of summing N floating-point numbers depends on the order of operations. Matrix multiplication implementations on different hardware reorder operations for efficiency, producing different results.

**Fused Multiply-Add (FMA).** Hardware that supports FMA computes `a*b + c` with a single rounding. Hardware without FMA computes two operations with two roundings. Results differ by up to 1 ULP.

**Transcendental functions.** `exp()`, `tanh()`, `sqrt()` have implementation-defined precision. The same input can produce different bit patterns on Intel vs AMD vs ARM vs NVIDIA.

**CUDA non-determinism.** NVIDIA's documentation explicitly states that certain CUDA operations (including parallel reductions used in matrix multiplication) are non-deterministic by default. `torch.use_deterministic_algorithms(True)` mitigates this within a single vendor but does not address cross-vendor or cross-generation differences.

### 1.2 Why This Matters

| Application | Requirement |
|---|---|
| Regulated financial AI | Audit trail of model decisions |
| FDA-regulated medical AI | Reproducible model behaviour |
| Zero-Knowledge ML proofs | Integer circuits required |
| Distributed training | Reproducibility across heterogeneous clusters |
| AI safety research | Reproducible edge-case analysis |
| Legal AI systems | Explainability and consistency |
| Blockchain / on-chain AI | Verifiable inference |

### 1.3 Current State of the Art

No existing standard addresses cross-hardware determinism for neural network inference.

| Approach | Limitation |
|---|---|
| PyTorch deterministic mode | NVIDIA only, not cross-vendor |
| INT8 quantisation | Optimises for speed, not cross-hardware bit-exactness |
| ZK-ML circuits | Correct approach but requires full ZK proof overhead |
| Vendor-specific flags | Hardware-dependent, no portability |

---

## 2. Specification

### 2.1 Numerical Format

| Property | Value |
|---|---|
| Format | Q16.16 fixed-point |
| Scale factor | 65536 (2¹⁶) |
| Accumulator type | int64 (64-bit signed integer) |
| Rounding | Round-half-up before right-shift |

**Encoding:** A floating-point value `v` is represented as the integer `round(v × 65536)`.

**Decoding:** An integer value `n` represents the floating-point value `n / 65536`.

### 2.2 Matrix Multiplication

Given input matrices A (M×K) and B (K×N), a compliant implementation must:

1. **Encode** — Convert A and B to int64 by multiplying by `SCALE_FACTOR` and rounding to nearest integer.
2. **Accumulate** — For each output element `C[i,j]`, compute the dot product of row `i` of A and column `j` of B using sequential int64 accumulation in row-major order.
3. **Round** — Apply round-half-up before the right-shift: `val = (acc + 32768) >> 16`
4. **Decode** — The float result is `val / SCALE_FACTOR`.

### 2.3 Reference Implementation

The following Python function is the ground truth. Any implementation producing a SHA-256 hash that matches this function's output for all canonical test cases is compliant.

```python
import numpy as np

SCALE_FACTOR = 65536

def reference_matmul(A: np.ndarray, B: np.ndarray,
                     use_relu: bool = False) -> np.ndarray:
    """
    A, B   : float32 input matrices
    Returns: int64 array (pre-decoded)

    Encode inputs first:
        A_f = (A * SCALE_FACTOR).astype(np.int64)
        B_f = (B * SCALE_FACTOR).astype(np.int64)
    Then call reference_matmul(A_f, B_f, use_relu).
    """
    M, K = A.shape
    _, N = B.shape
    Out  = np.zeros((M, N), dtype=np.int64)
    half = np.int64(32768)

    for r in range(M):
        for c in range(N):
            acc = np.int64(0)
            for k in range(K):
                acc += A[r, k] * B[k, c]
            val = (acc + half) >> 16
            Out[r, c] = np.int64(0) if (use_relu and val < 0) else val

    return Out
```

This is the specification. The inner loop order (`k = 0, 1, ..., K-1`) is normative. Any parallel implementation must produce results identical to this sequential order.

### 2.4 Accumulation Order

The reference uses **sequential row-major accumulation**. Parallelism across output elements `C[i,j]` is permitted — each output element is independent. Reordering the K-length inner loop accumulation is **not** permitted.

### 2.5 SHA-256 Verification

The compliance hash for a given (A, B) pair is computed as:

```python
import hashlib
import numpy as np

def compliance_hash(A_int64: np.ndarray) -> str:
    """
    A_int64 : the raw int64 output of reference_matmul (before decoding)
    Returns : 64-character hex SHA-256 string
    """
    return hashlib.sha256(
        np.ascontiguousarray(A_int64).tobytes()
    ).hexdigest()
```

The hash is taken over the **raw int64 bytes**, before dividing by `SCALE_FACTOR`. This ensures the hash is independent of floating-point representation.

### 2.6 Overflow Constraint

A compliant implementation is only guaranteed correct when inputs satisfy:

```
|A[i,j]| ≤ sqrt(INT64_MAX / (2 × K × SCALE_FACTOR²))
```

where `INT64_MAX = 9_223_372_036_854_775_807`.

For common values of K:

| K | Safe absolute input value |
|---|---|
| 64 | < 4096 |
| 256 | < 2048 |
| 768 | < 1186 |
| 1024 | < 1024 |
| 4096 | < 512 |

Neural network weights and activations normalised by layer normalisation typically satisfy these bounds for K ≤ 4096.

### 2.7 Optional Operations

The following operations used in transformer inference must also be implemented deterministically for full DIS compliance:

**Layer Normalisation** — Sequential scalar computation. Mean and variance computed by sequential summation. No reduction reordering permitted.

**Softmax** — Sequential computation. Numerically stable via max subtraction. No reduction reordering permitted.

**GELU activation** — Element-wise via:
```
0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715x³)))
```

**Embedding lookup** — Direct table lookup. Inherently deterministic.

---

## 3. Canonical Test Suite

### 3.1 Test Cases

A DIS v1.0 compliant implementation must pass all 31 test cases:

| Group | Seeds | Shapes | Count |
|---|---|---|---|
| Normal distribution | 42, 137, 999, 31415, 271828 | 64×64×64, 256×256×256, 512×128×512, 128×512×128, 511×255×767 | 25 |
| All-zeros | 0 | 64×64×64 | 1 |
| All-ones | 0 | 64×64×64 | 1 |
| Tiny values (±0.001) | 7 | 64×64×64 | 1 |
| Large values (±10) | 7 | 64×32×64 | 1 |
| Negative-heavy (±5 skewed) | 55 | 128×64×128 | 1 |
| Non-power-of-2 rectangular | 42 | 300×150×200 | 1 |

### 3.2 Input Generation

```python
import numpy as np

def make_matrices(M, K, N, seed, kind, relu):
    rng = np.random.default_rng(seed)

    if kind == "zeros":
        return np.zeros((M, K), np.float32), np.zeros((K, N), np.float32)
    if kind == "ones":
        return np.ones((M, K), np.float32), np.ones((K, N), np.float32)
    if kind == "tiny":
        return (rng.uniform(-1e-3, 1e-3, (M, K)).astype(np.float32),
                rng.uniform(-1e-3, 1e-3, (K, N)).astype(np.float32))
    if kind == "large":
        return (rng.uniform(-10, 10, (M, K)).astype(np.float32),
                rng.uniform(-10, 10, (K, N)).astype(np.float32))
    if kind == "neg_heavy":
        return (rng.uniform(-5, 1, (M, K)).astype(np.float32),
                rng.uniform(-5, 1, (K, N)).astype(np.float32))
    # default: normal
    return (rng.standard_normal((M, K)).astype(np.float32),
            rng.standard_normal((K, N)).astype(np.float32))
```

### 3.3 Test Case Keys

Each test case is identified by the string:

```
{M}x{K}x{N}_{kind}_seed{seed}_{relu|norelu}
```

Example: `256x256x256_normal_seed42_relu`

### 3.4 Reference Hashes

The following hashes were generated by the `cpu_reference()` implementation and verified on four independent hardware platforms (see [`manifests/hash_manifest.json`](manifests/hash_manifest.json)).

Selected hashes (seed=42, ReLU enabled):

| Test case | SHA-256 |
|---|---|
| `64x64x64_normal_seed42_relu` | `bc071e9c97c9bc2d3d480da4cbb687d2da83ecd92f1205d609b19276e4894b83` |
| `256x256x256_normal_seed42_relu` | `0a957271d84451bed064a258a4ee9e933f8ed05752319d7dc4f55f0a4c53fe50` |
| `512x128x512_normal_seed42_relu` | `c119f75f9a981019ceb7a13658c68cc4a01202076421319b5cd832d9f99ce330` |
| `128x512x128_normal_seed42_relu` | `56983316d2fa28d2d3b8622cb75491bf940dcd0ef8fa496cf4e79f2393171b88` |
| `511x255x767_normal_seed42_relu` | `571bae0b212da5d3aa89cd968e4af28eeac5293a58a0ffdf86fb9c1337c87f84` |

Full reference hashes for all 31 cases are in [`manifests/hash_manifest.json`](manifests/hash_manifest.json).

### 3.5 Verification Procedure

For each test case:

1. Generate input matrices using the specified seed and kind
2. Encode to int64 by multiplying by `SCALE_FACTOR`
3. Run the matmul through the candidate implementation
4. Compute SHA-256 of the raw int64 output bytes
5. Compare against the reference hash

An implementation is **DIS v1.0 compliant** if and only if all 31 hashes match.

---

## 4. Verified Platforms

| Platform | SM | OS | Status |
|---|---|---|---|
| NVIDIA GeForce GTX 1050 | sm_61 | Windows 10 | ✅ Compliant |
| NVIDIA Tesla T4 | sm_75 | Linux 6.6 | ✅ Compliant |
| NVIDIA Tesla P100 | sm_60 | Linux 6.6 | ✅ Compliant |
| CPU-only (x86-64) | — | Linux 6.6 | ✅ Compliant |

To add a platform to this table, run the benchmark and open an issue with your manifest. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 5. Relationship to Existing Standards

### 5.1 IEEE 754

IEEE 754 defines floating-point arithmetic at the operation level but explicitly does not guarantee identical results for sequences of operations across implementations. DIS v1.0 addresses the gap by replacing floating-point with integer arithmetic for the matrix multiplication primitive.

### 5.2 ZK-ML

Zero-Knowledge ML proof systems (EZKL, Giza, Modulus Labs) require integer arithmetic for circuit construction. DIS v1.0 provides the standardised integer arithmetic primitive that ZK-ML systems need, with a lighter verification mechanism (SHA-256 hash comparison) for cases where full ZK proofs are not required.

### 5.3 ONNX

The ONNX standard defines model interchange but does not guarantee identical numerical results across backends. A DIS-compliant ONNX backend would produce identical outputs across hardware.

### 5.4 PyTorch Deterministic Mode

`torch.use_deterministic_algorithms(True)` enforces determinism within a single GPU vendor but does not address cross-vendor or cross-generation differences. DIS v1.0 is hardware-agnostic.

---

## 6. Rationale for Q16.16

Q16.16 was chosen because:

**Sufficient precision for inference.** Neural network weights can be represented in Q16.16 with quantisation error bounded by `K × (0.5/65536) × √(2/π)` per output element, within acceptable bounds for classification and generation tasks.

**No overflow for typical workloads.** For K ≤ 4096 and inputs bounded by layer normalisation (typically |v| < 5), the int64 accumulator has approximately 40,000× headroom before overflow.

**Universal hardware support.** int64 multiplication and addition are supported on every modern CPU and GPU without special instructions or vendor-specific intrinsics.

**Simplicity.** Encoding is a single multiply by 65536. Decoding is a single right-shift. Rounding is a single add of 32768. The entire arithmetic fits in a handful of lines of any language.

---

## 7. Future Work

**DIS v1.1** (planned):
- Convolution (Conv2D, DepthwiseConv)
- Attention with variable-length sequences
- Batch normalisation
- AMD ROCm verification
- Intel oneAPI verification
- Apple Silicon (Metal) verification

**DIS v2.0** (research):
- Block floating-point extension for extended dynamic range
- Formal verification of kernel correctness (Lean/Coq)
- ZK circuit generation from DIS-compliant kernels

---

## Appendix: Glossary

| Term | Definition |
|---|---|
| DIS | Deterministic Inference Standard |
| Q16.16 | Fixed-point format: 16 bits integer part, 16 bits fractional part |
| SCALE_FACTOR | 65536 (2¹⁶) — the encoding multiplier |
| Compliant | An implementation passing all 31 canonical test cases |
| Manifest | A JSON file recording SHA-256 hashes for the canonical test suite |
| Master hash | SHA-256 of all 31 individual hashes concatenated |
| Cross-hardware proof | Two or more manifests from different hardware showing identical hashes |
