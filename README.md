# detmatmul

**Deterministic cross-hardware matrix multiplication with SHA-256 verification.**

`detmatmul` guarantees that the same computation produces **bit-exact results** regardless of GPU brand, GPU generation, or operating system - verified by SHA-256 hash comparison.

This is achieved by replacing IEEE-754 floating point with **Q16.16 fixed-point integer arithmetic** (int64). Integer arithmetic is fully associative: there is no hardware-dependent rounding, no non-determinism.

---

## Proven results

Every SHA-256 hash below is identical across all four platforms. Zero mismatches across 31 test cases.

| Platform | Compute | OS | Result |
|---|---|---|---|
| NVIDIA GeForce GTX 1050 | sm_61 | Windows 10 | ✅ All 31 match |
| NVIDIA Tesla T4 | sm_75 | Linux 6.6 | ✅ All 31 match |
| NVIDIA Tesla P100 | sm_60 | Linux 6.6 | ✅ All 31 match |
| CPU-only (no GPU) | - | Linux 6.6 | ✅ All 31 match |

The raw proof is in [`manifests/hash_manifest.json`](manifests/hash_manifest.json). Clone the repo, run the benchmark, and add your platform.

---

## Why this matters

**IEEE-754 floating point is not deterministic across hardware.** The same neural network inference on two different GPU generations can produce different outputs. This is documented, accepted, and largely ignored - because for most applications it doesn't matter.

For some applications it is critical:

| Application | Requirement |
|---|---|
| Regulated financial AI | Audit trail of model decisions |
| FDA-regulated medical AI | Reproducible model behavior |
| Zero-Knowledge ML (ZK-ML) | Integer circuits required |
| Distributed training | Reproducibility across heterogeneous clusters |
| AI safety research | Reproducible edge case analysis |
| Blockchain / on-chain AI | Verifiable inference |

`detmatmul` solves this at the kernel level. No application-level changes required beyond swapping the matmul call.

---

## Quick start

```bash
pip install detmatmul
```

Or from source:

```bash
git clone https://github.com/yourname/detmatmul
cd detmatmul
pip install -e .
```

**Requirements:** Python 3.10+, NumPy, Numba. CUDA optional - falls back to CPU automatically.

```python
import numpy as np
from detmatmul import matmul, spec_hash, verify_hash

A = np.random.randn(512, 512).astype(np.float32)
B = np.random.randn(512, 512).astype(np.float32)

# Deterministic matmul - identical result on any hardware
C = matmul(A, B)

# SHA-256 fingerprint of the result
h = spec_hash(A, B)
print(h)  # e.g. "bc071e9c97c9bc2d..."

# Verify this machine matches a known-good hash
assert verify_hash(A, B, expected_hash=h)
```

---

## Cross-hardware compliance verification

```python
from detmatmul.manifest import build_manifest, load_manifest, merge_manifests, compare_manifests

# On machine A
m = build_manifest()
save_manifest(m, "machine_a.json")

# On machine B
m = build_manifest()
save_manifest(m, "machine_b.json")

# Compare
merged = merge_manifests([load_manifest("machine_a.json"),
                          load_manifest("machine_b.json")])
result = compare_manifests(merged)

print(f"{result['agreed']}/{result['total']} cases agree")
print(f"Compliant: {result['compliant']}")
```

---

## GPT-2 demo

The world's first demonstrably deterministic language model inference. Run this on two different machines - the SHA-256 hashes will match.

```bash
# Requires: pip install transformers tiktoken
python examples/demo_gpt2.py

# Verify against a result from another machine
python examples/demo_gpt2.py --verify gpt2_results.json

# Interactive chat with hash display
python examples/demo_gpt2.py --interactive
```

---

## PyTorch drop-in replacement

```python
from detmatmul.torch_compat import deterministic_mode

model = torch.load("my_model.pt")

with deterministic_mode():
    output = model(input_tensor)
    # Every matmul inside used the Q16.16 kernel
```

---

## Command line

```bash
# Full audit - runs all 31 test cases, saves manifest
python examples/benchmark.py

# CPU-only (no CUDA required)
python examples/benchmark.py --cpu

# Verify this machine against the repo's manifest
python examples/benchmark.py --verify manifests/hash_manifest.json

# Merge two manifests from different machines
python examples/benchmark.py --merge machine_a.json machine_b.json
```

---

## REST API

```bash
pip install fastapi uvicorn
python detmatmul/api/server.py
# → http://localhost:8000/docs
```

Every `/generate` response includes an `output_hash` field. Send the same request to a server on different hardware - the hash is identical.

---

## Specification

The full specification is in [`SPEC.md`](SPEC.md).

- **Format:** Q16.16 fixed-point (SCALE_FACTOR = 65536)
- **Accumulator:** int64
- **Rounding:** round-half-up before right-shift: `(acc + 32768) >> 16`
- **Version:** 1.0

The specification is the `cpu_reference()` function in `detmatmul/core.py`. Any implementation - in any language, on any hardware - that produces matching SHA-256 hashes for the 31 canonical test cases is **DIS v1.0 compliant**.

---

## Performance

On production GPUs the deterministic kernel outperforms standard NumPy float32.

| Size | GTX 1050 | Tesla T4 | Tesla P100 |
|---|---|---|---|
| 256×256 | ~equal | 1.3× faster | - |
| 1024×1024 | 1.7× slower | **5.7× faster** | **3× faster** |
| 2048×2048 | 1.8× slower | **6.2× faster** | **2.7× faster** |

The GTX 1050 is a consumer card from 2016. On any modern data-centre GPU, determinism is free.

---

## Contribute

The most useful thing you can do right now is **run the benchmark on new hardware and post your manifest**.

1. Clone the repo
2. Run `python examples/benchmark.py --cpu` (or without `--cpu` for GPU)
3. Open an issue titled `"Verification: [your hardware]"` and attach your `hash_manifest.json`

Every new platform that agrees strengthens the cross-hardware proof. AMD GPUs, Apple Silicon, Intel Arc, and ARM CPUs are all unverified territory.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

---

## Related work

- **PyTorch deterministic mode** - NVIDIA only, not cross-vendor
- **INT8 quantization** - optimises for speed, does not guarantee cross-hardware bit-exactness  
- **ZK-ML (EZKL, Giza, Modulus Labs)** - correct approach but requires full ZK proof overhead; `detmatmul` provides the integer primitive these systems need
- **ONNX Runtime** - defines model interchange, does not guarantee identical numerics across backends

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).
