# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-03-25

Initial public release.

### Specification

- **DIS v1.0** - Deterministic Inference Standard formally defined
- Q16.16 fixed-point format (SCALE_FACTOR = 65536)
- int64 accumulator with round-half-up rounding
- 31-case canonical test suite established
- SHA-256 verification protocol defined

### Core (`detmatmul/core.py`)

- `matmul()` - deterministic fixed-point matrix multiplication
- `spec_hash()` - SHA-256 fingerprint of the raw int64 result
- `verify_hash()` - compliance check against a known-good hash
- `safe_input_range()` - maximum safe input value for a given K
- `check_overflow()` - pre-flight int64 overflow analysis
- GPU kernel: 4×4 register-blocked tiled implementation (64×64 tiles, TILE_K=16)
- CPU reference: sequential scalar (`cpu_reference`)
- CPU parallel: deterministic multi-threaded (`cpu_reference_parallel`)
- Automatic CPU fallback when CUDA is not available

### Manifest (`detmatmul/manifest.py`)

- `build_manifest()` - run all 31 test cases and record hashes
- `load_manifest()` / `save_manifest()` - JSON persistence
- `merge_manifests()` - combine manifests from multiple machines
- `compare_manifests()` - cross-hardware hash comparison with compliance report

### GPT-2 (`detmatmul/gpt2.py`)

- `DeterministicGPT2` - GPT-2 inference with every matmul through the Q16.16 kernel
- Deterministic layer normalisation, softmax, GELU
- Seeded sampling - identical token sequences across hardware
- SHA-256 output hash for cross-hardware generation proof
- Supports: gpt2, gpt2-medium, gpt2-large, gpt2-xl

### PyTorch compatibility (`detmatmul/torch_compat.py`)

- `deterministic_mode()` - context manager: patches `torch.mm`, `torch.matmul`, `torch.bmm`
- `patch_torch()` / `unpatch_torch()` - permanent patch/restore
- `@deterministic` - function decorator
- `tensor_hash()` - SHA-256 of a tensor's values
- `model_output_hash()` - run model and return (output, hash)

### API (`detmatmul/api/server.py`)

- FastAPI REST server
- `POST /generate` - text generation with `output_hash` in response
- `POST /verify` - cross-machine hash compliance check
- `POST /batch` - batch generation with master hash manifest
- `GET /spec` - specification version and parameters
- `GET /` - health check and hardware info

### Verified platforms

| Platform | SM | OS |
|---|---|---|
| NVIDIA GeForce GTX 1050 | sm_61 | Windows 10 |
| NVIDIA Tesla T4 | sm_75 | Linux 6.6 |
| NVIDIA Tesla P100 | sm_60 | Linux 6.6 |
| CPU-only (x86-64) | - | Linux 6.6 |

All 31 SHA-256 hashes match across all four platforms.

---

## Roadmap

### [1.1.0] - planned

- AMD ROCm kernel
- Apple Silicon (Metal) support
- Intel oneAPI support
- Convolution operations (Conv2D)
- Attention with variable-length sequences
- Batch normalisation

### [2.0.0] - research

- Block floating-point extension for extended dynamic range
- Formal verification of kernel correctness
- ZK circuit generation from DIS-compliant kernels
