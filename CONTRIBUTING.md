# Contributing to detmatmul

Thank you for your interest. This project is a cross-hardware determinism standard - its value grows with every new platform that verifies it. The most useful contribution you can make right now requires no code at all.

---

## The most useful thing: run the benchmark on new hardware

Every platform that agrees strengthens the proof. AMD GPUs, Apple Silicon, Intel Arc, ARM CPUs, and older NVIDIA architectures are all unverified territory.

**Step 1 - Install**

```bash
git clone https://github.com/yourname/detmatmul
cd detmatmul
pip install -e .
```

Or just download `examples/benchmark.py` and `manifests/hash_manifest.json` directly - the benchmark is self-contained.

**Step 2 - Run**

```bash
# With a CUDA GPU
python examples/benchmark.py --verify manifests/hash_manifest.json

# CPU-only (works on any machine, including AMD/Apple/ARM)
python examples/benchmark.py --cpu --verify manifests/hash_manifest.json
```

**Step 3 - Open an issue**

Title it: `Verification: [your hardware]` - for example, `Verification: AMD RX 7900 XTX` or `Verification: Apple M3 Pro (CPU)`.

Attach your `hash_manifest.json` and paste the summary line from the output. That's it. If it passes, you've extended the proof to a new platform.

---

## Reporting a hash mismatch

If the benchmark reports a failure - your hashes don't match the reference - that is an important finding.

Open an issue titled: `Mismatch: [your hardware]`

Include:
- Your `hash_manifest.json`
- The full terminal output of the benchmark
- Your Python version (`python --version`)
- Your Numba version (`python -c "import numba; print(numba.__version__)"`)
- Your CUDA version if applicable (`nvcc --version`)

Do not assume it is a bug in your setup. A genuine mismatch on a new platform is exactly the kind of finding this project exists to surface.

---

## Code contributions

### What is in scope

- **New platform support** - ROCm (AMD), Metal (Apple Silicon), oneAPI (Intel), OpenCL
- **New language implementations** - A Rust, C, or Julia implementation that passes the 31-case test suite
- **Performance improvements** - Faster GPU kernels that still produce identical hashes
- **New test cases** - Additional edge cases for the canonical suite (requires updating the spec version)
- **Bug fixes** - Anything that causes a hash mismatch or incorrect overflow detection

### What is out of scope

- Changing the accumulation order or rounding behaviour of the reference implementation - this would break all existing manifests
- Approximate or probabilistic variants - the entire point is bit-exactness
- Dependencies that are not available on all target platforms

### Before opening a PR

1. Run the full test suite: `pytest tests/ -v`
2. Run the benchmark and confirm your hashes match: `python examples/benchmark.py --cpu --verify manifests/hash_manifest.json`
3. If you are adding a new platform kernel, include its manifest output in the PR description

### Coding style

- Follow the existing style in `detmatmul/core.py` - clarity over cleverness
- All new kernels must include a comment stating which test cases they were verified against
- No new required dependencies without discussion first

---

## Updating the specification

Changes to `SPEC.md` that alter the accumulation order, rounding rule, or scale factor would increment the spec version and invalidate all existing manifests. This is a high bar intentionally.

Proposals to change the spec should be opened as an issue for discussion before any code is written. Include:
- The motivation
- The specific change to the reference implementation
- An analysis of the impact on existing verified platforms

---

## Questions

Open an issue. There are no dumb questions about determinism - it is a genuinely subtle subject.
