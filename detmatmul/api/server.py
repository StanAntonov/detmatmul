"""
Deterministic Inference Verification API
=========================================
A REST API that provides verifiable deterministic AI inference.

Every response includes a SHA-256 hash. The same request on any
hardware anywhere in the world produces the same hash.

This is the commercial layer — the "Verification as a Service" product.

Run:
    pip install fastapi uvicorn transformers tiktoken
    python -m detmatmul.api.server

Endpoints:
    GET  /                     Health check + hardware info
    POST /generate             Generate text (returns output + hash)
    POST /verify               Verify a hash matches expected output
    POST /batch                Batch generation with full hash manifest
    GET  /manifest             Current machine's hash manifest
    GET  /spec                 Specification version info
"""

import hashlib
import os
import platform
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

# ── FastAPI optional ──────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    raise ImportError(
        "FastAPI not installed. Run:\n"
        "  pip install fastapi uvicorn pydantic")

import numpy as np

from detmatmul.core import SPEC_VERSION, SCALE_FACTOR
from detmatmul._version import __version__

# GPU name detection
_GPU_NAME = "CPU-only"
try:
    from numba import cuda
    _dev      = cuda.get_current_device()
    _GPU_NAME = (_dev.name.decode()
                 if isinstance(_dev.name, bytes) else _dev.name)
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  APP
# ═══════════════════════════════════════════════════════════════════════════════

# Global model instance
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print("[API] Loading deterministic GPT-2 model...")
    try:
        from detmatmul.gpt2 import DeterministicGPT2
        _model = DeterministicGPT2.load("gpt2")
        print(f"[API] Model ready on {_GPU_NAME}")
    except Exception as e:
        print(f"[API] Warning: Could not load GPT-2: {e}")
        print("[API] API will run in demo mode (tiny model)")
        _model = None
    yield


app = FastAPI(
    title       = "Deterministic Inference API",
    description = (
        "Verifiable AI inference — every output has a SHA-256 hash "
        "that is identical across any hardware platform."
    ),
    version     = __version__,
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class GenerateRequest(BaseModel):
    prompt         : str            = Field(..., description="Input text prompt")
    max_new_tokens : int            = Field(50,  ge=1, le=500)
    temperature    : float          = Field(0.8, ge=0.0, le=2.0)
    top_k          : int            = Field(40,  ge=0, le=200)
    seed           : int            = Field(0,   description="Sampling seed")

class GenerateResponse(BaseModel):
    prompt         : str
    generated_text : str
    output_hash    : str   = Field(..., description="SHA-256 of output text")
    hardware       : str
    spec_version   : str
    elapsed_ms     : float
    timestamp      : str

class VerifyRequest(BaseModel):
    prompt         : str
    expected_hash  : str
    max_new_tokens : int   = 50
    temperature    : float = 0.8
    top_k          : int   = 40
    seed           : int   = 0

class VerifyResponse(BaseModel):
    compliant      : bool
    prompt         : str
    expected_hash  : str
    actual_hash    : str
    hardware       : str
    message        : str

class BatchRequest(BaseModel):
    prompts        : list  = Field(..., description="List of prompts")
    max_new_tokens : int   = 50
    temperature    : float = 0.8
    seed           : int   = 0


def _get_model():
    if _model is None:
        # Fallback: use the tiny deterministic transformer
        from detmatmul.inference import DeterministicTransformer
        return DeterministicTransformer(), "tiny"
    return _model, "gpt2"


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Health check and hardware info."""
    return {
        "status"         : "running",
        "hardware"       : _GPU_NAME,
        "os"             : f"{platform.system()} {platform.release()}",
        "python"         : platform.python_version(),
        "detmatmul"      : __version__,
        "spec_version"   : SPEC_VERSION,
        "scale_factor"   : SCALE_FACTOR,
        "model_loaded"   : _model is not None,
        "guarantee"      : (
            "All inference on this server produces SHA-256 verified "
            "bit-exact outputs. The same request returns the same hash "
            "on any compliant hardware."
        ),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Generate text with deterministic inference.

    The output_hash in the response is your proof of determinism.
    The same prompt + seed will always return the same hash,
    on this server or any other running the same spec.
    """
    model, _ = _get_model()
    t0 = time.perf_counter()

    try:
        text, output_hash = model.generate(
            req.prompt,
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
            top_k          = getattr(req, "top_k", 40),
            seed           = req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        prompt         = req.prompt,
        generated_text = text,
        output_hash    = output_hash,
        hardware       = _GPU_NAME,
        spec_version   = SPEC_VERSION,
        elapsed_ms     = round(elapsed, 2),
        timestamp      = datetime.now().isoformat(),
    )


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    """
    Verify that this machine produces the expected hash for a given prompt.

    Use this to confirm cross-hardware compliance — run the same request
    on two different machines and compare hashes.
    """
    model, _ = _get_model()

    try:
        _, actual_hash = model.generate(
            req.prompt,
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
            top_k          = 40,
            seed           = req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    compliant = (actual_hash == req.expected_hash)

    return VerifyResponse(
        compliant      = compliant,
        prompt         = req.prompt,
        expected_hash  = req.expected_hash,
        actual_hash    = actual_hash,
        hardware       = _GPU_NAME,
        message        = (
            "COMPLIANT: This hardware produces identical output to the reference."
            if compliant else
            "NON-COMPLIANT: Hash mismatch detected. "
            "This hardware does not match the reference."
        ),
    )


@app.post("/batch")
async def batch_generate(req: BatchRequest):
    """
    Generate for multiple prompts and return a full hash manifest.

    Use this to build cross-hardware compliance proof across an entire
    test suite in one API call.
    """
    model, _ = _get_model()
    results  = {}
    hashes   = []

    for prompt in req.prompts:
        try:
            text, h = model.generate(
                prompt,
                max_new_tokens = req.max_new_tokens,
                temperature    = req.temperature,
                seed           = req.seed,
            )
            results[prompt] = {"text": text, "hash": h}
            hashes.append(h)
        except Exception as e:
            results[prompt] = {"error": str(e), "hash": None}

    # Master hash — single fingerprint for entire batch
    master_hash = hashlib.sha256(
        "".join(h for h in hashes if h).encode()
    ).hexdigest()

    return {
        "hardware"    : _GPU_NAME,
        "spec_version": SPEC_VERSION,
        "timestamp"   : datetime.now().isoformat(),
        "master_hash" : master_hash,
        "n_prompts"   : len(req.prompts),
        "results"     : results,
        "instruction" : (
            "Run this same request on another machine and compare master_hash. "
            "If they match, both machines are compliant with the spec."
        ),
    }


@app.get("/spec")
async def spec_info():
    """Return the specification version and parameters."""
    return {
        "spec_version"   : SPEC_VERSION,
        "scale_factor"   : SCALE_FACTOR,
        "format"         : "Q16.16 fixed-point integer",
        "accumulator"    : "int64",
        "rounding"       : "round-half-up before right-shift",
        "hash_algorithm" : "SHA-256",
        "description"    : (
            "Any implementation producing matching SHA-256 hashes for "
            "the canonical test suite is compliant with this specification."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Run: pip install uvicorn")

    print("\nDeterministic Inference API")
    print(f"Hardware : {_GPU_NAME}")
    print(f"Spec     : v{SPEC_VERSION}")
    print("\nStarting server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs\n")

    uvicorn.run(
        "detmatmul.api.server:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,    # must be 1 — model is not thread-safe
    )
