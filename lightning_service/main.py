"""Small FastAPI service run ON the Lightning AI Studio to expose GPU
inference to the dashboard -- LightningGPUProvider's (src/dashboard/app.py)
counterpart to Kaggle's Jupyter-kernel-gateway approach. Unlike Kaggle's
setup (a raw Jupyter kernel driven over a websocket wire protocol), this
is a genuine HTTP API, since the Studio can run a normal long-lived
service directly.

Auth: a single bearer token from the LIGHTNING_SERVICE_TOKEN env var,
never committed. If that env var is unset, auth is skipped entirely --
fine for an internal CPU-tier test reached only via SSH+localhost, but
this MUST be set before this service is ever exposed publicly.

Endpoints:
  GET  /health   -- liveness + whether a GPU is actually available right now,
                    and which (if any) models are currently loaded
  GET  /models   -- same model list as /health, standalone
  POST /chat     -- generate a reply from a model, lazy-loading it on first
                    use. Defaults to sshleifer/tiny-gpt2 (a few MB, built by
                    HF specifically for fast tests) so this first pass can
                    prove the real torch/transformers/CUDA pipeline works
                    end-to-end without the time/GPU-hour cost of a real LLM.
                    NOTE: tiny-gpt2 has effectively random, untrained
                    weights -- expect gibberish output. This endpoint is
                    proving the *plumbing* works, not reply quality; wiring
                    a real model is a later, separate step.
  POST /cancel   -- best-effort cancel of an in-flight generation (no
                    actual generations are tracked as cancellable yet in
                    this first pass, since generate() here is synchronous
                    and short-lived -- kept as a real endpoint matching the
                    spec, honestly returns "nothing to cancel" rather than
                    faking success)
"""
from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NVIDIA Lightning Inference Service")

AUTH_TOKEN = os.environ.get("LIGHTNING_SERVICE_TOKEN", "")
DEFAULT_MODEL = "sshleifer/tiny-gpt2"

# In-memory model registry -- populated lazily by /chat. Empty on a fresh
# start; there is no fake "always ready" model list here.
_MODELS: dict = {}


def _check_auth(authorization: str | None) -> None:
    if not AUTH_TOKEN:
        return  # no token configured -- open; must be set before any public exposure
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _gpu_info():
    try:
        import torch
        available = torch.cuda.is_available()
        return {
            "gpu_available": available,
            "device_name": torch.cuda.get_device_name(0) if available else None,
            "torch_version": torch.__version__,
        }
    except ImportError:
        return {"gpu_available": False, "device_name": None, "torch_version": None}


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_MODELS.keys()), **_gpu_info()}


@app.get("/models")
def models(authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    return {"models": list(_MODELS.keys())}


class ChatRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL
    max_new_tokens: int = 64


def _ensure_model_loaded(model_name: str):
    if model_name in _MODELS:
        return _MODELS[model_name]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        mod = mod.to("cuda")
    _MODELS[model_name] = (tok, mod)
    return tok, mod


@app.post("/chat")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    import torch

    try:
        tok, mod = _ensure_model_loaded(req.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Couldn't load model '{req.model}': {e}")

    inputs = tok(req.prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    out = mod.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False)
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "request_id": str(uuid.uuid4()),
        "model": req.model,
        "reply": reply,
        "gpu_used": torch.cuda.is_available(),
    }


@app.post("/cancel")
def cancel(request_id: str, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    # generate() above is synchronous and short-lived in this first pass --
    # nothing is actually tracked as in-flight/cancellable yet. Honest
    # no-op rather than pretending a cancel took effect.
    return {"cancelled": False, "request_id": request_id, "reason": "No in-flight requests are tracked yet in this first pass."}
