"""Small FastAPI service run ON the Lightning AI Studio to expose GPU
inference to the dashboard -- LightningGPUProvider's (src/dashboard/app.py)
counterpart to Kaggle's Jupyter-kernel-gateway approach. Unlike Kaggle's
setup (a raw Jupyter kernel driven over a websocket wire protocol), this
is a genuine HTTP API, since the Studio can run a normal long-lived
service directly.

Auth: a single shared secret from the LIGHTNING_SERVICE_TOKEN env var,
checked on EVERY request (including /health) by the middleware below.
The Studio exposes this service on a public URL with no network-level
protection, so the token is the only thing standing in front of it --
it is mandatory, not optional. If the env var is unset the service still
starts but every request returns 503, so a misconfigured deploy fails
loudly instead of silently serving an open endpoint.

The dashboard sends the token as the `X-Service-Token` header; an
`Authorization: Bearer <token>` header is also accepted.

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
                    torch/transformers are NOT in the default requirements
                    (see requirements-inference.txt) -- /chat returns 503
                    until they're installed, on purpose.
  POST /cancel   -- best-effort cancel of an in-flight generation (no
                    actual generations are tracked as cancellable yet in
                    this first pass, since generate() here is synchronous
                    and short-lived -- kept as a real endpoint matching the
                    spec, honestly returns "nothing to cancel" rather than
                    faking success)
"""
from __future__ import annotations

import hmac
import os
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="NVIDIA Lightning Inference Service")

SERVICE_TOKEN = os.environ.get("LIGHTNING_SERVICE_TOKEN", "")
DEFAULT_MODEL = "sshleifer/tiny-gpt2"

# In-memory model registry -- populated lazily by /chat. Empty on a fresh
# start; there is no fake "always ready" model list here.
_MODELS: dict = {}


def _presented_token(request: Request) -> str:
    tok = request.headers.get("x-service-token", "")
    if not tok:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            tok = auth[len("Bearer "):]
    return tok


@app.middleware("http")
async def require_service_token(request: Request, call_next):
    """Mandatory shared-secret gate in front of every route."""
    if not SERVICE_TOKEN:
        return JSONResponse(
            status_code=503,
            content={"detail": "LIGHTNING_SERVICE_TOKEN is not configured on the service."},
        )
    # hmac.compare_digest is constant-time -- avoids leaking the token
    # length/prefix through response timing.
    if not hmac.compare_digest(_presented_token(request), SERVICE_TOKEN):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing X-Service-Token."},
        )
    return await call_next(request)


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
def models():
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
def chat(req: ChatRequest):
    try:
        import torch
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Inference deps not installed (see requirements-inference.txt). "
                   "This service only serves /health + /models today.",
        )

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
def cancel(request_id: str):
    # generate() above is synchronous and short-lived in this first pass --
    # nothing is actually tracked as in-flight/cancellable yet. Honest
    # no-op rather than pretending a cancel took effect.
    return {"cancelled": False, "request_id": request_id, "reason": "No in-flight requests are tracked yet in this first pass."}
