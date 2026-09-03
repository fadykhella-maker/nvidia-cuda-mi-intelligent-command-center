# MI Command Center — live (v3: full dashboard UI, wired to the real Kaggle kernel)
#
# This merges two things that were separate until now:
#   1. The rich multi-tab visual design from the claude.ai Artifact prototype
#      (Overview / Topology / GPU / Models / Agents / Tokens / About) — copied
#      from mi-command-center.html. (v3.1: the cosmetic floating Bond 001 "fab"
#      button/panel was removed — it overlapped the real native Bond panel
#      below the embedded dashboard; its model-picker content now lives in the
#      Models tab instead. v3.2: Bond's native panel can now load a real model
#      — Qwen2.5-7B-Instruct, 4-bit, on a kernel kept alive for the session —
#      and generate real replies instead of just pinging the kernel.)
#   2. The real, live Kaggle-kernel connection from the v2 vertical slice
#      (connect-once-via-query-params, run_remote() over the Jupyter kernel
#      wire protocol, no manual toggle anywhere).
#
# The visual shell renders via st.components.v1.html — Streamlit itself does
# the real check on every page load/refresh (same as v2) and injects the real
# values into the HTML via plain string .replace() on {{TOKEN}} markers
# BEFORE rendering (never Python str.format()/f-string on this HTML — the CSS
# is full of single-brace {…} blocks that would collide with format()).
#
# Still honest about what's real vs. not: the kernel-benchmark numbers, the
# Models/Agents/Tokens tabs, and the topology diagram's historical facts stay
# as accurate static record (those roadmap items genuinely aren't live yet).
# What's now dynamic: the GPU-backend status pill, the system-status KPIs,
# and the "detected hardware" tab — all driven by an actual kernel call, not
# a demo default.
#
# Run locally first (same as before):
#   cd live-vertical-slice
#   python3 -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#   streamlit run app.py

import datetime
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# Community Cloud executes the entry point from the repository root. Add the
# script directory explicitly so the sibling authentication module resolves
# consistently in both local and hosted launches.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from viewer_portal import require_viewer

ASSET_DIR = Path(__file__).resolve().parent / "assets"
NVIDIA_ICON_PATH = ASSET_DIR / "nvidia-favicon.png"
NVIDIA_ICON_DATA_URI = (
    "data:image/png;base64," + base64.b64encode(NVIDIA_ICON_PATH.read_bytes()).decode("ascii")
)

st.set_page_config(
    page_title="NVIDIA Intelligent Cloud Control",
    page_icon=Image.open(NVIDIA_ICON_PATH),
    layout="wide",
)

# Public deployment contract: authenticate before reading connection secrets,
# checking remote compute, or rendering the operational dashboard.
VIEWER_IDENTITY = require_viewer()
PUBLIC_VIEWER_MODE = True

# A real (if unglamorous) way for a button INSIDE the decorative dashboard's
# iframe to trigger a real Python-side action: it navigates the REAL outer
# page (window.top.location) to a URL with ?mi_action=... appended, which
# Streamlit picks up as a genuine query param on the resulting page load --
# the same channel the connect form already uses for jupyter_url/token.
# This is a full-page-reload round trip, not a seamless one, but it's a real
# working control physically inside that nav, not just documentation
# pointing elsewhere -- and it needed no custom Streamlit Component/build
# pipeline to get there. Processed and cleared immediately so a later
# refresh doesn't replay the same action.
_mi_action = st.query_params.get("mi_action")
if not PUBLIC_VIEWER_MODE and _mi_action == "show_chrome":
    st.session_state["mi_show_chrome"] = True
elif not PUBLIC_VIEWER_MODE and _mi_action == "hide_chrome":
    st.session_state["mi_show_chrome"] = False
elif _mi_action == "forget":
    st.query_params.clear()
if _mi_action is not None and "mi_action" in st.query_params:
    del st.query_params["mi_action"]

# Hide Streamlit Cloud's own viewer chrome (Share/GitHub/menu toolbar up top,
# the "Manage app" badge at the bottom) for regular viewers by default. This
# st.markdown() call injects straight into the REAL outer page (unlike the
# dashboard body below, which renders inside components.html()'s iframe) --
# so this is decided in plain Python from real session_state, no cross-frame
# JS/localStorage bridge needed at all. The Settings expander below sets
# mi_show_chrome via a real checkbox to bring the chrome back on request.
# stHeader is the CONTAINER the toolbar sits in -- hiding the toolbar button
# itself left the header's own reserved height/padding behind as blank space
# at the top, so the container itself is collapsed too, not just its contents.
_chrome_css = ".block-container{padding:0 !important;max-width:100% !important} iframe{border:none !important}"
if not st.session_state.get("mi_show_chrome", False):
    _chrome_css += (
        " [data-testid='stHeader']{height:0 !important;min-height:0 !important;"
        "padding:0 !important;margin:0 !important}"
        " [data-testid='stToolbar'],"
        " [data-testid='stStatusWidget'],"
        " [data-testid='stDecoration'],"
        " [data-testid='stAppDeployButton'],"
        " [class*='viewerBadge'],"
        " a[href*='streamlit.io'],"
        " #MainMenu,"
        " footer"
        " {display:none !important}"
    )
st.markdown(f"<style>{_chrome_css}</style>", unsafe_allow_html=True)

# --- Connection state: read from the URL's query params, not session_state,
# so a page REFRESH (not just a rerun) keeps it and re-checks automatically.
params = st.query_params


def get_secret(name: str, default=""):
    """Read a Streamlit secret without requiring a local secrets.toml.

    Streamlit Cloud always has the secrets file, while a fresh local checkout
    may not.  In that case the dashboard should still open and show its normal
    connection form instead of crashing before the UI renders.
    """
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


jupyter_url = get_secret("JUPYTER_URL", "")
jupyter_token = get_secret("JUPYTER_TOKEN", "")

# --- Kaggle API wake trigger + activity-based keep-alive -------------------
# Auto-wakes Kaggle by triggering a fresh run via its REST API (kaggle kernels
# push) when the dashboard finds itself offline on load -- no button click
# needed. The Kaggle-side notebook's last cell watches a shared
# "last_activity.txt" timestamp and lets the run end on its own once nobody's
# actually used the dashboard for a while, instead of a flat multi-hour
# reservation. Every real kernel call below (status check, Bond messages)
# touches that same file so real usage keeps the session alive naturally.
KAGGLE_API_TOKEN = get_secret("KAGGLE_API_TOKEN", "")
KAGGLE_KERNEL = "confidentialnvidia/confidential-nvidia-cuda-01"
KAGGLE_KERNEL_PATH = "kaggle_kernel"  # relative to this deployed repo's root
KAGGLE_WORKDIR = "/kaggle/working/AI_Lab"

ACTIVITY_TOUCH_CODE = (
    "import time as _t, os as _o\n"
    f"with open(_o.path.join({KAGGLE_WORKDIR!r}, 'last_activity.txt'), 'w') as _f:\n"
    "    _f.write(str(_t.time()))\n"
)


def wake_kaggle():
    """Trigger a fresh Kaggle run via its API (kaggle kernels push) -- the
    only channel that exists independent of the tunnel, since when Kaggle is
    off there's no tunnel yet to send a 'start' command through.

    kaggle_secrets.UserSecretsClient doesn't work in an API-triggered run (it
    needs the interactive editor's own consent state, which a push doesn't
    carry -- verified directly against a live push, which failed with
    ConnectionError/HTTP 400 on get_secret()). So the *committed* notebook
    has placeholder tokens, never real ones, and this function builds a
    temporary copy with the real values substituted in from Streamlit's own
    secrets, then pushes THAT -- the real values never touch git.

    Returns (ok, message)."""
    if not KAGGLE_API_TOKEN:
        return False, "Kaggle API token isn't configured (KAGGLE_API_TOKEN secret)."
    ngrok_authtoken = get_secret("NGROK_AUTHTOKEN", "")
    if not jupyter_token or not ngrok_authtoken:
        return False, "JUPYTER_TOKEN and/or NGROK_AUTHTOKEN secrets aren't configured."

    src_dir = KAGGLE_KERNEL_PATH
    nb_name = "confidential-nvidia-cuda-01.ipynb"
    try:
        with open(os.path.join(src_dir, nb_name), encoding="utf-8") as f:
            nb_text = f.read()
    except Exception as e:
        return False, f"Couldn't read the committed notebook: {e}"

    nb_text = nb_text.replace("__JUPYTER_TOKEN_PLACEHOLDER__", jupyter_token)
    nb_text = nb_text.replace("__NGROK_AUTHTOKEN_PLACEHOLDER__", ngrok_authtoken)
    # GITHUB_TOKEN is optional -- GPU-hour tracking (CLAUDE_CODE_HANDOFF_3.md)
    # degrades to "not configured, skip" inside the notebook if this is
    # blank, so a missing token here must never block the wake trigger.
    nb_text = nb_text.replace("__GITHUB_TOKEN_PLACEHOLDER__", get_secret("GITHUB_TOKEN", ""))

    tmp_dir = tempfile.mkdtemp(prefix="kaggle_wake_")
    try:
        with open(os.path.join(tmp_dir, nb_name), "w", encoding="utf-8") as f:
            f.write(nb_text)
        shutil.copy(os.path.join(src_dir, "kernel-metadata.json"), tmp_dir)

        env = {**os.environ, "KAGGLE_API_TOKEN": KAGGLE_API_TOKEN}
        try:
            r = subprocess.run(
                ["kaggle", "kernels", "push", "-p", tmp_dir],
                capture_output=True, text=True, timeout=60, env=env,
            )
        except Exception as e:
            return False, f"Couldn't reach Kaggle's API: {e}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if r.returncode != 0:
        return False, f"Kaggle push failed:\n{r.stdout}\n{r.stderr}"
    return True, "Kernel run triggered — this takes roughly 1-2 minutes before the tunnel comes up."


def get_kaggle_status():
    """Ask Kaggle itself what the last triggered run is actually doing --
    the real ground truth ('running', 'queued', 'complete', 'error', ...),
    not a guess from the tunnel being unreachable. A push returning
    successfully only means the upload was accepted; it says nothing about
    whether the run itself is booting, still going, or already crashed --
    this is what closes that gap.

    Returns (status, raw_output). status is 'unknown' if the check itself
    couldn't be completed (no token, no network, unparseable output)."""
    if not KAGGLE_API_TOKEN:
        return "unknown", "Kaggle API token isn't configured."
    env = {**os.environ, "KAGGLE_API_TOKEN": KAGGLE_API_TOKEN}
    try:
        r = subprocess.run(
            ["kaggle", "kernels", "status", KAGGLE_KERNEL],
            capture_output=True, text=True, timeout=20, env=env,
        )
    except Exception as e:
        return "unknown", f"Couldn't reach Kaggle's API: {e}"
    raw = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        return "unknown", raw or "status check failed"
    # Newer kaggle CLI versions print the raw enum repr, e.g.
    # `has status "KernelWorkerStatus.COMPLETE"` -- verified directly
    # against a live call. [a-zA-Z]+ alone stops at the first non-letter
    # and was matching "KernelWorkerStatus" (the enum class name) instead
    # of "COMPLETE" (the actual value) -- confirmed live in production as
    # the "KAGGLE SESSION KERNELWORKERSTATUS" pill text. Capture the whole
    # dotted token, then take the piece after the last "." if there is one
    # -- handles both this format and a plain bare word with no class
    # prefix, whichever this CLI version happens to print.
    m = re.search(r'has status "?([\w.]+)"?', raw)
    return (m.group(1).rsplit(".", 1)[-1].lower() if m else "unknown"), raw


def get_kaggle_error_log(max_chars: int = 2000) -> str:
    """Pull the tail of the actual run log from Kaggle's own API -- used
    when get_kaggle_status() reports 'error', so the dashboard can show the
    real crash reason (a dropped import, a bad substitution, an OOM, quota
    hit, ...) instead of leaving the user to guess or dig through the
    Kaggle website by hand. Best-effort: returns '' on any failure rather
    than raising, since this is a diagnostic extra, not the wake path
    itself."""
    if not KAGGLE_API_TOKEN:
        return ""
    env = {**os.environ, "KAGGLE_API_TOKEN": KAGGLE_API_TOKEN}
    tmp_dir = tempfile.mkdtemp(prefix="kaggle_log_")
    try:
        subprocess.run(
            ["kaggle", "kernels", "output", KAGGLE_KERNEL, "-p", tmp_dir],
            capture_output=True, text=True, timeout=30, env=env,
        )
        log_files = sorted(Path(tmp_dir).glob("*.log"))
        if not log_files:
            return ""
        text = log_files[0].read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
    except Exception:
        return ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- GPU provider registry --------------------------------------------------
# Kaggle is the only backend actually wired up today (KaggleGPUProvider wraps
# the wake/status/log functions above, unchanged behavior). This registry
# exists so that adding a second real backend -- Azure, AWS, or GCP spot GPUs
# -- later means writing one class with wake()/get_status() and registering
# it below, not touching the wake/status/UI plumbing that already works for
# Kaggle. UnconfiguredGPUProvider is a real, honest placeholder (never claims
# to be online or wired up) rather than hardcoding "Azure/AWS/GCP" strings
# wherever the dashboard currently checks for Kaggle specifically.
class GPUProvider:
    """Interface a GPU backend implements to plug into the dashboard's
    wake/status flow."""

    name = "unknown"

    def is_configured(self) -> bool:
        raise NotImplementedError

    def wake(self):
        """Returns (ok: bool, message: str)."""
        raise NotImplementedError

    def get_status(self):
        """Returns (status: str, raw_output: str)."""
        raise NotImplementedError


class KaggleGPUProvider(GPUProvider):
    name = "Kaggle"

    def is_configured(self) -> bool:
        return bool(KAGGLE_API_TOKEN)

    def wake(self):
        return wake_kaggle()

    def get_status(self):
        return get_kaggle_status()

    def get_error_log(self, max_chars: int = 2000) -> str:
        return get_kaggle_error_log(max_chars)


class UnconfiguredGPUProvider(GPUProvider):
    """Placeholder for a cloud GPU backend (Azure/AWS/GCP) that isn't wired
    up yet -- see the multi-provider roadmap item in
    ai-infra-agent-platform.md. Always reports itself as unconfigured and
    never claims to have woken or checked anything real."""

    def __init__(self, name: str):
        self.name = name

    def is_configured(self) -> bool:
        return False

    def wake(self):
        return False, f"{self.name} isn't wired up yet — no credentials or trigger mechanism configured."

    def get_status(self):
        return "not_configured", f"{self.name} isn't wired up yet."


GPU_PROVIDERS = {
    "kaggle": KaggleGPUProvider(),
    "azure": UnconfiguredGPUProvider("Azure"),
    "aws": UnconfiguredGPUProvider("AWS"),
    "gcp": UnconfiguredGPUProvider("GCP"),
}
# Which provider the dashboard's wake/status flow actually drives today.
# Swapping this later (once a real Azure/AWS/GCP provider is registered
# above) is a one-line config change, not a rewrite.
ACTIVE_GPU_PROVIDER = get_secret("GPU_PROVIDER", "kaggle")

# --- GPU-hour usage tracking (CLAUDE_CODE_HANDOFF_3.md) ---------------------
# The dashboard side only ever READS -- the Kaggle notebook's own keep-alive
# loop is what writes real observed runtime to data/gpu_usage_log.json in
# this repo (see the heartbeat cell added to kaggle_kernel/*.ipynb). This is
# this app's own *observed* estimate of GPU time, not Kaggle's official
# quota meter -- Kaggle exposes no API for that, so this never claims to be
# more than what it is.
GITHUB_REPO_SLUG = "fadykhella-maker/nvidia-cuda-mi-intelligent-command-center"
GPU_HOUR_BUDGET = 30.0  # the one configurable constant; matches Kaggle's free-tier weekly cap
GPU_USAGE_PRUNE_DAYS = 7


@st.cache_data(ttl=60)
def read_gpu_hours_used():
    """Sums observed runtime intervals from the last GPU_USAGE_PRUNE_DAYS
    days. Returns 0.0 on any failure (file doesn't exist yet, network
    hiccup, bad JSON) -- this is a nice-to-have status number, never
    something that should break the page if it's unreachable."""
    url = f"https://raw.githubusercontent.com/{GITHUB_REPO_SLUG}/main/data/gpu_usage_log.json"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        log = r.json()
    except Exception:
        return 0.0
    now = time.time()
    cutoff = now - GPU_USAGE_PRUNE_DAYS * 86400
    try:
        seconds = sum(
            iv["end"] - iv["start"] for iv in log.get("intervals", [])
            if iv.get("end", 0) >= cutoff
        )
    except Exception:
        return 0.0
    return round(seconds / 3600, 1)


def run_remote(code: str, timeout: int = 20):
    """Execute `code` on the live Kaggle kernel over the Jupyter kernel
    protocol. Returns (ok, output_text). ok=False on ANY failure — a
    connection error, a timeout, a bad token — never guessed as True."""
    if not jupyter_url or not jupyter_token:
        return False, "not connected"

    base = jupyter_url.rstrip("/")
    headers = {"Authorization": f"token {jupyter_token}"}

    try:
        r = requests.post(f"{base}/api/kernels", headers=headers, timeout=8)
        r.raise_for_status()
        kernel_id = r.json()["id"]
    except Exception as e:
        return False, f"Couldn't reach the Kaggle tunnel: {e}"

    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?token={jupyter_token}"

    msg_id = str(uuid.uuid4())
    execute_request = {
        "header": {
            "msg_id": msg_id, "username": "mi-command-center", "session": str(uuid.uuid4()),
            "msg_type": "execute_request", "version": "5.3",
        },
        "parent_header": {}, "metadata": {},
        "content": {
            "code": ACTIVITY_TOUCH_CODE + code, "silent": False, "store_history": False,
            "user_expressions": {}, "allow_stdin": False, "stop_on_error": True,
        },
        "channel": "shell",
    }

    output = []
    ok = True
    try:
        import websocket  # from websocket-client
        ws = websocket.create_connection(ws_url, timeout=timeout)
        ws.send(json.dumps(execute_request))
        start = time.time()
        while time.time() - start < timeout:
            msg = json.loads(ws.recv())
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
            content = msg.get("content", {})
            if mtype == "stream":
                output.append(content.get("text", ""))
            elif mtype == "error":
                ok = False
                output.append("\n".join(content.get("traceback", [])))
            elif mtype == "execute_reply":
                if content.get("status") == "error":
                    ok = False
                break
        ws.close()
    except Exception as e:
        return False, f"Kernel execution failed: {e}"
    finally:
        try:
            requests.delete(f"{base}/api/kernels/{kernel_id}", headers=headers, timeout=8)
        except Exception:
            pass

    return ok, ("".join(output).strip() or "(no output)")


def open_kernel():
    """Create a new kernel and return its id WITHOUT deleting it — used for
    the persistent Bond-model kernel, since a loaded model has to survive
    across multiple Send messages (the ad-hoc run_remote() above creates and
    destroys a fresh, empty kernel every single call, which is correct for a
    quick nvidia-smi/ping check but would mean re-downloading and re-loading
    the whole model on every message otherwise)."""
    if not jupyter_url or not jupyter_token:
        return None, "not connected"
    base = jupyter_url.rstrip("/")
    headers = {"Authorization": f"token {jupyter_token}"}
    try:
        r = requests.post(f"{base}/api/kernels", headers=headers, timeout=8)
        r.raise_for_status()
        return r.json()["id"], None
    except Exception as e:
        return None, f"Couldn't reach the Kaggle tunnel: {e}"


def run_on_kernel(kernel_id: str, code: str, timeout: int = 60):
    """Same Jupyter kernel wire protocol as run_remote(), but targets an
    EXISTING kernel_id and never deletes it afterward, so whatever the code
    defines (a loaded model, a tokenizer) stays in memory for the next call."""
    if not jupyter_url or not jupyter_token:
        return False, "not connected"

    base = jupyter_url.rstrip("/")
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/api/kernels/{kernel_id}/channels?token={jupyter_token}"

    msg_id = str(uuid.uuid4())
    execute_request = {
        "header": {
            "msg_id": msg_id, "username": "mi-command-center", "session": str(uuid.uuid4()),
            "msg_type": "execute_request", "version": "5.3",
        },
        "parent_header": {}, "metadata": {},
        "content": {
            "code": ACTIVITY_TOUCH_CODE + code, "silent": False, "store_history": False,
            "user_expressions": {}, "allow_stdin": False, "stop_on_error": True,
        },
        "channel": "shell",
    }

    output = []
    ok = True
    try:
        import websocket
        ws = websocket.create_connection(ws_url, timeout=timeout)
        ws.send(json.dumps(execute_request))
        start = time.time()
        while time.time() - start < timeout:
            msg = json.loads(ws.recv())
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
            content = msg.get("content", {})
            if mtype == "stream":
                output.append(content.get("text", ""))
            elif mtype == "error":
                ok = False
                output.append("\n".join(content.get("traceback", [])))
            elif mtype == "execute_reply":
                if content.get("status") == "error":
                    ok = False
                break
        ws.close()
    except Exception as e:
        return False, f"Kernel execution failed: {e}"

    return ok, ("".join(output).strip() or "(no output)")


def close_kernel(kernel_id: str):
    if not (jupyter_url and jupyter_token and kernel_id):
        return
    base = jupyter_url.rstrip("/")
    headers = {"Authorization": f"token {jupyter_token}"}
    try:
        requests.delete(f"{base}/api/kernels/{kernel_id}", headers=headers, timeout=8)
    except Exception:
        pass


# Runs once, on demand — installs deps, then downloads/loads FOUR ungated
# models in 4-bit onto the persistent kernel (skips Llama-3.1/Gemma-2 from
# the Models-tab candidate list since those are gated on HF and need a
# license-accepted token this Kaggle session doesn't have configured).
# ~4-5GB each in 4-bit x4 fits across the two T4s' 32GB combined via
# device_map="auto". Real download+load time (minutes), not simulated.
# Each model is tried independently so one failure doesn't sink the rest —
# BOND_READY lists what actually loaded, BOND_FAILED lists what didn't.
BOND_MODEL_IDS = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "Zephyr-7b-beta": "HuggingFaceH4/zephyr-7b-beta",
}

# Default model when Bond first loads -- the smallest/fastest of the four,
# so the first message doesn't wait on a 7B download. Switching to a
# different model in the picker loads only THAT one, on demand -- not all
# four upfront, which is what made the first message slow before.
BOND_DEFAULT_MODEL = "Phi-3.5-mini-instruct"


def bond_load_one_code(label: str, model_id: str) -> str:
    """Code sent to the persistent Bond kernel to load exactly one model,
    adding it to that kernel's BOND_MODELS dict (creating it on first call)
    rather than replacing whatever's already loaded there -- so switching
    models in the picker accumulates them on the same kernel instead of
    forcing a reload of ones already fetched this session."""
    return f'''
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate", "bitsandbytes"], check=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if "BOND_MODELS" not in globals():
    BOND_MODELS = {{}}

_bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
)

_label = {label!r}
_model_id = {model_id!r}
try:
    _tok = AutoTokenizer.from_pretrained(_model_id)
    _mod = AutoModelForCausalLM.from_pretrained(_model_id, quantization_config=_bnb, device_map="auto")
    BOND_MODELS[_label] = (_tok, _mod)
    print("BOND_READY:" + _label)
except Exception as _e:
    print("BOND_FAILED:" + _label + "=" + str(_e)[:200])
'''


def bond_tools_setup_code() -> str:
    """Code sent ONCE per kernel, right after it's opened (before any model
    load) -- defines Bond's web-search tool and the tool-calling
    bond_generate() shared by all four models, so it isn't redefined (and
    re-run its pip install) on every single model load.

    web_search uses ddgs (DuckDuckGo search) -- free, keyless, no signup,
    nothing to substitute at push time. Bond's own LLM decides when to call
    it and writes the final answer from the raw result snippets; there's no
    synthesized "answer" field like a paid search API would provide, which
    is the deliberate trade-off for zero credentials/zero billing risk."""
    return r'''
import sys, subprocess, json, re
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ddgs"], check=True)

def web_search(query: str) -> str:
    """
    Search the live internet for current information -- news, prices, facts,
    anything that requires up-to-date knowledge beyond training data.

    Args:
        query: What to search for, as a natural search query.
    """
    from ddgs import DDGS
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return f"No results found for: {query}"
    parts = []
    for item in results:
        parts.append(f"- {item.get('title', '')} ({item.get('href', '')}): "
                      f"{item.get('body', '')[:300]}")
    return "\n".join(parts)

BOND_TOOLS = [web_search]

# Qwen's chat template natively understands apply_chat_template(tools=...)
# and emits <tool_call>...</tool_call> on its own -- verified directly.
# Phi-3.5-mini's template does NOT (verified directly: it just answered
# "I don't have real-time access" instead of ever calling the tool). Rather
# than depending on each of the four models' own template having built-in
# tool-schema support, the exact tag format is taught directly in the
# system prompt instead -- model-agnostic, works whether or not the
# tokenizer's template itself understands `tools=`.
BOND_SYSTEM_PROMPT = (
    "You are Bond 001, a helpful assistant with access to one tool:\n\n"
    "web_search(query: str) -- searches the live internet for current, "
    "real-world information (news, prices, facts, anything needing "
    "up-to-date data beyond your training).\n\n"
    "To use it, respond with EXACTLY this and nothing else:\n"
    '<tool_call>{"name": "web_search", "arguments": {"query": "..."}}</tool_call>\n\n'
    "Use it whenever a question needs current, real-world, or "
    "time-sensitive information you can't be certain of from memory -- "
    "prices, news, scores, current events, specific facts you're not fully "
    "sure of. Don't guess when you can look it up. For anything else (math, "
    "general knowledge, conversation), answer directly without using the "
    "tool."
)

def bond_generate(prompt, model_label, max_new_tokens=90):
    if model_label not in BOND_MODELS:
        return f"[{model_label} isn't loaded on this kernel]"
    tok, mod = BOND_MODELS[model_label]
    messages = [
        {"role": "system", "content": BOND_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    def _ask(msgs):
        # Deliberately NOT passing tools=BOND_TOOLS here. Verified directly:
        # combining tools= with a system message whose CONTENT contains
        # literal {}'s (the <tool_call>{"name": ...} example in
        # BOND_SYSTEM_PROMPT above) intermittently throws a Jinja
        # UndefinedError inside transformers' render_jinja_template on this
        # transformers version -- reproduced 4/4 times on Phi-3.5-mini, but
        # NOT when the system prompt had no braces in it, and NOT when
        # tools= was omitted. Qwen's own tool-use ability doesn't actually
        # depend on the tools= kwarg's extra schema injection -- it's a
        # strong enough instruction-follower to reach for the tool from the
        # system prompt's taught format alone (this was verified live too),
        # so dropping tools= entirely removes the buggy path without losing
        # anything, uniformly across all four models.
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(mod.device)
        out = mod.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    reply = _ask(messages)
    for _ in range(2):
        # Qwen reliably wraps its call in <tool_call>...</tool_call> as
        # instructed. Phi-3.5-mini (verified directly) follows the JSON
        # shape correctly but drops the wrapper tags entirely, emitting just
        # the bare {"name": ..., "arguments": {...}} object -- so a second,
        # tag-less pattern is tried as a fallback rather than requiring the
        # tags, which would otherwise silently return that raw JSON as if it
        # were Bond's final answer instead of ever calling the tool.
        m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", reply, re.DOTALL)
        if not m:
            m = re.search(r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\})', reply, re.DOTALL)
        if not m:
            break
        try:
            call = json.loads(m.group(1))
        except Exception:
            break
        tool_fn = next((t for t in BOND_TOOLS if t.__name__ == call.get("name")), None)
        if not tool_fn:
            break
        try:
            result = str(tool_fn(**call.get("arguments", {})))
        except Exception as _e:
            result = f"Tool error: {_e}"
        # Deliberately NOT using the OpenAI-style {"role": "assistant",
        # "tool_calls": [...]} / {"role": "tool", ...} message shape here.
        # Verified directly: Phi-3.5-mini's chat template has no real
        # understanding of a "tool" role or a content-less "tool_calls"
        # message (at best it errors on the missing content key, at worst it
        # silently fails to treat the result as usable context and just
        # re-emits the same tool call again instead of a final answer).
        # Plain "user"/"assistant" messages are something every chat
        # template, tool-aware or not, already knows how to render
        # correctly -- so the tool round-trip is framed as an ordinary
        # conversation turn instead of depending on tool-schema support.
        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": f"Tool result for {call.get('name')}:\n{result}\n\n"
                       "Now answer my original question using this information. "
                       "Reply in plain language -- don't call the tool again.",
        })
        reply = _ask(messages)
    return reply
'''


def esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


connected = bool(jupyter_url and jupyter_token)

# --- Not connected yet: a plain, honest connect form (same as v2). ----------
if not connected:
    st.title("NVIDIA Intelligent Command Center")
    st.warning(
        "The secure dashboard is available, but the Kaggle GPU connection is "
        "currently offline or not configured. No viewer action is required."
    )
    st.caption("Connection details and compute controls are restricted to the private operator environment.")
    st.stop()

# --- Connected: run the REAL check on every load/refresh, no button needed. -
# A real Settings control lives in Streamlit's own NATIVE sidebar (st.sidebar)
# -- a genuine, fully backend-connected collapsible panel on the left, with
# its own built-in expand/collapse arrow. This is unrelated to the
# decorative left nav inside the dashboard body below (that one renders
# inside components.html()'s iframe, pure static HTML/JS with no connection
# back to this Python process at all -- its own "Settings" icon there is
# just for show and can't host a genuinely working toggle no matter how it's
# styled). st.sidebar sidesteps that limitation entirely since it isn't
# part of that iframe.
if not PUBLIC_VIEWER_MODE:
 with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.checkbox(
        "Show Streamlit Cloud toolbar & footer",
        key="mi_show_chrome",
        help=(
            'Reveals Streamlit\'s own native "⋮" menu up top (which has its own '
            "Settings dialog — wide mode, app theme, and more, separate from "
            'this custom dashboard), the Share/GitHub links, and the "Manage '
            'app" badge (owner-only regardless of this toggle). Hidden by '
            "default for regular viewers."
        ),
    )
    if st.button("Forget connection"):
        st.query_params.clear()
        st.rerun()

with st.spinner("Checking the live Kaggle kernel..."):
    info_ok, info_out = run_remote(
        "import subprocess, json, torch\n"
        "smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout\n"
        "info = {\n"
        "  'cuda_available': torch.cuda.is_available(),\n"
        "  'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,\n"
        "  'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "  'torch_version': torch.__version__,\n"
        "  'smi': smi,\n"
        "}\n"
        "print(json.dumps(info))"
    )

info = None
if info_ok:
    try:
        info = json.loads(info_out)
    except Exception:
        info_ok = False

online = bool(info_ok and info and info.get("cuda_available"))

# --- Auto-wake: if we're offline, trigger the active GPU provider
# automatically -- no button needed. Routed through GPU_PROVIDERS so this
# same block drives whichever provider is active (still only Kaggle for
# real today; Azure/AWS/GCP would plug in here once wired up, no changes
# needed to this logic).
#
# Real fix (2026-09-01): this used to push a fresh trigger purely on a
# 180s cooldown, with no idea whether the *previous* trigger was still
# actually booting. That's a genuine bug, not just a cosmetic gap: Kaggle's
# own boot-to-tunnel time (~1-2 min) is close enough to the old 180s
# cooldown that a slightly slow boot would get hit with ANOTHER push before
# it finished -- restarting the container and never actually reaching the
# tunnel-launch cell. Matches exactly what was observed: "Auto-wake
# triggered" showing while the backend stayed offline indefinitely. Now we
# ask Kaggle's own status first and only push when nothing is already in
# flight.
active_provider = GPU_PROVIDERS.get(ACTIVE_GPU_PROVIDER, GPU_PROVIDERS["kaggle"])
wake_message = None
provider_status, provider_status_raw = ("unknown", "")
provider_error_log = ""
# Read-only status poll (kaggle kernels status -- no push, no GPU-hours
# spent, just asking what the last triggered run is actually doing) runs
# for EVERY viewer, regardless of PUBLIC_VIEWER_MODE -- there's no cost or
# security reason to hide "is it running/queued/complete/error" behind the
# owner gate. Only the ACTIONS below (triggering a real push, pulling the
# crash log) stay owner-gated, unchanged.
if active_provider.is_configured():
    provider_status, provider_status_raw = active_provider.get_status()

if not PUBLIC_VIEWER_MODE and not online and active_provider.is_configured():
    if provider_status in ("running", "queued"):
        # A run is already in flight -- pushing again would restart it, not
        # help it. Just report the real status and wait.
        wake_message = (
            True,
            f'{active_provider.name} already reports status "{provider_status}" from the last '
            "trigger — not pushing again (that would restart the container and lose "
            'progress). Give it a bit longer, then hit "Recheck now."',
        )
    else:
        last_wake = st.session_state.get("last_wake_trigger_at", 0)
        WAKE_COOLDOWN_SECONDS = 240
        if time.time() - last_wake > WAKE_COOLDOWN_SECONDS:
            wake_ok, wake_msg = active_provider.wake()
            st.session_state["last_wake_trigger_at"] = time.time()
            if wake_ok:
                wake_msg += f' (last known status before this push: "{provider_status}")'
            wake_message = (wake_ok, wake_msg)
        else:
            wait_left = int(WAKE_COOLDOWN_SECONDS - (time.time() - last_wake))
            wake_message = (True, f"Already triggered a wake-up recently — give it up to {wait_left}s more, then refresh.")

    if provider_status == "error" and hasattr(active_provider, "get_error_log"):
        provider_error_log = active_provider.get_error_log()

smi_text = (info or {}).get("smi", "") or info_out
driver_m = re.search(r"Driver Version:\s*([\d.]+)", smi_text or "")
cuda_m = re.search(r"CUDA Version:\s*([\d.]+)", smi_text or "")
driver_version = driver_m.group(1) if driver_m else "—"
cuda_version = cuda_m.group(1) if cuda_m else "—"
device_count = (info or {}).get("device_count", 0)
device_name = (info or {}).get("device_name") or "—"
torch_version = (info or {}).get("torch_version") or "—"
compute_cap = "sm_75" if "T4" in (device_name or "") else "—"
checked_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

status_text = "ONLINE" if online else "OFFLINE"
pill_class = "on" if online else "off"
kpi_class = "g" if online else "mu"
gpu_count_name = f"{device_count}× {device_name.replace('Tesla ', '')}" if online else "— offline —"
cuda_kpi = cuda_version if online else "—"

# Kaggle's OWN reported run status ("running"/"queued"/"complete"/"error"/
# "unknown") -- a genuinely different signal from GPU BACKEND ONLINE/OFFLINE
# above, which only says whether the tunnel is currently reachable. This
# tells you WHY it isn't: still booting (queued/running but tunnel not up
# yet), cleanly idle (complete, nothing to wake), or actually broken
# (error). provider_status is computed unconditionally above (read-only
# poll, no PUBLIC_VIEWER_MODE gate) so this is real for every viewer.
_PROVIDER_STATUS_PILL_CLASS = {"running": "on", "queued": "warn", "error": "off"}
provider_status_pill_class = _PROVIDER_STATUS_PILL_CLASS.get(provider_status, "")
provider_status_text = provider_status.upper() if provider_status != "unknown" else "STATUS UNKNOWN"
provider_status_title = (
    f'{active_provider.name} reports "{provider_status}" for the last triggered run'
    if provider_status != "unknown"
    else f"{active_provider.name}'s own status couldn't be checked (not configured, or the check itself failed)"
)

if online:
    sync_html = f"""
      <div class="syncbox-row">
        <span class="pill on"><span class="dot"></span>GPU BACKEND ONLINE — real check, {esc(checked_at)}</span>
        <button class="syncbtn on" id="recheckBtn">Recheck now</button>
      </div>
      <pre style="font-family:var(--mono);font-size:10.5px;color:var(--muted);background:var(--panel2);
        border:1px solid var(--line);border-radius:8px;padding:10px 12px;max-height:160px;overflow:auto;margin:0">{esc(smi_text)[:1200]}</pre>
      <div class="tip">This is a real, live Streamlit server (not a static claude.ai Artifact), so it can actually reach your Kaggle tunnel. The status above comes from an <code>execute_request</code> sent to the live kernel on this exact page load — refresh any time to re-check for real.</div>
    """
else:
    if wake_message is not None:
        wake_ok, wake_msg = wake_message
        wake_html = (
            f'<div class="tip{"" if wake_ok else " warn"}"><b>'
            f'{"🔁 Auto-wake triggered — " if wake_ok else "⚠️ Auto-wake failed — "}</b>{esc(wake_msg)}</div>'
        )
    else:
        wake_html = (
            f'<div class="tip warn">Auto-wake isn\'t configured for {esc(active_provider.name)} — '
            "wake it manually instead.</div>"
        )

    # Real, provider-reported status -- replaces the old "stale tunnel, bad
    # token, or stopped session" guess with what the provider actually says
    # is happening right now.
    status_note_html = ""
    if provider_status != "unknown":
        status_note_html = (
            f'<div class="tip">{esc(active_provider.name)} itself reports status '
            f'<b>"{esc(provider_status)}"</b> for the last triggered run.</div>'
        )
    error_log_html = ""
    if provider_error_log:
        error_log_html = f"""
      <div class="tip warn"><b>Real crash log from the last run (status: error):</b></div>
      <pre style="font-family:var(--mono);font-size:10.5px;color:var(--muted);background:var(--panel2);
        border:1px solid var(--line);border-radius:8px;padding:10px 12px;max-height:220px;overflow:auto;margin:0">{esc(provider_error_log)}</pre>
    """

    sync_html = f"""
      <div class="syncbox-row">
        <span class="pill off"><span class="dot"></span>GPU BACKEND OFFLINE — real check, {esc(checked_at)}</span>
        <button class="syncbtn" id="recheckBtn">Recheck now</button>
      </div>
      <pre style="font-family:var(--mono);font-size:10.5px;color:var(--muted);background:var(--panel2);
        border:1px solid var(--line);border-radius:8px;padding:10px 12px;max-height:160px;overflow:auto;margin:0">{esc(str(info_out))[:1200]}</pre>
      {wake_html}
      {status_note_html}
      {error_log_html}
      <div class="tip warn">Tunnel check failure reason above is usually just "no tunnel yet" while a run boots or before one's been triggered — the status line above (when available) is the real ground truth for what {esc(active_provider.name)} itself is doing, not a guess.</div>
    """

# --- GPU-hour budget card values --------------------------------------------
gpu_hours_used = read_gpu_hours_used()
gpu_hours_pct = min(100, round(gpu_hours_used / GPU_HOUR_BUDGET * 100)) if GPU_HOUR_BUDGET else 0
if gpu_hours_pct >= 90:
    gpu_hours_fill_color = "var(--st-crit)"
    gpu_hours_warning_html = (
        '<div class="tip warn"><b>90%+ of the weekly GPU-hour budget used</b> — '
        "expect to hit Kaggle's real 30hr/week quota soon; pushes may start failing "
        "with a quota error until it resets.</div>"
    )
elif gpu_hours_pct >= 75:
    gpu_hours_fill_color = "var(--st-warn)"
    gpu_hours_warning_html = (
        '<div class="tip warn"><b>75%+ of the weekly GPU-hour budget used.</b></div>'
    )
elif gpu_hours_pct >= 50:
    gpu_hours_fill_color = "var(--st-warn)"
    gpu_hours_warning_html = (
        '<div class="tip"><b>50%+ of the weekly GPU-hour budget used.</b></div>'
    )
else:
    gpu_hours_fill_color = "var(--nv)"
    gpu_hours_warning_html = ""
gpu_hours_note_html = (
    "Kaggle doesn't expose your official quota via API — this tracks actual session "
    "runtime observed via a heartbeat the Kaggle notebook writes to this repo every "
    f"~5 minutes it's running, against a manual default of {GPU_HOUR_BUDGET:g}h/week. "
    "This is this app's own estimate, not Kaggle's official meter."
)

HTML_TEMPLATE = r"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Hanken+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  color-scheme: light;
  --void:#f4f7f2; --panel:#ffffff; --panel2:#eef2ea; --raised:#e6ede0;
  --line:#d9e0d2; --line-soft:#e4ead9;
  --ink:#10140f; --muted:#48594c; --faint:#7c8c7e;
  --nv:#4a7300; --nv-hi:#5b8f00; --nv-dim:#e8f1dd; --nv-glow:rgba(91,143,0,.20);
  --c-blue:#2a78d6; --c-orange:#eb6834; --c-aqua:#1baf7a; --c-yellow:#c98500;
  --st-idle:#7c8c7e; --st-running:#2a78d6; --st-good:#1f9d47; --st-warn:#a8730b; --st-crit:#d13a3f;
  --grad1:#e6eedb; --grad2:#dde6ef;
  --display:'Manrope',-apple-system,'Segoe UI',sans-serif;
  --body:'Hanken Grotesk',-apple-system,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono','SFMono-Regular',ui-monospace,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --void:#06090a; --panel:#0d1310; --panel2:#121a16; --raised:#182720;
    --line:#223129; --line-soft:#182019;
    --ink:#edf2ee; --muted:#93a89c; --faint:#55695f;
    --nv:#76b900; --nv-hi:#a6e000; --nv-dim:#16210a; --nv-glow:rgba(118,185,0,.38);
    --c-blue:#3987e5; --c-orange:#d95926; --c-aqua:#2bb489; --c-yellow:#c98500;
    --st-idle:#55695f; --st-running:#3987e5; --st-good:#2fb356; --st-warn:#d9a53f; --st-crit:#e5484d;
    --grad1:#0d1a0f; --grad2:#0a1420;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --void:#06090a; --panel:#0d1310; --panel2:#121a16; --raised:#182720;
  --line:#223129; --line-soft:#182019;
  --ink:#edf2ee; --muted:#93a89c; --faint:#55695f;
  --nv:#76b900; --nv-hi:#a6e000; --nv-dim:#16210a; --nv-glow:rgba(118,185,0,.38);
  --c-blue:#3987e5; --c-orange:#d95926; --c-aqua:#2bb489; --c-yellow:#c98500;
  --st-idle:#55695f; --st-running:#3987e5; --st-good:#2fb356; --st-warn:#d9a53f; --st-crit:#e5484d;
  --grad1:#0d1a0f; --grad2:#0a1420;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{
  background:
    radial-gradient(1000px 480px at 86% -10%, var(--grad1) 0%, transparent 58%),
    radial-gradient(760px 380px at -6% 108%, var(--grad2) 0%, transparent 55%),
    var(--void);
  color:var(--ink); font-family:var(--body); -webkit-font-smoothing:antialiased;
  display:grid; grid-template-columns:76px 1fr; height:100vh; overflow:hidden;
  transition:background-color .2s,color .2s;
}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#1c2620;border-radius:6px}
a{color:var(--c-blue)}
h1,h2,h3{font-family:var(--display);text-wrap:balance;margin:0}
.mono{font-family:var(--mono)}
@media (prefers-reduced-motion: reduce){*{animation-duration:.001ms !important;transition-duration:.001ms !important}}
nav.rail{background:linear-gradient(180deg,#080b09,#030403);border-right:1px solid var(--line);
  display:flex;flex-direction:column;align-items:center;padding:16px 0 14px;gap:4px;overflow-y:auto}
.mark{width:56px;height:44px;border-radius:11px;margin-bottom:14px;flex:none;position:relative}
.mark img{width:100%;height:100%;display:block;object-fit:contain}
.rail button.nav{width:60px;padding:9px 0;border-radius:11px;border:1px solid transparent;background:transparent;
  color:var(--faint);cursor:pointer;display:grid;place-items:center;gap:4px;position:relative;
  transition:color .15s,background .15s;font-family:var(--body)}
.rail button.nav .ico{width:17px;height:17px}
.rail button.nav .ico svg{width:100%;height:100%;stroke:currentColor;fill:none;stroke-width:1.6}
.rail button.nav .cap{font-size:8px;letter-spacing:.05em;text-transform:uppercase;font-family:var(--mono)}
.rail button.nav:hover{color:var(--muted);background:var(--panel)}
.rail button.nav.active{color:var(--nv-hi);background:var(--panel2);border-color:var(--line)}
.rail button.nav.active::before{content:"";position:absolute;left:-1px;top:9px;bottom:9px;width:3px;border-radius:3px;
  background:var(--nv);box-shadow:0 0 12px var(--nv-glow)}
.rail .spacer{flex:1}
main{overflow:hidden;display:flex;flex-direction:column;min-width:0}
header.top{height:62px;flex:none;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:16px;
  padding:0 clamp(16px,3vw,36px);background:linear-gradient(90deg,var(--panel),transparent)}
header .brand{display:flex;align-items:center;gap:10px}
header .brand img{width:30px;height:30px;object-fit:contain;flex:none}
.nvidia-eye{filter:drop-shadow(0 0 9px var(--nv-glow))}
header .titles{display:flex;flex-direction:column;gap:1px;justify-content:center}
header h1{font-size:15.5px;font-weight:650;letter-spacing:.01em}
header h1 .g{color:var(--nv-hi)}
header .crumb{font-family:var(--mono);font-size:9.5px;color:var(--faint);letter-spacing:.14em;text-transform:uppercase}
header .right{margin-left:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.pill{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:10.5px;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:6px 12px;background:var(--panel)}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--faint);flex:none}
.pill.off .dot{background:var(--st-crit);box-shadow:0 0 7px rgba(229,72,77,.6)}
.pill.on .dot{background:var(--st-good);box-shadow:0 0 7px rgba(47,179,86,.6)}
.pill.warn .dot{background:var(--st-warn);box-shadow:0 0 7px rgba(217,165,63,.6)}
.nvlogo{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:10px;color:var(--faint);
  border-left:1px solid var(--line);padding-left:12px}
.nvlogo b{color:var(--nv);font-family:var(--body);font-weight:700;letter-spacing:.03em}
.themebtn{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:999px;background:var(--panel);
  padding:6px 10px 6px 6px;cursor:pointer;font-family:var(--mono);font-size:10px;color:var(--muted)}
.themebtn .knob{width:26px;height:15px;border-radius:999px;background:var(--line-soft);position:relative;flex:none}
.themebtn .knob i{position:absolute;top:2px;left:2px;width:11px;height:11px;border-radius:50%;background:var(--nv);transition:left .18s}
.themebtn.dark .knob i{left:13px}
.themebtn svg{width:12px;height:12px;stroke:currentColor;fill:none;stroke-width:1.8}
.view{flex:1;overflow-y:auto;padding:26px clamp(16px,3vw,36px) 60px;display:none}
.view.active{display:block}
.lead{max-width:800px;margin:0 0 22px}
.lead h2{font-size:22px;margin:0 0 7px;font-weight:650}
.lead h2 .g{color:var(--nv-hi)}
.lead p{color:var(--muted);font-size:13.5px;margin:0;line-height:1.65}
.group{margin-bottom:30px}
.group-title{display:flex;align-items:center;gap:10px;margin:0 0 13px}
.group-title .bar{width:3px;height:15px;border-radius:3px;background:var(--nv)}
.group-title h3{margin:0;font-size:12px;font-family:var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
.group-title .note{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-left:auto}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:14px}
.kbox{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.kbox .n{font-family:var(--mono);font-size:25px;font-weight:600;font-variant-numeric:tabular-nums}
.kbox .n.g{color:var(--nv-hi)}.kbox .n.b{color:var(--c-blue)}.kbox .n.gr{color:var(--st-good)}.kbox .n.mu{color:var(--faint)}
.kbox .l{font-size:10.5px;color:var(--faint);font-family:var(--mono);margin-top:5px;text-transform:uppercase;letter-spacing:.06em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:15px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:17px;display:flex;flex-direction:column;gap:12px}
.card .head{display:flex;align-items:flex-start;gap:11px}
.badge{width:36px;height:36px;border-radius:9px;flex:none;display:grid;place-items:center;font-family:var(--mono);font-weight:700;font-size:13px;
  background:var(--nv-dim);color:var(--nv-hi);border:1px solid var(--line)}
.card .t{font-weight:600;font-size:14px;line-height:1.3}
.card .s{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:2px}
.statepill{font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:999px;
  border:1px solid var(--line);color:var(--faint);display:inline-flex;align-items:center;gap:5px;margin-left:auto;white-space:nowrap}
.statepill .d{width:6px;height:6px;border-radius:50%;background:currentColor}
.statepill.idle{color:var(--st-idle)}
.statepill.running{color:var(--st-running);border-color:rgba(57,135,229,.35)}
.statepill.good{color:var(--st-good);border-color:rgba(47,179,86,.35)}
.statepill.warn{color:var(--st-warn);border-color:rgba(217,165,63,.35)}
.card .rows{display:flex;flex-direction:column;gap:7px;font-family:var(--mono);font-size:11.5px}
.card .row{display:flex;justify-content:space-between;gap:10px;color:var(--muted)}
.card .row b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.card .foot{margin-top:auto;font-family:var(--mono);font-size:10.5px;color:var(--faint);border-top:1px solid var(--line-soft);padding-top:10px}
.meter{height:7px;border-radius:99px;background:var(--line-soft);overflow:hidden;position:relative}
.meter i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--nv),var(--nv-hi))}
.meter.b i{background:linear-gradient(90deg,#1c5cab,var(--c-blue))}
.syncbox{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-top:12px;display:flex;flex-direction:column;gap:11px}
.syncbox-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.syncbtn{font-family:var(--mono);font-size:10.5px;letter-spacing:.02em;padding:7px 12px;border-radius:8px;border:1px solid var(--line);background:var(--panel2);color:var(--muted);cursor:pointer}
.syncbtn:hover{border-color:var(--nv);color:var(--ink)}
.syncbtn.on{color:var(--st-good);border-color:rgba(47,179,86,.35)}
.tip{font-family:var(--mono);font-size:11px;color:var(--faint);border:1px solid var(--line);border-left:2px solid var(--nv);
  border-radius:7px;padding:10px 14px;background:var(--panel);max-width:800px;line-height:1.65}
.tip.warn{border-left-color:var(--st-warn)}
.tip b{color:var(--muted)}
.bench{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:20px 22px}
.bench h3{font-size:12px;font-family:var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:600;margin-bottom:2px}
.bench .sub{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-bottom:18px}
.barrow{display:grid;grid-template-columns:120px 1fr 96px;align-items:center;gap:14px;margin-bottom:13px}
.barrow:last-child{margin-bottom:0}
.barrow .lbl{font-family:var(--mono);font-size:11.5px;color:var(--muted);text-align:right}
.barrow .lbl b{color:var(--ink);display:block;font-family:var(--body);font-size:12.5px;font-weight:600;text-align:left}
.track{height:20px;border-radius:5px;background:var(--line-soft);position:relative;overflow:hidden}
.track .fill{position:absolute;left:0;top:0;bottom:0;border-radius:5px 3px 3px 5px}
.barrow .val{font-family:var(--mono);font-size:12px;color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.barrow .val small{color:var(--faint);font-size:10px}
.legend-note{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:16px;line-height:1.6}
.tbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px}
.tbl th{text-align:left;color:var(--faint);font-weight:500;text-transform:uppercase;letter-spacing:.06em;font-size:10px;
  padding:0 10px 8px;border-bottom:1px solid var(--line)}
.tbl td{padding:9px 10px;border-bottom:1px solid var(--line-soft);color:var(--muted);font-variant-numeric:tabular-nums}
.tbl td.hi{color:var(--nv-hi);font-weight:600}
.tbl tr:last-child td{border-bottom:none}
.scroll-x{overflow-x:auto}
.topo-wrap{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:8px}
.neural-fabric{position:relative;overflow:hidden;background:linear-gradient(135deg,var(--panel),color-mix(in srgb,var(--panel2) 72%,transparent));border:1px solid var(--line);border-radius:16px;padding:18px 20px 15px;margin:0 0 28px;isolation:isolate}
.neural-fabric::before{content:"";position:absolute;inset:0;background-image:linear-gradient(var(--line-soft) 1px,transparent 1px),linear-gradient(90deg,var(--line-soft) 1px,transparent 1px);background-size:42px 42px;opacity:.24;z-index:-2}
.neural-fabric::after{content:"";position:absolute;width:340px;height:340px;left:50%;top:52%;transform:translate(-50%,-50%);border-radius:50%;background:radial-gradient(circle,var(--nv-glow),transparent 67%);filter:blur(4px);z-index:-1;animation:fabric-breathe 4.8s ease-in-out infinite}
.fabric-head{display:flex;align-items:flex-start;gap:16px;margin-bottom:10px}.fabric-head h3{font-size:17px}.fabric-kicker{font:9px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--faint);margin-top:4px}.fabric-live{margin-left:auto;font:9.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--st-good);white-space:nowrap}.fabric-live::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:currentColor;box-shadow:0 0 9px currentColor;margin-right:7px;animation:live-pulse 1.5s ease-in-out infinite}
.fabric-stage{position:relative;height:260px}.fabric-stage svg{width:100%;height:100%;overflow:visible}.fabric-link{fill:none;stroke:var(--line);stroke-width:1.2}.fabric-flow{fill:none;stroke:url(#signalGradient);stroke-width:2;stroke-linecap:round;stroke-dasharray:7 18;animation:signal-flow 2.1s linear infinite}.fabric-flow.delay{animation-delay:-1.05s}.fabric-node circle.outer{fill:var(--panel);stroke:var(--nv);stroke-width:1.4;filter:drop-shadow(0 0 8px var(--nv-glow))}.fabric-node circle.core{fill:var(--nv);opacity:.88}.fabric-node text{font-family:var(--mono);fill:var(--ink);font-size:10px;text-anchor:middle}.fabric-node text.sub{fill:var(--faint);font-size:7.5px;letter-spacing:.08em}.fabric-node text.live{fill:var(--st-good)}.fabric-node.primary circle.outer{stroke-width:2;animation:node-orbit 3.8s ease-in-out infinite}.fabric-node.primary circle.core{animation:core-pulse 1.8s ease-in-out infinite}.spark{fill:var(--nv-hi);filter:drop-shadow(0 0 5px var(--nv));animation:spark-hop 2.4s ease-in-out infinite}.fabric-caption{display:flex;justify-content:space-between;gap:12px;border-top:1px solid var(--line-soft);padding-top:10px;font:9px var(--mono);color:var(--faint);letter-spacing:.07em;text-transform:uppercase}.fabric-caption b{color:var(--nv-hi);font-weight:500}
@keyframes signal-flow{to{stroke-dashoffset:-50}}
@keyframes fabric-breathe{0%,100%{opacity:.46;transform:translate(-50%,-50%) scale(.86)}50%{opacity:1;transform:translate(-50%,-50%) scale(1.08)}}
@keyframes live-pulse{50%{opacity:.35;transform:scale(.72)}}
@keyframes node-orbit{50%{stroke:var(--nv-hi);filter:drop-shadow(0 0 16px var(--nv-glow))}}
@keyframes core-pulse{50%{r:8;opacity:.42}}
@keyframes spark-hop{0%,100%{opacity:.25}50%{opacity:1}}
@media(max-width:760px){.fabric-stage{height:340px}.fabric-stage svg{transform:scale(1.12)}.fabric-caption{flex-direction:column}.fabric-head{flex-wrap:wrap}.fabric-live{margin-left:0;width:100%}}
figure{margin:0}
figcaption{font-family:var(--mono);font-size:10.5px;color:var(--faint);padding:12px 14px 4px;line-height:1.6}
.node-box{fill:var(--panel2, #121a16)}
::selection{background:var(--nv);color:#0b1400}
.gpuproviders{display:flex;align-items:center;gap:9px;border:1px solid var(--line);border-radius:999px;background:var(--panel);
  padding:6px 12px;cursor:pointer;font-family:var(--mono);font-size:10px;color:var(--muted);transition:border-color .15s}
.gpuproviders:hover{border-color:var(--nv)}
.gplabel{color:var(--faint);letter-spacing:.04em;text-transform:uppercase;font-size:9px;padding-right:8px;border-right:1px solid var(--line)}
.gpitem{display:flex;align-items:center;gap:5px;white-space:nowrap}
.gpitem .gpdot{width:6px;height:6px;border-radius:50%;background:var(--faint);opacity:.45;flex:none}
.gpitem.on .gpdot{background:var(--st-good);opacity:1;box-shadow:0 0 6px rgba(47,179,86,.6)}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(14px);background:var(--raised);
  border:1px solid var(--line);color:var(--ink);font-size:12.5px;padding:12px 18px;border-radius:11px;
  box-shadow:0 16px 40px -10px rgba(0,0,0,.5);opacity:0;pointer-events:none;transition:opacity .25s,transform .25s;
  z-index:90;max-width:min(440px,86vw);text-align:left;line-height:1.55}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast b{color:var(--nv-hi)}
</style>

<nav class="rail">
  <button class="nav active" data-view="overview"><span class="ico"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/></svg></span><span class="cap">Overview</span></button>
  <button class="nav" data-view="topology"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="2.3"/><circle cx="19" cy="6" r="2.3"/><circle cx="19" cy="18" r="2.3"/><path d="M7 12 L16.7 7 M7 12 L16.7 17"/></svg></span><span class="cap">Topology</span></button>
  <button class="nav" data-view="gpu"><span class="ico"><svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 6V3M12 6V3M17 6V3M7 21v-3M12 21v-3M17 21v-3"/></svg></span><span class="cap">GPU</span></button>
  <button class="nav" data-view="models"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="5" r="1.6"/><circle cx="20" cy="5" r="1.6"/><circle cx="4" cy="19" r="1.6"/><circle cx="20" cy="19" r="1.6"/><path d="M9.6 10.2 5.2 6M14.4 10.2 18.8 6M9.6 13.8 5.2 18M14.4 13.8 18.8 18"/></svg></span><span class="cap">Models</span></button>
  <button class="nav" data-view="agents"><span class="ico"><svg viewBox="0 0 24 24"><rect x="4" y="4" width="7" height="7" rx="1.4"/><rect x="13" y="4" width="7" height="7" rx="1.4"/><rect x="4" y="13" width="7" height="7" rx="1.4"/><rect x="13" y="13" width="7" height="7" rx="1.4"/></svg></span><span class="cap">Agents</span></button>
  <button class="nav" data-view="tokens"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5v9M9 10h4.2a1.8 1.8 0 0 1 0 3.6H9m2-7v1.2m0 9.6V17.4"/></svg></span><span class="cap">Tokens</span></button>
  <div class="spacer"></div>
  <button class="nav" data-view="settings"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></span><span class="cap">Settings</span></button>
  <button class="nav" data-view="about"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.1"/></svg></span><span class="cap">About</span></button>
</nav>

<main>
<header class="top">
  <div class="brand">
    <img class="nvidia-eye" src="{{NVIDIA_ICON_DATA_URI}}" alt="NVIDIA"/>
    <div class="titles">
      <h1>NVIDIA <span class="g">Intelligent Cloud Control</span></h1>
      <div class="crumb">nvidia · cuda · agentic gpu infrastructure</div>
    </div>
  </div>
  <div class="right">
    <span class="pill {{PILL_CLASS}}" id="headerGpuPill"><span class="dot"></span><span id="headerGpuText">GPU BACKEND {{STATUS_TEXT}}</span></span>
    <span class="pill {{PROVIDER_STATUS_PILL_CLASS}}" title="{{PROVIDER_STATUS_TITLE}}"><span class="dot"></span>KAGGLE SESSION {{PROVIDER_STATUS_TEXT}}</span>
    <span class="pill"><span class="dot"></span>APP-TRACKED GPU HOURS {{GPU_HOURS_USED}}/{{GPU_HOURS_BUDGET}}h</span>
    <div class="gpuproviders" id="gpuProviders" role="button" tabindex="0" aria-label="GPU backend providers" title="GPU backend providers">
      <span class="gplabel">NVIDIA GPU</span>
      <span class="gpitem {{PILL_CLASS}}"><i class="gpdot"></i>Kaggle</span>
      <span class="gpitem off"><i class="gpdot"></i>AWS</span>
      <span class="gpitem off"><i class="gpdot"></i>Azure</span>
      <span class="gpitem off"><i class="gpdot"></i>GCP</span>
    </div>
    <button class="themebtn" id="themeBtn" aria-label="Toggle bright / dark mode" title="Toggle bright / dark mode">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v3M12 18.5v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2.5 12h3M18.5 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>
      <span class="knob"><i></i></span>
      <svg viewBox="0 0 24 24"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4 7 7 0 0 0 20 14.5Z"/></svg>
    </button>
    <div class="nvlogo"><b>NVIDIA</b>CUDA {{CUDA_KPI}}</div>
  </div>
</header>

<section class="view active" id="overview">
  <div class="lead">
    <h2><span class="g">NVIDIA</span> Intelligent Cloud Control</h2>
    <p>A control plane for a Kaggle-hosted GPU rig, wired up from a Mac via VS Code — real CUDA, real PyTorch, an agentic layer being built on top. This view shows what's actually confirmed working versus what's still a placeholder; nothing here is simulated data dressed up as live telemetry. The GPU status below is a live server checking your kernel right now, not a static demo.</p>
  </div>

  <div class="neural-fabric" aria-label="Animated NVIDIA neural compute fabric">
    <div class="fabric-head">
      <div><h3>NVIDIA Neural Compute Fabric</h3><div class="fabric-kicker">intent → orchestration → accelerated intelligence → outcomes</div></div>
      <div class="fabric-live">fabric {{STATUS_TEXT_LOWER}}</div>
    </div>
    <div class="fabric-stage">
      <svg viewBox="0 0 1000 260" role="img" aria-label="Live animated neural network linking the dashboard, control plane, CUDA cores, AI models, agents, and application outputs">
        <defs><linearGradient id="signalGradient" x1="0" x2="1"><stop offset="0" stop-color="var(--c-aqua)"/><stop offset=".5" stop-color="var(--nv-hi)"/><stop offset="1" stop-color="var(--c-blue)"/></linearGradient></defs>
        <g>
          <path class="fabric-link" d="M90 130C175 130 195 70 275 70S390 130 500 130 625 60 720 60 830 105 915 105"/>
          <path class="fabric-link" d="M90 130C175 130 195 190 275 190S390 130 500 130 625 200 720 200 830 155 915 155"/>
          <path class="fabric-link" d="M275 70C360 70 395 200 500 130M275 190C360 190 395 60 500 130M720 60C790 60 835 155 915 155M720 200C790 200 835 105 915 105"/>
          <path class="fabric-flow" d="M90 130C175 130 195 70 275 70S390 130 500 130 625 60 720 60 830 105 915 105"/>
          <path class="fabric-flow delay" d="M90 130C175 130 195 190 275 190S390 130 500 130 625 200 720 200 830 155 915 155"/>
          <circle class="spark" cx="186" cy="89" r="3"/><circle class="spark" cx="392" cy="151" r="3" style="animation-delay:-.8s"/><circle class="spark" cx="616" cy="90" r="3" style="animation-delay:-1.5s"/><circle class="spark" cx="820" cy="129" r="3" style="animation-delay:-2s"/>
        </g>
        <g class="fabric-node" transform="translate(90 130)"><circle class="outer" r="39"/><circle class="core" r="5"/><text y="-7">STREAMLIT</text><text class="sub live" y="18">LIVE INPUT</text></g>
        <g class="fabric-node" transform="translate(275 70)"><circle class="outer" r="37"/><circle class="core" r="5"/><text y="-7">CONTROL</text><text class="sub" y="18">JUPYTER</text></g>
        <g class="fabric-node" transform="translate(275 190)"><circle class="outer" r="37"/><circle class="core" r="5"/><text y="-7">KAGGLE</text><text class="sub" y="18">GPU RUNTIME</text></g>
        <g class="fabric-node primary" transform="translate(500 130)"><circle class="outer" r="52"/><circle class="core" r="6"/><text y="-9" style="font-size:12px;font-weight:600">CUDA CORE</text><text class="sub" y="17">TESLA T4 × 2</text></g>
        <g class="fabric-node" transform="translate(720 60)"><circle class="outer" r="37"/><circle class="core" r="5"/><text y="-7">MODELS</text><text class="sub" y="18">4-BIT LLM</text></g>
        <g class="fabric-node" transform="translate(720 200)"><circle class="outer" r="37"/><circle class="core" r="5"/><text y="-7">AGENTS</text><text class="sub" y="18">ORCHESTRATE</text></g>
        <g class="fabric-node" transform="translate(915 105)"><circle class="outer" r="33"/><circle class="core" r="4"/><text y="-6">BOND 001</text><text class="sub" y="16">INFERENCE</text></g>
        <g class="fabric-node" transform="translate(915 155)"><circle class="outer" r="33"/><circle class="core" r="4"/><text y="-6">INSIGHTS</text><text class="sub" y="16">OUTPUT</text></g>
      </svg>
    </div>
    <div class="fabric-caption"><span><b>signals in motion</b> · requests flow across the fabric</span><span>real topology · status-aware · GPU accelerated</span></div>
  </div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>System status</h3><span class="note">last checked {{CHECKED_AT}}</span></div>
    <div class="kpi">
      <div class="kbox"><div class="n {{KPI_CLASS}}" id="kpiGpuBackend">{{STATUS_TEXT}}</div><div class="l">GPU backend</div></div>
      <div class="kbox"><div class="n g">{{GPU_COUNT_NAME}}</div><div class="l">Kaggle GPUs (live)</div></div>
      <div class="kbox"><div class="n b">{{CUDA_KPI}}</div><div class="l">CUDA toolkit</div></div>
      <div class="kbox"><div class="n gr">6.7×</div><div class="l">Best kernel speedup</div></div>
      <div class="kbox"><div class="n mu">0.0 / 30</div><div class="l">GPU hours (app-tracked)</div></div>
    </div>

    <div class="syncbox" id="syncbox">
      {{SYNC_HTML}}
    </div>
  </div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>What's real vs. pending</h3></div>
    <div class="grid">
      <div class="card">
        <div class="head"><div class="badge">✓</div><div><div class="t">CUDA + PyTorch verified</div><div class="s">live kernel · torch {{TORCH_VERSION}}</div></div><span class="statepill good"><span class="d"></span>confirmed</span></div>
        <div class="rows">
          <div class="row">torch<b>{{TORCH_VERSION}}</b></div>
          <div class="row">cuda.is_available<b>{{CUDA_AVAILABLE}}</b></div>
          <div class="row">device count<b>{{DEVICE_COUNT}}</b></div>
          <div class="row">device name<b>{{DEVICE_NAME}}</b></div>
        </div>
      </div>
      <div class="card">
        <div class="head"><div class="badge">✓</div><div><div class="t">Kernel benchmark verified</div><div class="s">matmul_v2.cu · 1024³ GEMM</div></div><span class="statepill good"><span class="d"></span>confirmed</span></div>
        <div class="rows">
          <div class="row">register-tiled vs naive<b>6.7×</b></div>
          <div class="row">register-tiled vs cuBLAS<b>2.77×</b></div>
          <div class="row">correctness check<b>passed</b></div>
        </div>
        <div class="foot">full breakdown → GPU &amp; CUDA tab</div>
      </div>
      <div class="card">
        <div class="head"><div class="badge">4</div><div><div class="t">Pick a model, it loads on demand</div><div class="s">4-bit, bitsandbytes</div></div><span class="statepill good"><span class="d"></span>on demand</span></div>
        <div class="rows">
          <div class="row">models<b>Qwen2.5-7B · Phi-3.5-mini · Mistral-7B · Zephyr-7B</b></div>
          <div class="row">default<b>Phi-3.5-mini (fastest)</b></div>
          <div class="row">install + load<b>only the picked model, not all four</b></div>
        </div>
        <div class="foot">click the Bond 001 button (bottom-right), pick a model from the dropdown — only that one loads, ready by the time you reach the chat box</div>
      </div>
      <div class="card">
        <div class="head"><div class="badge">…</div><div><div class="t">Agent orchestration</div><div class="s">6 planned agents</div></div><span class="statepill idle"><span class="d"></span>idle</span></div>
        <div class="rows">
          <div class="row">framework<b>hand-built</b></div>
          <div class="row">status<b>design done, code pending</b></div>
        </div>
        <div class="foot">see Agents tab for the planned roster</div>
      </div>
    </div>
  </div>

  <div class="tip"><b>Why this can go offline —</b> Kaggle sessions aren't always-on; the GPU backend only exists while a session + tunnel are actively running. When the tunnel's down, this page says OFFLINE honestly rather than showing a stale cached "online."</div>
</section>

<section class="view" id="topology">
  <div class="lead">
    <h2>System <span class="g">topology</span></h2>
    <p>This dashboard is now the live path itself — a real Streamlit server checking the kernel directly. Claude Code remains the one driving GitHub/Kaggle setup on the Mac; this page just talks to the kernel on your behalf whenever it's open.</p>
  </div>

  <div class="topo-wrap">
    <figure>
      <svg viewBox="0 0 920 360" role="img" aria-label="Topology diagram: this Streamlit dashboard connects directly through an ngrok tunnel to a Jupyter server on Kaggle, which drives the Tesla T4 GPUs running CUDA and PyTorch. A GitHub repository holds the source of truth and deploys to Streamlit Community Cloud, which is this page.">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
          </marker>
        </defs>
        <g font-family="IBM Plex Mono, monospace" font-size="11" fill="currentColor" opacity=".95">
          <rect x="20" y="60" width="150" height="64" rx="9" class="node-box" stroke="var(--nv)" stroke-width="1.3"/>
          <text x="95" y="86" text-anchor="middle" font-weight="600" font-size="12">This dashboard</text>
          <text x="95" y="103" text-anchor="middle" fill-opacity=".65">Streamlit Cloud, live</text>

          <line x1="170" y1="92" x2="288" y2="92" stroke="var(--nv)" stroke-width="1.6" marker-end="url(#arrow)"/>
          <text x="229" y="82" text-anchor="middle" fill="var(--nv-hi)" font-size="10">ngrok tunnel</text>

          <rect x="290" y="60" width="160" height="64" rx="9" class="node-box" stroke="var(--nv)" stroke-width="1.3"/>
          <text x="370" y="86" text-anchor="middle" font-weight="600" font-size="12">Kaggle notebook</text>
          <text x="370" y="103" text-anchor="middle" fill-opacity=".65">standalone Jupyter server</text>

          <line x1="450" y1="92" x2="568" y2="92" stroke="var(--nv)" stroke-width="1.6" marker-end="url(#arrow)"/>
          <text x="509" y="82" text-anchor="middle" fill="var(--nv-hi)" font-size="10">execute_request</text>

          <rect x="570" y="60" width="170" height="64" rx="9" class="node-box" stroke="var(--nv)" stroke-width="1.3"/>
          <text x="655" y="82" text-anchor="middle" font-weight="600" font-size="12">Tesla T4 × 2</text>
          <text x="655" y="98" text-anchor="middle" fill-opacity=".65">CUDA · PyTorch</text>
          <text x="655" y="113" text-anchor="middle" fill="var(--nv-hi)" font-size="9.5">{{STATUS_TEXT_LOWER}}</text>

          <rect x="20" y="180" width="150" height="60" rx="9" fill="none" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="4 3"/>
          <text x="95" y="205" text-anchor="middle" fill-opacity=".7" font-size="11">Mac + Claude Code</text>
          <text x="95" y="221" text-anchor="middle" fill-opacity=".5">develops the code</text>

          <line x1="170" y1="205" x2="288" y2="205" stroke="currentColor" stroke-opacity=".45" stroke-width="1.3" marker-end="url(#arrow)"/>
          <text x="229" y="195" text-anchor="middle" fill-opacity=".6" font-size="9.5">git push</text>

          <rect x="290" y="180" width="150" height="60" rx="9" class="node-box" stroke="currentColor" stroke-opacity=".4"/>
          <text x="365" y="205" text-anchor="middle" font-weight="600" font-size="12">GitHub repo</text>
          <text x="365" y="221" text-anchor="middle" fill-opacity=".65">source of truth</text>

          <line x1="440" y1="210" x2="568" y2="205" stroke="var(--c-blue)" stroke-width="1.6" marker-end="url(#arrow)"/>
          <text x="504" y="196" text-anchor="middle" fill="var(--c-blue)" font-size="10">auto-deploys</text>

          <rect x="570" y="180" width="170" height="60" rx="9" class="node-box" stroke="var(--c-blue)" stroke-width="1.3"/>
          <text x="655" y="202" text-anchor="middle" font-weight="600" font-size="12">Streamlit Cloud</text>
          <text x="655" y="218" text-anchor="middle" fill-opacity=".65">this page, always-on</text>
        </g>
      </svg>
      <figcaption>Every box on this diagram is real: this page → ngrok → Kaggle's Jupyter kernel → the T4 GPUs is the live path checked on every refresh. Separately, code changes flow Mac/Claude Code → GitHub → Streamlit Cloud, which auto-redeploys this exact page.</figcaption>
    </figure>
  </div>

  <div class="group" style="margin-top:26px">
    <div class="group-title"><span class="bar"></span><h3>Why Claude Code still does the deploy-side work</h3></div>
    <div class="tip">Claude Cowork (the cloud agent that designs and edits this dashboard's code) has no general internet access from its own sandbox, and the connected-folder device bridge to this Mac was confirmed to be separately network-restricted too — so neither can push to GitHub or test against the live tunnel directly. Claude Code, running locally in VS Code, has real shell + internet access and is the one that actually pushes commits and lets Streamlit Cloud pick them up.</div>
  </div>
</section>

<section class="view" id="gpu">
  <div class="lead">
    <h2>GPU <span class="g">&amp; CUDA</span></h2>
    <p>Hardware detection is live (checked {{CHECKED_AT}}). The benchmark numbers below are a real one-time run captured 2026-08-26 through the kernel bridge — not sample data, but not re-run on every refresh either (that would burn GPU-hours for no reason).</p>
  </div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Detected hardware</h3><span class="note">live, this page load</span></div>
    <div class="kpi">
      <div class="kbox"><div class="n {{KPI_CLASS}}">{{DEVICE_NAME}}</div><div class="l">GPU model</div></div>
      <div class="kbox"><div class="n b">{{DEVICE_COUNT}}</div><div class="l">Device count</div></div>
      <div class="kbox"><div class="n">{{CUDA_KPI}}</div><div class="l">CUDA version</div></div>
      <div class="kbox"><div class="n">{{TORCH_VERSION}}</div><div class="l">PyTorch</div></div>
      <div class="kbox"><div class="n mu">{{COMPUTE_CAP}}</div><div class="l">Compute capability</div></div>
    </div>
  </div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Telemetry field schema</h3><span class="note">not yet collecting — naming decided</span></div>
    <div class="tip">Rather than inventing our own metric names, the telemetry collector (Phase 1, not yet built) will read the same fields <a href="https://github.com/NVIDIA/dcgm-exporter" target="_blank" rel="noopener">NVIDIA's own dcgm-exporter</a> exposes — the tool behind Grafana's official <a href="https://grafana.com/grafana/dashboards/12239-nvidia-dcgm-exporter-dashboard/" target="_blank" rel="noopener">NVIDIA DCGM Exporter dashboard</a>. Collected via <span class="mono">pynvml</span> (DCGM itself isn't installable on Kaggle's containers), but named identically — so this data could be pushed into a real Grafana/Grafana&nbsp;Cloud instance later without a rename.</div>
    <div class="scroll-x" style="margin-top:14px">
      <table class="tbl">
        <thead><tr><th>Field (DCGM-aligned)</th><th>Meaning</th><th>Source here</th></tr></thead>
        <tbody>
          <tr><td>DCGM_FI_DEV_GPU_UTIL</td><td>GPU utilization %</td><td>pynvml</td></tr>
          <tr><td>DCGM_FI_DEV_FB_USED</td><td>framebuffer (VRAM) used</td><td>pynvml</td></tr>
          <tr><td>DCGM_FI_DEV_GPU_TEMP</td><td>GPU temperature</td><td>pynvml</td></tr>
          <tr><td>DCGM_FI_DEV_POWER_USAGE</td><td>power draw (W)</td><td>pynvml</td></tr>
          <tr><td>DCGM_FI_DEV_SM_CLOCK</td><td>streaming-multiprocessor clock</td><td>pynvml</td></tr>
          <tr><td>DCGM_FI_PROF_PIPE_TENSOR_ACTIVE</td><td>tensor-core activity (0-1)</td><td>pynvml (approx.)</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="tip warn"><b>GPU-hour discipline —</b> reading these fields is essentially instant and costs nothing meaningful; what actually burns the 30 hr/week Kaggle budget is running compute (benchmarks, model inference). So telemetry gets captured around real workloads — never a background daemon polling continuously just to keep a chart moving.</div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Matrix-multiply benchmark</h3><span class="note">1024×1024×1024 GEMM · 20 iterations · matmul_v2.cu · captured 2026-08-26</span></div>
    <div class="bench">
      <h3>Throughput by kernel</h3>
      <div class="sub">log-scaled bars — the gap from naive to cuBLAS spans over 18×</div>
      <div class="barrow">
        <div class="lbl"><b>Naive</b>unoptimized</div>
        <div class="track"><div class="fill" style="width:29%;background:var(--c-blue)"></div></div>
        <div class="val">346.7<small> GFLOP/s</small></div>
      </div>
      <div class="barrow">
        <div class="lbl"><b>Tiled (v1)</b>shared memory</div>
        <div class="track"><div class="fill" style="width:32%;background:var(--c-orange)"></div></div>
        <div class="val">379.9<small> GFLOP/s</small></div>
      </div>
      <div class="barrow">
        <div class="lbl"><b>Register-tiled (v2)</b>our best kernel</div>
        <div class="track"><div class="fill" style="width:71%;background:var(--c-aqua)"></div></div>
        <div class="val">2311.7<small> GFLOP/s</small></div>
      </div>
      <div class="barrow">
        <div class="lbl"><b>cuBLAS</b>vendor reference</div>
        <div class="track"><div class="fill" style="width:100%;background:var(--c-yellow)"></div></div>
        <div class="val">6396.4<small> GFLOP/s</small></div>
      </div>
      <div class="legend-note">Register-tiling: <b style="color:var(--nv-hi)">6.7×</b> faster than naive, closing the gap to cuBLAS from ~16.8× (v1 tiled) down to <b style="color:var(--nv-hi)">~2.77×</b>. Verified correct — max difference vs. cuBLAS 0.00055, within float tolerance.</div>
    </div>
  </div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Run log</h3></div>
    <div class="scroll-x">
      <table class="tbl">
        <thead><tr><th>Kernel</th><th>Time (ms)</th><th>GFLOP/s</th><th>vs. naive</th><th>vs. cuBLAS</th></tr></thead>
        <tbody>
          <tr><td>Naive</td><td>6.19</td><td>346.7</td><td>1.00×</td><td>0.054×</td></tr>
          <tr><td>Tiled (v1)</td><td>5.65</td><td>379.9</td><td>1.10×</td><td>0.059×</td></tr>
          <tr><td class="hi">Register-tiled (v2)</td><td class="hi">0.93</td><td class="hi">2311.7</td><td class="hi">6.67×</td><td class="hi">0.361×</td></tr>
          <tr><td>cuBLAS</td><td>0.34</td><td>6396.4</td><td>18.45×</td><td>1.00×</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<section class="view" id="models">
  <div class="lead">
    <h2>Model <span class="g">inference</span></h2>
    <p>The local LLM under test on this rig. Nothing here is live yet — shown as the honest pending state rather than mocked numbers.</p>
  </div>
  <div class="group">
    <div class="grid">
      <div class="card">
        <div class="head"><div class="badge">φ</div><div><div class="t">Phi-3-mini-4k-instruct</div><div class="s">microsoft/phi-3-mini-4k-instruct</div></div><span class="statepill warn"><span class="d"></span>pending</span></div>
        <div class="rows">
          <div class="row">quantization<b>4-bit (bitsandbytes)</b></div>
          <div class="row">compute dtype<b>float16</b></div>
          <div class="row">target device<b>cuda:0</b></div>
          <div class="row">parameters<b>3.8B</b></div>
        </div>
        <div class="foot">install → load → generate: none of the three steps have run yet</div>
      </div>
      <div class="card">
        <div class="head"><div class="badge">＋</div><div><div class="t">Next model candidate</div><div class="s">not yet selected</div></div><span class="statepill idle"><span class="d"></span>idle</span></div>
        <div class="rows">
          <div class="row">candidates<b>Llama, Mistral</b></div>
          <div class="row">blocker<b>license accept step</b></div>
        </div>
        <div class="foot">gated on HuggingFace, deferred until Phi-3 baseline works</div>
      </div>
    </div>
  </div>
  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>What this tab will show once live</h3></div>
    <div class="tip">Prompt/response pane, tokens-in, tokens-out, tokens/sec, latency, and GPU memory consumed per request — each pulled from a real generation call through the kernel bridge, not simulated.</div>
  </div>
  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Candidate models for Bond 001</h3><span class="note">pick one in the dropdown — only that one loads · 2 excluded (gated)</span></div>
    <div class="grid">
      <div class="card"><div class="head"><div class="badge">Q</div><div><div class="t">Qwen2.5-7B-Instruct</div><div class="s">7B · Alibaba · ungated · strong tool-use</div></div><span class="statepill good"><span class="d"></span>pick to load</span></div><div class="foot">Strong at tool-calling for the agent layer</div></div>
      <div class="card"><div class="head"><div class="badge">Φ</div><div><div class="t">Phi-3.5-mini-instruct</div><div class="s">3.8B · microsoft · ungated</div></div><span class="statepill warn"><span class="d"></span>default</span></div><div class="foot">Smallest/fastest — loaded automatically unless you pick a different one</div></div>
      <div class="card"><div class="head"><div class="badge">M</div><div><div class="t">Mistral-7B-Instruct-v0.3</div><div class="s">7B · ungated · function-calling</div></div><span class="statepill good"><span class="d"></span>pick to load</span></div><div class="foot">Loads on its own once picked, alongside whatever's already loaded on this kernel</div></div>
      <div class="card"><div class="head"><div class="badge">Z</div><div><div class="t">Zephyr-7b-beta</div><div class="s">7B · ungated · easy fallback</div></div><span class="statepill good"><span class="d"></span>pick to load</span></div><div class="foot">Loads on its own once picked, alongside whatever's already loaded on this kernel</div></div>
      <div class="card"><div class="head"><div class="badge">L</div><div><div class="t">Llama-3.1-8B-Instruct</div><div class="s">8B · Meta · gated — accept license on HF</div></div><span class="statepill idle"><span class="d"></span>excluded for now</span></div><div class="foot">Needs a license-accepted HF token configured on the Kaggle kernel first — not wired up yet</div></div>
      <div class="card"><div class="head"><div class="badge">G</div><div><div class="t">Gemma-2-9b-it</div><div class="s">9B · Google · gated — accept license on HF</div></div><span class="statepill idle"><span class="d"></span>excluded for now</span></div><div class="foot">Needs a license-accepted HF token configured on the Kaggle kernel first — not wired up yet</div></div>
    </div>
    <div class="tip">Bond's floating panel (bottom-right) loads only the model you pick from its dropdown, not all four — Phi-3.5-mini loads by default since it's fastest. Picking a different model loads it fresh; models already loaded this session stay in memory on the same kernel, so switching back to one you already used is instant. The two gated models are left out until a licensed HF token is set up on the Kaggle side.</div>
  </div>
</section>

<section class="view" id="agents">
  <div class="lead">
    <h2>Agent <span class="g">roster</span></h2>
    <p>Six planned agents, designed but not yet coded. Shown idle rather than invented "running" states.</p>
  </div>
  <div class="group">
    <div class="grid">
      <div class="card"><div class="head"><div class="badge">O</div><div><div class="t">Orchestrator</div><div class="s">task routing</div></div><span class="statepill idle"><span class="d"></span>idle</span></div><div class="rows"><div class="row">receives requests, assigns tasks, collects results</div></div></div>
      <div class="card"><div class="head"><div class="badge">C</div><div><div class="t">CUDA Agent</div><div class="s">hardware + kernels</div></div><span class="statepill idle"><span class="d"></span>idle</span></div><div class="rows"><div class="row">detects GPUs, runs benchmarks, reports metrics</div></div></div>
      <div class="card"><div class="head"><div class="badge">M</div><div><div class="t">Model Agent</div><div class="s">inference</div></div><span class="statepill idle"><span class="d"></span>idle</span></div><div class="rows"><div class="row">loads models, runs generation, measures tokens/sec</div></div></div>
      <div class="card"><div class="head"><div class="badge">P</div><div><div class="t">Performance Agent</div><div class="s">telemetry</div></div><span class="statepill idle"><span class="d"></span>idle</span></div><div class="rows"><div class="row">GPU/VRAM utilization, latency, throughput</div></div></div>
      <div class="card"><div class="head"><div class="badge">X</div><div><div class="t">Optimization Agent</div><div class="s">autonomous tuning</div></div><span class="statepill idle"><span class="d"></span>idle</span></div><div class="rows"><div class="row">tests batch size / precision, compares vs. baseline</div></div></div>
      <div class="card"><div class="head"><div class="badge">R</div><div><div class="t">Report Agent</div><div class="s">summaries</div></div><span class="statepill idle"><span class="d"></span>idle</span></div><div class="rows"><div class="row">writes experiment + bottleneck summaries</div></div></div>
    </div>
  </div>
</section>

<section class="view" id="tokens">
  <div class="lead">
    <h2>Tokens <span class="g">&amp; GPU hours</span></h2>
    <p>Accounting for two different budgets — model tokens and Kaggle GPU time — kept separate and honestly labeled.</p>
  </div>
  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>GPU-hour budget</h3><span class="note">application-tracked, not Kaggle's official meter</span></div>
    <div class="bench" style="max-width:640px">
      <div class="barrow" style="grid-template-columns:110px 1fr 100px">
        <div class="lbl"><b>Used</b>this week</div>
        <div class="track"><div class="fill" style="width:{{GPU_HOURS_PCT}}%;background:{{GPU_HOURS_FILL_COLOR}}"></div></div>
        <div class="val">{{GPU_HOURS_USED}}<small> hrs</small></div>
      </div>
      <div class="barrow" style="grid-template-columns:110px 1fr 100px">
        <div class="lbl"><b>Budget</b>configurable</div>
        <div class="track"><div class="fill" style="width:100%;background:var(--line)"></div></div>
        <div class="val">{{GPU_HOURS_BUDGET}}<small> hrs</small></div>
      </div>
      <div class="legend-note">{{GPU_HOURS_NOTE}}</div>
      {{GPU_HOURS_WARNING_HTML}}
    </div>
  </div>
  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Token accounting schema</h3><span class="note">per-request record, not yet populated</span></div>
    <div class="scroll-x">
      <table class="tbl">
        <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td>input_tokens</td><td>tokens in the prompt</td></tr>
          <tr><td>output_tokens</td><td>tokens generated</td></tr>
          <tr><td>total_tokens</td><td>input + output</td></tr>
          <tr><td>latency_ms</td><td>wall-clock time for the request</td></tr>
          <tr><td>tokens_per_second</td><td>output_tokens ÷ (latency_ms / 1000)</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<section class="view" id="settings">
  <div class="lead">
    <h2><span class="g">Settings</span></h2>
    <p>This nav is decorative HTML with no direct connection back to the Python server — but the two buttons below still genuinely work, by reloading the page with the change encoded in the URL, the same channel the connect form already uses.</p>
  </div>
  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Platform chrome &amp; connection</h3></div>
    <div class="card" style="max-width:520px">
      <div class="head">
        <div><div class="t">⚙️ Toolbar &amp; footer, connection</div>
        <div class="s">Real controls — reload the page to apply, same as the sidebar versions</div></div>
      </div>
      <div class="syncbox-row">
        <button class="syncbtn" onclick="window.miTriggerAction('{{CHROME_ACTION}}')">{{CHROME_BTN_LABEL}}</button>
        <button class="syncbtn" onclick="if(confirm('Forget this connection and disconnect?')) window.miTriggerAction('forget')">Forget connection</button>
      </div>
      <div class="tip">Same effect as the sidebar's Settings panel (open it on the left edge of the browser to see them without a reload) — turning the toolbar on also reveals Streamlit's own native "⋮" menu, which has its own Settings dialog (wide mode, app theme, and more) separate from this custom page.</div>
    </div>
    <div class="card" style="max-width:520px;margin-top:14px">
      <div class="head">
        <div><div class="t">"Manage app" floating button</div>
        <div class="s">Streamlit Cloud's own overlay — not controlled by this app</div></div>
      </div>
      <div class="tip">Same category as the sidebar toggle above: platform chrome that lives outside this app's own page entirely, so nothing in <span class="mono">app.py</span> can move, style, or hide it. It's Streamlit Community Cloud's own owner/collaborator control, visible only when the app's Streamlit account owner (or a collaborator) is logged in and viewing the app in a browser — regular anonymous visitors never see it at all.</div>
    </div>
  </div>

  <div class="group">
    <div class="group-title"><span class="bar"></span><h3>Header status, mirrored here</h3><span class="note">so it's not stranded only in the top bar</span></div>
    <div class="card" style="max-width:520px">
      <div class="head">
        <div><div class="t">Theme</div>
        <div class="s">Real, working control — pure client-side, no backend needed</div></div>
      </div>
      <button class="themebtn" id="themeBtn2" aria-label="Toggle bright / dark mode" title="Toggle bright / dark mode" style="width:fit-content">
        <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v3M12 18.5v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2.5 12h3M18.5 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>
        <span class="knob"><i></i></span>
        <svg viewBox="0 0 24 24"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4 7 7 0 0 0 20 14.5Z"/></svg>
      </button>
    </div>
    <div class="card" style="max-width:520px;margin-top:14px">
      <div class="head">
        <div><div class="t">GPU backend &amp; CUDA</div>
        <div class="s">Real data, mirrored from the header — not a control either place</div></div>
      </div>
      <div class="syncbox-row">
        <span class="pill {{PILL_CLASS}}"><span class="dot"></span>GPU BACKEND {{STATUS_TEXT}}</span>
        <div class="nvlogo" style="border-left:none;padding-left:0"><b>NVIDIA</b>CUDA {{CUDA_KPI}}</div>
      </div>
    </div>
    <div class="card" style="max-width:520px;margin-top:14px">
      <div class="head">
        <div><div class="t">APP-TRACKED GPU HOURS</div>
        <div class="s">{{GPU_HOURS_USED}} / {{GPU_HOURS_BUDGET}}h this week</div></div>
      </div>
      <div class="tip">Mirrors the real number in the header — see the Tokens &amp; GPU hours tab for the full bar, color thresholds, and the note on how this is measured.</div>
    </div>
    <div class="card" style="max-width:520px;margin-top:14px">
      <div class="head">
        <div><div class="t">NVIDIA GPU providers</div>
        <div class="s">Kaggle status is real; AWS/Azure/GCP are placeholders</div></div>
      </div>
      <div class="syncbox-row">
        <span class="gpitem {{PILL_CLASS}}"><i class="gpdot"></i>Kaggle</span>
        <span class="gpitem off"><i class="gpdot"></i>AWS</span>
        <span class="gpitem off"><i class="gpdot"></i>Azure</span>
        <span class="gpitem off"><i class="gpdot"></i>GCP</span>
      </div>
      <div class="tip">Kaggle's LED reflects the real live check, same as GPU BACKEND above. AWS/Azure/GCP are always off — there's no multi-cloud switching built yet, this is just marking where it would go once that's actually scoped. Clicking the row doesn't switch providers; it only shows a note about Kaggle's own start/stop limitations.</div>
    </div>
  </div>
</section>

<section class="view" id="about">
  <div class="lead">
    <h2>About this <span class="g">command center</span></h2>
    <p>Built to demonstrate CUDA, PyTorch GPU workloads, agentic orchestration, and honest observability — not a chatbot demo. Runs Kaggle T4×2 as the GPU backend, developed from a Mac via VS Code, hosted live on Streamlit Community Cloud so the visual dashboard and the real connection are finally the same page.</p>
  </div>
  <div class="tip warn"><b>Design principle carried through every tab —</b> if it isn't confirmed live, it says so. GPU-hours are labeled app-tracked rather than official. Pending features show as idle/pending rather than invented numbers.</div>
</section>
</main>

<div class="toast" id="toast"></div>

<script>
(function(){
  var buttons = document.querySelectorAll('.rail button.nav');
  var views = document.querySelectorAll('.view');
  buttons.forEach(function(b){
    b.addEventListener('click', function(){
      buttons.forEach(function(x){x.classList.remove('active')});
      views.forEach(function(v){v.classList.remove('active')});
      b.classList.add('active');
      document.getElementById(b.dataset.view).classList.add('active');
    });
  });
  // Two theme buttons now (header + Settings tab, added so the Settings
  // page mirrors the header instead of stranding a real control up top
  // only) -- both are pure client-side (localStorage + a data-theme
  // attribute on this iframe's own <html>), so keeping them in sync is
  // just a matter of updating both on every change, no backend involved.
  var themeBtns = [document.getElementById('themeBtn'), document.getElementById('themeBtn2')].filter(Boolean);
  var root = document.documentElement;
  function applyTheme(t){
    if(t){ root.setAttribute('data-theme', t); } else { root.removeAttribute('data-theme'); }
    var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var isDark = t ? t === 'dark' : sysDark;
    themeBtns.forEach(function(b){ b.classList.toggle('dark', isDark); });
  }
  var saved = null;
  try{ saved = localStorage.getItem('mi-cc-theme'); }catch(e){}
  applyTheme(saved);
  themeBtns.forEach(function(btn){
    btn.addEventListener('click', function(){
      var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
      var current = root.getAttribute('data-theme') || (sysDark ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      try{ localStorage.setItem('mi-cc-theme', next); }catch(e){}
    });
  });

  var toast = document.getElementById('toast'); var toastTimer = null;
  function showToast(html){
    toast.innerHTML = html;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function(){ toast.classList.remove('show'); }, 6500);
  }

  // Real, working control physically inside this decorative nav, without a
  // custom Streamlit Component build: navigates the REAL outer page
  // (window.top, same-origin) to a URL with ?mi_action=... appended --
  // Python reads and clears that param on the resulting page load, the same
  // channel the connect form already uses for jupyter_url/token. A full
  // reload, not a seamless partial update, by deliberate choice -- this
  // needed no new build tooling in exchange for that one trade-off.
  window.miTriggerAction = function(action){
    try{
      var url = new URL(window.top.location.href);
      url.searchParams.set('mi_action', action);
      window.top.location.href = url.toString();
    }catch(e){ showToast('Could not apply that from here — try the Settings panel in the sidebar instead.'); }
  };

  document.getElementById('gpuProviders').addEventListener('click', function(){
    showToast('Kaggle is the only backend actually wired up right now — AWS/Azure/GCP are shown for the roadmap, not connected. Kaggle also has no API to start or stop a session remotely; go start/stop it on kaggle.com directly, then reconnect here with the fresh tunnel URL/token.');
  });

  var recheckBtn = document.getElementById('recheckBtn');
  if(recheckBtn){
    recheckBtn.addEventListener('click', function(){
      try{ window.top.location.reload(); }catch(e){ showToast('Refresh this browser tab to recheck.'); }
    });
  }

})();
</script>
"""

html = HTML_TEMPLATE
html = html.replace("{{NVIDIA_ICON_DATA_URI}}", NVIDIA_ICON_DATA_URI)
html = html.replace("{{PILL_CLASS}}", pill_class)
html = html.replace("{{PROVIDER_STATUS_PILL_CLASS}}", provider_status_pill_class)
html = html.replace("{{PROVIDER_STATUS_TEXT}}", esc(provider_status_text))
html = html.replace("{{PROVIDER_STATUS_TITLE}}", esc(provider_status_title))
html = html.replace("{{STATUS_TEXT}}", status_text)
html = html.replace("{{STATUS_TEXT_LOWER}}", "confirmed live" if online else "not reachable right now")
html = html.replace("{{KPI_CLASS}}", kpi_class)
html = html.replace("{{GPU_COUNT_NAME}}", esc(gpu_count_name))
html = html.replace("{{CUDA_KPI}}", esc(cuda_kpi))
html = html.replace("{{CHECKED_AT}}", esc(checked_at))
html = html.replace("{{DEVICE_NAME}}", esc(device_name))
html = html.replace("{{DEVICE_COUNT}}", str(device_count))
html = html.replace("{{TORCH_VERSION}}", esc(torch_version))
html = html.replace("{{CUDA_AVAILABLE}}", "True" if online else "False")
html = html.replace("{{COMPUTE_CAP}}", esc(compute_cap))
html = html.replace("{{SYNC_HTML}}", sync_html)
html = html.replace("{{GPU_HOURS_USED}}", f"{gpu_hours_used:.1f}")
html = html.replace("{{GPU_HOURS_BUDGET}}", f"{GPU_HOUR_BUDGET:.1f}")
html = html.replace("{{GPU_HOURS_PCT}}", str(gpu_hours_pct))
html = html.replace("{{GPU_HOURS_FILL_COLOR}}", gpu_hours_fill_color)
html = html.replace("{{GPU_HOURS_NOTE}}", esc(gpu_hours_note_html))
html = html.replace("{{GPU_HOURS_WARNING_HTML}}", gpu_hours_warning_html)
_chrome_shown = st.session_state.get("mi_show_chrome", False)
html = html.replace("{{CHROME_ACTION}}", "hide_chrome" if _chrome_shown else "show_chrome")
html = html.replace("{{CHROME_BTN_LABEL}}", "Hide Streamlit Cloud toolbar & footer" if _chrome_shown else "Show Streamlit Cloud toolbar & footer")

components.html(html, height=980, scrolling=False)

# --- Floating Bond 001 widget — a real, native Streamlit chat, not the old
# JS-only cosmetic panel. It's built from plain st widgets (button, text_input,
# chat_message) wrapped in keyed containers that CSS pins to the *real* page
# viewport with position:fixed. That's the key difference from the removed
# v3 panel: that one lived inside the st.components.v1.html() iframe, so
# "fixed" meant fixed to the IFRAME's small box, which is what caused it to
# sit on top of the content right below it. This one is outside the iframe
# entirely, fixed to the actual browser window, so it floats over whatever
# is currently scrolled underneath it — the normal, correct behavior for a
# floating chat button — and it can call Python directly (open a kernel,
# load the model, generate) with no JS bridging needed.
st.markdown(
    """
    <style>
    /* bottom offset pushed up from 22px to clear Streamlit Cloud's own
       "Manage app" badge, which anchors to that same bottom-right corner
       for the logged-in owner -- confirmed via screenshot that it was
       overlapping this button's clickable area, making it fiddly to hit. */
    div.st-key-bond_fab { position: fixed; right: 22px; bottom: 86px; z-index: 999998; width: 56px; }
    div.st-key-bond_fab button {
        width: 56px; height: 56px; border-radius: 50% !important;
        background: radial-gradient(circle at 32% 28%, #a6e000, #76b900 60%) !important;
        border: 1px solid #223129 !important; color: #0b1400 !important;
        font-weight: 800 !important; font-size: 15px !important;
        box-shadow: 0 10px 26px -8px rgba(118,185,0,.55);
    }
    div.st-key-bond_panel {
        position: fixed !important; right: 22px; bottom: 154px; z-index: 999999;
        width: 380px; max-width: calc(100vw - 44px);
        max-height: min(640px, calc(100vh - 194px)); overflow-y: auto;
        background: #0d1310; border: 1px solid #223129; border-radius: 14px;
        box-shadow: 0 30px 70px -20px #000; padding: 16px 18px;
    }
    div.st-key-bond_panel h3, div.st-key-bond_panel p, div.st-key-bond_panel label,
    div.st-key-bond_panel span, div.st-key-bond_panel div { color: #edf2ee; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_one_bond_model(label: str):
    """Load exactly one model (opening the persistent kernel first if this is
    the first one) and add it to bond_loaded_models on success. Returns
    (ok, error_message_or_None)."""
    kernel_id = st.session_state.get("bond_kernel_id")
    is_new_kernel = not kernel_id
    if not kernel_id:
        with st.spinner("Opening a dedicated Kaggle kernel for Bond..."):
            kernel_id, err = open_kernel()
        if not kernel_id:
            return False, f"Couldn't open a kernel: {err}"
        st.session_state["bond_kernel_id"] = kernel_id
    if is_new_kernel:
        with st.spinner("Setting up Bond's web-search tool..."):
            tools_ok, tools_out = run_on_kernel(kernel_id, bond_tools_setup_code(), timeout=120)
        if not tools_ok:
            return False, f"Couldn't set up Bond's tools: {tools_out}"
    with st.spinner(
        f"Loading {label} onto the Kaggle T4 (installing transformers/bitsandbytes, "
        "downloading weights, loading in 4-bit) — this can take a couple minutes the "
        "first time this Kaggle session does it..."
    ):
        lok, lout = run_on_kernel(kernel_id, bond_load_one_code(label, BOND_MODEL_IDS[label]), timeout=480)
    if lok and f"BOND_READY:{label}" in lout:
        st.session_state.setdefault("bond_loaded_models", [])
        if label not in st.session_state["bond_loaded_models"]:
            st.session_state["bond_loaded_models"].append(label)
        return True, None
    return False, f"Load failed:\n\n{lout}"


# --- Fragments below: st.fragment scopes both the rerun AND Streamlit's
# whole-page dim/spinner overlay to just the fragment's own area, instead of
# the entire script (which includes the slow, network-bound status check up
# top). This is what actually fixes "clicking Bond takes time to pop up" and
# "the whole page dims until it replies" -- those were the outer script's own
# full rerun, not anything Bond-specific being slow.


@st.fragment(run_every=2)
def bond_autoload_fragment():
    """Loads ALL FOUR candidate models in the background, automatically,
    default-model first -- so switching the picker to any of the other
    three is instant instead of triggering a fresh multi-minute on-demand
    load (that on-demand path used to fire whenever someone picked a
    model other than the auto-loading default, which is exactly what
    "Qwen2.5-7B-Instruct... still loading" was).

    IMPORTANT: st.fragment(run_every=...) runs its body INLINE, blocking,
    on the very first call -- it only becomes an independently-scheduled
    tick starting from the *second* call onward. Doing the real (multi-
    minute) model loads on that first call would block the entire initial
    page render, including Bond's own button -- exactly the 'agent
    disappeared' symptom. So the first tick only sets a flag and returns
    immediately (page finishes rendering normally); the loads themselves
    only start on the second tick, ~2s later, by which point this
    fragment's reruns are already isolated from the rest of the page."""
    if st.session_state.get("bond_autoload_done"):
        return
    if not st.session_state.get("bond_autoload_started"):
        st.session_state["bond_autoload_started"] = True
        return
    if st.session_state.get("bond_selected_model") not in BOND_MODEL_IDS:
        st.session_state["bond_selected_model"] = BOND_DEFAULT_MODEL
    order = [BOND_DEFAULT_MODEL] + [m for m in BOND_MODEL_IDS if m != BOND_DEFAULT_MODEL]
    for label in order:
        if label in st.session_state.get("bond_loaded_models", []):
            continue
        if st.session_state.get(f"bond_load_failed_{label}"):
            continue
        lok, lerr = load_one_bond_model(label)
        if not lok:
            st.session_state[f"bond_load_failed_{label}"] = lerr
    st.session_state["bond_autoload_done"] = True


@st.fragment(run_every=2)
def bond_widget_fragment():
    # run_every=2 here (added so the chat input's disabled/placeholder state
    # below updates live once the background autoloader finishes, without
    # requiring the user to click anything first) is safe from the same
    # "first tick blocks the whole page" trap bond_autoload_fragment hit --
    # unlike that one, this fragment's own body never does slow blocking
    # work on a periodic tick; the only slow path (run_on_kernel when
    # actually sending a message) only runs in direct response to the
    # user's own click, same as before.
    with st.container(key="bond_fab"):
        if st.button("B1", key="bond_toggle_btn", help="Bond 001"):
            st.session_state["bond_panel_open"] = not st.session_state.get("bond_panel_open", False)

    if not st.session_state.get("bond_panel_open"):
        return

    with st.container(key="bond_panel"):
        st.markdown("### Bond 001")

        model_options = list(BOND_MODEL_IDS.keys())
        if st.session_state.get("bond_selected_model") not in model_options:
            st.session_state["bond_selected_model"] = BOND_DEFAULT_MODEL
        selected = st.selectbox("Model", model_options, key="bond_selected_model")

        # All four models autoload in the background (bond_autoload_fragment)
        # -- no per-model status text shown here by design, it's just noise
        # once loading is silent and automatic. A failed load is still worth
        # surfacing since it's actionable (retry button).
        model_ready = selected in st.session_state.get("bond_loaded_models", [])
        failed_key = f"bond_load_failed_{selected}"
        if st.session_state.get(failed_key):
            st.caption(f"⚠ {selected} failed to load: {st.session_state[failed_key]}")
            if st.button(f"Retry loading {selected}", key="bond_retry_load"):
                st.session_state.pop(failed_key, None)
                st.rerun()

        if "bond_messages" not in st.session_state:
            st.session_state["bond_messages"] = [
                {"role": "assistant", "content": "Welcome Mr Khella, how may I assist you today?"}
            ]

        for m in st.session_state.get("bond_messages", []):
            st.chat_message(m["role"]).write(m["content"])

        if st.session_state.get("bond_kernel_id"):
            if st.button("Unload all", key="bond_float_unload", use_container_width=True):
                close_kernel(st.session_state.get("bond_kernel_id"))
                for k in [k for k in st.session_state if k.startswith("bond_load_failed_")]:
                    st.session_state.pop(k, None)
                st.session_state.pop("bond_kernel_id", None)
                st.session_state.pop("bond_loaded_models", None)
                st.session_state["bond_autoload_done"] = False
                st.rerun()

        # st.chat_input submits on Enter like a normal chat, no separate
        # Send button/click check needed (unlike st.text_input, which only
        # commits its value on Enter/blur but doesn't trigger any action).
        # Disabled with a plain "warming up" placeholder while the selected
        # model isn't ready yet -- clearer than letting someone type into a
        # chat that can't actually reply yet, without showing Kaggle-specific
        # loading jargon. The run_every=2 above keeps this state current
        # without needing a click.
        fmsg = st.chat_input(
            "Message Bond 001…" if model_ready else "Bond is warming up…",
            key="bond_float_input",
            disabled=not model_ready,
        )

        if fmsg:
            st.session_state.setdefault("bond_messages", []).append({"role": "user", "content": fmsg})
            if selected in st.session_state.get("bond_loaded_models", []):
                with st.spinner(f"Generating with {selected} on the live Kaggle kernel..."):
                    gok, gout = run_on_kernel(
                        st.session_state["bond_kernel_id"],
                        f"print(bond_generate({json.dumps(fmsg)}, {json.dumps(selected)}))",
                        timeout=90,
                    )
                if gok:
                    st.session_state["bond_messages"].append({"role": "assistant", "content": f"**{selected}:** {gout}"})
                else:
                    st.session_state["bond_messages"].append({
                        "role": "assistant",
                        "content": f"The model kernel stopped responding — {gout}\n\n"
                                   "It likely died on the Kaggle side. Click \"Unload all\" then send another message to reload.",
                    })
            else:
                st.session_state["bond_messages"].append({
                    "role": "assistant",
                    "content": f"Still finishing setup for {selected} in the background — give it a little longer, then send that again.",
                })
            st.rerun()


if not PUBLIC_VIEWER_MODE:
    bond_autoload_fragment()
    bond_widget_fragment()
