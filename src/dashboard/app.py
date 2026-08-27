# MI Command Center — live vertical slice (Streamlit)
#
# v2 (2026-08-26): no manual "mark online/offline" anywhere. Status is decided
# by actually calling the live Kaggle kernel, every time this page loads or is
# refreshed — never by a button the user clicks to assert a state.
#
# "Login" here means: the ngrok URL + token for the current Kaggle tunnel
# (Kaggle has no account-level API for checking a notebook's live status —
# there's nothing to "log into" beyond that; see the README for why). Enter
# them once and they're kept in this browser's address bar (query params), so
# refreshing the page reconnects automatically instead of asking again. Click
# "Forget connection" to clear them — a stand-in for the real login/logout
# system planned later.
#
# Run it locally first, on the Mac, via Claude Code (it has real bash +
# internet there already):
#
#   cd live-vertical-slice
#   python3 -m venv venv && source venv/bin/activate
#   pip install -r requirements.txt
#   streamlit run app.py

import json
import time
import uuid

import requests
import streamlit as st

st.set_page_config(page_title="MI Command Center — Live", page_icon="\U0001F5A5️", layout="wide")

# --- Connection state: read from the URL's query params, not session_state,
# so a page REFRESH (not just a rerun) keeps it and re-checks automatically.
params = st.query_params
jupyter_url = params.get("jupyter_url", "")
jupyter_token = params.get("jupyter_token", "")


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
            "code": code, "silent": False, "store_history": False,
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


st.title("\U0001F5A5️ MI Command Center — live")

connected = bool(jupyter_url and jupyter_token)

# --- Not connected yet: ask once, like a login. -----------------------------
if not connected:
    st.warning(
        "Not connected — this checks the real Kaggle kernel, so it needs the "
        "current tunnel address once. Nothing is \"marked\" online here; the "
        "status below always comes from an actual call."
    )
    with st.form("connect"):
        st.write("**Connect to your Kaggle GPU session**")
        url_in = st.text_input("ngrok URL", placeholder="https://xxxx.ngrok-free.app")
        token_in = st.text_input("Token", type="password")
        st.caption(
            "From the Kaggle notebook's tunnel-launch cell (runbook section 5). "
            "This isn't a Kaggle account login — Kaggle has no API for that; "
            "it's the address of the specific live tunnel this session opened."
        )
        submitted = st.form_submit_button("Connect")
    if submitted and url_in and token_in:
        st.query_params["jupyter_url"] = url_in
        st.query_params["jupyter_token"] = token_in
        st.rerun()
    st.stop()

# --- Connected: run the REAL check on every load/refresh, no button needed. -
top = st.columns([3, 1])
with top[1]:
    if st.button("Forget connection"):
        st.query_params.clear()
        st.rerun()

with st.spinner("Checking the live Kaggle kernel..."):
    ok, smi_out = run_remote(
        "import subprocess; "
        "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)"
    )

status_col, kpi_col = st.columns([1, 3])
with status_col:
    if ok:
        st.success("GPU BACKEND ONLINE")
    else:
        st.error("GPU BACKEND OFFLINE")
    st.caption("Re-checked automatically on every page load / refresh — never a manual switch.")

with kpi_col:
    st.metric("Backend", "ONLINE" if ok else "OFFLINE")

st.subheader("nvidia-smi (live)")
if ok:
    st.code(smi_out, language="text")
else:
    st.code(smi_out, language="text")
    st.caption(
        "This is the real reason it's offline — a stale/expired tunnel URL, "
        "the Kaggle session having stopped, or a bad token. Get a fresh URL+token "
        "from the Kaggle notebook and reconnect, don't just retry."
    )

st.divider()

st.subheader("Bond 001")
st.caption(
    "Not yet connected to a served model — loading Phi-3-mini / Qwen2.5-7B-Instruct "
    "is the next roadmap step. What's real right now: sending a message pings the "
    "live Kaggle kernel and back, a genuine round trip through this exact bridge."
)
msg = st.text_input("Message Bond 001")
if st.button("Send"):
    with st.spinner("Reaching the live Kaggle kernel..."):
        pok, pout = run_remote(
            "import datetime, torch; "
            "print('pong from Kaggle at', datetime.datetime.utcnow().isoformat(), "
            "'| CUDA available:', torch.cuda.is_available())"
        )
    if pok:
        st.chat_message("assistant").write(
            f"Kernel reachable — {pout}\n\n"
            f"(Your message — “{msg}” — wasn't sent to a language model; none is loaded yet.)"
        )
    else:
        st.chat_message("assistant").write(f"Couldn't reach the kernel: {pout}")
