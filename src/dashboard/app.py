# MI Command Center — live (v3: full dashboard UI, wired to the real Kaggle kernel)
#
# This merges two things that were separate until now:
#   1. The rich multi-tab visual design from the claude.ai Artifact prototype
#      (Overview / Topology / GPU / Models / Agents / Tokens / About + the
#      Bond 001 floating panel) — copied verbatim from mi-command-center.html.
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
import json
import re
import time
import uuid

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="MI Command Center — live", page_icon="\U0001F5A5️", layout="wide")
st.markdown(
    "<style>.block-container{padding:0 !important;max-width:100% !important} "
    "iframe{border:none !important}</style>",
    unsafe_allow_html=True,
)

# --- Connection state: read from the URL's query params, not session_state,
# so a page REFRESH (not just a rerun) keeps it and re-checks automatically.
params = st.query_params
jupyter_url = st.secrets.get("JUPYTER_URL", params.get("jupyter_url", ""))
jupyter_token = st.secrets.get("JUPYTER_TOKEN", params.get("jupyter_token", ""))


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


def esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


connected = bool(jupyter_url and jupyter_token)

# --- Not connected yet: a plain, honest connect form (same as v2). ----------
if not connected:
    st.title("\U0001F5A5️ MI Command Center — live")
    st.warning(
        "Not connected — this checks the real Kaggle kernel, so it needs the "
        "current tunnel address once. Nothing is \"marked\" online here; the "
        "full dashboard below always reflects an actual call."
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
top = st.columns([6, 1])
with top[1]:
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
    sync_html = f"""
      <div class="syncbox-row">
        <span class="pill off"><span class="dot"></span>GPU BACKEND OFFLINE — real check, {esc(checked_at)}</span>
        <button class="syncbtn" id="recheckBtn">Recheck now</button>
      </div>
      <pre style="font-family:var(--mono);font-size:10.5px;color:var(--muted);background:var(--panel2);
        border:1px solid var(--line);border-radius:8px;padding:10px 12px;max-height:160px;overflow:auto;margin:0">{esc(str(info_out))[:1200]}</pre>
      <div class="tip warn">This is the real reason it's offline — a stale/expired tunnel URL, the Kaggle session having stopped, or a bad token. Get a fresh URL+token from the Kaggle notebook, click "Forget connection" above, and reconnect — don't just retry.</div>
    """

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
.mark{width:44px;height:44px;border-radius:11px;margin-bottom:14px;flex:none;position:relative}
.mark svg{width:100%;height:100%;display:block}
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
header .titles{display:flex;flex-direction:column;gap:1px}
header h1{font-size:15.5px;font-weight:650;letter-spacing:.01em}
header h1 .g{color:var(--nv-hi)}
header .crumb{font-family:var(--mono);font-size:9.5px;color:var(--faint);letter-spacing:.14em;text-transform:uppercase}
header .right{margin-left:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.pill{display:flex;align-items:center;gap:7px;font-family:var(--mono);font-size:10.5px;color:var(--muted);
  border:1px solid var(--line);border-radius:999px;padding:6px 12px;background:var(--panel)}
.pill .dot{width:7px;height:7px;border-radius:50%;background:var(--faint);flex:none}
.pill.off .dot{background:var(--st-crit);box-shadow:0 0 7px rgba(229,72,77,.6)}
.pill.on .dot{background:var(--st-good);box-shadow:0 0 7px rgba(47,179,86,.6)}
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
figure{margin:0}
figcaption{font-family:var(--mono);font-size:10.5px;color:var(--faint);padding:12px 14px 4px;line-height:1.6}
.node-box{fill:var(--panel2, #121a16)}
#fab{position:fixed;right:22px;bottom:22px;z-index:60;width:54px;height:54px;border-radius:50%;cursor:pointer;border:1px solid var(--line);
  background:radial-gradient(circle at 32% 28%,var(--nv-hi),var(--nv) 60%);box-shadow:0 10px 26px -8px var(--nv-glow);display:grid;place-items:center}
#fab svg{width:22px;height:22px;stroke:#0b1400;fill:none;stroke-width:2}
#fabpanel{position:fixed;right:22px;bottom:86px;z-index:60;width:380px;max-width:calc(100vw - 44px);background:var(--panel);
  border:1px solid var(--line);border-radius:14px;box-shadow:0 30px 70px -20px #000;display:none;flex-direction:column;
  max-height:min(640px,calc(100vh - 130px))}
#fabpanel.open{display:flex}
.fab-head{display:flex;align-items:center;gap:10px;padding:13px 14px;border-bottom:1px solid var(--line);background:var(--panel2);flex:none}
.fab-badge{width:30px;height:30px;border-radius:8px;flex:none;background:var(--nv-dim);border:1px solid var(--line);display:grid;place-items:center;
  font-family:var(--mono);font-weight:700;font-size:11px;color:var(--nv-hi)}
.fab-head .t{font-weight:600;font-size:13px}
.fab-head .s{font-family:var(--mono);font-size:9.5px;color:var(--faint);text-transform:uppercase}
.fab-head .x{margin-left:auto;cursor:pointer;color:var(--faint);background:none;border:none;font-size:16px}
.fab-scroll{overflow-y:auto;padding:14px;flex:1}
.fab-body{font-size:12px;color:var(--muted);line-height:1.6}
.fab-body .stat{font-family:var(--mono);font-size:10.5px;color:var(--st-warn);margin-top:10px;padding:8px 10px;border:1px solid var(--line);
  border-left:2px solid var(--st-warn);border-radius:6px;background:var(--panel2)}
.modelpicker{margin-top:13px;display:flex;flex-direction:column;gap:7px}
.modelrow{display:flex;align-items:center;gap:10px;padding:9px 10px;border:1px solid var(--line);border-radius:9px;background:var(--panel2);cursor:pointer}
.modelrow.sel{border-color:var(--nv)}
.modelrow .rb{width:14px;height:14px;border-radius:50%;border:2px solid var(--faint);flex:none}
.modelrow.sel .rb{border-color:var(--nv);background:radial-gradient(circle,var(--nv) 42%,transparent 46%)}
.modelrow .nm{font-weight:600;font-size:12px;color:var(--ink)}
.modelrow .mt{font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:1px}
.modelrow .tag{margin-left:auto;font-family:var(--mono);font-size:9px;padding:2px 7px;border-radius:999px;border:1px solid var(--line);color:var(--faint);white-space:nowrap}
.modelrow .tag.rec{color:var(--nv-hi);border-color:var(--nv)}
.bond-stream{display:flex;flex-direction:column;gap:9px;margin-top:14px;padding-top:14px;border-top:1px solid var(--line-soft)}
.bmsg{max-width:88%;font-size:12px;line-height:1.5;padding:8px 11px;border-radius:11px}
.bmsg.me{align-self:flex-end;background:var(--nv-dim);color:var(--ink);border:1px solid var(--line)}
.bmsg.sys{align-self:stretch;font-family:var(--mono);font-size:10px;color:var(--st-warn);background:var(--panel2);border:1px solid var(--line);border-left:2px solid var(--st-warn);border-radius:6px}
.fab-input{display:flex;gap:8px;padding:11px 12px;border-top:1px solid var(--line);flex:none}
.fab-input input{flex:1;background:var(--panel2);border:1px solid var(--line);color:var(--ink);border-radius:9px;padding:9px 11px;font-size:12.5px}
.fab-input input:focus{border-color:var(--nv);outline:none}
.fab-input button{width:36px;height:36px;border-radius:9px;border:none;cursor:pointer;background:linear-gradient(145deg,var(--nv-hi),var(--nv));color:#0b1400;font-size:14px}
::selection{background:var(--nv);color:#0b1400}
.powerbtn{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:999px;background:var(--panel);
  padding:6px 12px;cursor:pointer;font-family:var(--mono);font-size:10.5px;color:var(--muted);transition:border-color .15s}
.powerbtn:hover{border-color:var(--nv)}
.powerbtn svg{width:13px;height:13px;stroke:var(--st-crit);fill:none;stroke-width:2}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(14px);background:var(--raised);
  border:1px solid var(--line);color:var(--ink);font-size:12.5px;padding:12px 18px;border-radius:11px;
  box-shadow:0 16px 40px -10px rgba(0,0,0,.5);opacity:0;pointer-events:none;transition:opacity .25s,transform .25s;
  z-index:90;max-width:min(440px,86vw);text-align:left;line-height:1.55}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.toast b{color:var(--nv-hi)}
</style>

<nav class="rail">
  <div class="mark" title="MI Command Center">
    <svg viewBox="0 0 44 44" role="img" aria-label="MI Command Center mark">
      <polygon points="22,3 39,13 39,31 22,41 5,31 5,13" fill="var(--nv-dim)" stroke="var(--nv)" stroke-width="1.4"/>
      <path d="M13 27 L18 16 L22 24 L26 15 L31 27" fill="none" stroke="var(--nv-hi)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <button class="nav active" data-view="overview"><span class="ico"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/></svg></span><span class="cap">Overview</span></button>
  <button class="nav" data-view="topology"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="5" cy="12" r="2.3"/><circle cx="19" cy="6" r="2.3"/><circle cx="19" cy="18" r="2.3"/><path d="M7 12 L16.7 7 M7 12 L16.7 17"/></svg></span><span class="cap">Topology</span></button>
  <button class="nav" data-view="gpu"><span class="ico"><svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 6V3M12 6V3M17 6V3M7 21v-3M12 21v-3M17 21v-3"/></svg></span><span class="cap">GPU</span></button>
  <button class="nav" data-view="models"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="5" r="1.6"/><circle cx="20" cy="5" r="1.6"/><circle cx="4" cy="19" r="1.6"/><circle cx="20" cy="19" r="1.6"/><path d="M9.6 10.2 5.2 6M14.4 10.2 18.8 6M9.6 13.8 5.2 18M14.4 13.8 18.8 18"/></svg></span><span class="cap">Models</span></button>
  <button class="nav" data-view="agents"><span class="ico"><svg viewBox="0 0 24 24"><rect x="4" y="4" width="7" height="7" rx="1.4"/><rect x="13" y="4" width="7" height="7" rx="1.4"/><rect x="4" y="13" width="7" height="7" rx="1.4"/><rect x="13" y="13" width="7" height="7" rx="1.4"/></svg></span><span class="cap">Agents</span></button>
  <button class="nav" data-view="tokens"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5v9M9 10h4.2a1.8 1.8 0 0 1 0 3.6H9m2-7v1.2m0 9.6V17.4"/></svg></span><span class="cap">Tokens</span></button>
  <div class="spacer"></div>
  <button class="nav" data-view="about"><span class="ico"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.1"/></svg></span><span class="cap">About</span></button>
</nav>

<main>
<header class="top">
  <div class="titles">
    <h1>MI <span class="g">Command Center</span></h1>
    <div class="crumb">nvidia · cuda · agentic gpu infrastructure</div>
  </div>
  <div class="right">
    <span class="pill {{PILL_CLASS}}" id="headerGpuPill"><span class="dot"></span><span id="headerGpuText">GPU BACKEND {{STATUS_TEXT}}</span></span>
    <span class="pill"><span class="dot"></span>APP-TRACKED GPU HOURS</span>
    <button class="powerbtn" id="powerBtn" aria-label="Turn the Kaggle GPU session on or off" title="Turn the Kaggle GPU session on or off">
      <svg viewBox="0 0 24 24"><path d="M12 3v8"/><path d="M6.3 6.3a8 8 0 1 0 11.4 0"/></svg>
      KAGGLE GPU
    </button>
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
    <h2><span class="g">MI</span> Command Center</h2>
    <p>A control plane for a Kaggle-hosted GPU rig, wired up from a Mac via VS Code — real CUDA, real PyTorch, an agentic layer being built on top. This view shows what's actually confirmed working versus what's still a placeholder; nothing here is simulated data dressed up as live telemetry. The GPU status below is a live server checking your kernel right now, not a static demo.</p>
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
        <div class="head"><div class="badge">…</div><div><div class="t">Phi-3-mini inference</div><div class="s">4-bit, bitsandbytes</div></div><span class="statepill warn"><span class="d"></span>pending</span></div>
        <div class="rows">
          <div class="row">install step<b>queued</b></div>
          <div class="row">load step<b>queued</b></div>
        </div>
        <div class="foot">next thing being wired up — Bond 001 below still just pings the kernel, no model loaded</div>
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
        <div class="track"><div class="fill" style="width:0%;background:var(--nv)"></div></div>
        <div class="val">0.0<small> hrs</small></div>
      </div>
      <div class="barrow" style="grid-template-columns:110px 1fr 100px">
        <div class="lbl"><b>Budget</b>configurable</div>
        <div class="track"><div class="fill" style="width:100%;background:var(--line)"></div></div>
        <div class="val">30.0<small> hrs</small></div>
      </div>
      <div class="legend-note">Kaggle doesn't expose your official quota via API — this tracks actual session runtime this app has observed, against a manual default of 30h/week. Not yet actually tracking (Phase 1 item 7). Warnings fire at 50% / 75% / 90% once tracking is live.</div>
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

<section class="view" id="about">
  <div class="lead">
    <h2>About this <span class="g">command center</span></h2>
    <p>Built to demonstrate CUDA, PyTorch GPU workloads, agentic orchestration, and honest observability — not a chatbot demo. Runs Kaggle T4×2 as the GPU backend, developed from a Mac via VS Code, hosted live on Streamlit Community Cloud so the visual dashboard and the real connection are finally the same page.</p>
  </div>
  <div class="tip warn"><b>Design principle carried through every tab —</b> if it isn't confirmed live, it says so. GPU-hours are labeled app-tracked rather than official. Pending features show as idle/pending rather than invented numbers.</div>
</section>
</main>

<div class="toast" id="toast"></div>
<button id="fab" aria-label="Open Bond 001"><svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 1 8 8"/><circle cx="9" cy="12" r="1"/><circle cx="13" cy="12" r="1"/><circle cx="17" cy="12" r="1"/></svg></button>
<div id="fabpanel">
  <div class="fab-head">
    <div class="fab-badge">B1</div>
    <div><div class="t">Bond 001</div><div class="s">agentic console · model not yet connected</div></div>
    <button class="x" id="fabclose" aria-label="Close">×</button>
  </div>
  <div class="fab-scroll">
    <div class="fab-body">
      Bond 001 will route requests to the Orchestrator Agent once the agent layer + a served model exist. Pick which model it should run on the Kaggle T4s — these are six that fit comfortably in 4-bit on 16GB of VRAM without burning through the weekly GPU-hour budget too fast. The real, working Bond ping (live round trip through this exact kernel) is in the Streamlit panel below this dashboard — not yet wired into this floating panel directly.
      <div class="modelpicker" id="modelPicker">
        <div class="modelrow" data-model="Phi-3.5-mini-instruct"><span class="rb"></span><div><div class="nm">Phi-3.5-mini-instruct</div><div class="mt">3.8B · microsoft · ungated</div></div><span class="tag">fastest</span></div>
        <div class="modelrow sel" data-model="Qwen2.5-7B-Instruct"><span class="rb"></span><div><div class="nm">Qwen2.5-7B-Instruct</div><div class="mt">7B · Alibaba · ungated · strong tool-use</div></div><span class="tag rec">recommended</span></div>
        <div class="modelrow" data-model="Mistral-7B-Instruct-v0.3"><span class="rb"></span><div><div class="nm">Mistral-7B-Instruct-v0.3</div><div class="mt">7B · ungated · function-calling</div></div><span class="tag">solid</span></div>
        <div class="modelrow" data-model="Llama-3.1-8B-Instruct"><span class="rb"></span><div><div class="nm">Llama-3.1-8B-Instruct</div><div class="mt">8B · Meta · gated — accept license on HF</div></div><span class="tag">popular</span></div>
        <div class="modelrow" data-model="Gemma-2-9b-it"><span class="rb"></span><div><div class="nm">Gemma-2-9b-it</div><div class="mt">9B · Google · gated — accept license on HF</div></div><span class="tag">quality</span></div>
        <div class="modelrow" data-model="Zephyr-7b-beta"><span class="rb"></span><div><div class="nm">Zephyr-7b-beta</div><div class="mt">7B · ungated · easy fallback</div></div><span class="tag">fallback</span></div>
      </div>
      <div class="stat" id="modelNote">If serving with vLLM: one model loaded at a time, not all six — <b>Qwen2.5-7B-Instruct</b> selected as the default pick for now (ungated, fast enough to conserve GPU-hours, strong at tool-calling for the agent layer). Phi-3-mini-4k-instruct is still the one already queued for the first basic load-and-verify test — separate from this pick.</div>
      <div class="bond-stream" id="bondStream"></div>
    </div>
  </div>
  <div class="fab-input">
    <input type="text" id="bondInput" placeholder="Message Bond 001…" disabled>
    <button id="bondSend" aria-label="Go to live Bond 001">↓</button>
  </div>
</div>

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
  var fab = document.getElementById('fab'), panel = document.getElementById('fabpanel'), close = document.getElementById('fabclose');
  fab.addEventListener('click', function(){ panel.classList.toggle('open'); });
  close.addEventListener('click', function(){ panel.classList.remove('open'); });

  var themeBtn = document.getElementById('themeBtn');
  var root = document.documentElement;
  function applyTheme(t){
    if(t){ root.setAttribute('data-theme', t); } else { root.removeAttribute('data-theme'); }
    var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var isDark = t ? t === 'dark' : sysDark;
    themeBtn.classList.toggle('dark', isDark);
  }
  var saved = null;
  try{ saved = localStorage.getItem('mi-cc-theme'); }catch(e){}
  applyTheme(saved);
  themeBtn.addEventListener('click', function(){
    var sysDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    var current = root.getAttribute('data-theme') || (sysDark ? 'dark' : 'light');
    var next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try{ localStorage.setItem('mi-cc-theme', next); }catch(e){}
  });

  var toast = document.getElementById('toast'); var toastTimer = null;
  function showToast(html){
    toast.innerHTML = html;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function(){ toast.classList.remove('show'); }, 6500);
  }
  document.getElementById('powerBtn').addEventListener('click', function(){
    showToast('Kaggle has no API to start or stop a session remotely — that\'s a Kaggle limitation, not this page\'s. Go start/stop it on kaggle.com directly, then reconnect here with the fresh tunnel URL/token.');
  });

  var recheckBtn = document.getElementById('recheckBtn');
  if(recheckBtn){
    recheckBtn.addEventListener('click', function(){
      try{ window.top.location.reload(); }catch(e){ showToast('Refresh this browser tab to recheck.'); }
    });
  }

  document.getElementById('bondSend').addEventListener('click', function(){
    try{ window.top.document.getElementById('bond-native-anchor').scrollIntoView({behavior:'smooth'}); }
    catch(e){ showToast('Scroll down on the page — the real Bond 001 box is below this dashboard.'); }
  });

  var models = document.querySelectorAll('.modelrow');
  models.forEach(function(m){
    m.addEventListener('click', function(){
      models.forEach(function(x){x.classList.remove('sel')});
      m.classList.add('sel');
    });
  });
})();
</script>
"""

html = HTML_TEMPLATE
html = html.replace("{{PILL_CLASS}}", pill_class)
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

components.html(html, height=980, scrolling=False)

# --- Native, proven Bond 001 panel (real round trip) — lives just below the
# rich visual dashboard above, since it needs a genuine Streamlit round trip
# that a JS-only floating panel inside an iframe can't do on its own yet.
st.markdown('<div id="bond-native-anchor"></div>', unsafe_allow_html=True)
st.divider()
st.subheader("Bond 001 — live (the real one)")
st.caption(
    "Not yet connected to a served model — loading Phi-3-mini / Qwen2.5-7B-Instruct is the next roadmap step. "
    "What's real right now: sending a message pings the live Kaggle kernel and back, a genuine round trip "
    "through this exact bridge. (The floating Bond panel in the dashboard above is visual/cosmetic for now — "
    "this box below is the one that actually reaches the kernel.)"
)
msg = st.text_input("Message Bond 001")
if st.button("Send"):
    with st.spinner("Reaching the live Kaggle kernel..."):
        pok, pout = run_remote(
            "import datetime, torch; "
            "print('pong from Kaggle at', datetime.datetime.now(datetime.timezone.utc).isoformat(), "
            "'| CUDA available:', torch.cuda.is_available())"
        )
    if pok:
        st.chat_message("assistant").write(
            f"Kernel reachable — {pout}\n\n"
            f"(Your message — “{msg}” — wasn't sent to a language model; none is loaded yet.)"
        )
    else:
        st.chat_message("assistant").write(f"Couldn't reach the kernel: {pout}")
