"""NVIDIA viewer-only login gate for the public Streamlit application."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth


def _auth_secret(name: str, default=""):
    try:
        return st.secrets.get("auth", {}).get(name, default)
    except FileNotFoundError:
        return default


def _login_scene() -> None:
    logo = Path(__file__).with_name("assets") / "nvidia-logo-official.png"
    logo_uri = "data:image/png;base64," + base64.b64encode(logo.read_bytes()).decode("ascii")
    map_path = Path(__file__).with_name("assets") / "nvidia-global-network-map.png"
    map_uri = "data:image/png;base64," + base64.b64encode(map_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
<style>
[data-testid="stHeader"],[data-testid="stSidebar"],footer{{display:none!important}}
div.st-key-viewer_logout{{position:fixed;right:18px;top:14px;z-index:999999;width:auto}}div.st-key-viewer_logout button{{background:#08100a!important;color:#9ee62b!important;border:1px solid #29402e!important;padding:.3rem .8rem!important}}
.stApp{{background:#010302;color:#f2f5f2}}[data-testid="stMainBlockContainer"],.main .block-container{{width:min(560px,calc(100vw - 36px))!important;max-width:560px!important;margin:11vh auto 0!important;padding:30px 38px 28px!important;position:relative;z-index:5;background:rgba(8,14,10,.97);border:1px solid #29402e;border-radius:24px;box-shadow:0 25px 90px #000}}
[data-testid="stMainBlockContainer"]:before,.main .block-container:before{{content:"";display:block;width:140px;height:66px;margin:0 auto 14px;background:url('{logo_uri}') center/contain no-repeat}}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"],.main .block-container>[data-testid="stVerticalBlock"]{{gap:.65rem!important}}
.nv-world{{position:fixed;inset:0;z-index:0;overflow:hidden;background:#010302}}.nv-world .world-map-image{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;display:block;opacity:.82;filter:saturate(.94) brightness(.7)}}.nv-world:after{{content:"";position:absolute;inset:0;z-index:1;background:linear-gradient(rgba(1,5,2,.06),rgba(1,4,2,.27));pointer-events:none}}
.nv-world svg{{position:absolute;z-index:2;inset:10% 0 0;width:100%;height:85%;opacity:.4}}.nv-head{{position:absolute;z-index:3;top:26px;left:28px;display:flex;flex-direction:column;color:#76b900;font:14px ui-monospace,monospace;letter-spacing:1px}}.nv-head b{{font:700 22px Inter,sans-serif}}.nv-head span{{color:#68716a}}
.grid{{stroke:#17311e;stroke-width:1;fill:none;opacity:.28}}.land{{display:none}}.route{{fill:none;stroke:#76b900;stroke-width:1.5;stroke-dasharray:8 10;animation:flow 8s linear infinite;filter:drop-shadow(0 0 4px #76b900)}}.route.alt{{stroke:#32a8ff;animation-duration:11s}}.node{{fill:#76b900;stroke:#d9ff9c;stroke-width:2;filter:drop-shadow(0 0 7px #76b900);animation:pulse 2s ease-in-out infinite alternate}}
@keyframes flow{{to{{stroke-dashoffset:-144}}}}@keyframes pulse{{to{{r:8;opacity:.5}}}}
[data-testid="stForm"]{{background:transparent;border:0;padding:0;box-shadow:none}}
[data-testid="stForm"] h1,[data-testid="stForm"] h2,[data-testid="stForm"] h3{{text-align:center;color:#87cf13}}.stTextInput input{{background:#020503!important;border-color:#304334!important;color:#fff!important}}[data-testid="stFormSubmitButton"] button{{background:linear-gradient(90deg,#9ee62b,#76b900)!important;color:#071000!important;border:0!important;font-weight:800!important}}
.portal-note{{text-align:center;color:#8a958c;font:12px ui-monospace,monospace;letter-spacing:1.5px;margin:-10px 0 16px}}
@media(max-width:700px){{[data-testid="stMainBlockContainer"],.main .block-container{{margin-top:7vh!important;padding:24px 22px!important}}.nv-world svg{{width:180%;left:-40%}}.nv-head span{{display:none}}}}
</style>
<div class="nv-world" aria-hidden="true"><img class="world-map-image" src="{map_uri}" alt="" /><div class="nv-head"><b>NVIDIA Accelerated Intelligence</b><span>Kaggle GPU · CUDA Engineering · Global Compute Fabric</span></div>
<svg viewBox="0 0 1120 500" preserveAspectRatio="xMidYMid slice"><g class="grid"><path d="M0 100H1120M0 200H1120M0 300H1120M0 400H1120M140 0V500M280 0V500M420 0V500M560 0V500M700 0V500M840 0V500M980 0V500"/></g><g class="land"><path d="M75 146l28-30 49-11 42 9 20 21 44 4 31 31-12 23-36 9-19 35-25 10-8 42-29 38-22-33-28-12-13-46-32-25 15-30z"/><path d="M281 114l13-18 25 3 7 15-20 14z"/><path d="M323 230l37 4 31 28 15 47-17 54-28 53-24-31 3-42-23-36-12-48z"/><path d="M505 145l28-31 48-12 34 14 38-7 40 13 55-6 36 18 57 2 43 24 67 16 34 31-21 25-55-4-28 22-50-7-33 20-51-10-28-38-33 9-22-28-43 5-31-26-37 9-34-21z"/><path d="M566 225l48 10 31 38-8 55-30 73-39-22-18-51-20-43z"/><path d="M874 342l40-22 56 12 31 34-15 43-67 10-49-30z"/><path d="M1008 389l18-9 16 15-13 17z"/></g><g><path class="route" d="M115 225Q360 45 610 210"/><path class="route alt" d="M175 265Q460 480 790 245"/><path class="route" d="M380 170Q650 5 900 205"/><path class="route alt" d="M95 330Q510 120 980 345"/></g><g><circle class="node" cx="145" cy="210" r="5"/><circle class="node" cx="420" cy="178" r="5"/><circle class="node" cx="655" cy="180" r="5"/><circle class="node" cx="925" cy="218" r="5"/><circle class="node" cx="960" cy="385" r="5"/></g></svg></div>
<div class="portal-note">NVIDIA INTELLIGENT COMMAND CENTER · SECURE TEAM VIEW</div>
""",
        unsafe_allow_html=True,
    )


def require_viewer() -> dict[str, str]:
    username = str(_auth_secret("viewer_username")).strip()
    password_hash = str(_auth_secret("viewer_password_hash")).strip()
    cookie_key = str(_auth_secret("cookie_key")).strip()
    if not username or not password_hash or not cookie_key:
        st.error("Viewer access is not configured. Add the [auth] values in Streamlit Secrets.")
        st.stop()

    credentials = {"usernames": {username: {"name": str(_auth_secret("viewer_name", "Team Viewer")), "password": password_hash, "roles": ["viewer"]}}}
    authenticator = stauth.Authenticate(credentials, str(_auth_secret("cookie_name", "nvidia_ai_viewer")), cookie_key, 30, auto_hash=False)
    authenticator.login(location="unrendered")
    if st.session_state.get("authentication_status") is True:
        if "viewer" not in (st.session_state.get("roles") or []):
            st.error("This account does not have viewer access.")
            st.stop()
        st.markdown(
            """
            <style>
            [data-testid="stHeader"],[data-testid="stSidebar"],footer{display:none!important}
            div.st-key-viewer_logout{position:fixed;right:18px;top:14px;z-index:999999;width:auto}
            div.st-key-viewer_logout button{background:#08100a!important;color:#9ee62b!important;border:1px solid #29402e!important;padding:.3rem .8rem!important}
            [data-testid="stMainBlockContainer"],.main .block-container{width:100%!important;max-width:100%!important;margin:0!important;padding:0!important;background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important}
            [data-testid="stMainBlockContainer"]:before,.main .block-container:before{display:none!important;content:none!important}
            [data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"],.main .block-container>[data-testid="stVerticalBlock"]{gap:0!important;padding:0!important;margin:0!important}
            div.st-key-viewer_logout{position:fixed!important;margin:0!important;padding:0!important;height:auto!important}
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(key="viewer_logout"):
            authenticator.logout("Sign out", location="main", key="nvidia_viewer_logout")
        return {"username": username, "role": "viewer"}

    _login_scene()
    remember = st.checkbox("Remember this trusted device for 30 days", value=False)
    authenticator.login(location="main", max_login_attempts=5, fields={"Form name": "Team View", "Username": "Username", "Password": "Password", "Login": "Sign in"})
    status = st.session_state.get("authentication_status")
    if status is False:
        st.error("Username or password is incorrect.")
    if status is not True:
        st.stop()
    if "viewer" not in (st.session_state.get("roles") or []):
        st.error("This account does not have viewer access.")
        st.stop()
    if not remember:
        authenticator.cookie_controller.delete_cookie()
    st.rerun()
