"""NVIDIA viewer-only login gate for the public Streamlit application."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import extra_streamlit_components as stx


# --- Cookie hardening ------------------------------------------------------
# streamlit-authenticator persists its re-auth cookie through
# extra-streamlit-components' CookieManager, which defaults to
# SameSite=Strict and gives Authenticate() no way to override it. Strict
# means the browser withholds the cookie whenever the app is *arrived at*
# via a cross-site navigation -- a share.streamlit.io redirect, an external
# link, an embed -- so that first load shows the login screen even with a
# valid 30-day cookie sitting in the browser. Lax still blocks the cookie
# on cross-site POSTs (all we need for CSRF safety on a re-auth cookie) but
# sends it on top-level cross-site GETs. Patch the default once, at import.
if not getattr(stx.CookieManager.set, "_nv_lax", False):
    _cm_set_orig = stx.CookieManager.set

    def _cm_set_lax(*args, **kwargs):
        kwargs.setdefault("same_site", "lax")
        return _cm_set_orig(*args, **kwargs)

    _cm_set_lax._nv_lax = True
    stx.CookieManager.set = _cm_set_lax


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
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{{background:transparent!important;color:#f2f5f2}}[data-testid="stMainBlockContainer"],.main .block-container{{width:min(650px,calc(100vw - 36px))!important;max-width:650px!important;margin:15vh auto 0!important;padding:20px 34px 52px!important;position:relative;z-index:5;background:rgba(8,14,10,.86)!important;border:1.5px solid rgba(255,255,255,.78)!important;border-radius:18px;box-shadow:0 25px 90px #000,inset 0 0 0 1px rgba(255,255,255,.05)}}
[data-testid="stMainBlockContainer"]:before,.main .block-container:before{{content:"";display:block;position:relative;z-index:3;width:106px;height:46px;margin:0 auto 6px;background:url('{logo_uri}') center/contain no-repeat}}
[data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"],.main .block-container>[data-testid="stVerticalBlock"]{{position:relative;z-index:3;gap:.35rem!important}}
.nv-world{{position:fixed;inset:0;z-index:-1;overflow:hidden;background:#010302}}.nv-world .world-map-image{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:center;display:block;opacity:.96;filter:saturate(1.18) brightness(.9) contrast(1.08) drop-shadow(0 0 14px rgba(118,185,0,.28))}}.nv-world:after{{content:"";position:absolute;inset:0;z-index:1;background:linear-gradient(rgba(1,5,2,.02),rgba(1,4,2,.14));pointer-events:none}}
.nv-world svg{{position:absolute;z-index:2;inset:10% 0 0;width:100%;height:85%;opacity:.62;filter:drop-shadow(0 0 7px rgba(118,185,0,.48))}}.nv-head{{position:absolute;z-index:3;top:26px;left:28px;display:flex;flex-direction:column;color:#76b900;font:14px ui-monospace,monospace;letter-spacing:1px}}.nv-head b{{font:700 22px Inter,sans-serif}}.nv-head span{{color:#818b83}}
.login-card-surface{{display:none!important}}
.grid{{stroke:#17311e;stroke-width:1;fill:none;opacity:.28}}.land{{display:none}}.route{{fill:none;stroke:#76b900;stroke-width:1.5;stroke-dasharray:8 10;animation:flow 8s linear infinite;filter:drop-shadow(0 0 4px #76b900)}}.route.alt{{stroke:#32a8ff;animation-duration:11s}}.node{{fill:#76b900;stroke:#d9ff9c;stroke-width:2;filter:drop-shadow(0 0 7px #76b900);animation:pulse 2s ease-in-out infinite alternate}}
@keyframes flow{{to{{stroke-dashoffset:-144}}}}@keyframes pulse{{to{{r:8;opacity:.5}}}}
[data-testid="stForm"]{{background:transparent;border:0;padding:0;box-shadow:none}}
[data-testid="stForm"] h1,[data-testid="stForm"] h2,[data-testid="stForm"] h3{{text-align:center;color:#87cf13}}.stTextInput input{{background:#020503!important;border:1px solid #d9dde1!important;color:#fff!important}}.stTextInput label,.stCheckbox label,[data-testid="stForm"] p{{color:#f5f7f8!important;font-family:Inter,Arial,sans-serif!important}}[data-testid="stForm"] [data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]){{display:block!important;width:100%!important}}[data-testid="stFormSubmitButton"],[data-testid="stFormSubmitButton"]>div{{display:flex!important;justify-content:center!important;align-items:center!important;width:100%!important;margin:0!important;padding:0!important}}[data-testid="stFormSubmitButton"] button{{position:static!important;transform:none!important;margin:4px auto 0!important;background:linear-gradient(90deg,#9ee62b,#76b900)!important;color:#071000!important;border:0!important;font-weight:800!important}}[data-testid="stCheckbox"]{{display:flex!important;justify-content:center!important;width:100%!important;margin:2px 0 0!important}}[data-testid="stCheckbox"] label{{width:auto!important;margin:0 auto!important}}
.portal-note{{text-align:center;color:#e1e5e8;font:12px ui-monospace,monospace;letter-spacing:1.5px;margin:-10px 0 16px}}
@media(max-width:700px){{[data-testid="stMainBlockContainer"],.main .block-container{{margin-top:7vh!important;padding:24px 22px 58px!important}}.nv-world svg{{width:180%;left:-40%}}.nv-head span{{display:none}}}}
</style>
<div class="nv-world" aria-hidden="true"><img class="world-map-image" src="{map_uri}" alt="" /><div class="nv-head"><b>NVIDIA Accelerated Intelligence</b><span>NVIDIA GPU · CUDA Engineering · Global Compute Fabric</span></div>
<svg viewBox="0 0 1120 500" preserveAspectRatio="xMidYMid slice"><g class="grid"><path d="M0 100H1120M0 200H1120M0 300H1120M0 400H1120M140 0V500M280 0V500M420 0V500M560 0V500M700 0V500M840 0V500M980 0V500"/></g><g class="land"><path d="M75 146l28-30 49-11 42 9 20 21 44 4 31 31-12 23-36 9-19 35-25 10-8 42-29 38-22-33-28-12-13-46-32-25 15-30z"/><path d="M281 114l13-18 25 3 7 15-20 14z"/><path d="M323 230l37 4 31 28 15 47-17 54-28 53-24-31 3-42-23-36-12-48z"/><path d="M505 145l28-31 48-12 34 14 38-7 40 13 55-6 36 18 57 2 43 24 67 16 34 31-21 25-55-4-28 22-50-7-33 20-51-10-28-38-33 9-22-28-43 5-31-26-37 9-34-21z"/><path d="M566 225l48 10 31 38-8 55-30 73-39-22-18-51-20-43z"/><path d="M874 342l40-22 56 12 31 34-15 43-67 10-49-30z"/><path d="M1008 389l18-9 16 15-13 17z"/></g><g><path class="route" d="M115 225Q360 45 610 210"/><path class="route alt" d="M175 265Q460 480 790 245"/><path class="route" d="M380 170Q650 5 900 205"/><path class="route alt" d="M95 330Q510 120 980 345"/></g><g><circle class="node" cx="145" cy="210" r="5"/><circle class="node" cx="420" cy="178" r="5"/><circle class="node" cx="655" cy="180" r="5"/><circle class="node" cx="925" cy="218" r="5"/><circle class="node" cx="960" cy="385" r="5"/></g></svg></div>
<div class="login-card-surface" aria-hidden="true"></div>
<div class="portal-note">NVIDIA Intelligent Command Center · Secure Team View</div>
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
    remember_key = "nvidia_remember_viewer"
    remember = bool(st.session_state.get(remember_key, True))
    cookie_expiry_days = 30 if remember else 0
    authenticator = stauth.Authenticate(
        credentials,
        str(_auth_secret("cookie_name", "nvidia_ai_viewer")),
        cookie_key,
        cookie_expiry_days,
        auto_hash=False,
    )

    # "Remember" unchecked must actively forget the device, not merely skip
    # writing a fresh cookie: without this, an earlier remember=True cookie
    # keeps the session alive against the user's explicit choice, because
    # cookie_expiry_days=0 only makes streamlit-authenticator's set_cookie()
    # a no-op -- it never clears what's already there.
    if not remember:
        try:
            authenticator.cookie_controller.delete_cookie()
        except Exception:
            pass

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
    authenticator.login(location="main", max_login_attempts=5, fields={"Form name": "Team View", "Username": "Username", "Password": "Password", "Login": "Sign in"})
    st.checkbox(
        "Remember this trusted device for 30 days",
        value=True,
        key=remember_key,
    )
    status = st.session_state.get("authentication_status")
    if status is False:
        st.error("Username or password is incorrect.")
    if status is not True:
        st.stop()
    if "viewer" not in (st.session_state.get("roles") or []):
        st.error("This account does not have viewer access.")
        st.stop()

    # The form login just succeeded on THIS run. streamlit-authenticator only
    # fires its own confirming st.rerun() for YAML-*file* credentials
    # (self.path); with dict credentials it returns straight on, and app.py
    # immediately renders the full ~2000-line dashboard in the same run. The
    # CookieManager's document.cookie write then races that render, and a
    # refresh in the first moment after login lands before it persists ->
    # "log in, refresh now, still logged out".
    #
    # Fix: pause briefly so the component's set message reaches and runs on
    # the frontend, then rerun once. authentication_status is already True in
    # session_state, so the next run returns immediately at the cookie-restore
    # branch above and the dashboard renders then -- run N stays small and the
    # cookie write gets a clean, uncluttered window. Guarded so it happens at
    # most once per session (no loop if the write still needs another visit).
    if remember and not st.session_state.get("_nv_cookie_flush_done"):
        st.session_state["_nv_cookie_flush_done"] = True
        time.sleep(0.6)
        st.rerun()

    return {"username": username, "role": "viewer"}
