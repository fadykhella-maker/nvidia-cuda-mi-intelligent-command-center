# Reflex migration feasibility — assessment only

No code was migrated, no Reflex project was scaffolded, and no new
dependencies were added for this. This is a read of `src/dashboard/app.py`
as it stands today (2,670 lines) plus its two helper modules
(`viewer_portal.py`, `lightning_service/main.py`), written to answer one
question: is replacing Streamlit with Reflex worth doing, and when.

## 1. What actually has to move

`app.py` is really two different programs sharing one Python process, and
they'd migrate completely differently.

**Program A — a ~1,200-line HTML/CSS/JS document** (`HTML_TEMPLATE`,
`hw_status_chip()`/`hw_status_row()` string builders, `esc()`, ~30
`{{PLACEHOLDER}}` substitutions), rendered once per script run via
`st.components.v1.html(html, height=980)` into a sandboxed iframe. This is
the entire visible dashboard: the nav rail, all nine tabs (Overview,
Topology, GPU, Models, Agents, Tokens, Lightning, Settings, About), the
animated neural-fabric SVG, the topology diagram, benchmark tables, the
theme toggle, toast notifications, and the tab-switching JS. It has **no
live connection back to Python** except full-page reloads via
`window.top.location.href` query-param tricks (how the chrome-toggle and
"forget connection" buttons work) — everything else is static once
rendered.

This is the part that does **not port**. Reflex has no equivalent of "hand
it a giant pre-built HTML string and let it live in an iframe" — Reflex
pages are a tree of Python-defined components that compile to React. `rx.html()`
exists as an escape hatch for raw markup, but using it to drop in the same
2,670-line HTML blob throws away everything Reflex is actually for (typed
components, real reactive bindings, no manual `{{TOKEN}}` string surgery)
and still has to be rebuilt tab-by-tab as real `rx.*` components to be
worth doing at all. By line count, this is the majority of the file.

**Program B — real, backend-connected native widgets**, all added or
touched this session, living *outside* that iframe:
- `viewer_portal.require_viewer()` — `streamlit-authenticator` +
  `extra-streamlit-components` cookie-based login, currently mid-debug.
- `gpu_providers_sidebar_fragment()` — `@st.fragment(run_every=15)`,
  live Kaggle/Lightning status + hardware chips in `st.sidebar`.
- `bond_autoload_fragment()` / `bond_widget_fragment()` — `@st.fragment(run_every=2)`,
  Bond 001's chat (`st.chat_message`/`st.chat_input`) and background model loading.
- The Settings sidebar (chrome toggle, forget-connection button).

This is the part that ports **conceptually** but not **mechanically**.
`st.fragment(run_every=N)` is a Streamlit-specific polling primitive with
no Reflex equivalent; Reflex's version of "keep a panel live" is an
`rx.State` event handler with `yield` inside a loop, or a background task —
same idea (poll a cached function, push an update), different API, has to
be re-written per fragment, not copy-pasted.

**Program C — pure Python business logic, zero Streamlit coupling**, and
the one genuinely portable piece of this whole file: the `GPUProvider`
interface (`is_configured()`/`wake()`/`get_status()`), `KaggleGPUProvider`/
`LightningGPUProvider`/`UnconfiguredGPUProvider`, the `GPU_PROVIDERS`
registry, `resolve_active_gpu_provider()`, `wake_kaggle()`/
`get_kaggle_status()`/`get_kaggle_error_log()`, `wake_lightning()`/
`get_lightning_detail()`/`get_lightning_health()`,
`compute_kaggle_hw_state()`/`compute_lightning_hw_state()`, `run_remote()`
(raw Jupyter kernel-gateway wire protocol over `websocket-client`), and
`read_gpu_hours_used()`. None of these call any `st.*` API directly (only
`get_secret()` wraps `st.secrets`, trivially swappable). This is the actual
substance of "what's real" in the dashboard, and it would carry over to
Reflex, or to anything else, unchanged.

**Bottom line on scope**: roughly a third of this file (Program C) is a
clean lift. The other two-thirds (Programs A and B) are a full rewrite,
not a port — A because Reflex has no iframe-HTML-string primitive to
receive it, B because the specific live-update mechanism doesn't exist in
Reflex under the same name or shape.

## 2. Does "remember me" survive the move, or start over?

Starts over completely. `streamlit-authenticator`'s cookie flow is built
directly on Streamlit APIs that don't exist elsewhere: `st.session_state`
for the login form's round-trip, `st.context.cookies` to read the
re-auth cookie server-side, and a Streamlit custom component
(`extra_streamlit_components.CookieManager`) to write it client-side.
There is no Reflex package that reimplements this specific flow.

Reflex has its own session model — each browser tab holds an `rx.State`
instance tied to a WebSocket connection, identified by a `reflex_session_id`
cookie Reflex manages internally for *state* identity, not user identity —
and no built-in username/password + "remember this device" login system.
Getting one means either pulling in a community auth package (immature,
this reviewer found nothing as established as `streamlit-authenticator`)
or hand-rolling cookie/JWT auth against Reflex's FastAPI backend from
scratch.

Concretely: **the exact bug currently being chased (cookie SameSite /
write-timing race) would not be fixed by migrating.** It would be replaced
by a different, equally from-scratch set of cookie/session edge cases,
just under a new API. Migrating is not a shortcut past this bug.

## 3. Effort and risk

- **Scope**: full rewrite of Programs A and B (the visible dashboard and
  every live-updating panel), reusing Program C as-is. Not incremental —
  Reflex can't host the iframe-HTML approach side-by-side with real Reflex
  components in any way that's less work than just rebuilding the pages.
- **Hosting is a second, separate migration.** Streamlit Community Cloud
  hosts this app free today, reading `st.secrets` directly. Reflex doesn't
  run on Streamlit Cloud — it needs its own host (Reflex Cloud's free tier,
  or self-hosting on something like Fly.io/Render), its own secrets
  mechanism (no `st.secrets`), and its own deploy pipeline. This was
  already flagged as an open question in this project's own prior handoff
  notes ("confirm truly free online hosting limits before deciding") and
  it is still unresolved — a real unknown, not a rounding error.
- **What breaks that isn't obvious until attempted**: the nine tabs' worth
  of hand-tuned CSS (the neural-fabric animation, the topology SVG, the
  theme-toggle transition, the provider-strip layout that was just
  reworked this session) all have to be re-expressed in Reflex's styling
  model: getting them to look identical is real design work, not a
  mechanical translation, and is exactly the kind of thing that eats far
  more time than the estimate assumes.
- **Opportunity cost**: this project's actual open work right now is GPU
  behavior — Lightning inference, Kaggle quota safety, the auth bug — all
  of which live in Program C or B and are unaffected by a UI framework
  swap. A full front-end rewrite is a multi-week-plus distraction from that,
  for a payoff (no Streamlit footer, more layout control) that is purely
  cosmetic.

## 4. Recommendation

**Not worth doing now.** Two-thirds of the file has no path to a partial
migration, the one thing under active debugging (login persistence) gets
fully rebuilt rather than fixed, hosting is a second unresolved migration
riding on top of the UI one, and the actual value of this project — real
Kaggle/Lightning GPU status, honest quota/tier reporting — lives entirely
in the part that already doesn't care which framework renders it.

Revisit only if a *functional* reason shows up — something Streamlit
genuinely can't do that blocks real work — not for the footer or for
"more control," which are the reasons on the table today. If that day
comes, Program C is already a clean, ready-to-reuse core; that's the one
piece of prep worth keeping in mind, not acting on.
