# NVIDIA CUDA MI Intelligent Command Center

A live dashboard for a Kaggle-hosted dual Tesla T4 GPU backend, reached over an
ngrok tunnel exposing the Jupyter kernel gateway.

## What's actually working

`src/dashboard/app.py` is a Streamlit app that:
- Takes the current ngrok URL + token for the live Kaggle tunnel (submitted
  once, kept in the page's URL query params so a refresh reconnects
  automatically instead of asking again).
- On every page load/refresh, opens a real connection to that tunnel over the
  standard Jupyter kernel wire protocol (`POST /api/kernels` → open a `wss://`
  channel with the token → send an `execute_request` → collect `stream`
  messages until `execute_reply`) and runs `nvidia-smi` on the live kernel,
  showing the real output. Status ("ONLINE"/"OFFLINE") reflects whether that
  call actually succeeded — never a manually-clicked toggle.
- Has a "Bond 001" send box that pings the same kernel for a real
  `torch.cuda.is_available()` + timestamp round trip. Bond is **not** yet
  wired to a served language model (see Roadmap).

This was tested end-to-end locally against a live Kaggle tunnel before being
pushed here — both `nvidia-smi` and the Bond ping returned real output from
an actual Tesla T4 ×2 Kaggle kernel.

## Why the tunnel is manual input, not a stored secret

Kaggle has no account-level API to check a notebook's live status, and the
ngrok URL/token are ephemeral — they change every time the Kaggle session or
tunnel restarts. So "connecting" here means pasting the *current* tunnel's
address, not logging into an account. There's nothing durable to store as a
secret yet; see Roadmap for the planned auto-refresh mechanism.

## Run locally

```bash
cd src/dashboard
python3 -m venv venv && source venv/bin/activate
pip install -r ../../requirements.txt
streamlit run app.py
```

Paste the current ngrok URL + token from the Kaggle notebook's tunnel-launch
cell into the form on first load.

## Repo layout

```
src/
  agents/       # planned: orchestration layer
  cuda/         # CUDA kernels (naive/tiled/register-blocked GEMM, benchmarks)
  models/       # planned: served model loading (Bond 001)
  monitoring/   # planned: pynvml/DCGM-style telemetry capture
  dashboard/    # the live Streamlit app (app.py)
  storage/      # planned: persisted run history/metrics
  utils/        # shared helpers
notebooks/      # Kaggle/Colab notebooks
configs/        # runtime configuration
data/           # datasets (gitignored contents expected)
experiments/    # benchmark run outputs
reports/        # write-ups
scripts/        # one-off operational scripts
```

## Roadmap

- Load Phi-3-mini-4k-instruct or Qwen2.5-7B-Instruct on the Kaggle kernel so
  Bond 001 can actually converse, not just ping.
- A live tunnel auto-refresh mechanism, so the dashboard doesn't need the
  URL/token re-pasted after every Kaggle/ngrok restart.
- Deployed as a Hugging Face Space (Streamlit, free CPU tier) for a
  standing URL — paste the current tunnel in whenever you have a live
  Kaggle session, rather than running this locally each time.
