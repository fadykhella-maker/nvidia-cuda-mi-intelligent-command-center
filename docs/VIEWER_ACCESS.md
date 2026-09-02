# Secure Global Viewer Portal

The hosted NVIDIA dashboard authenticates visitors before it reads connection secrets, checks remote compute, or renders operational content. Its public role is deliberately read-only.

## Public viewer boundary

- View GPU, CUDA, model, agent, and topology status.
- See Kaggle as online, starting, unavailable, or offline without fabricated telemetry.
- Cannot enter or reveal tunnel credentials.
- Cannot wake Kaggle, open or unload kernels, autoload models, or expose operator controls.
- Cannot deploy, edit, restart, configure, or delete the Streamlit application through dashboard code.

Repository and deployment ownership remain controlled by GitHub and Streamlit Community Cloud.

## Create the secrets

After installing the repository requirements, run:

```powershell
python scripts/generate_viewer_secrets.py
```

The password is entered invisibly. Paste the generated TOML into **Streamlit Community Cloud → App settings → Secrets**. Never commit the generated block.

To change the password, generate and replace `viewer_password_hash`. Rotate `cookie_key` at the same time when every remembered browser must be signed out. AMD and NVIDIA must use different cookie keys and cookie names.

## Acceptance checks

1. Wrong credentials are rejected and login attempts are limited.
2. Correct credentials open the dashboard.
3. The remember option survives a browser restart for up to 30 days.
4. No tunnel token or management action appears to a viewer.
5. An offline Kaggle backend produces an honest offline view without blocking login.
6. Rotating `cookie_key` invalidates remembered sessions.
7. Desktop and mobile login layouts remain usable.
