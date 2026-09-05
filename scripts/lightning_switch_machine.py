"""Switch the Lightning AI Studio's machine tier directly via lightning_sdk.

Standalone script -- run locally (not through the dashboard) for a quick,
one-off machine-tier change without waiting on a full Streamlit wake-cycle.
Reads LIGHTNING_API_KEY / LIGHTNING_USER_ID the same way app.py's
get_secret() does: top-level keys in .streamlit/secrets.toml (not nested
under a [table] header -- see get_secret()'s docstring in
src/dashboard/app.py for why that distinction matters).

Usage:
    python3 scripts/lightning_switch_machine.py T4     # switch to a T4 GPU
    python3 scripts/lightning_switch_machine.py CPU    # switch back to CPU

Studio/teamspace/org constants are copied from src/dashboard/app.py's
LIGHTNING_STUDIO_NAME/LIGHTNING_TEAMSPACE/LIGHTNING_ORG -- keep these two
in sync if the Studio is ever renamed or moved to a different teamspace.

Lightning's free tier gives a limited number of T4 GPU-hours -- remember to
switch back to CPU once you're done testing so it doesn't idle on the clock.
"""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"

# Keep in sync with src/dashboard/app.py.
LIGHTNING_STUDIO_NAME = "coastal-salmon-q96l"
LIGHTNING_TEAMSPACE = "general"
LIGHTNING_ORG = "mnbd-org"


def _load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        raise SystemExit(
            f"Missing {SECRETS_PATH} -- copy .streamlit/secrets.example.toml "
            "and fill in real values, or add LIGHTNING_API_KEY/LIGHTNING_USER_ID "
            "there directly."
        )
    with open(SECRETS_PATH, "rb") as f:
        return tomllib.load(f)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1].upper() not in ("T4", "CPU"):
        raise SystemExit("Usage: python3 scripts/lightning_switch_machine.py [T4|CPU]")
    target = sys.argv[1].upper()

    secrets = _load_secrets()
    api_key = secrets.get("LIGHTNING_API_KEY", "")
    user_id = secrets.get("LIGHTNING_USER_ID", "")
    if not api_key or not user_id:
        raise SystemExit(
            "LIGHTNING_API_KEY / LIGHTNING_USER_ID missing from secrets.toml "
            "(must be top-level keys, not nested under a [table] header)."
        )
    # lightning_sdk reads credentials from os.environ, not a value passed in.
    os.environ["LIGHTNING_API_KEY"] = api_key
    os.environ["LIGHTNING_USER_ID"] = user_id

    from lightning_sdk import Machine, Studio
    from lightning_sdk.status import Status

    studio = Studio(name=LIGHTNING_STUDIO_NAME, teamspace=LIGHTNING_TEAMSPACE, org=LIGHTNING_ORG)
    machine = Machine.T4 if target == "T4" else Machine.CPU

    status = studio.status
    print(f"Current status: {status} / machine: {studio.machine}")

    if status != Status.Running:
        print(f"Studio isn't running -- starting it directly on {target} ...")
        studio.start(machine=machine)
    else:
        print(f"Switching running Studio to {target} (blocks until provisioned) ...")
        studio.switch_machine(machine)

    print(f"Done. Status: {studio.status} / machine: {studio.machine}")
    if target == "T4":
        print(
            "Reminder: switch back with "
            "`python3 scripts/lightning_switch_machine.py CPU` when you're "
            "done testing, so it doesn't idle on GPU hours."
        )


if __name__ == "__main__":
    main()
