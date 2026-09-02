"""Generate Streamlit viewer secrets locally without echoing the password."""

from getpass import getpass
from pathlib import Path
from secrets import token_urlsafe

import bcrypt


username = input("Viewer username: ").strip()
password = getpass("Viewer password: ")
confirm = getpass("Confirm password: ")
if not username or not password:
    raise SystemExit("Username and password are required.")
if password != confirm:
    raise SystemExit("Passwords do not match.")

password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
target = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    "[auth]\n"
    f'viewer_username = "{username}"\n'
    'viewer_name = "Team Viewer"\n'
    f'viewer_password_hash = "{password_hash}"\n'
    'cookie_name = "nvidia_ai_viewer"\n'
    f'cookie_key = "{token_urlsafe(48)}"\n',
    encoding="utf-8",
)
print(f"Viewer secrets saved privately to {target}")
print("The password, hash, and cookie key were not printed. Do not commit or share this file.")
