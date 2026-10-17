"""Load local secrets from .env (gitignored). No external dependency."""

from pathlib import Path

_ENV = Path(__file__).resolve().parents[2] / ".env"


def load_env() -> dict:
    out: dict[str, str] = {}
    if _ENV.exists():
        for line in _ENV.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def get(key: str) -> str:
    v = load_env().get(key)
    if not v:
        raise RuntimeError(f"{key} not set in {_ENV}")
    return v
