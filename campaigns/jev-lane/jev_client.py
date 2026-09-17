"""Jev (TypeSafe) client for the jev-lane guard readouts.

Vendored from the wave-5 beer-game harness
(~/projects/agent-bullwhip-replication/agent_bullwhip/jev_agent.py) and
trimmed to what a guard readout needs: one POST to /v1/systemone with a
state + typed questions; retries on 429/529 and transient errors.

Key resolution: TYPESAFE_API_KEY from the environment, else .env.local next
to this file (gitignored).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env.local"
API_URL = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai") + "/v1/systemone"
DEFAULT_MODEL = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")


def _load_key() -> str | None:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def ask(state, questions: dict, model: str | None = None,
        timeout: float = 60.0, retries: int = 4) -> dict:
    """One POST to /v1/systemone. Retries 429/529 with exponential backoff."""
    key = _load_key()
    if not key:
        raise RuntimeError(
            "TYPESAFE_API_KEY not set — put it in the environment or in "
            f"{ENV_FILE} as TYPESAFE_API_KEY=... (dashboard: console.typesafe.ai)")
    payload = {"state": state, "model": model or DEFAULT_MODEL, "questions": questions}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=timeout,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"})
            if r.status_code in (429, 529):
                raise requests.HTTPError(f"HTTP {r.status_code} (rate/overload)")
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 — retry anything transient
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"TypeSafe call failed after {retries} attempts: {last_err}")


# -- question builders ------------------------------------------------------ #
def noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


def score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def choice(instructions: str, options: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}
