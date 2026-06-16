"""
Hit all four endpoints with the generated sample payloads and print the JSON responses.
Uses only the stdlib (urllib) so it needs no extra deps.

Prereqs:
    py tests/make_samples.py                       # writes tests/samples.json
    py -m uvicorn api.main:app                      # in another terminal
Then:
    py tests/smoke_test.py
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
SAMPLES = Path(__file__).resolve().parent / "samples.json"


def call(method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def show(title, status, payload):
    print(f"\n===== {title}  (HTTP {status}) =====")
    print(json.dumps(payload, indent=2))


def main():
    samples = json.loads(SAMPLES.read_text())

    s, j = call("GET", "/health")
    show("GET /health", s, j)

    s, j = call("POST", "/forecast", samples["forecast"])
    show("POST /forecast", s, j)

    s, j = call("POST", "/waste-check", samples["waste_wasteful"])
    show("POST /waste-check (wasteful instance)", s, j)

    s, j = call("POST", "/waste-check", samples["waste_healthy"])
    show("POST /waste-check (healthy instance)", s, j)

    s, j = call("POST", "/anomaly", samples["anomaly"])
    show("POST /anomaly", s, j)

    print("\nAll endpoints called.")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"Could not reach {BASE} — is the server running? ({e})")
        sys.exit(1)
