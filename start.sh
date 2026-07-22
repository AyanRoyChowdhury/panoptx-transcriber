#!/bin/bash
set -e
cd "$(dirname "$0")"

VENV="$HOME/.venvs/parakeet"

if [ ! -d "$VENV" ]; then
  echo "==> Creating virtual environment at $VENV ..."
  python3 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source "$VENV/bin/activate"
  # Install any newly added deps without reinstalling everything
  pip install -q -r requirements.txt
fi

echo ""
echo "Open http://localhost:8000 in your browser"
echo "Server auto-reloads on Python file changes — just refresh the browser for HTML changes."
echo ""
# TLS: browser mic capture (getUserMedia) requires a secure origin, so serve
# HTTPS when the self-signed cert is present (cert.pem/key.pem, SAN=tailscale IP).
if [ -f cert.pem ] && [ -f key.pem ]; then
  uvicorn server:app --reload --host 0.0.0.0 --port 8000 --log-level info \
    --ssl-keyfile key.pem --ssl-certfile cert.pem
else
  uvicorn server:app --reload --host 0.0.0.0 --port 8000 --log-level info
fi
