#!/usr/bin/env bash
# ─── MNIST Neural Network Explorer — Startup Script ────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"
VENV="$SCRIPT_DIR/.venv"

echo ""
echo "🧠  MNIST Neural Network Explorer"
echo "────────────────────────────────────────"

# ── Create virtualenv if missing ──────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "📦  Creating Python virtual environment…"
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# ── Install / upgrade deps ────────────────────────────────────────────────────
echo "📦  Checking dependencies…"
pip install -q --upgrade pip
pip install -q -r "$BACKEND/requirements.txt"

echo ""
echo "✅  Dependencies ready."
echo "🌐  Starting server at http://localhost:8000"
echo "    Press Ctrl+C to stop."
echo ""

# ── Launch FastAPI ─────────────────────────────────────────────────────────────
cd "$BACKEND"
python main.py
