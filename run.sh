#!/usr/bin/env bash
# One command to run the app:   ./run.sh        (or: make run)
# Run the tests instead:        ./run.sh test   (or: make test)
set -euo pipefail
cd "$(dirname "$0")"

# Find a Python >= 3.11.
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ ! -x .venv/bin/python ]; then
  if [ -z "$PY" ]; then
    echo "Python 3.11 or newer is required." >&2
    exit 1
  fi
  "$PY" -m venv .venv
fi

# Reinstall only when requirements.txt changed since the last install.
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  .venv/bin/pip install -q --disable-pip-version-check -r requirements.txt
  touch .venv/.installed
fi

if [ "${1:-}" = "test" ]; then
  exec .venv/bin/python -m pytest -q
fi

if [ ! -f .env ]; then
  echo "Note: no .env file found. Copy .env.example to .env and set NUCLEO_API_TOKEN." >&2
fi
echo "Starting on http://localhost:${PORT:-8000}"
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}"
