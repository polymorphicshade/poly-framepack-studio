#!/usr/bin/env bash
# Launches the poly-framepack-studio server. Extra arguments are passed to studio.py.
set -e
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment not found at .venv. See README.md for setup."
    exit 1
fi

source ".venv/bin/activate"
exec python studio.py --port 7860 "$@"
