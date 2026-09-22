#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

rm -f .coverage
find . -name '.coverage.*' -delete 2>/dev/null || true

for test_file in tests/test_*.py; do
  coverage run --parallel-mode --source=weather_ai -m pytest "$test_file" -q
done

coverage combine
coverage report --fail-under=90 -m
