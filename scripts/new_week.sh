#!/usr/bin/env bash
# Scaffold a new week folder from the templates.
# Usage: ./scripts/new_week.sh 01 transformer-foundations
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <week-number> <topic-slug>" >&2
  echo "Example: $0 01 transformer-foundations" >&2
  exit 1
fi

NUM="$1"
SLUG="$2"
DIR="week-${NUM}-${SLUG}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -d "${ROOT}/${DIR}" ]; then
  echo "Error: ${DIR} already exists." >&2
  exit 1
fi

mkdir -p "${ROOT}/${DIR}/src"
sed "s/Week NN/Week ${NUM}/g" "${ROOT}/templates/week-README.md" > "${ROOT}/${DIR}/README.md"
sed "s/Week NN/Week ${NUM}/g" "${ROOT}/templates/notes.md"       > "${ROOT}/${DIR}/notes.md"
cp "${ROOT}/templates/requirements.txt" "${ROOT}/${DIR}/requirements.txt"

echo "Created ${DIR}/ (README.md, notes.md, requirements.txt, src/)"
echo "Next: fill the README, add code under src/, then update the master index."
