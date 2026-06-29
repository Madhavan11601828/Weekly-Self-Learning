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
T="${ROOT}/templates"

if [ -d "${ROOT}/${DIR}" ]; then
  echo "Error: ${DIR} already exists." >&2
  exit 1
fi

mkdir -p "${ROOT}/${DIR}/src"

# README index (substitute the week number)
sed "s/Week NN/Week ${NUM}/g" "${T}/week-README.md" > "${ROOT}/${DIR}/README.md"

# The four-part docs
for f in plan progress verify notes concept; do
  sed "s/Week NN/Week ${NUM}/g" "${T}/${f}.md" > "${ROOT}/${DIR}/${f}.md"
done

cp "${T}/requirements.txt" "${ROOT}/${DIR}/requirements.txt"

echo "Created ${DIR}/ with:"
echo "  README.md  plan.md  progress.md  verify.md  notes.md  requirements.txt  src/"
echo "Next: fill plan.md, then work through progress.md day by day."
