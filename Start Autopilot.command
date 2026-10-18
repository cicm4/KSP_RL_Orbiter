#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print -u2 'Set up the Python environment using the Run instructions in README.md, then open this launcher again.'
  read -r '?Press Return to close.'
  exit 1
fi
exec .venv/bin/python -m orbiter.control \
  --project-root "$PWD" \
  --launch-model "$PWD/models/transfer_ascent_ppo.zip" \
  --capture-model "$PWD/models/moon_capture.zip" \
  --run-root "$PWD/runs/control"
