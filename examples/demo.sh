#!/usr/bin/env bash
# The README tour on synthetic files, in a fresh directory:  bash examples/demo.sh [DIR]
set -euo pipefail

DEMO="${1:-pseudorca-demo}"
"${PYTHON:-python3}" "$(dirname "$0")/make_demo_files.py" "$DEMO"
cd "$DEMO"
git init -q
export PSEUDORCA_ACTOR="${PSEUDORCA_ACTOR:-alice}"

pseudorca init --competition word-problems

pseudorca idea add short-cot    --title "Shorter reasoning instruction" --status adopted \
    --fingerprint "Think step by step, briefly."
pseudorca idea add retry-parse  --title "Retry when no answer parses"   --status adopted \
    --patch retry-parse.diff
pseudorca idea add prefix-cache --title "vLLM prefix caching"           --status testing \
    --fingerprint "enable_prefix_caching=True"
pseudorca idea add temp-06      --title "Sampling temperature 0.6"      --status rejected \
    --fingerprint 're:temperature\s*=\s*0\.6'

pseudorca nb add base.ipynb --tag baseline
pseudorca nb add next.ipynb --parent base.ipynb --applies short-cot --applies retry-parse --tag candidate
pseudorca nb tree next.ipynb

pseudorca audit latest
pseudorca audit base.ipynb
pseudorca check next.ipynb
pseudorca check base.ipynb || echo "(check failed as expected: base.ipynb lacks the adopted ideas)"
pseudorca status

pseudorca run add runs/base.csv --nb base.ipynb --label base --protocol p60x3
pseudorca run add runs/next.csv --nb next.ipynb --label next --protocol p60x3
pseudorca compare base next

pseudorca sub add 1001 --nb base.ipynb --score 0.412 --date 2026-01-05
pseudorca sub add 1002 --nb base.ipynb --score 0.398 --date 2026-01-06
pseudorca sub add 1003 --nb next.ipynb --score 0.437 --date 2026-01-07
pseudorca lb --diff 1001 1003

pseudorca report
pseudorca log -n 5
