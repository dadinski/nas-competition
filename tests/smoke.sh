#!/usr/bin/env bash
# End-to-end smoke run over a chosen subset of datasets at a short budget.
# Mirrors the Makefile's build/run/score exactly (same main.py, same layout),
# only with fewer datasets and a smaller per-dataset time_limit, so it exercises
# the real harness path rather than a hand-rolled approximation.
#
# Dataset ORDER matters for this run: it is what exposes the cross-dataset
# batch-size collapse, so a memory-heavy dataset must come before a small one.
#
# NOTE: like the Makefile, this COPIES the datasets into smoke_pkg/ - that was
# 9.7 GB for AddNIST+Myofibre+Sudoku. `rm -rf smoke_pkg` when done.
#
# usage: smoke.sh <minutes-per-dataset> <dataset> [dataset...]
set -euo pipefail

# Repo root is this script's parent directory, so the script works from anywhere.
# PY defaults to whatever `python` resolves to; override for a specific env, e.g.
#   PY=/path/to/python bash tests/smoke.sh 8 AddNIST
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SUB="${SUB:-submission_baseline}"
MINUTES="$1"; shift

cd "$ROOT"
rm -rf smoke_pkg
mkdir -p smoke_pkg/predictions smoke_pkg/datasets
for d in "$@"; do
  cp -R "datasets/$d" smoke_pkg/datasets/
done
find smoke_pkg/datasets -name "test_y.npy" -delete
cp evaluation/main.py smoke_pkg/main.py
cp $SUB/*.py smoke_pkg/

# main.py hardcodes metadata['time_limit'] = 1 (hours) as a local testing
# scaffold - see CLAUDE.md 3/6. Rewrite that single line for the smoke budget.
HOURS=$($PY -c "print($MINUTES/60.0)")
$PY - "$HOURS" <<'EOF'
import sys, io, re
h = sys.argv[1]
p = 'smoke_pkg/main.py'
s = open(p, encoding='utf-8').read()
# tolerate any numeric literal - the scaffold has been 1, 1.0 and 3.0 at times
pat = re.compile(r"^(\s*)metadata\['time_limit'\]\s*=\s*[0-9.]+\s*$", re.M)
assert len(pat.findall(s)) == 1, 'time_limit scaffold line not found as expected'
s = pat.sub(lambda m: "%smetadata['time_limit'] = %s" % (m.group(1), h), s)
open(p, 'w', encoding='utf-8').write(s)
print('smoke budget set to %s h (%.0f s) per dataset' % (h, float(h) * 3600))
EOF

cd smoke_pkg && "$PY" -u main.py
