#!/usr/bin/env bash
# Scores the previous closed session, backfills actuals, and logs to logs/score.log.
# flock skips the run if another one is still going. Reruns for the same as-of date write nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/score.lock
if ! flock -n 9; then
    echo "$(date -Is) skipped, another run is in progress" >> logs/score.log
    exit 0
fi
echo "$(date -Is) start" >> logs/score.log
if .venv/bin/natgas score >> logs/score.log 2>&1; then
    echo "$(date -Is) done" >> logs/score.log
else
    status=$?
    echo "$(date -Is) failed with exit code $status" >> logs/score.log
    exit "$status"
fi
