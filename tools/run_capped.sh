#!/usr/bin/env bash
# Run a command under hard memory and CPU ceilings at the lowest priority, so
# that a mistake kills or slows the command instead of freezing the machine.
#
#   tools/run_capped.sh .venv/bin/python tools/some_tool.py
#   MEM_MAX=4G CPU_MAX=200% tools/run_capped.sh <command>
#
# Defaults: 3 GB of memory, no swap, four cores' worth of CPU, and two threads
# per numeric library. It refuses to start when the machine is already busy
# (load above 12 or under 8 GB available); set FORCE=1 to start anyway.
set -euo pipefail
if [[ "${FORCE:-0}" != "1" ]]; then
  load=$(cut -d' ' -f1 /proc/loadavg)
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if awk -v l="$load" 'BEGIN {exit !(l > 12)}'; then
    echo "run_capped: load is $load (over 12); not starting. Wait, or set FORCE=1." >&2
    exit 75
  fi
  if (( avail_kb < 8 * 1024 * 1024 )); then
    echo "run_capped: only $((avail_kb / 1024)) MB available (under 8 GB); not starting. Wait, or set FORCE=1." >&2
    exit 75
  fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${MEM_MAX:-3G}" -p MemorySwapMax=0 -p CPUQuota="${CPU_MAX:-400%}" \
  nice -n 19 "$@"
