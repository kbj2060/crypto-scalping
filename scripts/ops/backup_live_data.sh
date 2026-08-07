#!/usr/bin/env bash
# Daily backup of data/live and data/ensemble to a separate drive (D:\, mounted
# at /mnt/d under WSL) -- the only copy of live state/ledgers/duckdb files
# otherwise lives on the single WSL root disk with no off-host redundancy.
#
# Intentionally does NOT use rsync --delete: a file removed locally (accidental
# rm, a bug) must stay in the backup rather than being mirrored away on the
# next run. This means the backup only grows -- acceptable given the current
# ~11.5G source size against hundreds of GB free on the destination drive.
#
# Usage: run daily from cron, e.g.:
#   0 4 * * * cd /home/llewyn/crypto-scalping && /bin/bash scripts/ops/backup_live_data.sh >> logs/backup_live_data_cron.log 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${BACKUP_DEST:-/mnt/d/crypto-scalping-backups}"

if [[ ! -d "$(dirname "$DEST")" ]]; then
  echo "[$(date -Iseconds)] backup destination drive not mounted, skipping ($DEST)"
  exit 0
fi

mkdir -p "$DEST/data/live" "$DEST/data/ensemble"

echo "[$(date -Iseconds)] backup starting -> $DEST"
for dir in data/live data/ensemble; do
  rsync -a --exclude '*.tmp' "$ROOT/$dir/" "$DEST/$dir/"
done
echo "[$(date -Iseconds)] backup done"
