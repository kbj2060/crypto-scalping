#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/llewyn/crypto-scalping"
PY="/home/llewyn/miniconda3/envs/quant_ai/bin/python"
RUN="$ROOT/scripts/train_dsac_feature_variant_20260521.py"
SUM="$ROOT/scripts/summarize_dsac_feature_screen_runs_20260521.py"
SPEC="$ROOT/tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
CSV="$ROOT/tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv"

COMPARE_OUT="${1:-$ROOT/tmp/causal_regen_20260516/dsac_feature_screen_regime_fixed_fastgpu2_20260521}"
INITIAL_PID="${2:-}"
COMPARE_EPISODES="${3:-70}"
RETRAIN_OUT="${4:-$ROOT/tmp/causal_regen_20260516/dsac_feature_retrain_regime_fixed_fastgpu2_20260521}"
RETRAIN_EPISODES="${5:-200}"

mkdir -p "$COMPARE_OUT" "$RETRAIN_OUT"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$COMPARE_OUT/batch.log"
}

wait_for_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
}

run_summary() {
  python3 "$SUM" --runs-dir "$1" --episode-cutoff "$2" >/dev/null 2>&1 || true
}

variant_completed() {
  local runs_dir="$1"
  local variant="$2"
  local target_ep="$3"
  python3 - "$runs_dir" "$variant" "$target_ep" <<'PY'
import csv, sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
variant = sys.argv[2]
target_ep = int(sys.argv[3])
csv_path = runs_dir / f"screen_run_summary_ep{target_ep}.csv"
if not csv_path.exists():
    print("0")
    raise SystemExit(0)
with csv_path.open() as fh:
    for row in csv.DictReader(fh):
        if row.get("variant") != variant:
            continue
        try:
            latest = int(float(row.get("latest_episode") or "0"))
        except Exception:
            latest = 0
        print("1" if latest >= target_ep else "0")
        raise SystemExit(0)
print("0")
PY
}

choose_winner() {
  local runs_dir="$1"
  local target_ep="$2"
  local out_json="$3"
  python3 - "$runs_dir" "$target_ep" "$out_json" <<'PY'
import csv, json, sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
target_ep = int(sys.argv[2])
out_json = Path(sys.argv[3])
csv_path = runs_dir / f"screen_run_summary_ep{target_ep}.csv"
rows = []
with csv_path.open() as fh:
    for row in csv.DictReader(fh):
        if row.get("variant") not in {"stable48_global_pca32", "current_tail111"}:
            continue
        rows.append(row)

def f(row, key):
    try:
        return float(row.get(key) or "-inf")
    except Exception:
        return float("-inf")

def i(row, key):
    try:
        return int(float(row.get(key) or "0"))
    except Exception:
        return 0

rows = [r for r in rows if i(r, "latest_episode") >= target_ep]
if not rows:
    raise SystemExit("no completed compare variants available for winner selection")

# Stability-first rule for the 70-episode compare:
# latest_val_score -> best_val_score -> latest_val_pnl -> best_val_pnl
rows.sort(
    key=lambda r: (
        f(r, "latest_val_score"),
        f(r, "best_val_score"),
        f(r, "latest_val_pnl"),
        f(r, "best_val_pnl"),
    ),
    reverse=True,
)
winner = rows[0]
out_json.write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")
print(winner["variant"])
PY
}

run_variant() {
  local variant="$1"
  local out_root="$2"
  local episodes="$3"
  mkdir -p "$out_root/$variant"
  log "START variant=$variant episodes=$episodes"
  PYTHONUNBUFFERED=1 "$PY" "$RUN" \
    --variant "$variant" \
    --spec-dir "$SPEC" \
    --input-csv "$CSV" \
    --out-root "$out_root" \
    --episodes "$episodes" \
    --fresh-start \
    --device auto \
    > "$out_root/$variant/launcher.log" 2>&1
  log "END variant=$variant episodes=$episodes"
}

wait_for_pid "$INITIAL_PID"
run_summary "$COMPARE_OUT" "$COMPARE_EPISODES"

if [[ "$(variant_completed "$COMPARE_OUT" "current_tail111" "$COMPARE_EPISODES")" != "1" ]]; then
  run_variant "current_tail111" "$COMPARE_OUT" "$COMPARE_EPISODES"
fi

run_summary "$COMPARE_OUT" "$COMPARE_EPISODES"
WINNER_JSON="$COMPARE_OUT/winner_selection_ep${COMPARE_EPISODES}.json"
WINNER="$(choose_winner "$COMPARE_OUT" "$COMPARE_EPISODES" "$WINNER_JSON")"
log "SELECT winner=$WINNER compare_episodes=$COMPARE_EPISODES retrain_episodes=$RETRAIN_EPISODES"

run_variant "$WINNER" "$RETRAIN_OUT" "$RETRAIN_EPISODES"
run_summary "$RETRAIN_OUT" "$RETRAIN_EPISODES"
log "RETRAIN COMPLETE winner=$WINNER retrain_episodes=$RETRAIN_EPISODES"
