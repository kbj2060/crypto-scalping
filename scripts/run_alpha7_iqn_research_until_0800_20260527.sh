#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/llewyn/crypto-scalping"
OUT_ROOT="$ROOT/tmp/causal_regen_20260516/alpha7_iqn_until_0800_20260527"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate quant_ai

cutoff_ts="$(date -d 'today 08:00' +%s)"
summary_csv="$OUT_ROOT/iqn_research_ranking.csv"
echo "name,oos_cost3_pnl,oos_cost3_mdd,oos_cost3_trades,oos_cost3_wr,val_cost3_pnl,val_cost3_trades,delta_vs_baseline,cvar_min,edge_min,out_dir" > "$summary_csv"

run_variant() {
  local name="$1"
  shift
  local now_ts
  now_ts="$(date +%s)"
  if [[ "$now_ts" -ge "$cutoff_ts" ]]; then
    echo "[STOP] cutoff reached before $name"
    return 1
  fi
  local out_dir="$OUT_ROOT/$name"
  local log="$LOG_DIR/alpha7_iqn_${name}_20260527.log"
  echo "[RUN] $name -> $out_dir"
  python "$ROOT/scripts/train_eval_alpha7_iqn_fallback_20260527.py" \
    --out-dir "$out_dir" \
    --epochs 8 \
    "$@" 2>&1 | tee "$log"
  python - "$name" "$out_dir" "$summary_csv" <<'PY'
import csv, json, sys
from pathlib import Path
name, out_dir, summary_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
s = json.loads((out_dir / "summary.json").read_text())
b = s["best_by_selection"]
row = {
    "name": name,
    "oos_cost3_pnl": b["oos_metrics"]["cost3"]["pnl"],
    "oos_cost3_mdd": b["oos_metrics"]["cost3"]["mdd"],
    "oos_cost3_trades": b["oos_metrics"]["cost3"]["trades"],
    "oos_cost3_wr": b["oos_metrics"]["cost3"]["wr"],
    "val_cost3_pnl": b["val_metrics"]["cost3"]["pnl"],
    "val_cost3_trades": b["val_metrics"]["cost3"]["trades"],
    "delta_vs_baseline": b["delta_vs_baseline_oos_cost3_pnl"],
    "cvar_min": b["cvar_min"],
    "edge_min": b["edge_min"],
    "out_dir": str(out_dir),
}
with summary_csv.open("a", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=list(row)).writerow(row)
print("[RESULT]", row)
PY
}

run_variant "hurdle020_cql080_noaf" \
  --entry-hurdle 0.020 --theta-penalty 0.004 --cql-alpha 0.080 --anti-flat-lambda 0.000 \
  --tail-tau-mix 0.60 --tail-sample-weight 1.60 --min-val-fallback-trades 5 || true

run_variant "hurdle030_cql120_noaf" \
  --entry-hurdle 0.030 --theta-penalty 0.006 --cql-alpha 0.120 --anti-flat-lambda 0.000 \
  --tail-tau-mix 0.70 --tail-sample-weight 2.00 --min-val-fallback-trades 3 || true

run_variant "micro_tp_hurdle010" \
  --take-profit 0.030 --stop-loss 0.022 --max-hold 6 --entry-hurdle 0.010 --theta-penalty 0.006 \
  --cql-alpha 0.080 --anti-flat-lambda 0.000 --tail-tau-mix 0.60 --min-val-fallback-trades 8 || true

run_variant "asym_rr_hurdle015" \
  --take-profit 0.050 --stop-loss 0.030 --max-hold 8 --entry-hurdle 0.015 --theta-penalty 0.004 \
  --cql-alpha 0.100 --anti-flat-lambda 0.000 --tail-tau-mix 0.65 --min-val-fallback-trades 5 || true

run_variant "strict_cvar_hurdle025" \
  --entry-hurdle 0.025 --theta-penalty 0.010 --cql-alpha 0.180 --anti-flat-lambda 0.000 \
  --tail-tau-mix 0.80 --tail-tau-max 0.18 --tail-sample-weight 2.50 --min-val-fallback-trades 1 || true

run_variant "anti_flat_small_hurdle020" \
  --entry-hurdle 0.020 --theta-penalty 0.004 --cql-alpha 0.080 --anti-flat-lambda 0.015 --anti-flat-edge 0.020 \
  --tail-tau-mix 0.60 --tail-sample-weight 1.60 --min-val-fallback-trades 5 || true

python - "$summary_csv" <<'PY'
import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p)
if len(df):
    df = df.sort_values(["oos_cost3_pnl", "val_cost3_pnl"], ascending=[False, False])
    df.to_csv(p, index=False)
    print(df.to_string(index=False))
PY
