#!/usr/bin/env bash
set -eo pipefail

ROOT="/home/llewyn/crypto-scalping"
OUT_ROOT="$ROOT/tmp/causal_regen_20260516/alpha7_iqn_until_0800_20260527_seed_sweep_bg"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate quant_ai

cutoff_ts="$(date -d 'today 08:00' +%s)"
summary_csv="$OUT_ROOT/seed_sweep_ranking.csv"
echo "name,oos_cost3_pnl,oos_cost3_mdd,oos_cost3_trades,oos_cost3_wr,val_cost3_pnl,val_cost3_trades,delta_vs_baseline,cvar_min,edge_min,out_dir" > "$summary_csv"

run_variant() {
  local name="$1"
  shift
  if [[ "$(date +%s)" -ge "$cutoff_ts" ]]; then
    echo "[STOP] cutoff reached before $name"
    return 1
  fi
  local out_dir="$OUT_ROOT/$name"
  local log="$LOG_DIR/alpha7_iqn_seed_${name}_20260527.log"
  echo "[RUN] $name"
  python "$ROOT/scripts/train_eval_alpha7_iqn_fallback_20260527.py" \
    --out-dir "$out_dir" \
    --epochs 8 \
    --take-profit 0.050 \
    --stop-loss 0.030 \
    --max-hold 8 \
    --anti-flat-lambda 0 \
    --tail-tau-mix 0.65 \
    --tail-sample-weight 1.60 \
    --min-val-fallback-trades 1 \
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

for seed in 52729 52730 52731 52732 52733 52734 52735; do
  run_variant "h012_s${seed}" --seed "$seed" --entry-hurdle 0.012 --theta-penalty 0.003 --cql-alpha 0.090 || true
  run_variant "h015_s${seed}" --seed "$seed" --entry-hurdle 0.015 --theta-penalty 0.004 --cql-alpha 0.100 || true
  run_variant "h018_s${seed}" --seed "$seed" --entry-hurdle 0.018 --theta-penalty 0.005 --cql-alpha 0.110 || true
done

python - "$summary_csv" <<'PY'
import pandas as pd, sys
p = sys.argv[1]
df = pd.read_csv(p)
if len(df):
    df = df.sort_values(["oos_cost3_pnl", "val_cost3_pnl"], ascending=[False, False])
    df.to_csv(p, index=False)
    print(df.to_string(index=False))
PY
