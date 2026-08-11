"""h384판과 동일한 always-short 대조 + 게이트 전/후 편향 분해를, 라이브 라벨(h48_conservative
원본 48bar, threshold=0.50) 재학습판에 적용."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [260620, 481003, 26611, 903174, 155827]
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
TAG = "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h48orig_20260811_r30000_s"

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0

def forced_side_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out

rows, bias_rows = [], []
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, oof, fname, prefix in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
    ]:
        src = pd.read_csv(d / fname, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned)

        dec_model = omega._to_fixed_decisions(src_aligned, oof=oof)
        dec_short = forced_side_dec(dec_model, -1)

        m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
        m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
        rows.append({"seed": seed, "split": split_name,
                      "model_pnl": m_model["pnl"], "model_trades": m_model["trades"],
                      "always_short_pnl": m_short["pnl"]})

        dir_action = src_aligned[f"{prefix}_dir_action"]
        final_action = src_aligned[f"{prefix}_final_action"]
        dir_active = dir_action[dir_action != 0]
        final_active = final_action[final_action != 0]
        bias_rows.append({"seed": seed, "split": split_name,
                           "dir_raw_short_pct": (dir_active == 2).mean() * 100 if len(dir_active) else np.nan,
                           "final_short_pct": (final_active == 2).mean() * 100 if len(final_active) else np.nan,
                           "gate_survival_pct": len(final_active) / max(len(dir_active), 1) * 100})

df = pd.DataFrame(rows)
bias = pd.DataFrame(bias_rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print()
for split in ["VAL", "OOS"]:
    sub = df[df.split == split]
    t, p = stats.ttest_rel(sub.model_pnl, sub.always_short_pnl)
    print(f"=== {split} (5시드) === 모델={sub.model_pnl.mean():.3f}+-{sub.model_pnl.std():.3f}  "
          f"always_short={sub.always_short_pnl.mean():.3f}+-{sub.always_short_pnl.std():.3f}  "
          f"이긴시드={int((sub.model_pnl>sub.always_short_pnl).sum())}/5  paired t={t:.3f} p={p:.4f}")
    bsub = bias[bias.split == split]
    print(f"    게이트 전(dir_action) 숏비중={bsub.dir_raw_short_pct.mean():.1f}%  "
          f"게이트 후(final_action) 숏비중={bsub.final_short_pct.mean():.1f}%  통과율={bsub.gate_survival_pct.mean():.1f}%")
