from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

# train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py::_prepare_frames()가
# 무조건 이렇게 덮어씀 (기본값 max_hold=72/cooldown=6는 이 파이프라인에 안 씀 -- TP/SL로만 청산)
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [260620, 481003, 26611, 903174, 155827, 44452, 51724, 179660, 240382, 375044, 378518, 692713, 711841, 750878, 821662]
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
TAG = "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s"

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

rows = []
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
    ]:
        src = pd.read_csv(d / fname, parse_dates=["timestamp"])
        # align frame to src's timestamps exactly (positional match required by _metrics)
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned), f"{seed} {split_name} length mismatch {len(f)} vs {len(src_aligned)}"

        dec_model = omega._to_fixed_decisions(src_aligned, oof=oof)
        dec_short = forced_side_dec(dec_model, -1)
        dec_long = forced_side_dec(dec_model, 1)

        m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
        m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
        m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

        active_bars = int(omega._active(dec_model).sum())
        rows.append({
            "seed": seed, "split": split_name, "active_signal_bars": active_bars,
            "model_pnl": m_model["pnl"], "model_trades": m_model["trades"],
            "model_long": m_model["long_entries"], "model_short": m_model["short_entries"],
            "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
            "always_long_pnl": m_long["pnl"], "always_long_trades": m_long["trades"],
        })

df = pd.DataFrame(rows)

# sanity check: recomputed model_pnl(unforced)이 원본 report.json과 맞는지 확인
import json
chk = json.load(open(RUN_ROOT / f"{TAG}260620" / "report.json"))["results"]["q0p50"]
for split, key in [("VAL", "validation"), ("OOS", "oos")]:
    orig = chk[key]
    mine = df[(df.seed == 260620) & (df.split == split)].iloc[0]
    print(f"[검증] seed=260620 {split}: 원본 pnl={orig['pnl']:.3f} trades={orig['trades']} L={orig['long_entries']}/S={orig['short_entries']}"
          f"  |  재현 pnl={mine['model_pnl']:.3f} trades={mine['model_trades']} L={mine['model_long']}/S={mine['model_short']}")
print()
out_path = "tmp/eth_h48qual_odyssey_regression_analysis_20260811/always_short_baseline_results.csv"
df.to_csv(out_path, index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print()
for split in ["VAL", "OOS"]:
    sub = df[df["split"] == split]
    print(f"=== {split} 15시드 평균 ===")
    print(f"  모델(direction_head 실제 선택): {sub['model_pnl'].mean():.3f} +- {sub['model_pnl'].std():.3f}")
    print(f"  always_short(같은 진입타이밍, 방향만 숏 고정): {sub['always_short_pnl'].mean():.3f} +- {sub['always_short_pnl'].std():.3f}")
    print(f"  always_long: {sub['always_long_pnl'].mean():.3f} +- {sub['always_long_pnl'].std():.3f}")
    beat = (sub['model_pnl'] > sub['always_short_pnl']).sum()
    print(f"  모델이 always_short을 이긴 시드 수: {beat}/15")
