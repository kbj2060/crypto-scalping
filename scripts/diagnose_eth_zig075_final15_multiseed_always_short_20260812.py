"""final15(zig075) N=5 다양시드 always-short 대조 요약. scripts/
diagnose_eth_zig075_final15_quality_trend_bias_20260812.py(단일시드 260620용)를 5개 시드
디렉터리에 대해 반복 -- q050 threshold만(표준), VAL+OOS 양쪽."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [260620, 481003, 26611, 903174, 155827]
BASE = "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmredesign_20260810_final15"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")


def bundle_dir(seed: int) -> Path:
    if seed == 260620:
        return ROOT / BASE
    return ROOT / f"{BASE}_seed{seed}"


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


print("VAL/OOS 가격 프레임 로드 중...", flush=True)
train_all = _read(TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
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
missing = []
for seed in SEEDS:
    d = bundle_dir(seed)
    val_f = d / "validation_predictions_q050.csv"
    oos_f = d / "oos_predictions_q050.csv"
    if not (val_f.exists() and oos_f.exists()):
        missing.append(seed)
        continue
    for split_name, frame, fpath, prefix in [
        ("VAL", val_raw, val_f, "omega1_regime3_expertdq_oof"),
        ("OOS", oos_raw, oos_f, "omega1_regime3_expertdq"),
    ]:
        src = pd.read_csv(fpath, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned), f"seed{seed}/{split_name} length mismatch"

        dec_model = omega._to_fixed_decisions(src_aligned, oof=("oof" in prefix))
        dec_short = forced_side_dec(dec_model, -1)

        m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
        m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)

        rows.append({
            "seed": seed, "split": split_name,
            "model_pnl": m_model["pnl"], "model_wr": m_model["wr"], "model_trades": m_model["trades"],
            "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
            "model_beats_always_short": m_model["pnl"] > m_short["pnl"],
        })

if missing:
    print(f"[대기 중] 아직 안 끝난 시드: {missing}")

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False))

if not missing:
    print(f"\n=== model이 always_short을 이긴 횟수: {int(df['model_beats_always_short'].sum())}/{len(df)} ===")
    print(df.groupby("split")[["model_pnl", "always_short_pnl"]].mean().to_string())
