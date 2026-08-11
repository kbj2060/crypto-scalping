"""라이브 실제 번들(102피쳐, true_3head_tabm_bundle.pt, 2026-06-30 학습)의 이미 저장된 예측
(validation_predictions_q050.csv/oos_predictions_q050.csv, 2026-06-30에 이미 export됨 -- 재학습
아님)로 always-short 대조 + 게이트 전/후 편향 분해. VAL 프레임은 이 번들 고유의 TRAIN_CSV(alpha5
regime4 계열, alpha6_current 아님 -- export json의 override_used=True로 확인)를 그대로 써야 정합."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

D = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"

# 이 번들 고유 TRAIN_CSV (alpha5 regime4 계열) / EVAL_CSV (alpha6_current, 우리가 늘 쓰던 것과 동일)
LIVE_TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


print("라이브 VAL 프레임(alpha5 regime4 계열) 로드 중... (387MB, 시간 걸릴 수 있음)", flush=True)
train_all = _read(LIVE_TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)
print(f"VAL n={len(val_raw)}  OOS n={len(oos_raw)}", flush=True)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def forced_side_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


rows = []
for split_name, frame, fname, prefix in [
    ("VAL", val_raw, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
    ("OOS", oos_raw, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
]:
    src = pd.read_csv(D / fname, parse_dates=["timestamp"])
    f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    assert len(f) == len(src_aligned), f"{split_name} length mismatch {len(f)} vs {len(src_aligned)} (frame={len(frame)}, src={len(src)})"

    dec_model = omega._to_fixed_decisions(src_aligned, oof=("oof" in prefix))
    dec_short = forced_side_dec(dec_model, -1)
    dec_long = forced_side_dec(dec_model, 1)

    m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
    m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
    m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

    dir_action = src_aligned[f"{prefix}_dir_action"]
    final_action = src_aligned[f"{prefix}_final_action"]
    dir_active = dir_action[dir_action != 0]
    final_active = final_action[final_action != 0]

    rows.append({
        "split": split_name, "n_bars": len(f),
        "model_pnl": m_model["pnl"], "model_mdd": m_model["mdd"], "model_trades": m_model["trades"],
        "model_long": m_model["long_entries"], "model_short": m_model["short_entries"],
        "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
        "always_long_pnl": m_long["pnl"], "always_long_trades": m_long["trades"],
        "dir_raw_short_pct": (dir_active == 2).mean() * 100 if len(dir_active) else np.nan,
        "final_short_pct": (final_active == 2).mean() * 100 if len(final_active) else np.nan,
        "gate_survival_pct": len(final_active) / max(len(dir_active), 1) * 100,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print()
print(df.to_string(index=False))
