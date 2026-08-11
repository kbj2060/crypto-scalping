"""라이브 실제 zig075 번들(102피쳐, true_3head_tabm_bundle.pt, 2026-06-29 학습, quality_threshold=0.75)의
이미 저장된 예측(validation_predictions_q075.csv/oos_predictions_q075.csv, 2026-06-30에 이미 export됨
-- 재학습 아님)로 always-short 대조 + 게이트 전/후 편향 분해. h48qual의
scripts/verify_eth_h48qual_always_short_baseline_live_bundle_20260811.py를 zig075 번들 경로/threshold만
바꿔 그대로 재사용(동일 ThreeHeadTabM 구조, 동일 quality_for_action 게이팅 공식 -- bundle state_dict
key/shape와 omega4_6_1_live.py의 _Component 로딩 코드로 2026-08-11 확인). VAL 프레임은 이 번들 고유의
TRAIN_CSV(alpha5 regime4 계열, alpha6_current 아님 -- export json의 train_eval_override_used=True로
확인, h48qual과 동일 경로)를 그대로 써야 정합.

참고: zig075의 quality_head는 h48qual의 독립적 48bar 배리어(h48_conservative) 라벨이 아니라
`quality_mode=same_as_direction`(report.json의 label_contract.quality_mode, 2026-08-11 확인) --
direction_head와 동일한 zigzag_action 라벨로 학습됨. 아키텍처와 게이팅 공식은 동일하지만 quality
라벨의 의미(독립 배리어 결과 vs direction 자기 자신에 대한 재확인)가 다르다는 점은 결과 해석 시
유의."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

D = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"

# 이 번들 고유 TRAIN_CSV (alpha5 regime4 계열) / EVAL_CSV (alpha6_current, 우리가 늘 쓰던 것과 동일)
# -- h48qual과 동일 경로 (prediction_export_q075_20260630.json의 train_csv/eval_csv로 확인)
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
    ("VAL", val_raw, "validation_predictions_q075.csv", "omega1_regime3_expertdq_oof"),
    ("OOS", oos_raw, "oos_predictions_q075.csv", "omega1_regime3_expertdq"),
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
