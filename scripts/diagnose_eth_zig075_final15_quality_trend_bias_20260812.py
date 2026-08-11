"""final15(zig075, JM 레짐+15개 선별피쳐, 단일시드 260620) 저장 예측으로 always-short 대조.
scripts/diagnose_eth_zig075_quality_trend_bias_20260811.py(구버전 라이브 zig075용)의 최소-diff
변형 -- final15는 alpha5 override가 아니라 표준 alpha6_current TRAIN_CSV/EVAL_CSV로 학습됐음
(report.json train.rows=78,568이 표준 소스의 2025-01~09 슬라이스와 일치, override 소스 아님 --
train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811.py가 TRAIN_CSV를
건드리지 않음, 즉 현재 레포 기본값 그대로 사용). 재학습 아님, q040~q060 5개 threshold 전부 대조."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

D = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmredesign_20260810_final15"

# final15는 override 없이 표준 alpha6_current TRAIN_CSV/EVAL_CSV로 학습됨 (report.json
# train.rows=78,568이 이 소스의 2025-01~09 슬라이스와 일치 -- alpha5 override(78,509/183,936과는
# 다른 숫자)가 아님을 확인 완료).
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


print("VAL/OOS 가격 프레임 로드 중...", flush=True)
train_all = _read(TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
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
for q_tag in ["q040", "q045", "q050", "q055", "q060"]:
    for split_name, frame, fname, prefix in [
        ("VAL", val_raw, f"validation_predictions_{q_tag}.csv", "omega1_regime3_expertdq_oof"),
        ("OOS", oos_raw, f"oos_predictions_{q_tag}.csv", "omega1_regime3_expertdq"),
    ]:
        src = pd.read_csv(D / fname, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned), f"{q_tag}/{split_name} length mismatch {len(f)} vs {len(src_aligned)} (frame={len(frame)}, src={len(src)})"

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
            "q": q_tag, "split": split_name, "n_bars": len(f),
            "model_pnl": m_model["pnl"], "model_mdd": m_model["mdd"], "model_wr": m_model["wr"], "model_trades": m_model["trades"],
            "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
            "always_long_pnl": m_long["pnl"], "always_long_trades": m_long["trades"],
            "dir_raw_short_pct": (dir_active == 2).mean() * 100 if len(dir_active) else np.nan,
            "final_short_pct": (final_active == 2).mean() * 100 if len(final_active) else np.nan,
            "gate_survival_pct": len(final_active) / max(len(dir_active), 1) * 100,
        })

df = pd.DataFrame(rows)
pd.set_option("display.width", 240)
print()
print(df.to_string(index=False))
