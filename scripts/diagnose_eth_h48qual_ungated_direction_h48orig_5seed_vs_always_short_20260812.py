"""candidate 9(eth_h48qual_quality_head_replacement_research_20260812.md)의 핵심 미해결 질문을 이
서브프로젝트의 표준 잣대(N>=5 다양한 시드)로 검증: quality_head 게이트 없이 direction_head 원본
(dir_action)만으로 거래하면 always-short/long을 이기는가? 단일 라이브 번들 실행
(diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py)은 VAL 동률/OOS 완패로
나왔으나 5시드가 아니라 확정 취급 불가했다. h48orig 5-seed 재현판(FINAL12 피쳐, 실제 h48qual
레시피 그대로 48bar, verify_eth_h48qual_always_short_baseline_h48orig_20260811.py와 동일
시드/경로/TRAIN_CSV)의 이미 저장된 예측을 재사용한다 -- 재학습 없음."""
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

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def ungated_decisions(src: pd.DataFrame, prefix: str, oof: bool) -> pd.DataFrame:
    """_to_fixed_decisions와 동일 로직이지만 final_action 대신 dir_action을 게이트 없이 사용."""
    src2 = src.copy()
    src2[f"{prefix}_final_action"] = src2[f"{prefix}_dir_action"]
    return omega._to_fixed_decisions(src2, oof=oof)


def forced_side_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


print("VAL/OOS 가격 프레임 로드 중...", flush=True)
train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)

rows = []
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, oof, fname, prefix in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
    ]:
        src = pd.read_csv(d / fname, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned), f"seed={seed} {split_name} length mismatch {len(f)} vs {len(src_aligned)}"

        dec_gated = omega._to_fixed_decisions(src_aligned, oof=oof)
        dec_ungated = ungated_decisions(src_aligned, prefix, oof)
        dec_short = forced_side_dec(dec_ungated, -1)
        dec_long = forced_side_dec(dec_ungated, 1)

        m_gated = omega._metrics(f, dec_gated, fee=fee, slip=slip, cost_mult=cost_mult)
        m_ungated = omega._metrics(f, dec_ungated, fee=fee, slip=slip, cost_mult=cost_mult)
        m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
        m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

        rows.append({
            "seed": seed, "split": split_name,
            "gated_pnl": m_gated["pnl"], "gated_trades": m_gated["trades"],
            "ungated_pnl": m_ungated["pnl"], "ungated_trades": m_ungated["trades"],
            "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
        })

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print()
print(df.to_string(index=False))
print()
for split in ["VAL", "OOS"]:
    sub = df[df.split == split]
    t, p = stats.ttest_rel(sub.ungated_pnl, sub.always_short_pnl)
    beats = int((sub.ungated_pnl > sub.always_short_pnl).sum())
    print(f"=== {split} (5시드) === ungated={sub.ungated_pnl.mean():.3f}+-{sub.ungated_pnl.std():.3f}  "
          f"always_short={sub.always_short_pnl.mean():.3f}+-{sub.always_short_pnl.std():.3f}  "
          f"이긴시드={beats}/5  paired t={t:.3f} p={p:.4f}")
