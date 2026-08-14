"""h48qual MFE 분위수 회귀 quality_head(train_eval_omega4_h48qual_mfe_quality_regression_
20260812.py, N=5 진짜 무작위 시드) 결과에 이 서브 프로젝트 전체가 써온 필수 검증 -- 모델의
실제 선택 pnl을 같은 active bar 집합에서 방향만 강제숏/강제롱한 always_short/always_long과
대조. MI/R^2 게이트를 통과했다고 해서 실전 pnl 우위가 보장되는 게 아니므로(이 서브 프로젝트가
반복 학습한 교훈) 절대 생략하지 않는다."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [13036874, 747899465, 799474674, 570627141, 842447243]
TAG = "omega4_quality_regression_20260621_h48qual_mfe_quality_reg_20260812_s"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
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
        ("VAL", val_raw, True, "validation_predictions_2025_quality_reg_q70.csv"),
        ("OOS", oos_raw, False, "oos_predictions_2026_quality_reg_q70.csv"),
    ]:
        src_path = d / fname
        if not src_path.exists():
            print(f"[스킵] seed={seed} {split_name}: {src_path} 없음")
            continue
        src = pd.read_csv(src_path, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned), f"{seed} {split_name} length mismatch"

        dec_model = omega._to_fixed_decisions(src_aligned, oof=oof)
        dec_short = forced_side_dec(dec_model, -1)
        dec_long = forced_side_dec(dec_model, 1)

        for cm, tag in [(1.0, "cost1"), (2.0, "cost2"), (3.0, "cost3")]:
            m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cm)
            m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cm)
            m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cm)
            rows.append({
                "seed": seed, "split": split_name, "cost": tag,
                "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
            })

df = pd.DataFrame(rows)
out_path = ROOT / "tmp/eth_h48qual_mfe_quality_reg_5seed_always_short_baseline_20260812/pnl_comparison.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
for split in ["VAL", "OOS"]:
    print(f"\n===================== {split} =====================")
    for cost in ["cost1", "cost2", "cost3"]:
        sub = df[(df.split == split) & (df.cost == cost)]
        if sub.empty:
            continue
        n = len(sub)
        beat_short = int((sub["model_pnl"] > sub["always_short_pnl"]).sum())
        beat_long = int((sub["model_pnl"] > sub["always_long_pnl"]).sum())
        try:
            _, p_short = stats.wilcoxon(sub["model_pnl"] - sub["always_short_pnl"])
        except Exception:
            p_short = float("nan")
        try:
            _, p_long = stats.wilcoxon(sub["model_pnl"] - sub["always_long_pnl"])
        except Exception:
            p_long = float("nan")
        print(f"[{split}/{cost}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}  "
              f"always_short={sub['always_short_pnl'].mean():+7.2f}  always_long={sub['always_long_pnl'].mean():+7.2f}  "
              f"승(short)={beat_short}/{n}  승(long)={beat_long}/{n}  "
              f"wilcoxon_p(short)={p_short:.4f}  wilcoxon_p(long)={p_long:.4f}")

print(f"\n=== 저장 ===\n{out_path}")
