# 재현성 참고: 이 스크립트는 h48qual 자체 리서치 패널(fa_features.parquet, 145컬럼)에 의존하는데,
# 이 파일은 세션 scratchpad에만 있고 레포에 커밋되지 않았다 -- 기존 FINAL12 dedup 작업과 동일한
# 재현성 갭. docs/experiments/eth_h48qual_final12_feature_selection_20260811.md 참고.
"""quality_head을 3-way 분류 대신 tb_quality(연속값)로 회귀시키면 실제로 도움이 되는지, 모델을
새로 학습하기 전에 먼저 오라클(실제 배리어 결과값을 안다고 가정)로 싸게 확인한다. direction_head의
원본 argmax(dir_action, 게이트 적용 전)는 그대로 쓰고, 게이트만 현재의 quality_head 분류기 대신
tb_long_quality/tb_short_quality(direction_head가 고른 쪽) > 0 으로 바꿔서 always_short/실제모델과
비교."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [260620, 481003, 26611, 903174, 155827, 44452, 51724, 179660, 240382, 375044, 378518, 692713, 711841, 750878, 821662]
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
TAG = "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s"

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")
TB_DIR = ROOT / "tmp/eth_h384_conservative_triple_barrier_labels_20260811"

def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)

# 캐노니컬 빌더의 validation/oos 분할은 SPLIT_TS 기준으로 우리 val_raw/oos_raw와 정확히 대응
tb_val = pd.read_csv(TB_DIR / "validation_triple_barrier_labels.csv", parse_dates=["timestamp"],
                      usecols=["timestamp", "tb_long_quality_h384_conservative", "tb_short_quality_h384_conservative"])
tb_oos = pd.read_csv(TB_DIR / "oos_triple_barrier_labels.csv", parse_dates=["timestamp"],
                      usecols=["timestamp", "tb_long_quality_h384_conservative", "tb_short_quality_h384_conservative"])

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def oracle_dec(dec_model, src_aligned, tb):
    """dec_model: 실제 모델 dec(게이트 후) -- notional/leverage/tp/sl/router_expert 등 템플릿 값만 재사용.
    side는 dir_action(direction_head 원본, 게이트 전)로 다시 만들고, tb_*_quality>0일 때만 active."""
    dir_action = pd.to_numeric(src_aligned["omega1_regime3_expertdq_dir_action" if "omega1_regime3_expertdq_dir_action" in src_aligned.columns
                                            else "omega1_regime3_expertdq_oof_dir_action"], errors="raise").to_numpy(dtype=np.int64)
    m = src_aligned[["timestamp"]].merge(tb, on="timestamp", how="left")
    long_q = m["tb_long_quality_h384_conservative"].to_numpy(dtype=np.float64)
    short_q = m["tb_short_quality_h384_conservative"].to_numpy(dtype=np.float64)
    side = np.zeros(len(dir_action), dtype=np.int64)
    side[(dir_action == 1) & (long_q > 0)] = 1
    side[(dir_action == 2) & (short_q > 0)] = -1
    active = side != 0
    out = dec_model.copy()
    out["side"] = side
    out["action"] = np.where(side == 1, omega.ACTION_LONG, np.where(side == -1, omega.ACTION_SHORT, omega.ACTION_CASH))
    out.loc[~active, "notional_exposure"] = 0.0
    out.loc[~active, "position_fraction"] = 0.0
    out.loc[~active, "take_profit"] = 0.0
    out.loc[~active, "stop_loss"] = 0.0
    return out


rows = []
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, oof, fname, tb in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv", tb_val),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv", tb_oos),
    ]:
        src = pd.read_csv(d / fname, parse_dates=["timestamp"])
        f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
        assert len(f) == len(src_aligned)

        dec_model = omega._to_fixed_decisions(src_aligned, oof=oof)
        dec_oracle = oracle_dec(dec_model, src_aligned, tb)
        dec_short = dec_model.copy()
        act = omega._active(dec_model)
        dec_short.loc[act, "side"] = -1
        dec_short.loc[act, "action"] = omega.ACTION_SHORT

        m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
        m_oracle = omega._metrics(f, dec_oracle, fee=fee, slip=slip, cost_mult=cost_mult)
        m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)

        rows.append({
            "seed": seed, "split": split_name,
            "model_pnl": m_model["pnl"], "model_trades": m_model["trades"],
            "oracle_pnl": m_oracle["pnl"], "oracle_trades": m_oracle["trades"],
            "oracle_long": m_oracle["long_entries"], "oracle_short": m_oracle["short_entries"],
            "always_short_pnl": m_short["pnl"],
        })

df = pd.DataFrame(rows)
out = "tmp/eth_h48qual_odyssey_regression_analysis_20260811/oracle_quality_gate_results.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print()
from scipy import stats
for split in ["VAL", "OOS"]:
    sub = df[df.split == split]
    print(f"=== {split} 15시드 ===")
    print(f"  현재 모델(quality분류기 게이트): {sub.model_pnl.mean():.3f} +- {sub.model_pnl.std():.3f}")
    print(f"  오라클 게이트(tb_quality>0):     {sub.oracle_pnl.mean():.3f} +- {sub.oracle_pnl.std():.3f}   숏비중={sub.oracle_short.sum()/(sub.oracle_short.sum()+sub.oracle_long.sum())*100:.1f}%")
    print(f"  always_short:                  {sub.always_short_pnl.mean():.3f} +- {sub.always_short_pnl.std():.3f}")
    t, p = stats.ttest_rel(sub.oracle_pnl, sub.always_short_pnl)
    print(f"  오라클 vs always_short paired t={t:.3f} p={p:.5f}  (오라클이 이긴 시드: {(sub.oracle_pnl > sub.always_short_pnl).sum()}/15)")
