"""zig075 실제 라이브 레시피 N=5 독립 시드 재학습(scripts/train_eval_omega4_3head_parent72_eth_
zig075_liverecipe_20260812.py, 서버 GPU) 결과에 h48qual과 동일한 short-only 격리 방법론
(scripts/diagnose_eth_h48qual_short_only_vs_always_short_20260811.py)을 적용 -- N=1 라이브
가중치 확인(2026-08-12)의 통계적 확정판."""
from pathlib import Path
import sys
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

SEEDS = [913588538, 702006280, 238746861, 689517735, 605384781]
TAG = "omega4_3head_parent72_loose_entry_quality_20260620_zig075_liverecipe_20260812_s"
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


def side_only_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    keep = active & (pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64) == side_value)
    drop = active & ~keep
    out.loc[drop, "notional_exposure"] = 0.0
    return out


rows = []
for seed in SEEDS:
    d = RUN_ROOT / f"{TAG}{seed}"
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q075.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q075.csv"),
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
        dec_short_forced = forced_side_dec(dec_model, -1)
        dec_short_only = side_only_dec(dec_model, -1)
        dec_long_only = side_only_dec(dec_model, 1)

        m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
        m_always_short = omega._metrics(f, dec_short_forced, fee=fee, slip=slip, cost_mult=cost_mult)
        m_short_only = omega._metrics(f, dec_short_only, fee=fee, slip=slip, cost_mult=cost_mult)
        m_long_only = omega._metrics(f, dec_long_only, fee=fee, slip=slip, cost_mult=cost_mult)

        rows.append({
            "seed": seed, "split": split_name,
            "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
            "always_short_pnl": m_always_short["pnl"], "always_short_trades": m_always_short["trades"], "always_short_wr": m_always_short["wr"],
            "short_only_pnl": m_short_only["pnl"], "short_only_trades": m_short_only["trades"], "short_only_wr": m_short_only["wr"],
            "long_only_pnl": m_long_only["pnl"], "long_only_trades": m_long_only["trades"], "long_only_wr": m_long_only["wr"],
        })

df = pd.DataFrame(rows)
out_path = ROOT / "tmp/eth_zig075_liverecipe_5seed_short_only_vs_always_short_20260812/short_only_vs_always_short.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
print(f"\n===================== zig075 liverecipe (N={len(SEEDS)}시드, 진짜 무작위) =====================")
for split in ["VAL", "OOS"]:
    sub = df[df["split"] == split]
    if sub.empty:
        continue
    n = len(sub)
    print(f"\n--- {split} ({n}시드) ---")
    print(sub[["seed", "model_pnl", "always_short_pnl", "short_only_pnl", "short_only_wr", "always_short_wr"]].to_string(index=False))
    print(f"  model(전체, 롱+숏 실제선택):   pnl={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}  trades={sub['model_trades'].mean():5.1f}  wr={sub['model_wr'].mean()*100:5.1f}%")
    print(f"  always_short(같은 active set, 방향 강제): pnl={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}  trades={sub['always_short_trades'].mean():5.1f}  wr={sub['always_short_wr'].mean()*100:5.1f}%")
    print(f"  model_short_only(실제 숏 선택만):  pnl={sub['short_only_pnl'].mean():+7.2f}±{sub['short_only_pnl'].std():5.2f}  trades={sub['short_only_trades'].mean():5.1f}  wr={sub['short_only_wr'].mean()*100:5.1f}%")
    print(f"  model_long_only(실제 롱 선택만):   pnl={sub['long_only_pnl'].mean():+7.2f}±{sub['long_only_pnl'].std():5.2f}  trades={sub['long_only_trades'].mean():5.1f}  wr={sub['long_only_wr'].mean()*100:5.1f}%")
    beat_as_pnl = int((sub['short_only_pnl'] > sub['always_short_pnl']).sum())
    beat_as_wr = int((sub['short_only_wr'] > sub['always_short_wr']).sum())
    print(f"  >> short_only가 always_short(pnl) 이긴 시드: {beat_as_pnl}/{n}   |   short_only가 always_short(승률) 이긴 시드: {beat_as_wr}/{n}")

    from scipy import stats
    try:
        stat, p = stats.wilcoxon(sub['short_only_pnl'] - sub['always_short_pnl'])
        print(f"  wilcoxon p (short_only vs always_short, pnl) = {p:.4f}")
    except Exception as e:
        print(f"  wilcoxon 계산 불가: {e}")

print("\n=== 저장 ===")
print(out_path)
