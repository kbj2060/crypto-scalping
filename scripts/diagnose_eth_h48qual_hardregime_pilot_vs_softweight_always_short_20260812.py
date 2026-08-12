"""사용자 제안(레짐별 TabM 완전 분리학습) 검증: 기존 h48orig는 이미 bull/bear/chop 3-expert를
regime route 확률로 soft-weight해 학습 중이었음이 밝혀져(train_eval_omega4_3head_parent72_
loose_entry_quality_20260620.py의 _fit_expert_omega4), --hard-regime-filter 플래그를 추가해
같은 시드(260620)로 hard filter(그 레짐으로 argmax된 bar만, 가중치 0/1) 버전을 파일럿 학습했다.
이 스크립트는 diagnose_eth_h48qual_ungated_direction_h48orig_5seed_vs_always_short_20260812.py와
동일 방법론(게이트 우회, always_short/long 동일 active set 강제)으로 hard-filter 번들 하나를
평가하고, 이미 알려진 h48orig soft-weight seed=260620 결과(VAL -7.98%, OOS +8.02%, gated)와
직접 대조한다. 단일시드 파일럿 -- N>=5 확장은 신호가 보일 때만."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h48orig_hardregime_20260812_r30000_s260620"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def ungated_decisions(src: pd.DataFrame, prefix: str, oof: bool) -> pd.DataFrame:
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
train_all = _read(TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)

rows = []
for split_name, frame, oof, fname, prefix in [
    ("VAL", val_raw, True, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
    ("OOS", oos_raw, False, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
]:
    src = pd.read_csv(BUNDLE_DIR / fname, parse_dates=["timestamp"])
    f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    assert len(f) == len(src_aligned), f"{split_name} length mismatch {len(f)} vs {len(src_aligned)}"

    dec_gated = omega._to_fixed_decisions(src_aligned, oof=oof)
    dec_ungated = ungated_decisions(src_aligned, prefix, oof)
    dec_short = forced_side_dec(dec_ungated, -1)
    dec_long = forced_side_dec(dec_ungated, 1)

    m_gated = omega._metrics(f, dec_gated, fee=fee, slip=slip, cost_mult=cost_mult)
    m_ungated = omega._metrics(f, dec_ungated, fee=fee, slip=slip, cost_mult=cost_mult)
    m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
    m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

    rows.append({
        "split": split_name,
        "gated_pnl": m_gated["pnl"], "gated_trades": m_gated["trades"], "gated_wr": m_gated.get("wr"),
        "ungated_pnl": m_ungated["pnl"], "ungated_trades": m_ungated["trades"],
        "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
        "always_long_pnl": m_long["pnl"], "always_long_trades": m_long["trades"],
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print()
print("=== hard-regime-filter 파일럿 (seed=260620) ===")
print(df.to_string(index=False))
print()
print("=== 참고: h48orig soft-weight 기존 결과 (동일 시드 260620) ===")
print("VAL  gated_pnl=-7.981600  gated_trades=45  always_short_pnl=8.251175")
print("OOS  gated_pnl=8.024482   gated_trades=24  always_short_pnl=22.501344")
