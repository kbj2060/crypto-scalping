"""사용자 질문: quality_head를 아예 제거하면 어떤가? 이 세션에서 한 번도 직접 답하지 않은 질문:
게이트(quality_head) 없이 direction_head의 원본 픽(dir_action)만으로 거래하면 always-short/
always-long을 이기는가? 지금까지의 always-short 대조는 전부 게이트 통과 후(final_action) 결과만
비교했음 -- 이건 그 앞 단계(원본 dir_action)를 직접 본다. 재학습 없음, 기존 저장 예측 재사용.
_to_fixed_decisions()가 {prefix}final_action을 읽는 지점에서 dir_action으로 바꿔치기.

2026-08-15 갱신: --bundle-dir/--out-csv 인자를 추가해 일반화(로직 변경 없음 -- 기존 하드코딩
BUNDLE_DIR를 argparse 기본값으로 옮긴 것뿐, zig075 쪽 diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py와
동일 패턴). 인자를 생략하면 기존 배포 번들(q050)을 그대로 검사한다."""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

# 라이브 h48qual 번들 -- VAL/OOS는 원본 그대로(이미 정확 확인됨), TRAIN은 2024-2025 전체 재생성판.
DEFAULT_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/trade_candidates_2024_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")

ap = argparse.ArgumentParser()
ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR, help="validation/oos_predictions_q050.csv를 포함하는 h48qual parent out_dir")
ap.add_argument("--out-csv", type=Path, default=None, help="결과 CSV 저장 경로 (기본: 화면 출력만, 저장 안 함)")
args = ap.parse_args()
BUNDLE_DIR = Path(args.bundle_dir)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def ungated_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """_to_fixed_decisions와 동일 로직이지만 final_action 대신 dir_action을 게이트 없이 사용."""
    src2 = src.copy()
    src2[f"{prefix}_final_action"] = src2[f"{prefix}_dir_action"]
    oof = "oof" in prefix
    return omega._to_fixed_decisions(src2, oof=oof)


def forced_side_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


print("VAL/OOS 가격 프레임 로드 중...", flush=True)
val_price = _read(TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
val_price = val_price[val_price["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_price = _read(EVAL_CSV)

rows = []
for split_name, price_frame, fname, prefix in [
    ("VAL", val_price, "validation_predictions_q050.csv", "omega1_regime3_expertdq_oof"),
    ("OOS", oos_price, "oos_predictions_q050.csv", "omega1_regime3_expertdq"),
]:
    src = pd.read_csv(BUNDLE_DIR / fname, parse_dates=["timestamp"])
    f = price_frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    assert len(f) == len(src_aligned), f"{split_name} length mismatch {len(f)} vs {len(src_aligned)}"

    dec_gated = omega._to_fixed_decisions(src_aligned, oof=("oof" in prefix))
    dec_ungated = ungated_decisions(src_aligned, prefix)
    dec_short = forced_side_dec(dec_ungated, -1)
    dec_long = forced_side_dec(dec_ungated, 1)

    m_gated = omega._metrics(f, dec_gated, fee=fee, slip=slip, cost_mult=cost_mult)
    m_ungated = omega._metrics(f, dec_ungated, fee=fee, slip=slip, cost_mult=cost_mult)
    m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
    m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

    rows.append({
        "split": split_name,
        "gated_pnl": m_gated["pnl"], "gated_trades": m_gated["trades"], "gated_wr": m_gated["wr"],
        "ungated_pnl": m_ungated["pnl"], "ungated_trades": m_ungated["trades"], "ungated_wr": m_ungated["wr"],
        "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"],
        "always_long_pnl": m_long["pnl"], "always_long_trades": m_long["trades"],
        "ungated_beats_always_short": m_ungated["pnl"] > m_short["pnl"],
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print()
print(f"bundle_dir: {BUNDLE_DIR}")
print(df.to_string(index=False))
if args.out_csv is not None:
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("\n저장:", args.out_csv)
