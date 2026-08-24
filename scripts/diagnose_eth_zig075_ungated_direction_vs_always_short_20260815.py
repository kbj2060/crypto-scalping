"""Odyssey1이 h48qual에 대해 확립한 "ungated direction_head" 스킬 테스트
(`scripts/diagnose_eth_h48qual_ungated_direction_vs_always_short_20260812.py`)를 zig075에
그대로 적용. h48qual의 direction_head는 N>=5 다양시드로 스킬 없음이 formal하게 확정됐지만
(`docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`), zig075는
같은 아키텍처(3-Head TabM, 같은 zigzag_action 라벨)를 공유하면서도 이 formal 테스트를 받은 적이
없다(기존 zig075 체크는 전부 quality_threshold=0.75로 이미 게이트된 active set 위에서
model vs always_short를 비교한 것 -- `scripts/diagnose_eth_zig075_short_only_vs_always_short_20260812.py`
등 -- gate 자체를 무시한 순수 direction_head 스킬 테스트는 아니었음).

quality_head를 완전히 무시(quality_threshold 미적용)하고 direction_head의 원본 argmax
(`dir_action`)만으로 매 bar 거래 시뮬레이션 -> always_long/always_short과 대조.

재학습 없음(기본 인자) -- ETH 라이브 zig075 번들(`FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH`,
quality_threshold=0.75)의 기존 저장 예측(q075.csv)만 재사용. 단일 인스턴스(N=1, 이미 배포된
시드 하나) -- Seed-Diversity Ensemble Promotion Gate 정책상 이 결과만으로 확정 결론을 낼 수
없고, Stage 2(N>=5 다양시드 formal 재학습) 필요 여부를 가늠하는 preliminary 신호로만 쓴다.

2026-08-15 Stage 2 갱신: --bundle-dir/--out-csv 인자를 추가해 다른 시드로 재학습된 zig075
번들 디렉토리(validation/oos_predictions_q075.csv를 포함하는 out_dir)에도 로직 변경 없이
재사용 가능하게 일반화. 인자를 생략하면 기존 배포 번들(N=1 Stage 1과 동일)을 그대로 검사한다."""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

# 라이브 ETH zig075 번들 (trading_bot_modules/runtime_config.py:
# FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH, quality_threshold=0.75 -- omega4_6_1_live.py:290).
DEFAULT_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"
LIVE_TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")  # zig075/h48qual 기존 진단 스크립트 전체와 동일한 라이브 VAL 시작점

ap = argparse.ArgumentParser()
ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR, help="validation/oos_predictions_q075.csv를 포함하는 zig075 parent out_dir")
ap.add_argument("--out-csv", type=Path, default=None, help="결과 CSV 저장 경로 (기본: tmp/eth_zig075_ungated_direction_vs_always_short_20260815/ungated_vs_always_short.csv)")
args = ap.parse_args()
BUNDLE_DIR = Path(args.bundle_dir)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def ungated_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """_to_fixed_decisions와 동일 로직이지만 final_action(quality gate 통과분) 대신
    dir_action(direction_head 원본 argmax, gate 없음)을 사용."""
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
val_price = _read(LIVE_TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
val_price = val_price[val_price["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_price = _read(EVAL_CSV)
print(f"VAL n={len(val_price)}  OOS n={len(oos_price)}", flush=True)

rows = []
for split_name, price_frame, fname, prefix in [
    ("VAL", val_price, "validation_predictions_q075.csv", "omega1_regime3_expertdq_oof"),
    ("OOS", oos_price, "oos_predictions_q075.csv", "omega1_regime3_expertdq"),
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
        "always_short_pnl": m_short["pnl"], "always_short_trades": m_short["trades"], "always_short_wr": m_short["wr"],
        "always_long_pnl": m_long["pnl"], "always_long_trades": m_long["trades"], "always_long_wr": m_long["wr"],
        "ungated_beats_always_short": m_ungated["pnl"] > m_short["pnl"],
        "ungated_beats_always_long": m_ungated["pnl"] > m_long["pnl"],
        "ungated_beats_max_baseline": m_ungated["pnl"] > max(m_short["pnl"], m_long["pnl"]),
    })

df = pd.DataFrame(rows)
out_path = Path(args.out_csv) if args.out_csv is not None else ROOT / "tmp/eth_zig075_ungated_direction_vs_always_short_20260815/ungated_vs_always_short.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
print()
print(f"bundle_dir: {BUNDLE_DIR}")
print(df.to_string(index=False))
print("\n저장:", out_path)
