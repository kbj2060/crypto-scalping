"""SOL 버전: `scripts/diagnose_eth_zig075_ungated_direction_vs_always_short_20260815.py`를
동일 로직으로 SOL 라이브 zig075-동형 번들(`sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720`,
FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH)에 적용.

quality_head를 완전히 무시(quality_threshold 미적용)하고 direction_head의 원본 argmax
(`dir_action`)만으로 매 bar 거래 시뮬레이션 -> always_long/always_short과 대조.

ETH판과의 차이(둘 다 근거 있음, 임의 변경 아님):
- 리스크/PnL 시뮬레이션은 SOL 전용 모듈(`train_eval_omega1_2_tabm_diffusion_risk_sol_20260707`)
  사용 -- SOL 자체 BASE_TEMPLATE(notional/leverage/TP/SL)와 fee/slip을 씀. ETH 상수를
  SOL에 재사용하지 않는다는 이 저장소 확립 규칙([[sol_adaptive_squeeze_v2_20260720]]).
- 결정(포지션/사이드) 구성 함수(`_to_fixed_decisions`)는 SOL 학습 스크립트 자신이 실제로
  호출하는 것과 동일하게 `parent._to_decisions`를 통해 간접 호출한다 -- 이 함수는
  `train_eval_omega1_2_tabm_3head_20260603.py` 내부에 하드코딩된 ETH 모듈의
  `_to_fixed_decisions`를 그대로 재사용하는 공유 아키텍처 레이어(자산별 fee/TP/SL과 무관,
  action 코드 -> 포지션 상태 변환 로직만 담당)라서 SOL 학습 파이프라인도 실제로 이 경로를
  탄다 -- 다르게 부르면 오히려 실제 학습/평가 경로와 불일치하게 된다.
- 가격 소스: SOL feature 빌드 산출물(`data/splits/year_oos_adaptive_squeeze_sol_20260720/
  sol_features_2025.csv`/`sol_features_2026.csv`, timestamp/open/high/low/close 포함) --
  예측 CSV 자체엔 가격 컬럼이 없어 별도 조인 필요(ETH판과 동일 패턴).
- VAL/OOS 구간: 기존 배포 번들의 저장 예측 CSV가 실제로 덮는 범위 그대로
  (VAL 2025-10-01~12-31, OOS는 2026-01-01~03-31을 1차, 04-01~06-30을 참고 2차로 분리 보고 --
  ETH formal 테스트는 OOS 창 1개였지만 SOL 번들 예측 CSV가 이미 07-12까지 존재해 추가 비용
  없이 참고창을 하나 더 보고할 수 있어 사전에 추가하기로 함, 사후선택 아님).

재학습 없음(기본 인자) -- 인자로 받은 bundle_dir의 기존 저장 예측만 재사용."""
import argparse
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

DEFAULT_BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_adaptive_squeeze_20260720"
VAL_PRICE_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2025.csv"
OOS_PRICE_CSV = ROOT / "data/splits/year_oos_adaptive_squeeze_sol_20260720/sol_features_2026.csv"
VAL_START = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2025-12-31 23:59:59")
OOSQ1_START = pd.Timestamp("2026-01-01")
OOSQ1_END = pd.Timestamp("2026-03-31 23:59:59")
OOSQ2_START = pd.Timestamp("2026-04-01")
OOSQ2_END = pd.Timestamp("2026-06-30 23:59:59")

ap = argparse.ArgumentParser()
ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR, help="validation/oos_predictions_qXXX.csv를 포함하는 SOL zig075 parent out_dir")
ap.add_argument("--pred-tag", default="q070", help="사용할 quality threshold 태그 (ungated 비교엔 threshold 자체는 무시되지만 파일 선택에 필요; 라이브 SOL은 q070)")
ap.add_argument("--out-csv", type=Path, default=None)
args = ap.parse_args()
BUNDLE_DIR = Path(args.bundle_dir)

fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def _read_price(path, lo, hi):
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"], low_memory=False)
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)]
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def ungated_decisions(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    src2 = src.copy()
    src2[f"{prefix}_final_action"] = src2[f"{prefix}_dir_action"]
    oof = "oof" in prefix
    return parent._to_decisions(src2, oof=oof)


def forced_side_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


print("VAL/OOS 가격 프레임 로드 중...", flush=True)
val_price = _read_price(VAL_PRICE_CSV, VAL_START, VAL_END)
oosq1_price = _read_price(OOS_PRICE_CSV, OOSQ1_START, OOSQ1_END)
oosq2_price = _read_price(OOS_PRICE_CSV, OOSQ2_START, OOSQ2_END)
print(f"VAL n={len(val_price)}  OOS-Q1 n={len(oosq1_price)}  OOS-Q2(참고) n={len(oosq2_price)}", flush=True)

rows = []
for split_name, price_frame, fname, prefix in [
    ("VAL", val_price, f"validation_predictions_{args.pred_tag}.csv", "omega1_regime3_expertdq_oof"),
    ("OOS-Q1", oosq1_price, f"oos_predictions_{args.pred_tag}.csv", "omega1_regime3_expertdq"),
    ("OOS-Q2(ref)", oosq2_price, f"oos_predictions_{args.pred_tag}.csv", "omega1_regime3_expertdq"),
]:
    src_full = pd.read_csv(BUNDLE_DIR / fname, parse_dates=["timestamp"])
    f = price_frame.merge(src_full[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    src_aligned = src_full.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    assert len(f) == len(src_aligned), f"{split_name} length mismatch {len(f)} vs {len(src_aligned)}"

    dec_gated = parent._to_decisions(src_aligned, oof=("oof" in prefix))
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
out_path = Path(args.out_csv) if args.out_csv is not None else ROOT / "tmp/sol_zig075_ungated_direction_vs_always_short_20260824/ungated_vs_always_short.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
print()
print(f"bundle_dir: {BUNDLE_DIR}")
print(df.to_string(index=False))
print("\n저장:", out_path)
