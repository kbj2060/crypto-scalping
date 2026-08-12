"""신규 탐색 축 스카우팅 문서(eth_h48qual_direction_skill_new_directions_scouting_20260812.md)의
1순위 후보(Tier 0, 재학습 불필요): TRAIN(2024-01~2025-09) 내부에서 2025 Q2~Q3(2025-04~2025-10,
+36%->+67% 연속 상승)를 홀드아웃해, direction_head 원본(게이트 없음)이 이 상승구간에서는
always-long/short 대비 다른 결과를 보이는지 확인한다. 인샘플(모델이 이 데이터로 학습함)이라
스킬 증명은 못 되지만, "VAL/OOS 하락장 특정적 실패"인지 "이 피쳐/라벨 조합엔 방향 스킬 자체가
없음"인지를 구분하는 값싼 정찰용 게이트. 재학습 없음 -- 이미 재생성된 fullwindow TRAIN 예측
(scripts/regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py 산출물) 재사용."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

BUNDLE_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630_fullwindow_predictions_recheck_20260812"
KLINE_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
PREFIX = "omega1_regime3_expertdq_oof"
WINDOW_START = pd.Timestamp("2025-04-01")
WINDOW_END = pd.Timestamp("2025-10-01")

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


print("가격 프레임 + TRAIN 예측 로드 중...", flush=True)
price = _read(KLINE_CSV, usecols=["timestamp", "open", "high", "low", "close"])
price = price[(price.timestamp >= WINDOW_START) & (price.timestamp < WINDOW_END)].reset_index(drop=True)
print(f"가격 구간: {price.timestamp.min()} ~ {price.timestamp.max()}  n={len(price)}  "
      f"시작종가={price.close.iloc[0]:.2f} 종료종가={price.close.iloc[-1]:.2f} "
      f"수익률={(price.close.iloc[-1]/price.close.iloc[0]-1)*100:.1f}%", flush=True)

src = pd.read_csv(BUNDLE_DIR / "train_predictions_q050.csv", parse_dates=["timestamp"])
f = price.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
assert len(f) == len(src_aligned), f"length mismatch {len(f)} vs {len(src_aligned)}"
print(f"매칭된 bar 수: {len(f)}", flush=True)

dec_gated = omega._to_fixed_decisions(src_aligned, oof=True)
dec_ungated = ungated_decisions(src_aligned, PREFIX, oof=True)
dec_short = forced_side_dec(dec_ungated, -1)
dec_long = forced_side_dec(dec_ungated, 1)

m_gated = omega._metrics(f, dec_gated, fee=fee, slip=slip, cost_mult=cost_mult)
m_ungated = omega._metrics(f, dec_ungated, fee=fee, slip=slip, cost_mult=cost_mult)
m_short = omega._metrics(f, dec_short, fee=fee, slip=slip, cost_mult=cost_mult)
m_long = omega._metrics(f, dec_long, fee=fee, slip=slip, cost_mult=cost_mult)

print()
print(f"{'구성':<14} {'pnl%':>10} {'trades':>8} {'wr':>8}")
for name, m in [("gated(라이브)", m_gated), ("ungated(원본)", m_ungated),
                 ("always_short", m_short), ("always_long", m_long)]:
    print(f"{name:<14} {m['pnl']:>10.3f} {m['trades']:>8d} {m.get('wr', float('nan')):>8.3f}")

best_baseline = max(m_short["pnl"], m_long["pnl"])
print(f"\nmax(always_long, always_short) = {best_baseline:.3f}%")
print(f"ungated이 max(always_long,always_short)를 이김? {m_ungated['pnl'] > best_baseline}")
