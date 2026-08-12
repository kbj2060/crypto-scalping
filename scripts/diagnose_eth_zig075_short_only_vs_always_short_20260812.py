"""zig075판 short-only 격리 테스트 -- h48qual에서 이미 돌린
`scripts/diagnose_eth_h48qual_short_only_vs_always_short_20260811.py`의 방법론(모델이 실제로
숏을 고른 bar만 골라 `model_short_only`를 만들고, 같은 active set 전체를 강제숏한
`always_short`와 pnl/승률로 직접 대조)을 zig075의 실제 라이브 번들(102피쳐,
true_3head_tabm_bundle.pt, 2026-06-29 학습, quality_threshold=0.75)에 적용.

데이터 로딩은 `scripts/diagnose_eth_zig075_quality_trend_bias_20260811.py`를 그대로 재사용
(TRAIN_CSV가 이 번들 고유의 alpha5 regime4 계열이라는 점, EVAL_CSV/SPLIT_TS 전부 h48qual과
동일하게 이미 검증됨). 이 번들은 재학습 없이 이미 저장된 예측(q075.csv)만 쓰므로 단일
인스턴스(N=1) 확인이며, h48qual의 5/15시드 재현 테스트만큼 통계적으로 강하지 않다 -- 결과
해석 시 명시할 것.

참고: zig075의 quality_head는 h48qual의 독립적 48bar 배리어 라벨이 아니라
`quality_mode=same_as_direction`(direction_head와 동일한 zigzag_action 라벨로 학습) --
결과 해석 시 이 차이를 유의."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

D = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"

LIVE_TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")


def _read(path, usecols=None):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False, usecols=usecols)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


print("라이브 VAL 프레임(alpha5 regime4 계열) 로드 중...", flush=True)
train_all = _read(LIVE_TRAIN_CSV, usecols=["timestamp", "open", "high", "low", "close"])
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


def side_only_dec(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    keep = active & (pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64) == side_value)
    drop = active & ~keep
    out.loc[drop, "notional_exposure"] = 0.0
    return out


rows = []
for split_name, frame, fname, prefix in [
    ("VAL", val_raw, "validation_predictions_q075.csv", "omega1_regime3_expertdq_oof"),
    ("OOS", oos_raw, "oos_predictions_q075.csv", "omega1_regime3_expertdq"),
]:
    src = pd.read_csv(D / fname, parse_dates=["timestamp"])
    f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    src_aligned = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
    assert len(f) == len(src_aligned), f"{split_name} length mismatch"

    dec_model = omega._to_fixed_decisions(src_aligned, oof=("oof" in prefix))
    dec_short_forced = forced_side_dec(dec_model, -1)
    dec_short_only = side_only_dec(dec_model, -1)
    dec_long_only = side_only_dec(dec_model, 1)

    m_model = omega._metrics(f, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
    m_always_short = omega._metrics(f, dec_short_forced, fee=fee, slip=slip, cost_mult=cost_mult)
    m_short_only = omega._metrics(f, dec_short_only, fee=fee, slip=slip, cost_mult=cost_mult)
    m_long_only = omega._metrics(f, dec_long_only, fee=fee, slip=slip, cost_mult=cost_mult)

    rows.append({
        "split": split_name,
        "model_pnl": m_model["pnl"], "model_trades": m_model["trades"], "model_wr": m_model["wr"],
        "always_short_pnl": m_always_short["pnl"], "always_short_trades": m_always_short["trades"], "always_short_wr": m_always_short["wr"],
        "short_only_pnl": m_short_only["pnl"], "short_only_trades": m_short_only["trades"], "short_only_wr": m_short_only["wr"],
        "long_only_pnl": m_long_only["pnl"], "long_only_trades": m_long_only["trades"], "long_only_wr": m_long_only["wr"],
    })

df = pd.DataFrame(rows)
out_path = ROOT / "tmp/eth_zig075_short_only_vs_always_short_20260812/short_only_vs_always_short.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

pd.set_option("display.width", 220)
print("\n===================== zig075 (live bundle, N=1 instance) =====================")
for _, r in df.iterrows():
    print(f"\n--- {r['split']} ---")
    print(f"  model(전체, 롱+숏 실제선택):              pnl={r['model_pnl']:+7.2f}  trades={r['model_trades']:5.0f}  wr={r['model_wr']*100:5.1f}%")
    print(f"  always_short(같은 active set, 방향 강제): pnl={r['always_short_pnl']:+7.2f}  trades={r['always_short_trades']:5.0f}  wr={r['always_short_wr']*100:5.1f}%")
    print(f"  model_short_only(실제 숏 선택만):         pnl={r['short_only_pnl']:+7.2f}  trades={r['short_only_trades']:5.0f}  wr={r['short_only_wr']*100:5.1f}%")
    print(f"  model_long_only(실제 롱 선택만):          pnl={r['long_only_pnl']:+7.2f}  trades={r['long_only_trades']:5.0f}  wr={r['long_only_wr']*100:5.1f}%")
    beats_pnl = r['short_only_pnl'] > r['always_short_pnl']
    beats_wr = r['short_only_wr'] > r['always_short_wr']
    print(f"  >> short_only가 always_short 이김? pnl={'YES' if beats_pnl else 'NO'}  승률={'YES' if beats_wr else 'NO'}")

print("\n=== 저장 ===")
print(out_path)
