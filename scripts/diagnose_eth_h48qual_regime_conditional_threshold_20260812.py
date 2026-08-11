"""팀장 리서치 문서 후보 3: quality_threshold를 레짐(bull/bear/chop, router_expert 하드 라우팅)별로
따로 두면 전역 0.40~0.80 스윕과 다른 최적점이 나오는가? quality_for_action은 threshold와 무관한
값이라(재계산 아님, 저장된 값 그대로) final_action = dir_action if quality_for_action>=threshold
else CASH를 직접 재구성해서 임의 threshold를 재학습 없이 테스트한다. 풀링(전 시드 합산)으로
표본을 확보 -- 레짐별로 쪼개면 시드당 거래수가 더 줄어든다."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0

RUN_ROOT = ROOT / "tmp/causal_regen_20260516"
VARIANTS = {
    "h48orig": {
        "tag": "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h48orig_20260811_r30000_s",
        "seeds": [260620, 481003, 26611, 903174, 155827],
    },
    "h384_v2": {
        "tag": "omega4_3head_parent72_loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s",
        "seeds": [260620, 481003, 26611, 903174, 155827, 44452, 51724, 179660, 240382, 375044, 378518, 692713, 711841, 750878, 821662],
    },
}
TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
REGIMES = ["bull", "bear", "chop"]


def _read(path):
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


train_all = _read(TRAIN_CSV)
val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
oos_raw = _read(EVAL_CSV)
fee, slip = omega._load_fee_slip()
cost_mult = 3.0


def decisions_at_threshold(src: pd.DataFrame, prefix: str, threshold: float, regime: str | None) -> pd.DataFrame:
    dir_action = pd.to_numeric(src[f"{prefix}_dir_action"], errors="raise").to_numpy(dtype=np.int64)
    qfa = pd.to_numeric(src[f"{prefix}_quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    passed = qfa >= float(threshold)
    action = np.where(passed, dir_action, omega.ACTION_CASH)
    if regime is not None:
        router = src[f"{prefix}_router_expert"].astype(str).to_numpy()
        action = np.where(router == regime, action, omega.ACTION_CASH)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action, "side": side,
        "notional_exposure": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "leverage": np.where(active, float(omega.BASE_TEMPLATE["leverage"]), 1.0),
        "position_fraction": np.where(active, float(omega.BASE_TEMPLATE["notional"]), 0.0),
        "take_profit": np.where(active, float(omega.BASE_TEMPLATE["take_profit"]), 0.0),
        "stop_loss": np.where(active, float(omega.BASE_TEMPLATE["stop_loss"]), 0.0),
        "max_hold_bars": np.where(active, int(omega.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
        "cooldown_bars": np.where(active, int(omega.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
    })


for variant_name, cfg in VARIANTS.items():
    print(f"\n{'=' * 78}\n{variant_name} ({len(cfg['seeds'])}시드 풀링)\n{'=' * 78}")
    for split_name, frame, oof, fname in [
        ("VAL", val_raw, True, "validation_predictions_q050.csv"),
        ("OOS", oos_raw, False, "oos_predictions_q050.csv"),
    ]:
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        aligned = []
        for seed in cfg["seeds"]:
            path = RUN_ROOT / f"{cfg['tag']}{seed}" / fname
            if not path.exists():
                continue
            src = pd.read_csv(path, parse_dates=["timestamp"])
            f = frame.merge(src[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            src_a = src.merge(f[["timestamp"]], on="timestamp", how="inner").reset_index(drop=True)
            aligned.append((f, src_a))
        if not aligned:
            continue
        print(f"\n--- {split_name} ---")
        header = f"{'threshold':>9} | {'전체 pnl':>9} {'trades':>7} | " + " | ".join(f"{r:>16}" for r in REGIMES)
        print(header)
        rows_out = []
        for th in THRESHOLDS:
            pnl_all, trades_all = [], []
            regime_stats = {r: {"pnl": [], "trades": []} for r in REGIMES}
            for f, src_a in aligned:
                dec = decisions_at_threshold(src_a, prefix, th, None)
                m = omega._metrics(f, dec, fee=fee, slip=slip, cost_mult=cost_mult)
                pnl_all.append(m["pnl"])
                trades_all.append(m["trades"])
                for r in REGIMES:
                    dec_r = decisions_at_threshold(src_a, prefix, th, r)
                    m_r = omega._metrics(f, dec_r, fee=fee, slip=slip, cost_mult=cost_mult)
                    regime_stats[r]["pnl"].append(m_r["pnl"])
                    regime_stats[r]["trades"].append(m_r["trades"])
            line = f"{th:>9.2f} | {np.mean(pnl_all):>+9.2f} {np.mean(trades_all):>7.1f} | "
            line += " | ".join(f"pnl={np.mean(regime_stats[r]['pnl']):>+7.2f}(n={np.mean(regime_stats[r]['trades']):>4.1f})" for r in REGIMES)
            print(line)
            row = {"threshold": th, "pooled_pnl": np.mean(pnl_all), "pooled_trades": np.mean(trades_all)}
            for r in REGIMES:
                row[f"{r}_pnl"] = np.mean(regime_stats[r]["pnl"])
                row[f"{r}_trades"] = np.mean(regime_stats[r]["trades"])
            rows_out.append(row)
        dfres = pd.DataFrame(rows_out)
        best_pooled = dfres.loc[dfres["pooled_pnl"].idxmax(), "threshold"]
        print(f"  풀링 최적 threshold: {best_pooled:.2f}")
        for r in REGIMES:
            best_r = dfres.loc[dfres[f"{r}_pnl"].idxmax(), "threshold"]
            diff_note = "  <-- 풀링과 다름" if abs(best_r - best_pooled) > 1e-9 else "  (풀링과 동일)"
            print(f"  {r:>4} 최적 threshold: {best_r:.2f}{diff_note}")
