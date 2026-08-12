"""One-vs-rest 독립 3모델의 POST_OOS 전체구간 결과(5/5시드, cost1/2/3 전부 p=0.0312)가 TCN이
겪었던 것과 같은 재현 안 되는 단일-draw 우연인지 확인 -- Optuna로 이미 확정된 winners.json은
그대로 재사용(HP 재탐색 없음), N=5 완전히 새로운 무작위 시드로만 재학습해 같은 결과가
나오는지 직접 검증한다."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PREV_OUT = ROOT / "tmp/eth_h48qual_onevsrest_specialist_20260812"
OUT_DIR = ROOT / "tmp/eth_h48qual_onevsrest_reproduction_check_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
POST_OOS_START, POST_OOS_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-08-04 23:59:59")
N_FINAL_SEEDS = 5
COST_MULTS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}


def log(msg: str) -> None:
    print(msg, flush=True)


RAW_WIDE = ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio",
            "cvd_12", "cvd_48", "cvd_288", "cvd_slope_12", "cvd_slope_48", "taker_acceleration", "big_trade_ratio",
            "funding_roc_12", "funding_roc_48", "ou_funding_z", "btc_ret_3", "btc_ret_12", "eth_btc_ret_spread_12",
            "parkinson_vol", "hurst_48", "kalman_velocity", "mtf_trend_1h", "mtf_trend_4h"]

winners = json.loads((PREV_OUT / "winners.json").read_text())
log(f"기존 winners.json 재사용(HP 재탐색 없음): {list(winners.keys())}")

log("zig075 소스 패널 + zigzag_action 라벨 로딩...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").reset_index(drop=True)

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")
label_map = labels.set_index("timestamp")["zigzag_action"]

df = panel.merge(label_map.reset_index(), on="timestamp", how="left").sort_values("timestamp").reset_index(drop=True)
X_all = df[RAW_WIDE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END) & df["zigzag_action"].notna()
val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END) & df["zigzag_action"].notna()
oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END) & df["zigzag_action"].notna()
post_oos_mask = (df["timestamp"] >= POST_OOS_START) & (df["timestamp"] <= POST_OOS_END)

y_action = pd.to_numeric(df["zigzag_action"], errors="coerce")
targets = {"cash": (y_action == 0).astype(int), "long": (y_action == 1).astype(int), "short": (y_action == 2).astype(int)}
ACTION_ORDER = ["cash", "long", "short"]
X_train = X_all[train_mask].reset_index(drop=True)

sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

fee, slip = omega._load_fee_slip()
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0


def build_dec(action):
    action = action.astype(np.int64)
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


def forced_side(dec, side_value):
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
log(f"완전히 새로운 N={N_FINAL_SEEDS}시드(무작위, 원래 실행과 무관): {seeds}")

months = pd.period_range("2024-06", "2026-08", freq="M")
pnl_rows, clf_rows, monthly_rows = [], [], []

for seed in seeds:
    probs = {}
    for head in ACTION_ORDER:
        p = {k: v for k, v in winners[head]["params"].items() if k != "class_weight_mode"}
        cw = winners[head]["class_weight_mode"]
        y_head_train = targets[head][train_mask].to_numpy()
        sw = compute_sample_weight("balanced", y_head_train) if cw == "balanced" else None
        model = lgb.LGBMClassifier(objective="binary", n_estimators=500, random_state=seed, verbosity=-1, n_jobs=-1, **p)
        model.fit(X_train, y_head_train, sample_weight=sw)
        probs[head] = model.predict_proba(X_all)[:, 1]
    prob_matrix = np.stack([probs[h] for h in ACTION_ORDER], axis=1)
    action_full = prob_matrix.argmax(axis=1)
    ts_all = df["timestamp"].to_numpy()

    for split_name, mask, has_label in [("VAL", val_mask, True), ("OOS", oos_mask, True), ("POST_OOS", post_oos_mask, False)]:
        m = mask.to_numpy()
        pred = action_full[m]
        if has_label:
            y_true = y_action[m].to_numpy().astype(np.int64)
            clf_rows.append({"seed": seed, "split": split_name, "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                              "macro_f1": f1_score(y_true, pred, average="macro")})
        ohlc = df.loc[m, ["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec = build_dec(pred)
        for cost_name, cost_mult in COST_MULTS.items():
            m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=cost_mult)
            m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=cost_mult)
            pnl_rows.append({"seed": seed, "split": split_name, "cost": cost_name, "model_pnl": m_model["pnl"],
                              "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                              "beats_short": m_model["pnl"] > m_short["pnl"], "beats_long": m_model["pnl"] > m_long["pnl"]})

    for month in months:
        m_start, m_end = month.start_time, month.end_time
        mm = (ts_all >= np.datetime64(m_start)) & (ts_all <= np.datetime64(m_end))
        if mm.sum() < 50:
            continue
        pred_m = action_full[mm]
        ohlc_m = df.loc[mm, ["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        dec_m = build_dec(pred_m)
        mm_model = omega._metrics(ohlc_m, dec_m, fee=fee, slip=slip, cost_mult=3.0)
        mm_short = omega._metrics(ohlc_m, forced_side(dec_m, -1), fee=fee, slip=slip, cost_mult=3.0)
        mm_long = omega._metrics(ohlc_m, forced_side(dec_m, 1), fee=fee, slip=slip, cost_mult=3.0)
        price_ret = (ohlc_m["close"].iloc[-1] / ohlc_m["close"].iloc[0] - 1) * 100
        monthly_rows.append({"seed": seed, "month": str(month), "in_sample": bool(m_start <= TRAIN_END),
                              "price_ret_pct": float(price_ret), "model_pnl": mm_model["pnl"],
                              "always_short_pnl": mm_short["pnl"], "always_long_pnl": mm_long["pnl"],
                              "beats_both": (mm_model["pnl"] > mm_short["pnl"]) and (mm_model["pnl"] > mm_long["pnl"])})
    log(f"  seed={seed} 완료")

pnl_df, clf_df, monthly_df = pd.DataFrame(pnl_rows), pd.DataFrame(clf_rows), pd.DataFrame(monthly_rows)
pnl_df.to_csv(OUT_DIR / "pnl_comparison.csv", index=False)
monthly_df.to_csv(OUT_DIR / "monthly_breakdown.csv", index=False)

log(f"\n{'='*100}\n재현성 검증(완전 새 시드) 결과\n{'='*100}")
for split_name in ["VAL", "OOS", "POST_OOS"]:
    for cost_name in COST_MULTS:
        sub = pnl_df[(pnl_df["split"] == split_name) & (pnl_df["cost"] == cost_name)]
        beat_s = int(sub["beats_short"].sum())
        diff = (sub["model_pnl"] - sub["always_short_pnl"]).to_numpy()
        wp = stats.wilcoxon(diff, alternative="greater")[1] if len(sub) >= 5 and np.any(diff != 0) else float("nan")
        log(f"  [{split_name}/{cost_name}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
            f"  always_short={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}"
            f"  승={beat_s}/{len(sub)}  wilcoxon_p={wp:.4f}")

log(f"\n--- POST_OOS 월별 분해(cost3) ---")
post_agg = monthly_df[(monthly_df["month"] >= "2026-03") & (monthly_df["month"] <= "2026-08")].groupby("month").agg(
    price_ret_pct=("price_ret_pct", "first"), model_pnl_mean=("model_pnl", "mean"),
    always_short_mean=("always_short_pnl", "mean"), beats_both_rate=("beats_both", "mean")).reset_index().sort_values("month")
for _, r in post_agg.iterrows():
    log(f"  {r['month']} 가격{r['price_ret_pct']:+6.1f}%  model={r['model_pnl_mean']:+7.2f}  short={r['always_short_mean']:+7.2f}  둘다승={r['beats_both_rate']*100:5.0f}%")
cm, cs = 1.0, 1.0
for _, r in post_agg.iterrows():
    cm *= (1 + r["model_pnl_mean"] / 100)
    cs *= (1 + r["always_short_mean"] / 100)
log(f"  월별 복리 누적: model={100*(cm-1):+.2f}%  always_short={100*(cs-1):+.2f}%")
log(f"\n출력: {OUT_DIR}")
