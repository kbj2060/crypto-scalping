"""왜 One-vs-rest 독립 3모델이 OOS(2026-01~02, 순수 하락)에서는 완패하고 POST_OOS(2026-03~08,
혼합/휩소)에서는 재현되는 양성 신호를 보이는가 -- 4가지 진단으로 파고든다.

(1) 피쳐 PSI drift: TRAIN 대비 OOS vs POST_OOS -- OOS가 POST_OOS보다 더 낯선(분포이동 큰)
    구간인지 확인.
(2) 모델의 CASH/LONG/SHORT 선택 비율: OOS vs POST_OOS.
(3) OOS 클래스별 정밀도/재현율(정답 라벨 있음) -- 특정 클래스가 유독 나쁜지.
(4) 주간 반등(bounce) vs 하락주 풀링 비교(TCN 후속검증과 동일 방법론) -- OOS와 POST_OOS 둘 다에
    적용해 "휩소 레짐에서 유리하다"는 메커니즘이 이번엔 일관되게 나타나는지 확인(TCN 때는
    VAL/OOS가 정반대로 나와 메커니즘이 기각됐었다)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PREV_OUT = ROOT / "tmp/eth_h48qual_onevsrest_specialist_20260812"
OUT_DIR = ROOT / "tmp/eth_h48qual_onevsrest_regime_diagnosis_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
POST_OOS_START, POST_OOS_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-08-04 23:59:59")
N_SEEDS = 5
ACTION_NAMES = {0: "CASH", 1: "LONG", 2: "SHORT"}


def log(msg: str) -> None:
    print(msg, flush=True)


RAW_WIDE = ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio",
            "cvd_12", "cvd_48", "cvd_288", "cvd_slope_12", "cvd_slope_48", "taker_acceleration", "big_trade_ratio",
            "funding_roc_12", "funding_roc_48", "ou_funding_z", "btc_ret_3", "btc_ret_12", "eth_btc_ret_spread_12",
            "parkinson_vol", "hurst_48", "kalman_velocity", "mtf_trend_1h", "mtf_trend_4h"]

winners = json.loads((PREV_OUT / "winners.json").read_text())

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


# ---------------------------------------------------------------------------
# 1. 피쳐 PSI drift: TRAIN 대비 OOS vs POST_OOS
# ---------------------------------------------------------------------------

def psi(train_vals, other_vals, bins=10):
    edges = np.quantile(train_vals, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    t_counts, _ = np.histogram(train_vals, bins=edges)
    o_counts, _ = np.histogram(other_vals, bins=edges)
    t_pct = np.clip(t_counts / max(len(train_vals), 1), 1e-6, None)
    o_pct = np.clip(o_counts / max(len(other_vals), 1), 1e-6, None)
    return float(np.sum((o_pct - t_pct) * np.log(o_pct / t_pct)))


log(f"\n{'='*90}\n(1) 피쳐 PSI drift: TRAIN 대비 OOS vs POST_OOS\n{'='*90}")
X_train_np = X_train.to_numpy()
X_oos_np = X_all[oos_mask.to_numpy()].to_numpy()
X_post_np = X_all[post_oos_mask.to_numpy()].to_numpy()
psi_rows = []
for i, c in enumerate(RAW_WIDE):
    psi_oos = psi(X_train_np[:, i], X_oos_np[:, i])
    psi_post = psi(X_train_np[:, i], X_post_np[:, i])
    psi_rows.append({"feature": c, "psi_oos": psi_oos, "psi_post_oos": psi_post, "oos_more_drifted": psi_oos > psi_post})
psi_df = pd.DataFrame(psi_rows).sort_values("psi_oos", ascending=False)
psi_df.to_csv(OUT_DIR / "feature_psi_drift.csv", index=False)
log(f"  평균 PSI: OOS={psi_df['psi_oos'].mean():.3f}  POST_OOS={psi_df['psi_post_oos'].mean():.3f}")
log(f"  OOS가 더 드리프트된 피쳐 수: {int(psi_df['oos_more_drifted'].sum())}/{len(RAW_WIDE)}")
log("  상위 8개(OOS 기준):")
for _, r in psi_df.head(8).iterrows():
    log(f"    {r['feature']:<28s} PSI(OOS)={r['psi_oos']:.3f}  PSI(POST_OOS)={r['psi_post_oos']:.3f}")

# ---------------------------------------------------------------------------
# 2~4. N=5 시드 학습 + 전체범위 예측 저장 (재사용)
# ---------------------------------------------------------------------------

seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_SEEDS)
log(f"\n최종 N={N_SEEDS}시드(무작위, 신규): {seeds}")

action_preds = {}  # seed -> full-range action array
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
    action_preds[seed] = prob_matrix.argmax(axis=1)
    log(f"  seed={seed} 학습+추론 완료")

# ---------------------------------------------------------------------------
# 2. 클래스 선택 비율: OOS vs POST_OOS
# ---------------------------------------------------------------------------

log(f"\n{'='*90}\n(2) 모델 예측 클래스 비율: OOS vs POST_OOS\n{'='*90}")
class_dist_rows = []
for split_name, mask in [("OOS", oos_mask), ("POST_OOS", post_oos_mask)]:
    m = mask.to_numpy()
    for seed in seeds:
        pred = action_preds[seed][m]
        vc = pd.Series(pred).value_counts(normalize=True).sort_index()
        row = {"split": split_name, "seed": seed}
        for k in (0, 1, 2):
            row[ACTION_NAMES[k]] = float(vc.get(k, 0.0))
        class_dist_rows.append(row)
class_dist_df = pd.DataFrame(class_dist_rows)
class_dist_df.to_csv(OUT_DIR / "class_distribution.csv", index=False)
for split_name in ["OOS", "POST_OOS"]:
    sub = class_dist_df[class_dist_df["split"] == split_name]
    log(f"  {split_name}: CASH={sub['CASH'].mean()*100:.1f}%  LONG={sub['LONG'].mean()*100:.1f}%  SHORT={sub['SHORT'].mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# 3. OOS 클래스별 정밀도/재현율
# ---------------------------------------------------------------------------

log(f"\n{'='*90}\n(3) OOS 클래스별 정밀도/재현율 (정답 라벨 있음)\n{'='*90}")
prf_rows = []
m_oos = oos_mask.to_numpy()
y_true_oos = y_action[m_oos].to_numpy().astype(np.int64)
for seed in seeds:
    pred = action_preds[seed][m_oos]
    prec, rec, f1, support = precision_recall_fscore_support(y_true_oos, pred, labels=[0, 1, 2], zero_division=0)
    for k in (0, 1, 2):
        prf_rows.append({"seed": seed, "class": ACTION_NAMES[k], "precision": prec[k], "recall": rec[k], "f1": f1[k], "support": int(support[k])})
prf_df = pd.DataFrame(prf_rows)
prf_df.to_csv(OUT_DIR / "oos_precision_recall.csv", index=False)
for k in (0, 1, 2):
    sub = prf_df[prf_df["class"] == ACTION_NAMES[k]]
    log(f"  {ACTION_NAMES[k]:<6s} precision={sub['precision'].mean():.3f}±{sub['precision'].std():.3f}  "
        f"recall={sub['recall'].mean():.3f}±{sub['recall'].std():.3f}  support={sub['support'].mean():.0f}")

# ---------------------------------------------------------------------------
# 4. 주간 반등 vs 하락주 풀링 비교 (OOS + POST_OOS, TCN 후속검증과 동일 방법론)
# ---------------------------------------------------------------------------

log(f"\n{'='*90}\n(4) 주간 반등(bounce) vs 하락주 풀링 비교 -- OOS와 POST_OOS 둘 다\n{'='*90}")
week_of = df["timestamp"].dt.to_period("W-MON")
week_close = df.groupby(week_of)["close"].agg(["first", "last"])
weekly_ret = (week_close["last"] / week_close["first"] - 1) * 100
week_sign = week_of.map(lambda w: "bounce" if weekly_ret.loc[w] > 0 else "down").to_numpy()

bounce_rows = []
for split_name, mask, s_start, s_end in [("OOS", oos_mask, OOS_START, OOS_END), ("POST_OOS", post_oos_mask, POST_OOS_START, POST_OOS_END)]:
    m = mask.to_numpy()
    for regime_tag in ["bounce", "down"]:
        r_mask = m & (week_sign == regime_tag)
        if r_mask.sum() < 50:
            log(f"  [{split_name}/{regime_tag}주] 표본 부족(n={r_mask.sum()}), 스킵")
            continue
        ohlc = df.loc[r_mask, ["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        for seed in seeds:
            pred_r = action_preds[seed][r_mask]
            dec = build_dec(pred_r)
            m_model = omega._metrics(ohlc, dec, fee=fee, slip=slip, cost_mult=3.0)
            m_short = omega._metrics(ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=3.0)
            m_long = omega._metrics(ohlc, forced_side(dec, 1), fee=fee, slip=slip, cost_mult=3.0)
            bounce_rows.append({"split": split_name, "regime": regime_tag, "seed": seed, "n_bars": int(r_mask.sum()),
                                 "model_pnl": m_model["pnl"], "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                                 "beats_short": m_model["pnl"] > m_short["pnl"], "beats_long": m_model["pnl"] > m_long["pnl"]})
bounce_df = pd.DataFrame(bounce_rows)
bounce_df.to_csv(OUT_DIR / "bounce_vs_down_weeks.csv", index=False)
for split_name in ["OOS", "POST_OOS"]:
    for regime_tag in ["bounce", "down"]:
        sub = bounce_df[(bounce_df.split == split_name) & (bounce_df.regime == regime_tag)]
        if sub.empty:
            continue
        beat_s, beat_l = int(sub["beats_short"].sum()), int(sub["beats_long"].sum())
        log(f"  [{split_name}/{regime_tag}주, n_bars={sub['n_bars'].iloc[0]}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
            f"  short={sub['always_short_pnl'].mean():+7.2f}  long={sub['always_long_pnl'].mean():+7.2f}"
            f"  승(short)={beat_s}/{len(sub)}  승(long)={beat_l}/{len(sub)}")

log(f"\n출력: {OUT_DIR}")
