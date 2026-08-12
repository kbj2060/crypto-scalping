"""사용자 제안: "롱과 숏과 캐시를 따로 데이터를 취합해서 모델을 따로 만드는" 것 -- 공유 3-class
softmax(CASH/LONG/SHORT) 대신 완전히 독립적인 3개의 이진분류기(LONG-vs-rest, SHORT-vs-rest,
CASH-vs-rest)를 각자 학습해, 세 확률을 비교(argmax)해서 최종 행동을 정한다.

근거: 2026-08-11 진단(direction confidence calibration)에서 공유 3-class 모델의 LONG이
유독 과소신(-15.0pp)되는 현상이 확인됐다 -- 하나의 공유 표현을 세 클래스가 경쟁하며 나눠 쓰는
구조적 요인일 수 있다는 가설. 독립 모델은 이 경쟁 자체를 없앤다.

단서(사전에 명시): 같은 문제를 겨냥한 사후보정(클래스별 temperature scaling)은 이미 시도돼
VAL에선 개선됐지만 OOS엔 일반화 안 됐다(같은 날 진단). 이번은 사후보정이 아니라 처음부터
별도 학습이라 메커니즘은 다르지만, 정보이론적으로 완전히 새 정보를 만들어내진 않는다는 한계는
여전 -- 기대치는 신중하게.

설계: GBDT(LightGBM) 베이스(오늘 가장 빠르고 잘 검증됨), zig075 소스 패널의 raw_wide 24피쳐
(오늘 TCN 다변량탐색에서 이미 검증된 풀 재사용). 각 이진타겟마다 독립 Optuna 라이트서치(15
trial) -> VAL 재평가로 채택 -> N=5 진짜 무작위 시드 최종검증. 오늘의 교훈 그대로: 거래
시뮬레이션 필수 + POST_OOS 월별분해를 같은 실행 안에 처음부터 포함."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
SMOKE = os.environ.get("OVR_SMOKE", "0") == "1"
OUT_DIR = ROOT / ("tmp/eth_h48qual_onevsrest_specialist_20260812_smoke" if SMOKE else "tmp/eth_h48qual_onevsrest_specialist_20260812")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
POST_OOS_START, POST_OOS_END = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-08-04 23:59:59")

N_TRIALS_PER_HEAD = 2 if SMOKE else 15
N_FINAL_SEEDS = 1 if SMOKE else 5
COST_MULTS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}


def log(msg: str) -> None:
    print(msg, flush=True)


RAW_WIDE = ["log_return", "volatility_z", "rsi", "macd_hist", "bb_width_z", "wick_ratio", "net_taker_ratio",
            "cvd_12", "cvd_48", "cvd_288", "cvd_slope_12", "cvd_slope_48", "taker_acceleration", "big_trade_ratio",
            "funding_roc_12", "funding_roc_48", "ou_funding_z", "btc_ret_3", "btc_ret_12", "eth_btc_ret_spread_12",
            "parkinson_vol", "hurst_48", "kalman_velocity", "mtf_trend_1h", "mtf_trend_4h"]

# ---------------------------------------------------------------------------
# 1. 데이터 로딩
# ---------------------------------------------------------------------------

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
log(f"  TRAIN={train_mask.sum()}  VAL={val_mask.sum()}  OOS={oos_mask.sum()}  POST_OOS={post_oos_mask.sum()}(라벨없음)")

y_action = pd.to_numeric(df["zigzag_action"], errors="coerce")
targets = {"cash": (y_action == 0).astype(int), "long": (y_action == 1).astype(int), "short": (y_action == 2).astype(int)}
ACTION_ORDER = ["cash", "long", "short"]  # argmax 순서 = ACTION_CASH(0)/LONG(1)/SHORT(2)와 일치

X_train, X_val, X_oos = X_all[train_mask].reset_index(drop=True), X_all[val_mask].reset_index(drop=True), X_all[oos_mask].reset_index(drop=True)
ts_train = df.loc[train_mask, "timestamp"].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. omega 거래 시뮬레이션 헬퍼
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402

fee, slip = omega._load_fee_slip()
omega.BASE_TEMPLATE["max_hold"] = 0
omega.BASE_TEMPLATE["cooldown"] = 0


def build_dec(action: np.ndarray) -> pd.DataFrame:
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


def forced_side(dec: pd.DataFrame, side_value: int) -> pd.DataFrame:
    out = dec.copy()
    active = omega._active(dec)
    out.loc[active, "side"] = side_value
    out.loc[active, "action"] = omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT
    return out


# ---------------------------------------------------------------------------
# 3. TRAIN 내부 월별 확장윈도우 CV (GBDT 백본 진단과 동일 관례)
# ---------------------------------------------------------------------------

months_train = sorted(ts_train.dt.to_period("M").unique())
MIN_TRAIN_MONTHS = 6
cv_folds = []
for i in range(MIN_TRAIN_MONTHS, len(months_train)):
    tr_mask = ts_train.dt.to_period("M").isin(months_train[:i]).to_numpy()
    va_mask = (ts_train.dt.to_period("M") == months_train[i]).to_numpy()
    tr_idx, va_idx = np.flatnonzero(tr_mask), np.flatnonzero(va_mask)
    if len(tr_idx) > 500 and len(va_idx) > 100:
        cv_folds.append((tr_idx[:-48] if len(tr_idx) > 48 else tr_idx, va_idx))
log(f"CV 폴드 {len(cv_folds)}개")

# ---------------------------------------------------------------------------
# 4. 헤드별 Optuna 라이트서치 -> VAL 재평가로 채택
# ---------------------------------------------------------------------------

winners = {}
for head in ACTION_ORDER:
    log(f"\n{'='*80}\n헤드: {head}\n{'='*80}")
    y_head = targets[head][train_mask].to_numpy()

    def objective(trial, y_head=y_head):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 0, 7),
        )
        class_weight_mode = trial.suggest_categorical("class_weight_mode", ["none", "balanced"])
        losses = []
        for tr_idx, va_idx in cv_folds:
            Xtr, ytr = X_train.iloc[tr_idx], y_head[tr_idx]
            Xv, yv = X_train.iloc[va_idx], y_head[va_idx]
            sw = compute_sample_weight("balanced", ytr) if class_weight_mode == "balanced" else None
            model = lgb.LGBMClassifier(objective="binary", n_estimators=500, random_state=0, verbosity=-1, n_jobs=-1, **params)
            model.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            losses.append(model.best_score_["valid_0"]["binary_logloss"])
        trial.set_user_attr("class_weight_mode", class_weight_mode)
        return float(np.mean(losses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=20260812))
    study.optimize(objective, n_trials=N_TRIALS_PER_HEAD, show_progress_bar=False)
    log(f"  Optuna 완료: best logloss={study.best_value:.4f}")
    winners[head] = {"params": study.best_params, "class_weight_mode": study.best_trial.user_attrs["class_weight_mode"]}

(OUT_DIR / "winners.json").write_text(json.dumps(winners, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 5. N=5 진짜 무작위 시드 최종 검증: 3개 헤드 독립 학습 -> argmax 결합
# ---------------------------------------------------------------------------

seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
log(f"\n최종 N={N_FINAL_SEEDS}시드(무작위): {seeds}")

months = pd.period_range("2024-06", "2026-08", freq="M")
pnl_rows, clf_rows, monthly_rows = [], [], []

for seed in seeds:
    probs = {}
    for head in ACTION_ORDER:
        p = winners[head]["params"]
        cw = winners[head]["class_weight_mode"]
        y_head_train = targets[head][train_mask].to_numpy()
        sw = compute_sample_weight("balanced", y_head_train) if cw == "balanced" else None
        model = lgb.LGBMClassifier(objective="binary", n_estimators=500, random_state=seed, verbosity=-1, n_jobs=-1, **p)
        model.fit(X_train, y_head_train, sample_weight=sw)
        probs[head] = model.predict_proba(X_all)[:, 1]  # 전체 패널(TRAIN~POST_OOS)에 대해 한번에 추론

    prob_matrix = np.stack([probs[h] for h in ACTION_ORDER], axis=1)  # (N, 3) = [cash, long, short]
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
                              "model_trades": m_model["trades"], "model_wr": m_model["wr"],
                              "always_short_pnl": m_short["pnl"], "always_long_pnl": m_long["pnl"],
                              "beats_short": m_model["pnl"] > m_short["pnl"], "beats_long": m_model["pnl"] > m_long["pnl"]})

    # POST_OOS 월별 분해도 처음부터 같이 (오늘의 교훈)
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
clf_df.to_csv(OUT_DIR / "classification.csv", index=False)
monthly_df.to_csv(OUT_DIR / "monthly_breakdown.csv", index=False)

# ---------------------------------------------------------------------------
# 6. 리포트
# ---------------------------------------------------------------------------

log(f"\n{'='*100}\nOne-vs-Rest 독립 3모델 -- N={N_FINAL_SEEDS}시드 최종 요약\n{'='*100}")
for split_name in ["VAL", "OOS", "POST_OOS"]:
    if split_name in ("VAL", "OOS"):
        csub = clf_df[clf_df["split"] == split_name]
        log(f"\n--- {split_name} 분류: balanced_acc={csub['balanced_accuracy'].mean():.3f}±{csub['balanced_accuracy'].std():.3f} ---")
    else:
        log(f"\n--- {split_name}(라벨없음, PnL만) ---")
    for cost_name in COST_MULTS:
        sub = pnl_df[(pnl_df["split"] == split_name) & (pnl_df["cost"] == cost_name)]
        beat_s, beat_l = int(sub["beats_short"].sum()), int(sub["beats_long"].sum())
        diff = (sub["model_pnl"] - sub["always_short_pnl"]).to_numpy()
        wp = stats.wilcoxon(diff, alternative="greater")[1] if len(sub) >= 5 and np.any(diff != 0) else float("nan")
        log(f"  [{cost_name}] model={sub['model_pnl'].mean():+7.2f}±{sub['model_pnl'].std():5.2f}"
            f"  always_short={sub['always_short_pnl'].mean():+7.2f}±{sub['always_short_pnl'].std():5.2f}"
            f"  always_long={sub['always_long_pnl'].mean():+7.2f}"
            f"  승(short)={beat_s}/{len(sub)}  승(long)={beat_l}/{len(sub)}  wilcoxon_p={wp:.4f}")

log(f"\n--- POST_OOS 월별 분해(cost3, 즉시 검증) ---")
post_agg = monthly_df[(monthly_df["month"] >= "2026-03") & (monthly_df["month"] <= "2026-08")].groupby("month").agg(
    price_ret_pct=("price_ret_pct", "first"), model_pnl_mean=("model_pnl", "mean"),
    always_short_mean=("always_short_pnl", "mean"), always_long_mean=("always_long_pnl", "mean"),
    beats_both_rate=("beats_both", "mean")).reset_index().sort_values("month")
for _, r in post_agg.iterrows():
    log(f"  {r['month']} 가격{r['price_ret_pct']:+6.1f}%  model={r['model_pnl_mean']:+7.2f}  "
        f"short={r['always_short_mean']:+7.2f}  long={r['always_long_mean']:+7.2f}  둘다승={r['beats_both_rate']*100:5.0f}%")
cm, cs = 1.0, 1.0
for _, r in post_agg.iterrows():
    cm *= (1 + r["model_pnl_mean"] / 100)
    cs *= (1 + r["always_short_mean"] / 100)
log(f"  POST_OOS 월별 복리 누적: model={100*(cm-1):+.2f}%  always_short={100*(cs-1):+.2f}%")

log(f"\n출력 디렉토리: {OUT_DIR}")
