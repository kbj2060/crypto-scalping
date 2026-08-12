"""TabM 백본 대체 후보 리서치(docs/experiments/eth_h48qual_tabm_backbone_replacement_model_research_
20260812.md) 1순위 권장안 실행: FINAL12 -> zigzag_action 3-class 방향 분류를 LightGBM(GBDT)으로
학습해 always-short/always-long과 비교한다. 목적은 "TabM 탓인가 데이터(피쳐/라벨/구간) 탓인가"를
가장 싸게 분리하는 진단이지, GBDT 승격 시도가 아니다.

파이프라인은 h48orig 학습 스크립트(train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_
20260811.py)의 _prepare_frames를 그대로 재사용한다 -- FINAL12 피쳐 구성(REGIME3_CURRENT 오버레이 +
diff1/dt288 파생 4개), 실제 라이브 h48orig 방향 라벨 디렉토리(zigzag_action_labels_20260531),
BASE_TEMPLATE(TP=2.6%/SL=1.4%/notional=0.45/leverage=2.0, max_hold/cooldown=0)을 완전히 그대로
쓴다. 단 TP/SL/notional/leverage는 이 스크립트에서 레짐별 expert_scale(0.75/0.90/0.90)을 적용하지
않는다 -- GBDT에는 대응하는 라우터가 없어서 GBDT 자신과 그 자신의 always-short/always-long
기준선(같은 active bar set 강제숏/강제롱) 사이의 비교는 완전히 대칭이지만, 계약 문서에 기록된
기존 TabM h48orig 숫자(레짐 스케일 포함)와 절대 PnL 크기를 직접 비교할 때는 이 차이를 감안해야
한다.

범위 주의(계약 문서 "함정 1" 참고): TRAIN은 h48orig 5시드 재현판과 동일한 2025-01~09(9개월,
isolated-verification 관례)이다 -- 라이브 번들의 진짜 21개월(2024-01~2025-09, 183,936행) TRAIN이
아니다. 기존 h48orig TabM 5시드 숫자(VAL -7.32±11.28/+8.51±1.03, OOS +3.58±8.70/+22.89±5.15)와
직접 비교 가능하게 하려고 의도적으로 같은 윈도우를 씀.

절차: (1) 데이터 분석(클래스 균형/PSI drift/상관/단변량 MI) -> (2) TRAIN 내부 월별 확장윈도우
CV(embargo 48bar)로 Optuna HP 탐색(목적함수: 평균 multi_logloss) -> (3) 상위 5개 CV 후보를 VAL
거래 시뮬레이션(cost_mult=3.0)으로 재평가해 always_short 대비 마진이 가장 큰 설정을 최종 채택
(select-on-validation-only) -> (4) 그 설정으로 N=8 진짜 무작위 시드(Seed-Diversity Gate) 최종
학습, VAL/OOS 둘 다 cost1/2/3에서 always_short/always_long과 대조. OOS는 이 스크립트에서 단 한
번, 최종 단계에서만 읽는다(blind).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "lightgbm이 이 conda env에 없습니다 -- `pip install lightgbm`으로 설치 후 재실행하세요."
    ) from exc
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "optuna가 이 conda env에 없습니다 -- `pip install optuna`으로 설치 후 재실행하세요."
    ) from exc

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats

import train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_20260811 as h48orig_mod  # noqa: E402

parent_script = h48orig_mod.parent_script
omega = h48orig_mod.omega
FINAL12 = h48orig_mod.FINAL12

# GBDT_SMOKE=1이면 파이프라인 정합성 확인용 축소 실행(HP 탐색/최종 결론에 쓰지 않음).
SMOKE = os.environ.get("GBDT_SMOKE", "0") == "1"
OUT_DIR = ROOT / ("tmp/eth_h48qual_gbdt_backbone_diagnostic_20260812_smoke" if SMOKE
                  else "tmp/eth_h48qual_gbdt_backbone_diagnostic_20260812")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_TRIALS = 3 if SMOKE else int(os.environ.get("GBDT_N_TRIALS", "80"))
N_FINAL_SEEDS = 2 if SMOKE else int(os.environ.get("GBDT_N_FINAL_SEEDS", "8"))
CV_EMBARGO_BARS = 48  # 4시간, 월경계 근처 롤링피쳐 스필오버 방지용 여유
TOP_K_CV_CANDIDATES = 2 if SMOKE else 5
COST_MULTS = {"cost1": 1.0, "cost2": 2.0, "cost3": 3.0}
HEADLINE_COST = "cost3"  # 이 레포 train_eval 스크립트들의 기본값(--cost-mult default=3.0)과 일치

ACTION_NAMES = {0: "CASH", 1: "LONG", 2: "SHORT"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. 데이터 로딩 (h48orig 학습 스크립트와 완전히 동일한 파이프라인)
# ---------------------------------------------------------------------------

log("프레임 로딩 (FINAL12 + h48orig zigzag_action 라벨)...")
frames = parent_script._prepare_frames(
    disable_tp_sl=False,
    direction_label_dir=ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531",
    quality_mode="quality_label_action",
    quality_label_dir=ROOT / "tmp/eth_h48_conservative_orig_padded_to_zigzag_timestamps_20260811",
    quality_min_edge=0.0010,
    quality_max_mae=0.0100,
    quality_min_mfe_mae=1.20,
    quality_max_hold_bars=288,
)
train_raw: pd.DataFrame = frames["train_raw"]
val_raw: pd.DataFrame = frames["val_raw"]
oos_raw: pd.DataFrame = frames["oos_raw"]
feature_cols = frames["feature_cols"]
assert list(feature_cols) == list(FINAL12), f"feature_cols mismatch: {feature_cols} vs {FINAL12}"

for name, df in [("TRAIN", train_raw), ("VAL", val_raw), ("OOS", oos_raw)]:
    ts = pd.to_datetime(df["timestamp"])
    log(f"  {name}: rows={len(df)}  {ts.min()} ~ {ts.max()}")

X_train_full = train_raw[FINAL12].astype(np.float64)
y_train_full = pd.to_numeric(train_raw["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
X_val = val_raw[FINAL12].astype(np.float64)
y_val = pd.to_numeric(val_raw["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
X_oos = oos_raw[FINAL12].astype(np.float64)
y_oos = pd.to_numeric(oos_raw["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)

fee, slip = omega._load_fee_slip()

# ---------------------------------------------------------------------------
# 2. 데이터 분석
# ---------------------------------------------------------------------------

log("데이터 분석: 클래스 균형 / PSI drift / 상관 / 단변량 MI...")


def class_balance(y: np.ndarray) -> dict:
    vc = pd.Series(y).value_counts().sort_index()
    total = len(y)
    return {ACTION_NAMES[int(k)]: {"count": int(v), "pct": float(v) / total} for k, v in vc.items()}


def psi(train_vals: np.ndarray, other_vals: np.ndarray, bins: int = 10) -> float:
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


data_analysis = {
    "class_balance": {"train": class_balance(y_train_full), "val": class_balance(y_val), "oos": class_balance(y_oos)},
    "feature_psi_train_vs_val": {c: psi(X_train_full[c].to_numpy(), X_val[c].to_numpy()) for c in FINAL12},
    "feature_psi_train_vs_oos": {c: psi(X_train_full[c].to_numpy(), X_oos[c].to_numpy()) for c in FINAL12},
    "feature_stats": {
        split_name: {c: {"mean": float(df[c].mean()), "std": float(df[c].std()), "min": float(df[c].min()), "max": float(df[c].max())} for c in FINAL12}
        for split_name, df in [("train", X_train_full), ("val", X_val), ("oos", X_oos)]
    },
}

corr = X_train_full.corr(method="spearman")
data_analysis["train_feature_spearman_corr"] = corr.round(4).to_dict()
max_offdiag = corr.where(~np.eye(len(FINAL12), dtype=bool)).abs().max().max()
log(f"  FINAL12 TRAIN 내 최대 |spearman| (대각 제외) = {max_offdiag:.3f} (mRMR/knockoff dedup 사후 재확인)")

mi = mutual_info_classif(X_train_full.to_numpy(), y_train_full, discrete_features=False, random_state=0)
data_analysis["train_mutual_info_vs_zigzag_action"] = {c: float(m) for c, m in zip(FINAL12, mi)}
log("  단변량 MI(TRAIN, zigzag_action 3-class) 상위 5:")
for c, m in sorted(zip(FINAL12, mi), key=lambda t: -t[1])[:5]:
    log(f"    {c}: {m:.4f}")

drift_flags = [c for c in FINAL12 if data_analysis["feature_psi_train_vs_val"][c] > 0.25 or data_analysis["feature_psi_train_vs_oos"][c] > 0.25]
log(f"  PSI>0.25(중간~심한 drift) 피쳐: {drift_flags if drift_flags else '없음'}")

(OUT_DIR / "data_analysis.json").write_text(json.dumps(data_analysis, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 3. TRAIN 내부 월별 확장윈도우 CV 폴드 (embargo 포함)
# ---------------------------------------------------------------------------

train_ts = pd.to_datetime(train_raw["timestamp"])
months = train_ts.dt.to_period("M")
uniq_months = sorted(months.unique())
MIN_TRAIN_MONTHS = 4
folds = []
for i in range(MIN_TRAIN_MONTHS, len(uniq_months)):
    val_month = uniq_months[i]
    train_mask = months.isin(uniq_months[:i]).to_numpy()
    val_mask = (months == val_month).to_numpy()
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) > CV_EMBARGO_BARS:
        train_idx = train_idx[:-CV_EMBARGO_BARS]
    if len(train_idx) > 500 and len(val_idx) > 100:
        folds.append((train_idx, val_idx, str(val_month)))
log(f"CV 폴드 {len(folds)}개 (확장윈도우, embargo={CV_EMBARGO_BARS}bar): {[f[2] for f in folds]}")


def sample_weight_for(y: np.ndarray, class_weight_mode: str, cash_mult: float) -> np.ndarray:
    if class_weight_mode == "balanced":
        sw = compute_sample_weight("balanced", y)
    else:
        sw = np.ones(len(y), dtype=np.float64)
    sw = sw.copy()
    sw[y == 0] *= cash_mult
    return sw


# ---------------------------------------------------------------------------
# 4. Optuna 하이퍼파라미터 탐색 (목적함수: CV 평균 multi_logloss)
# ---------------------------------------------------------------------------

log(f"Optuna HP 탐색 시작 ({N_TRIALS} trials x {len(folds)} folds)...")


def objective(trial: "optuna.Trial") -> float:
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 7, 255, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 0, 7),
    )
    class_weight_mode = trial.suggest_categorical("class_weight_mode", ["none", "balanced"])
    cash_weight_mult = trial.suggest_float("cash_weight_mult", 0.5, 3.0, log=True)

    loglosses, best_iters = [], []
    for train_idx, val_idx, _ in folds:
        Xtr, ytr = X_train_full.iloc[train_idx], y_train_full[train_idx]
        Xv, yv = X_train_full.iloc[val_idx], y_train_full[val_idx]
        sw = sample_weight_for(ytr, class_weight_mode, cash_weight_mult)
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=2000, random_state=0,
            verbosity=-1, n_jobs=-1, **params,
        )
        model.fit(
            Xtr, ytr, sample_weight=sw, eval_set=[(Xv, yv)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        loglosses.append(model.best_score_["valid_0"]["multi_logloss"])
        best_iters.append(model.best_iteration_ or model.n_estimators)
    trial.set_user_attr("mean_best_iter", float(np.mean(best_iters)))
    trial.set_user_attr("class_weight_mode", class_weight_mode)
    trial.set_user_attr("cash_weight_mult", cash_weight_mult)
    return float(np.mean(loglosses))


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=20260812))
t0 = time.time()
study.trials_dataframe()  # no-op, sanity
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
log(f"Optuna 완료: {N_TRIALS} trials, {time.time() - t0:.0f}s, best CV logloss={study.best_value:.4f}")
study.trials_dataframe().to_csv(OUT_DIR / "optuna_trials.csv", index=False)

# ---------------------------------------------------------------------------
# 5. 상위 K개 CV 후보를 VAL 거래 시뮬레이션으로 재평가 -> 최종 HP 채택
# ---------------------------------------------------------------------------


def build_dec(action: np.ndarray) -> pd.DataFrame:
    action = action.astype(np.int64)
    active = action != omega.ACTION_CASH
    side = np.where(action == omega.ACTION_LONG, 1, np.where(action == omega.ACTION_SHORT, -1, 0)).astype(np.int64)
    return pd.DataFrame({
        "action": action,
        "side": side,
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


def all_active_dec(n: int, side_value: int) -> pd.DataFrame:
    action = np.full(n, omega.ACTION_LONG if side_value > 0 else omega.ACTION_SHORT, dtype=np.int64)
    return build_dec(action)


def train_final(params: dict, class_weight_mode: str, cash_weight_mult: float, n_estimators: int, seed: int):
    sw = sample_weight_for(y_train_full, class_weight_mode, cash_weight_mult)
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=n_estimators, random_state=seed,
        bagging_seed=seed, feature_fraction_seed=seed, verbosity=-1, n_jobs=-1,
        importance_type="gain", **params,
    )
    model.fit(X_train_full, y_train_full, sample_weight=sw)
    return model


trials_sorted = sorted(study.trials, key=lambda t: t.value)
top_candidates = trials_sorted[:TOP_K_CV_CANDIDATES]
log(f"상위 {len(top_candidates)}개 CV 후보를 VAL 거래 시뮬레이션으로 재평가 (select-on-validation-only)...")

val_frame_ohlc = val_raw[["timestamp", "open", "high", "low", "close"]]
candidate_rows = []
for rank, trial in enumerate(top_candidates):
    lgb_params = {k: v for k, v in trial.params.items() if k not in ("class_weight_mode", "cash_weight_mult")}
    cw_mode = trial.user_attrs["class_weight_mode"]
    cash_mult = trial.user_attrs["cash_weight_mult"]
    n_est = max(10, int(round(trial.user_attrs["mean_best_iter"])))
    model = train_final(lgb_params, cw_mode, cash_mult, n_est, seed=0)
    pred = model.predict(X_val)
    dec = build_dec(pred)
    m_model = omega._metrics(val_frame_ohlc, dec, fee=fee, slip=slip, cost_mult=COST_MULTS[HEADLINE_COST])
    m_short = omega._metrics(val_frame_ohlc, forced_side(dec, -1), fee=fee, slip=slip, cost_mult=COST_MULTS[HEADLINE_COST])
    margin = m_model["pnl"] - m_short["pnl"]
    candidate_rows.append({
        "rank_by_cv": rank, "trial_number": trial.number, "cv_logloss": trial.value,
        "val_pnl": m_model["pnl"], "val_trades": m_model["trades"], "val_always_short_pnl": m_short["pnl"],
        "margin_vs_always_short": margin, "n_estimators": n_est,
        "class_weight_mode": cw_mode, "cash_weight_mult": cash_mult, "lgb_params": lgb_params,
    })
    log(f"  trial#{trial.number} cv_logloss={trial.value:.4f} VAL_pnl={m_model['pnl']:+.2f} always_short={m_short['pnl']:+.2f} margin={margin:+.2f}")

cand_df = pd.DataFrame(candidate_rows)
cand_df.to_csv(OUT_DIR / "top_candidates_val_reeval.csv", index=False)
winner = candidate_rows[int(np.argmax([r["margin_vs_always_short"] for r in candidate_rows]))]
log(f"채택된 HP: trial#{winner['trial_number']}  VAL margin vs always_short={winner['margin_vs_always_short']:+.2f}")
(OUT_DIR / "winning_hp.json").write_text(json.dumps(winner, indent=2, ensure_ascii=False, default=str))

# ---------------------------------------------------------------------------
# 6. 최종 HP로 N개 진짜 무작위 시드 학습, VAL/OOS 둘 다 평가 (OOS는 여기서 최초 1회만 읽음)
# ---------------------------------------------------------------------------

seeds = random.SystemRandom().sample(range(1_000_000, 999_000_000), N_FINAL_SEEDS)
log(f"최종 시드(N={N_FINAL_SEEDS}, 무작위 추출): {seeds}")

oos_frame_ohlc = oos_raw[["timestamp", "open", "high", "low", "close"]]
feature_importance_sum = np.zeros(len(FINAL12))
final_rows = []
clf_rows = []
for seed in seeds:
    model = train_final(winner["lgb_params"], winner["class_weight_mode"], winner["cash_weight_mult"], winner["n_estimators"], seed=seed)
    feature_importance_sum += model.feature_importances_

    for split_name, X, y_true, ohlc in [("VAL", X_val, y_val, val_frame_ohlc), ("OOS", X_oos, y_oos, oos_frame_ohlc)]:
        pred = model.predict(X)
        clf_rows.append({
            "seed": seed, "split": split_name,
            "balanced_accuracy": balanced_accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro"),
            "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        })
        dec_model = build_dec(pred)
        dec_short_matched = forced_side(dec_model, -1)
        dec_long_matched = forced_side(dec_model, 1)
        n = len(ohlc)
        dec_short_all = all_active_dec(n, -1)
        dec_long_all = all_active_dec(n, 1)
        row = {"seed": seed, "split": split_name}
        for cost_name, cost_mult in COST_MULTS.items():
            m_model = omega._metrics(ohlc, dec_model, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short_m = omega._metrics(ohlc, dec_short_matched, fee=fee, slip=slip, cost_mult=cost_mult)
            m_long_m = omega._metrics(ohlc, dec_long_matched, fee=fee, slip=slip, cost_mult=cost_mult)
            m_short_all = omega._metrics(ohlc, dec_short_all, fee=fee, slip=slip, cost_mult=cost_mult)
            m_long_all = omega._metrics(ohlc, dec_long_all, fee=fee, slip=slip, cost_mult=cost_mult)
            row[f"{cost_name}_gbdt_pnl"] = m_model["pnl"]
            row[f"{cost_name}_gbdt_trades"] = m_model["trades"]
            row[f"{cost_name}_gbdt_wr"] = m_model["wr"]
            row[f"{cost_name}_always_short_matched_pnl"] = m_short_m["pnl"]
            row[f"{cost_name}_always_long_matched_pnl"] = m_long_m["pnl"]
            row[f"{cost_name}_always_short_allbar_pnl"] = m_short_all["pnl"]
            row[f"{cost_name}_always_long_allbar_pnl"] = m_long_all["pnl"]
            row[f"{cost_name}_gbdt_beats_always_short_matched"] = m_model["pnl"] > m_short_m["pnl"]
        final_rows.append(row)
    log(f"  seed={seed} 완료")

final_df = pd.DataFrame(final_rows)
final_df.to_csv(OUT_DIR / "final_multiseed_results.csv", index=False)
clf_df = pd.DataFrame(clf_rows)
clf_df.to_csv(OUT_DIR / "final_classification_metrics.csv", index=False)

feature_importance = {c: float(v) for c, v in zip(FINAL12, feature_importance_sum / N_FINAL_SEEDS)}
(OUT_DIR / "feature_importance_gain_mean.json").write_text(json.dumps(feature_importance, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 7. 요약 리포트
# ---------------------------------------------------------------------------

print("\n" + "=" * 100)
print("GBDT(LightGBM) FINAL12 -> zigzag_action 백본 진단 -- 최종 요약")
print("=" * 100)
print(f"TRAIN rows={len(train_raw)}  VAL rows={len(val_raw)}  OOS rows={len(oos_raw)}")
print(f"채택 HP (trial#{winner['trial_number']}, CV logloss={winner['cv_logloss']:.4f}): {winner['lgb_params']}")
print(f"class_weight_mode={winner['class_weight_mode']}  cash_weight_mult={winner['cash_weight_mult']:.3f}  n_estimators={winner['n_estimators']}")
print(f"시드: {seeds}")

print(f"\n피쳐 중요도(gain, {N_FINAL_SEEDS}시드 평균, 내림차순):")
for c, v in sorted(feature_importance.items(), key=lambda kv: -kv[1]):
    print(f"  {c}: {v:.1f}")

for split in ["VAL", "OOS"]:
    sub = final_df[final_df["split"] == split]
    print(f"\n--- {split} ({len(sub)}시드) ---")
    for cost_name in COST_MULTS:
        beat = int(sub[f"{cost_name}_gbdt_beats_always_short_matched"].sum())
        diff = (sub[f"{cost_name}_gbdt_pnl"] - sub[f"{cost_name}_always_short_matched_pnl"]).to_numpy()
        if len(sub) >= 5 and np.any(diff != 0):
            _, wp = stats.wilcoxon(diff, alternative="greater")
        else:
            wp = float("nan")
        print(f"  [{cost_name}] gbdt={sub[f'{cost_name}_gbdt_pnl'].mean():+7.2f}±{sub[f'{cost_name}_gbdt_pnl'].std():5.2f}"
              f"  always_short(동일active)={sub[f'{cost_name}_always_short_matched_pnl'].mean():+7.2f}±{sub[f'{cost_name}_always_short_matched_pnl'].std():5.2f}"
              f"  always_long(동일active)={sub[f'{cost_name}_always_long_matched_pnl'].mean():+7.2f}±{sub[f'{cost_name}_always_long_matched_pnl'].std():5.2f}"
              f"  always_short(전체bar)={sub[f'{cost_name}_always_short_allbar_pnl'].mean():+7.2f}"
              f"  always_long(전체bar)={sub[f'{cost_name}_always_long_allbar_pnl'].mean():+7.2f}"
              f"  gbdt승={beat}/{len(sub)}  wilcoxon_p={wp:.4f}")
    csub = clf_df[clf_df["split"] == split]
    print(f"  분류: balanced_acc={csub['balanced_accuracy'].mean():.3f}±{csub['balanced_accuracy'].std():.3f}"
          f"  macro_f1={csub['macro_f1'].mean():.3f}±{csub['macro_f1'].std():.3f}")

print(f"\n출력 디렉토리: {OUT_DIR}")
print("=" * 100)
