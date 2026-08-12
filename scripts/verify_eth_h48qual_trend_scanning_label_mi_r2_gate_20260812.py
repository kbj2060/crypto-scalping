"""오라클 라벨 문헌 리서치(docs/experiments/eth_h48qual_oracle_label_design_literature_research_
20260812.md) 권장안 1+2 실행: trend-scanning(De Prado, t-value 기반 자동 호라이즌 선택) 라벨을
구성한 뒤, TabM/GBDT 풀 학습 전에 저비용 MI/R² 사전 게이트만 돌린다. 게이트를 통과 못 하면
이 라벨도 즉시 폐기 -- zigzag_action/h48_conservative와 같은 반복을 피하는 게 목적이다.

트렌드스캐닝 정의: 각 bar t에서 L in L_GRID(bar 단위)마다 log(close)를 [t, t+L] 구간에서
시간 인덱스에 대해 OLS 회귀, 기울기의 t-value를 계산. |t-value|가 최대인 L*을 선택(De Prado,
Machine Learning for Asset Managers). 연속 타겟은 그 t-value 자체; 이산 3-class 라벨은
|t-value| >= 임계값이면 부호에 따라 LONG/SHORT, 미만이면 CASH(임계값 민감도는 90/95/99%
근사치 3개를 모두 리포트).

TRAIN/VAL/OOS 구간과 FINAL12 피쳐는 h48orig 학습 파이프라인(_prepare_frames)을 그대로 재사용 --
GBDT 백본 진단(train_eval_eth_h48qual_final12_gbdt_backbone_diagnostic_20260812.py)과 동일 관례
(TRAIN=2025-01~09 isolated-verification 윈도우). GBM 홀드아웃 R² 게이트는 quality_head
회귀전환 시도(verify_eth_h48qual_quality_gbm_final12_20260811.py)와 정확히 동일한 두 설정
(약한 정규화 depth=5 / 강한 정규화 depth=2+early stopping)을 그대로 사용해 직접 비교 가능하게
한다. 이 스크립트는 의도적으로 가볍다 -- Optuna 없음, N=1 fit -- "TabM/GBDT 풀 학습을 정당화할
근거가 있는지"만 싸게 확인하는 사전 게이트이기 때문."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import r2_score, roc_auc_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_20260811 as h48orig_mod  # noqa: E402

parent_script = h48orig_mod.parent_script
omega = h48orig_mod.omega
FINAL12 = h48orig_mod.FINAL12

OUT_DIR = ROOT / "tmp/eth_h48qual_trend_scanning_mi_r2_gate_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_GRID = [8, 16, 24, 32, 48, 64, 80, 96]
T_THRESHOLDS = {"90pct_1.65": 1.65, "95pct_1.96": 1.96, "99pct_2.58": 2.58}
PRIMARY_THRESHOLD_KEY = "95pct_1.96"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. 트렌드스캐닝 t-value 계산 (벡터화, 연도별 원본 CSV에 독립 적용 -- zigzag_action과 동일 관례)
# ---------------------------------------------------------------------------

def trend_scan_tvalue(close: pd.Series, l_grid: list[int]) -> pd.DataFrame:
    n = len(close)
    y = np.log(close.to_numpy(dtype=np.float64))
    idx = np.arange(n, dtype=np.float64)
    y_s = pd.Series(y)
    y2_s = y_s ** 2
    iy_s = pd.Series(idx) * y_s

    best_abs_t = np.full(n, -np.inf)
    best_t = np.full(n, np.nan)
    best_L = np.full(n, np.nan)
    best_slope = np.full(n, np.nan)

    for L in l_grid:
        w = L + 1
        if w > n:
            continue
        xg = np.arange(w, dtype=np.float64)
        xbar = xg.mean()
        Sxx = float(np.sum((xg - xbar) ** 2))

        roll_y = y_s.rolling(w).sum().to_numpy()
        roll_y2 = y2_s.rolling(w).sum().to_numpy()
        roll_iy = iy_s.rolling(w).sum().to_numpy()

        Sxy = roll_iy - (idx - L + xbar) * roll_y
        b = Sxy / Sxx
        Syy = roll_y2 - (roll_y ** 2) / w
        sse = np.maximum(Syy - b * Sxy, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            se_b = np.sqrt(sse / max(w - 2, 1) / Sxx)
            t_stat = np.where(se_b > 1e-12, b / se_b, 0.0)

        # 윈도우는 [row-L, row]를 덮는다 -- 시작점(row-L)에 결과를 배정하려면 -L 시프트
        t_stat_shifted = pd.Series(t_stat).shift(-L).to_numpy()
        b_shifted = pd.Series(b).shift(-L).to_numpy()

        valid = ~np.isnan(t_stat_shifted)
        better = valid & (np.abs(t_stat_shifted) > best_abs_t)
        best_abs_t = np.where(better, np.abs(t_stat_shifted), best_abs_t)
        best_t = np.where(better, t_stat_shifted, best_t)
        best_L = np.where(better, L, best_L)
        best_slope = np.where(better, b_shifted, best_slope)

    return pd.DataFrame({"trend_tvalue": best_t, "trend_L": best_L, "trend_slope": best_slope})


def build_trend_labels(raw_csv_path: Path) -> pd.DataFrame:
    src = omega._read(raw_csv_path)
    scan = trend_scan_tvalue(src["close"], L_GRID)
    out = pd.concat([src[["timestamp"]].reset_index(drop=True), scan], axis=1)
    return out


log("트렌드스캐닝 라벨 계산 (2025 원본 CSV, 2026 원본 CSV 각각 독립 -- zigzag_action과 동일 관례)...")
trend_2025 = build_trend_labels(omega.TRAIN_CSV)
trend_2026 = build_trend_labels(omega.EVAL_CSV)
n_nan_2025 = trend_2025["trend_tvalue"].isna().sum()
n_nan_2026 = trend_2026["trend_tvalue"].isna().sum()
log(f"  2025: {len(trend_2025)}행, 라벨 없음(파일 끝 forward-window 부족) {n_nan_2025}행")
log(f"  2026: {len(trend_2026)}행, 라벨 없음(파일 끝 forward-window 부족) {n_nan_2026}행")

# ---------------------------------------------------------------------------
# 2. FINAL12 피쳐 프레임(h48orig 파이프라인) + 트렌드스캐닝 라벨 병합
# ---------------------------------------------------------------------------

log("FINAL12 프레임 로딩 (h48orig와 동일 파이프라인)...")
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
train_raw = frames["train_raw"].merge(trend_2025, on="timestamp", how="inner")
val_raw = frames["val_raw"].merge(trend_2025, on="timestamp", how="inner")
oos_raw = frames["oos_raw"].merge(trend_2026, on="timestamp", how="inner")

for name, before, after in [("train", frames["train_raw"], train_raw), ("val", frames["val_raw"], val_raw), ("oos", frames["oos_raw"], oos_raw)]:
    log(f"  {name}: FINAL12프레임 {len(before)}행 -> 트렌드라벨 병합 후 {len(after)}행 (끝쪽 truncation 반영)")

train_raw = train_raw.dropna(subset=["trend_tvalue"]).reset_index(drop=True)
val_raw = val_raw.dropna(subset=["trend_tvalue"]).reset_index(drop=True)
oos_raw = oos_raw.dropna(subset=["trend_tvalue"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. 이산 라벨 구성 + 데이터 분석
# ---------------------------------------------------------------------------

def discretize(tvalue: np.ndarray, threshold: float) -> np.ndarray:
    action = np.zeros(len(tvalue), dtype=np.int64)  # CASH
    action[tvalue >= threshold] = 1  # LONG
    action[tvalue <= -threshold] = 2  # SHORT
    return action


analysis = {"l_grid": L_GRID, "thresholds": T_THRESHOLDS}
log("\n=== 데이터 분석: 트렌드스캐닝 라벨 특성 ===")
for split_name, df in [("TRAIN", train_raw), ("VAL", val_raw), ("OOS", oos_raw)]:
    l_dist = df["trend_L"].value_counts().sort_index()
    log(f"  {split_name}: L* 분포 = {dict(l_dist)}")
    analysis[f"{split_name}_L_distribution"] = {str(k): int(v) for k, v in l_dist.items()}
    for tkey, tval in T_THRESHOLDS.items():
        action = discretize(df["trend_tvalue"].to_numpy(), tval)
        vc = pd.Series(action).value_counts().sort_index()
        pct = {["CASH", "LONG", "SHORT"][int(k)]: f"{v / len(action) * 100:.1f}%" for k, v in vc.items()}
        log(f"    threshold={tkey}: {pct}")
        analysis.setdefault(f"{split_name}_class_balance", {})[tkey] = {["CASH", "LONG", "SHORT"][int(k)]: int(v) for k, v in vc.items()}

# zigzag_action과의 방향 일치율(참고용, 게이트 판정에는 안 씀)
for split_name, df in [("TRAIN", train_raw), ("VAL", val_raw), ("OOS", oos_raw)]:
    trend_primary = discretize(df["trend_tvalue"].to_numpy(), T_THRESHOLDS[PRIMARY_THRESHOLD_KEY])
    zz = df["zigzag_action"].to_numpy()
    both_active = (trend_primary != 0) & (zz != 0)
    if both_active.sum() > 0:
        agree = (trend_primary[both_active] == zz[both_active]).mean()
        log(f"  {split_name}: 둘 다 active인 {both_active.sum()}행 중 방향 일치율 = {agree*100:.1f}%")
        analysis[f"{split_name}_direction_agreement_with_zigzag"] = {"n_both_active": int(both_active.sum()), "agree_pct": float(agree)}

(OUT_DIR / "trend_label_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False))

X_train = train_raw[FINAL12].astype(np.float64)
X_val = val_raw[FINAL12].astype(np.float64)
X_oos = oos_raw[FINAL12].astype(np.float64)
t_train = train_raw["trend_tvalue"].to_numpy()
t_val = val_raw["trend_tvalue"].to_numpy()
t_oos = oos_raw["trend_tvalue"].to_numpy()
label_train = discretize(t_train, T_THRESHOLDS[PRIMARY_THRESHOLD_KEY])
label_val = discretize(t_val, T_THRESHOLDS[PRIMARY_THRESHOLD_KEY])
label_oos = discretize(t_oos, T_THRESHOLDS[PRIMARY_THRESHOLD_KEY])

# ---------------------------------------------------------------------------
# 4. MI 게이트
# ---------------------------------------------------------------------------

log("\n=== MI 게이트 (TRAIN) ===")
mi_class = mutual_info_classif(X_train.to_numpy(), label_train, discrete_features=False, random_state=0)
mi_reg = mutual_info_regression(X_train.to_numpy(), t_train, discrete_features=False, random_state=0)
mi_report = {}
log(f"  {'피쳐':35s} {'MI(discrete label)':>20s} {'MI(continuous t-value)':>24s}")
for c, mc, mr in sorted(zip(FINAL12, mi_class, mi_reg), key=lambda x: -x[1]):
    log(f"  {c:35s} {mc:20.4f} {mr:24.4f}")
    mi_report[c] = {"mi_discrete_label": float(mc), "mi_continuous_tvalue": float(mr)}
(OUT_DIR / "mi_gate.json").write_text(json.dumps(mi_report, indent=2, ensure_ascii=False))
log(f"  참고: zigzag_action 대비 MI(discrete) 합계 비교는 GBDT 진단 문서의 zigzag MI 상위값(cvp_regime 0.414 등)과 대조할 것.")

# ---------------------------------------------------------------------------
# 5. GBM 홀드아웃 R^2 게이트 (quality_head 회귀전환 시도와 동일 두 설정)
# ---------------------------------------------------------------------------

log("\n=== GBM 홀드아웃 R^2 게이트 (연속 t-value 타겟) ===")
gbm_report = {}


def sign_auc(y_true, y_pred):
    sign_true = (y_true > 0).astype(int)
    if len(np.unique(sign_true)) < 2:
        return float("nan")
    return float(roc_auc_score(sign_true, y_pred))


configs = {
    "weak_reg_depth5": dict(max_depth=5, max_iter=300, random_state=260620),
    "strong_reg_depth2_earlystop": dict(max_depth=2, learning_rate=0.03, l2_regularization=2.0,
                                         early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                                         max_iter=1000, random_state=260620),
}
for cfg_name, cfg in configs.items():
    model = HistGradientBoostingRegressor(**cfg).fit(X_train, t_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_oos = model.predict(X_oos)
    r2_train, r2_val, r2_oos = r2_score(t_train, pred_train), r2_score(t_val, pred_val), r2_score(t_oos, pred_oos)
    auc_val, auc_oos = sign_auc(t_val, pred_val), sign_auc(t_oos, pred_oos)
    n_iter = getattr(model, "n_iter_", cfg.get("max_iter"))
    log(f"  [{cfg_name}] n_iter={n_iter}  TRAIN R2={r2_train:+.4f}  VAL R2={r2_val:+.4f}  OOS R2={r2_oos:+.4f}  VAL 부호AUC={auc_val:.3f}  OOS 부호AUC={auc_oos:.3f}")
    gbm_report[cfg_name] = {"n_iter": int(n_iter) if n_iter else None, "r2_train": float(r2_train), "r2_val": float(r2_val),
                             "r2_oos": float(r2_oos), "sign_auc_val": auc_val, "sign_auc_oos": auc_oos}
(OUT_DIR / "gbm_r2_gate.json").write_text(json.dumps(gbm_report, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 6. 보조: 가벼운 3-class 분류 홀드아웃 (Optuna 없음, 단일 fit -- 사전 게이트 성격 유지)
# ---------------------------------------------------------------------------

log("\n=== 보조: 가벼운 3-class 분류 홀드아웃 (튜닝 없음, 참고용) ===")
try:
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
                              num_leaves=31, random_state=260620, verbosity=-1)
    clf.fit(X_train, label_train, eval_set=[(X_val, label_val)], eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)])
    clf_report = {}
    for split_name, X, y_true in [("VAL", X_val, label_val), ("OOS", X_oos, label_oos)]:
        pred = clf.predict(X)
        bacc = balanced_accuracy_score(y_true, pred)
        mf1 = f1_score(y_true, pred, average="macro")
        log(f"  {split_name}: balanced_acc={bacc:.3f}  macro_f1={mf1:.3f}")
        clf_report[split_name] = {"balanced_accuracy": float(bacc), "macro_f1": float(mf1)}
    (OUT_DIR / "light_classification_holdout.json").write_text(json.dumps(clf_report, indent=2, ensure_ascii=False))
except ImportError:
    log("  lightgbm 없음 -- 보조 분류 홀드아웃 스킵(핵심 MI/R^2 게이트는 이미 완료됨)")

log(f"\n출력 디렉토리: {OUT_DIR}")
log("게이트 판정 기준(권장): VAL/OOS R^2가 유의미하게 0보다 크고(대략 >0.02~0.05 이상), 부호-AUC가"
    " 0.5를 뚜렷이 상회해야 TabM/GBDT 풀 학습으로 승격할 근거가 있다고 본다. 위 수치를 그 기준에"
    " 직접 대조해 판정할 것 -- 이 스크립트는 판정을 자동으로 내리지 않는다(임계값 자체가 project 합의 사항).")
