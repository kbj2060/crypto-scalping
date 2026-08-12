"""신규 탐색 축 스카우팅 (a)-2 후보: Fear & Greed Index 과거 백필(alternative.me 무료 공개
API, 2018년~ 일별 히스토리)을 TRAIN/VAL/OOS 전체에 조인해 `zigzag_action` 예측에 도움이
되는지 확인한다. 서브 프로젝트가 이미 F4-C 수집기로 F&G를 수집 중이지만(2026-08-10부터
라이브 전방향만) 이 API는 무료 과거 백필이 존재해 그 계획을 바꾸지 않고도 지금 바로 조인
가능 -- 다른 8개 신규 데이터소스 후보를 막았던 "라이브 duckdb가 2026-05 이후만 커버" 벽이
이 후보엔 적용되지 않는다.

표준 절차: 신규 raw-level 피쳐는 학습 전 corr(price)/corr(시간순번) 오염도부터 확인
(feedback_raw_feature_price_trend_contamination, 배제 기준 0.561 -- CapMVRVCur/whale_retail_ratio
에서 이미 두 번 걸림). 일별 값이라 5분봉 288개에 동일값이 반복되므로 bar 단위 정보량이
원천적으로 작을 것으로 예상(스카우팅 문서의 솔직한 사전 기대치) -- 그래도 비용이 사실상
0(API 호출 1회)이라 닫아두는 값어치가 있다."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

ROOT = Path("/home/kbj20/crypto-scalping")
OUT_DIR = ROOT / "tmp/eth_fear_greed_backfill_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
CONTAMINATION_THRESHOLD = 0.561


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Fear & Greed 전체 히스토리 다운로드
# ---------------------------------------------------------------------------

log("alternative.me Fear&Greed 전체 히스토리 다운로드...")
resp = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=30)
resp.raise_for_status()
data = resp.json()["data"]
fng = pd.DataFrame(data)
fng["date"] = pd.to_datetime(fng["timestamp"].astype(np.int64), unit="s").dt.normalize()
fng["fng_value"] = pd.to_numeric(fng["value"], errors="raise")
fng = fng[["date", "fng_value"]].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
log(f"  {len(fng)}일치, {fng['date'].min().date()} ~ {fng['date'].max().date()}")
fng.to_csv(OUT_DIR / "fear_greed_daily_raw.csv", index=False)

# 파생: day-over-day 변화, 7일 이동평균 대비 편차(가격추세 오염 완화 관례와 동일 정신)
fng["fng_diff1"] = fng["fng_value"].diff(1).fillna(0.0)
fng["fng_ma7_dev"] = (fng["fng_value"] - fng["fng_value"].rolling(7, min_periods=3).mean()).fillna(0.0)

# ---------------------------------------------------------------------------
# 2. 5분봉 프레임에 forward-fill 조인 (as-of, causal -- 그날 값은 그날 자정부터 유효하다고 가정)
# ---------------------------------------------------------------------------

log("\nzig075 소스 패널(close, zigzag_action) 로딩 후 F&G 조인...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv",
                     usecols=["timestamp", "close"], low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
panel["date"] = panel["timestamp"].dt.normalize()

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")

df = panel.merge(labels, on="timestamp", how="inner").merge(fng, on="date", how="left").sort_values("timestamp").reset_index(drop=True)
missing = df["fng_value"].isna().sum()
log(f"  병합 후 {len(df)}행, F&G 결측 {missing}행({missing/len(df)*100:.2f}%, F&G 히스토리 시작 이전 구간)")
df[["fng_value", "fng_diff1", "fng_ma7_dev"]] = df[["fng_value", "fng_diff1", "fng_ma7_dev"]].ffill().fillna(0.0)

train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)
log(f"  TRAIN n={train_mask.sum()}  VAL n={val_mask.sum()}  OOS n={oos_mask.sum()}")

# ---------------------------------------------------------------------------
# 3. 오염도 체크 (표준 절차) + MI relevance
# ---------------------------------------------------------------------------

log("\n=== 오염도 체크 (corr(price), corr(시간순번)) + MI(zigzag_action, TRAIN) ===")
bar_idx = np.arange(len(df))
report = {}
for col in ["fng_value", "fng_diff1", "fng_ma7_dev"]:
    x_train = df.loc[train_mask, col].to_numpy()
    close_train = df.loc[train_mask, "close"].to_numpy()
    idx_train = bar_idx[train_mask.to_numpy()]
    corr_price = float(np.corrcoef(x_train, close_train)[0, 1])
    corr_time, _ = spearmanr(x_train, idx_train)
    mi = mutual_info_classif(x_train.reshape(-1, 1), df.loc[train_mask, "zigzag_action"].to_numpy(),
                              discrete_features=False, random_state=260620)[0]
    contaminated = abs(corr_price) > CONTAMINATION_THRESHOLD
    log(f"  {col:<15s} corr(price)={corr_price:+.3f}  corr(시간순번)={corr_time:+.3f}  MI={mi:.4f}{'  [오염]' if contaminated else ''}")
    report[col] = {"corr_price": corr_price, "corr_time": float(corr_time), "mi": float(mi), "contaminated": contaminated}

(OUT_DIR / "contamination_and_mi_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

clean_cols = [c for c, r in report.items() if not r["contaminated"]]
log(f"\n오염도 통과: {clean_cols}")

# ---------------------------------------------------------------------------
# 4. 가벼운 홀드아웃: FINAL12(패널에서 구할 수 있는 9개) 단독 vs +F&G (튜닝 없음)
# ---------------------------------------------------------------------------

log("\n=== 가벼운 LightGBM 홀드아웃 비교 (튜닝 없음, 참고용) ===")
try:
    import lightgbm as lgb
    from sklearn.metrics import balanced_accuracy_score, f1_score

    full_panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
    full_panel["timestamp"] = pd.to_datetime(full_panel["timestamp"])
    full_panel = full_panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    df2 = full_panel.merge(labels, on="timestamp", how="inner")
    df2["date"] = df2["timestamp"].dt.normalize()
    df2 = df2.merge(fng, on="date", how="left").sort_values("timestamp").reset_index(drop=True)
    df2[["fng_value", "fng_diff1", "fng_ma7_dev"]] = df2[["fng_value", "fng_diff1", "fng_ma7_dev"]].ffill().fillna(0.0)
    df2["funding_pressure_diff1"] = pd.to_numeric(df2["funding_pressure"], errors="coerce").diff(1).fillna(0.0)

    FINAL12_AVAILABLE = ["cvp_regime", "funding_pressure_diff1", "ou_halflife", "realized_skewness",
                          "mta_funding", "vwap_dist_24", "funding_roc_48", "breakout_strength"]
    missing = [c for c in FINAL12_AVAILABLE if c not in df2.columns]
    FINAL12_AVAILABLE = [c for c in FINAL12_AVAILABLE if c in df2.columns]

    tm = (df2["timestamp"] >= TRAIN_START) & (df2["timestamp"] <= TRAIN_END)
    vm = (df2["timestamp"] >= VAL_START) & (df2["timestamp"] <= VAL_END)
    om = (df2["timestamp"] >= OOS_START) & (df2["timestamp"] <= OOS_END)
    ytr, yv, yo = df2.loc[tm, "zigzag_action"].to_numpy(), df2.loc[vm, "zigzag_action"].to_numpy(), df2.loc[om, "zigzag_action"].to_numpy()

    def fit_eval(cols, label):
        Xtr = df2.loc[tm, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xv = df2.loc[vm, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xo = df2.loc[om, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
                                  num_leaves=31, random_state=260620, verbosity=-1)
        clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="multi_logloss", callbacks=[lgb.early_stopping(30, verbose=False)])
        out = {}
        for sn, X, y_true in [("VAL", Xv, yv), ("OOS", Xo, yo)]:
            pred = clf.predict(X)
            out[sn] = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred)), "macro_f1": float(f1_score(y_true, pred, average="macro"))}
            log(f"  [{label}] {sn}: balanced_acc={out[sn]['balanced_accuracy']:.3f}  macro_f1={out[sn]['macro_f1']:.3f}")
        return out

    log(f"\n[FINAL12(패널가용 {len(FINAL12_AVAILABLE)}개) 단독]")
    r_base = fit_eval(FINAL12_AVAILABLE, "base")
    log(f"\n[FINAL12 + F&G(오염도 통과 {len(clean_cols)}개)]")
    r_fng = fit_eval(FINAL12_AVAILABLE + clean_cols, "base+fng")

    (OUT_DIR / "holdout_comparison.json").write_text(json.dumps({"base": r_base, "base_plus_fng": r_fng, "fng_cols_used": clean_cols}, indent=2, ensure_ascii=False))
except ImportError:
    log("  lightgbm 없음 -- 홀드아웃 비교 스킵")

log(f"\n출력 디렉토리: {OUT_DIR}")
