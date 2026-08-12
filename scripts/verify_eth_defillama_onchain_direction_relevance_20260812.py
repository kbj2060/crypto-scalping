"""사용자 지시("CoinGlass/Dune/DefiLlama/LunarCrush/Santiment 재검토") 실행 — 완전 무료+API키
불필요+전체 구간 백필 가능한 것은 DefiLlama뿐임을 사전 확인 후(다른 4개는 유료거나 계정가입/
과거구간 제한, 별도 문서 참고) DefiLlama의 ETH 체인 지표 3종을 zigzag_action 방향 relevance로
검증한다.

지표: (1) 체인 TVL(스마트컨트랙트 예치자산), (2) DEX 거래량(온체인 트레이딩 활동),
(3) fees/revenue(가스+프로토콜 매출, 네트워크 사용량 프록시). 전부 일별 해상도라 5분봉 288개에
동일값이 forward-fill되는 구조적 정보량 한계가 있음(Fear&Greed 실험과 동일 caveat, 사전
기대치를 낮게 잡음) -- 그래도 CapMVRVCur(candidate 6, corr(price)=0.95~0.97로 오염 확정)나
Fear&Greed(오염은 없으나 MI 미미)와는 경제적으로 다른 신호원이라 재검토 가치가 있음.

표준 절차: 신규 raw-level 피쳐는 학습 전 corr(price)/corr(시간순번) 오염도부터 확인(배제 기준
0.561), 원값뿐 아니라 detrend 파생(day-over-day diff, 7일 이동평균 대비 편차)도 함께 체크."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

ROOT = Path("/home/kbj20/crypto-scalping")
OUT_DIR = ROOT / "tmp/eth_defillama_onchain_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")
CONTAMINATION_THRESHOLD = 0.561


def log(msg: str) -> None:
    print(msg, flush=True)


def fetch_daily(url: str, value_key: str | None = None) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and "totalDataChart" in data:
        rows = data["totalDataChart"]
        df = pd.DataFrame(rows, columns=["ts", "value"])
    elif isinstance(data, list) and value_key:
        df = pd.DataFrame([{"ts": d["date"], "value": d.get(value_key)} for d in data])
    elif isinstance(data, list) and not value_key:
        # historicalChainTvl 형식: [{"date": ..., "tvl": ...}, ...]
        df = pd.DataFrame([{"ts": d["date"], "value": d.get("tvl")} for d in data])
    else:
        raise RuntimeError(f"예상 못한 응답 형식: {url}")
    df["date"] = pd.to_datetime(df["ts"].astype(np.int64), unit="s").dt.normalize()
    df = df[["date", "value"]].dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


log("DefiLlama ETH 체인 지표 3종 다운로드...")
tvl = fetch_daily("https://api.llama.fi/v2/historicalChainTvl/Ethereum").rename(columns={"value": "eth_chain_tvl"})
dex = fetch_daily("https://api.llama.fi/overview/dexs/ethereum?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true").rename(columns={"value": "eth_dex_volume"})
fees = fetch_daily("https://api.llama.fi/overview/fees/ethereum?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true").rename(columns={"value": "eth_fees_revenue"})
for name, df in [("TVL", tvl), ("DEX거래량", dex), ("fees/revenue", fees)]:
    log(f"  {name}: {len(df)}일치, {df['date'].min().date()} ~ {df['date'].max().date()}")

merged = tvl.merge(dex, on="date", how="outer").merge(fees, on="date", how="outer").sort_values("date").reset_index(drop=True)
raw_cols = ["eth_chain_tvl", "eth_dex_volume", "eth_fees_revenue"]

# 파생: day-over-day 변화율, 7일 이동평균 대비 편차 (표준 detrend 관례)
for c in raw_cols:
    merged[f"{c}_diff1pct"] = merged[c].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged[f"{c}_ma7dev"] = ((merged[c] - merged[c].rolling(7, min_periods=3).mean()) / merged[c].rolling(7, min_periods=3).mean().replace(0.0, np.nan)).fillna(0.0)

ALL_COLS = raw_cols + [f"{c}_diff1pct" for c in raw_cols] + [f"{c}_ma7dev" for c in raw_cols]

log("\nzig075 소스 패널(close, zigzag_action) 로딩 후 조인...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", usecols=["timestamp", "close"], low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
panel["date"] = panel["timestamp"].dt.normalize()

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")

df = panel.merge(labels, on="timestamp", how="inner").merge(merged, on="date", how="left").sort_values("timestamp").reset_index(drop=True)
missing = df[ALL_COLS].isna().any(axis=1).sum()
log(f"  병합 후 {len(df)}행, 결측 {missing}행({missing/len(df)*100:.2f}%)")
df[ALL_COLS] = df[ALL_COLS].ffill().fillna(0.0)

train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)
log(f"  TRAIN n={train_mask.sum()}  VAL n={val_mask.sum()}  OOS n={oos_mask.sum()}")

log("\n=== 오염도 체크(corr(price), corr(시간순번)) + MI(zigzag_action, TRAIN) ===")
bar_idx = np.arange(len(df))
report = {}
for col in ALL_COLS:
    x_train = df.loc[train_mask, col].to_numpy()
    close_train = df.loc[train_mask, "close"].to_numpy()
    idx_train = bar_idx[train_mask.to_numpy()]
    corr_price = float(np.corrcoef(x_train, close_train)[0, 1]) if np.std(x_train) > 1e-12 else 0.0
    corr_time, _ = spearmanr(x_train, idx_train)
    mi = mutual_info_classif(x_train.reshape(-1, 1), df.loc[train_mask, "zigzag_action"].to_numpy(),
                              discrete_features=False, random_state=260620)[0]
    contaminated = abs(corr_price) > CONTAMINATION_THRESHOLD
    log(f"  {col:<28s} corr(price)={corr_price:+.3f}  corr(시간순번)={corr_time:+.3f}  MI={mi:.4f}{'  [오염]' if contaminated else ''}")
    report[col] = {"corr_price": corr_price, "corr_time": float(corr_time), "mi": float(mi), "contaminated": contaminated}

(OUT_DIR / "contamination_and_mi_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
clean_cols = [c for c, r in report.items() if not r["contaminated"]]
log(f"\n오염도 통과({len(clean_cols)}/{len(ALL_COLS)}): {clean_cols}")

log("\n=== 가벼운 LightGBM 홀드아웃 비교 (튜닝 없음, 참고용) ===")
try:
    import lightgbm as lgb
    from sklearn.metrics import balanced_accuracy_score, f1_score

    full_panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
    full_panel["timestamp"] = pd.to_datetime(full_panel["timestamp"])
    full_panel = full_panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    df2 = full_panel.merge(labels, on="timestamp", how="inner")
    df2["date"] = df2["timestamp"].dt.normalize()
    df2 = df2.merge(merged, on="date", how="left").sort_values("timestamp").reset_index(drop=True)
    for c in raw_cols:
        df2[f"{c}_diff1pct"] = df2[c].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df2[f"{c}_ma7dev"] = ((df2[c] - df2[c].rolling(7, min_periods=3).mean()) / df2[c].rolling(7, min_periods=3).mean().replace(0.0, np.nan)).fillna(0.0)
    df2[ALL_COLS] = df2[ALL_COLS].ffill().fillna(0.0)
    df2["funding_pressure_diff1"] = pd.to_numeric(df2["funding_pressure"], errors="coerce").diff(1).fillna(0.0)

    FINAL12_AVAILABLE = ["cvp_regime", "funding_pressure_diff1", "ou_halflife", "realized_skewness",
                          "mta_funding", "vwap_dist_24", "funding_roc_48", "breakout_strength"]
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
    log(f"\n[FINAL12 + DefiLlama(오염도 통과 {len(clean_cols)}개)]")
    r_combo = fit_eval(FINAL12_AVAILABLE + clean_cols, "base+defillama")

    (OUT_DIR / "holdout_comparison.json").write_text(json.dumps({"base": r_base, "base_plus_defillama": r_combo, "defillama_cols_used": clean_cols}, indent=2, ensure_ascii=False))
except ImportError:
    log("  lightgbm 없음 -- 홀드아웃 비교 스킵")

log(f"\n출력 디렉토리: {OUT_DIR}")
