"""신규 탐색 축 스카우팅(docs/experiments/eth_h48qual_direction_skill_new_directions_scouting_
20260812.md) (a)-1 후보: FINAL12는 direction_head(`zigzag_action`) 전용으로 스크리닝된 적이
없다(quality 타겟과 섞여 병합됐고, 공통윈도우도 2025 상반기 6개월뿐이었다) — 이 스크립트는
`zigzag_action` 단독 relevance(mutual_info_classif)로 처음부터 다시 순위를 매긴다.

이전 rescreen(quality 회귀 대상, rescreen_eth_h48qual_quality_regression_*_20260811.py)이 쓰던
`fa_features.parquet`(2025-only, M7/AI teacher 컬럼 포함)는 레포 밖 세션 scratchpad에만 있고
위험했다(이번 세션에 tmp/eth_h48qual_fa_features_backup_20260812/로 백업 완료). 이 스크립트는
대신 커밋된 `data/splits/year_oos/eth_features_2024_2026_analysis.csv`(zig075 소스 패널, 145
raw 컬럼, 2024-06~2026-08 커버 -- fa_features.parquet의 2025-only보다 넓고 실제 VAL/OOS까지
포함)를 1차 풀로 쓴다. 이 패널엔 M7/AI teacher 컬럼이 아예 없어 Model Architect의 2026-05-27
정책("direction-family M7/AI outputs are removed from active/candidate inputs")을 굳이
필터링하지 않아도 자동으로 지킨다.

TRAIN 윈도우는 이 패널의 실제 커버리지(2024-06-01~)에 맞춰 2024-06-01~2025-09-30로 잡는다 --
계약 문서의 canonical TRAIN(2024-01~2025-09, 183,936행)보다 5개월 짧은 부분집합이다(2024
Jan~May 데이터가 이 패널에 없음). VAL/OOS는 canonical과 동일(2025-10~12/2026-01~02) -- 이
패널이 실제로 커버해서 이번엔 진짜 VAL+OOS 홀드아웃 확인이 가능하다(quality 회귀 rescreen은
2025 상반기 안에서만 확인했었음).

절차: (1) 이 세션 표준 deny-list/PRICE_LIKE/CONST/REPLACE(diff1/dt288) 그대로 적용 (2) TRAIN에서
zigzag_action 3-class MI로 relevance 계산 (3) mRMR(TRAIN 상관행렬) 순차 선택 + |r|>0.5 하드
중복제거 (4) 생존 후보 전부 corr(close) 오염도 체크(표준 절차, 배제 기준 0.561) (5) FINAL12
대비 비교 (6) 가벼운 LightGBM 홀드아웃(튜닝 없음)으로 FINAL12 단독 vs FINAL12+신규후보 VAL/OOS
분류 성능 대조 -- "MI 랭킹이 다르다"와 "실제로 도움이 된다"를 구분하기 위함."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

ROOT = Path("/home/kbj20/crypto-scalping")
OUT_DIR = ROOT / "tmp/eth_h48qual_direction_only_mrmr_rescreen_20260812"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:59:59")

FINAL12 = [
    "cvp_regime", "funding_pressure_diff1", "ou_halflife", "m7_vae_error_dt288",
    "realized_skewness", "mta_funding", "sig_whale_dt288", "sum_toptrader_long_short_ratio_dt288",
    "vwap_dist_24", "funding_roc_48", "breakout_strength",
    "regime3_current_sensitive_wide24_chop_prob",
]

DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")
NON_FEATURE = {"timestamp", "open", "high", "low", "close", "open_btc", "high_btc", "low_btc", "close_btc"}
PRICE_LIKE = ["sum_open_interest_value"]
CONST = []
# 이 세션에서 이미 가격추세/레벨 오염이 확인돼 detrend/diff1으로 치환하기로 한 것들(quality
# rescreen의 REPLACE와 동일 관례) -- m7_vae_error는 이 패널에 없어 제외.
REPLACE = {
    "funding_pressure": ("funding_pressure_diff1", "diff1"),
    "last_funding_rate": ("last_funding_rate_dt288", "dt288"),
    "squeeze_power": ("squeeze_power_dt288", "dt288"),
    "long_squeeze_risk": ("long_squeeze_risk_dt288", "dt288"),
    "funding_abs": ("funding_abs_dt288", "dt288"),
    "whale_retail_ratio": ("whale_retail_ratio_dt288", "dt288"),
    "count_long_short_ratio": ("count_long_short_ratio_dt288", "dt288"),
    "sum_toptrader_long_short_ratio": ("sum_toptrader_long_short_ratio_dt288", "dt288"),
}
CONTAMINATION_THRESHOLD = 0.561  # FINAL12 dedup에서 확정된 배제 기준


def is_candidate(col: str) -> bool:
    if col in NON_FEATURE or col in PRICE_LIKE or col in CONST or col in REPLACE:
        return False
    if any(col.startswith(p) for p in DENY_PREFIXES):
        return False
    if any(t in col for t in DENY_TOKENS):
        return False
    return True


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. 패널 + zigzag_action 라벨 로딩
# ---------------------------------------------------------------------------

log("zig075 소스 패널 로딩 (data/splits/year_oos/eth_features_2024_2026_analysis.csv)...")
panel = pd.read_csv(ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv", low_memory=False)
panel["timestamp"] = pd.to_datetime(panel["timestamp"])
panel = panel.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
log(f"  패널: {len(panel)}행, {panel['timestamp'].min()} ~ {panel['timestamp'].max()}, {len(panel.columns)-1}개 raw 컬럼")

LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
labels = pd.concat([
    pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{y}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    for y in (2024, 2025, 2026)
], ignore_index=True).drop_duplicates("timestamp", keep="last")

df = panel.merge(labels, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
log(f"  라벨 병합 후: {len(df)}행")

candidate_raw = [c for c in df.columns if is_candidate(c) and pd.api.types.is_numeric_dtype(df[c])]
replace_raw_needed = [r for r in REPLACE if r in df.columns]
log(f"  원시 후보 {len(candidate_raw)}개, REPLACE 치환 대상 {replace_raw_needed}")

for raw, (derived, kind) in REPLACE.items():
    if raw not in df.columns:
        continue
    src = pd.to_numeric(df[raw], errors="coerce").astype(np.float64)
    if kind == "diff1":
        df[derived] = src.diff(1).fillna(0.0)
    elif kind == "dt288":
        df[derived] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)

POOL = sorted(set(candidate_raw) | {d for d, _ in REPLACE.values() if d in df.columns})
log(f"  최종 후보 풀 {len(POOL)}개 (raw {len(candidate_raw)} - REPLACE원본 {len(replace_raw_needed)} + 파생 {len(REPLACE)})")

train_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= VAL_END)
oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)
log(f"  TRAIN(2024-06~2025-09) n={train_mask.sum()}  VAL n={val_mask.sum()}  OOS n={oos_mask.sum()}")

X_all = df[POOL].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_all = pd.to_numeric(df["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)

X_train, y_train = X_all[train_mask].reset_index(drop=True), y_all[train_mask]
X_val, y_val = X_all[val_mask].reset_index(drop=True), y_all[val_mask]
X_oos, y_oos = X_all[oos_mask].reset_index(drop=True), y_all[oos_mask]
close_train = df.loc[train_mask, "close"].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. direction-only relevance (MI vs zigzag_action, TRAIN)
# ---------------------------------------------------------------------------

log("\nMI(zigzag_action 3-class, TRAIN) 계산 중...")
mi = mutual_info_classif(X_train.to_numpy(), y_train, discrete_features=False, random_state=260620)
relevance = pd.Series(mi, index=POOL).sort_values(ascending=False)
relevance_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-12)

log("상위 20 (direction-only MI):")
for f, v in relevance.head(20).items():
    log(f"  {f:<45s} MI={v:.4f}")

# ---------------------------------------------------------------------------
# 3. mRMR 순차선택 + 하드 중복제거
# ---------------------------------------------------------------------------

corr = X_train.corr()


def mrmr_select(rel: pd.Series, corr_matrix: pd.DataFrame, k: int) -> list[str]:
    remaining = list(rel.index)
    selected: list[str] = []
    for _ in range(min(k, len(remaining))):
        if not selected:
            best = rel[remaining].idxmax()
        else:
            redund = corr_matrix.loc[remaining, selected].abs().mean(axis=1)
            score = rel[remaining] / (1.0 + redund)
            best = score.idxmax()
        selected.append(best)
        remaining.remove(best)
    return selected


K = 25
top_k = mrmr_select(relevance_norm, corr, K)
log(f"\nmRMR top{K} (relevance / corr(close)):")
mrmr_rows = []
for f in top_k:
    cc = float(X_train[f].corr(close_train))
    contaminated = abs(cc) > CONTAMINATION_THRESHOLD
    log(f"  {f:<45s} relevance={relevance_norm[f]:.3f}  corr(close)={cc:+.3f}{'  [오염]' if contaminated else ''}")
    mrmr_rows.append({"feature": f, "relevance": float(relevance_norm[f]), "corr_close": cc, "contaminated": contaminated})

contaminated_set = {r["feature"] for r in mrmr_rows if r["contaminated"]}
top_k_clean = [f for f in top_k if f not in contaminated_set]
if contaminated_set:
    log(f"  오염도 체크로 제외: {sorted(contaminated_set)}")

Ctop = X_train[top_k_clean].corr()
pairs = []
for i, a in enumerate(top_k_clean):
    for b in top_k_clean[i + 1:]:
        r = Ctop.loc[a, b]
        if abs(r) > 0.5:
            pairs.append((a, b, float(r)))

adj: dict[str, set[str]] = {}
for a, b, r in pairs:
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)
kept, dropped = [], []
for f in top_k_clean:
    conflicts = adj.get(f, set()) & set(kept)
    if not conflicts:
        kept.append(f)
    else:
        dropped.append((f, sorted(conflicts)))

log(f"\ntop{K}(오염도 통과 {len(top_k_clean)}개) -> |r|>0.5 중복제거 후 {len(kept)}개 생존:")
for f in kept:
    log(f"  {f}  relevance={relevance_norm[f]:.3f}")
log(f"탈락 {len(dropped)}개: {[d[0] for d in dropped]}")

overlap = [f for f in FINAL12 if f in kept]
only_final12 = [f for f in FINAL12 if f not in kept]
new_candidates = [f for f in kept if f not in FINAL12]
log(f"\nFINAL12 대비: 겹침 {len(overlap)}/12={overlap}")
log(f"FINAL12에만 있음(이번 direction-only 랭킹에선 탈락): {only_final12}")
log(f"신규 후보(FINAL12엔 없음): {new_candidates}")

(OUT_DIR / "direction_only_mrmr_result.json").write_text(json.dumps({
    "pool_size": len(POOL), "train_n": int(train_mask.sum()), "val_n": int(val_mask.sum()), "oos_n": int(oos_mask.sum()),
    "relevance_top20": {f: float(v) for f, v in relevance.head(20).items()},
    "mrmr_top_k": mrmr_rows, "contaminated_excluded": sorted(contaminated_set),
    "final_deduped": kept, "dropped_redundant": [(f, c) for f, c in dropped],
    "overlap_with_FINAL12": overlap, "only_FINAL12": only_final12, "new_candidates": new_candidates,
}, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------------
# 4. 가벼운 홀드아웃 검증: FINAL12 단독 vs FINAL12+신규후보 (튜닝 없음)
# ---------------------------------------------------------------------------

log("\n=== 가벼운 LightGBM 홀드아웃 비교 (튜닝 없음, 참고용) ===")
final12_available = [c for c in FINAL12 if c in df.columns]
missing_final12 = [c for c in FINAL12 if c not in df.columns]
if missing_final12:
    log(f"  주의: 이 패널엔 FINAL12 중 {missing_final12}가 없음(다른 피쳐-엔지니어링 세대) -- 비교에서 제외")

try:
    import lightgbm as lgb

    def fit_eval(feature_cols, label):
        Xtr = df.loc[train_mask, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xv = df.loc[val_mask, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xo = df.loc[oos_mask, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        clf = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=500, learning_rate=0.05,
                                  num_leaves=31, random_state=260620, verbosity=-1)
        clf.fit(Xtr, y_train, eval_set=[(Xv, y_val)], eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(30, verbose=False)])
        out = {}
        for split_name, X, y_true in [("VAL", Xv, y_val), ("OOS", Xo, y_oos)]:
            pred = clf.predict(X)
            out[split_name] = {"balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
                                "macro_f1": float(f1_score(y_true, pred, average="macro"))}
            log(f"  [{label}] {split_name}: balanced_acc={out[split_name]['balanced_accuracy']:.3f}  macro_f1={out[split_name]['macro_f1']:.3f}")
        return out, clf

    log("\n[FINAL12 단독]")
    r_final12, _ = fit_eval(final12_available, "FINAL12")

    expanded = sorted(set(final12_available) | set(new_candidates))
    log(f"\n[FINAL12 + 신규후보 {len(new_candidates)}개 = {len(expanded)}개]")
    r_expanded, _ = fit_eval(expanded, "FINAL12+신규")

    (OUT_DIR / "holdout_comparison.json").write_text(json.dumps({
        "final12_only": r_final12, "final12_plus_new": r_expanded, "new_candidates_used": new_candidates,
    }, indent=2, ensure_ascii=False))
except ImportError:
    log("  lightgbm 없음 -- 홀드아웃 비교 스킵")

log(f"\n출력 디렉토리: {OUT_DIR}")
