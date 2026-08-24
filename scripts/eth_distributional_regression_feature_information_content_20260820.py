#!/usr/bin/env python3
"""③분포적 회귀 후보 착수 전 cheap check -- 실제 분포모수 헤드+NLL loss+CRPS평가+백테스트
전체를 새로 짜기 전에("나머지 두 후보보다 훨씬 큰 작업"이라고 이미 문서화됨), 158개 캐노니컬
피쳐가 연속 forward log-return 타겟과 애초에 조금이라도 관계가 있는지부터 확인한다(DC/CUSUM
때 개별피쳐 정보량 체크를 학습 전에 먼저 한 것과 동일 원칙, 이번엔 이산라벨 대신 연속타겟이라
AUC 대신 Spearman IC 사용).

라벨: tmp/eth_distributional_regression_return_labels_20260819/fwd_return_labels_{year}.csv
(이미 빌드됨, 4개 horizon: h12/24/48/96bar). 158개 피쳐 각각을 각 horizon의 raw forward
log-return과 Spearman IC로 비교, TRAIN/VAL/OOS 3-split x 4horizon마다 "158개 중 최고IC"
permutation-null(라벨 2000회 셔플, 벡터화)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/eth_distributional_regression_return_labels_20260819"
RNG = np.random.default_rng(20260820)
N_PERM = 2000
HORIZONS = ["h12", "h24", "h48", "h96"]
SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31"),
    "VAL": ("2025-09-01", "2025-12-31"),
    "OOS": ("2026-01-01", "2026-03-31"),
}

import sys
sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega

BASE_158 = json.loads(Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad/dc_base_158_cols.json").read_text())


def _batch_ic_permnull(X: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """X: (n, k) 158개 피쳐, y: (n,) 연속타겟. 전체 컬럼에 대해 벡터화 랭크-행렬곱으로 실제IC +
    퍼뮤테이션당 '158개 중 최고|IC|' 귀무분포를 한번에 계산."""
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    n = len(y)
    rank_y = rankdata(y)
    mean_y, std_y = rank_y.mean(), rank_y.std()

    rank_X = np.apply_along_axis(rankdata, 0, np.where(np.isnan(X), np.nanmedian(X, axis=0), X))
    # 컬럼별 NaN은 중앙값으로 대체 후 랭크(피쳐별 결측이 소수인 경우의 근사, 전량 NaN 컬럼은 아래서 걸러짐)
    valid_col = ~np.all(np.isnan(X), axis=0)
    mean_x = rank_X.mean(axis=0)
    std_x = rank_X.std(axis=0)

    real_ic = (rank_X.T @ rank_y / n - mean_x * mean_y) / (std_x * std_y)
    real_ic = np.where(valid_col, real_ic, np.nan)

    null_max = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(rank_y)
        ic_perm = (rank_X.T @ y_perm / n - mean_x * mean_y) / (std_x * std_y)
        ic_perm = np.where(valid_col, ic_perm, 0.0)
        null_max[i] = np.nanmax(np.abs(ic_perm))

    best_idx = int(np.nanargmax(np.abs(real_ic)))
    real_best_abs = float(np.abs(real_ic[best_idx]))
    p95 = float(np.percentile(null_max, 95))
    empirical_p = float((null_max >= real_best_abs).mean())
    return {"n": n, "real_ic_all": real_ic, "best_idx": best_idx, "best_abs_ic": real_best_abs,
            "null_p95": p95, "empirical_p": empirical_p}


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    feat_frames = []
    for year, feat in ((2024, None), (2025, train), (2026, eval_df)):
        if year == 2024:
            continue
        f = feat[["timestamp", *BASE_158]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        feat_frames.append(f)
    # 2024는 omega 프레임에 없음(DC 캐노니컬은 2025/2026만) -- raw canonical CSV에서 직접 로드
    f2024 = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2024.csv", low_memory=False)
    f2024["timestamp"] = pd.to_datetime(f2024["timestamp"])
    missing_2024_cols = [c for c in BASE_158 if c not in f2024.columns]
    print(f"2024 CSV에 없는 피쳐(0-fill 대체): {missing_2024_cols}", flush=True)
    for c in missing_2024_cols:
        f2024[c] = 0.0
    f2024 = f2024[["timestamp", *BASE_158]]
    feat_frames.insert(0, f2024)

    feat_all = pd.concat(feat_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    lbl_frames = [pd.read_csv(LABEL_DIR / f"fwd_return_labels_{y}.csv", parse_dates=["timestamp"]) for y in (2024, 2025, 2026)]
    lbl_all = pd.concat(lbl_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df = feat_all.merge(lbl_all, on="timestamp", how="inner").set_index("timestamp")
    print(f"피쳐+라벨 병합: {len(df):,}행", flush=True)

    X_full = df[BASE_158].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    results = {}
    for split_name, (start, end) in SPLITS.items():
        mask = (df.index >= start) & (df.index <= end)
        X_sub = X_full[mask]
        for h in HORIZONS:
            y_sub = df.loc[mask, f"fwd_logret_{h}"].to_numpy(dtype=np.float64)
            r = _batch_ic_permnull(X_sub, y_sub, N_PERM, RNG)
            results[(split_name, h)] = r
            best_feat = BASE_158[r["best_idx"]]
            print(f"  {split_name:5s} {h}: n={r['n']:,} best_feat={best_feat:35s} "
                  f"|IC|={r['best_abs_ic']:.4f} null_p95={r['null_p95']:.4f} "
                  f"empirical_p={r['empirical_p']:.3f}", flush=True)

    n_sig = sum(1 for r in results.values() if r["empirical_p"] < 0.05)
    best_feats_raw = sorted({BASE_158[v["best_idx"]] for v in results.values()})
    print(f"\n순열귀무 p<0.05 통과: {n_sig}/{len(results)}칸", flush=True)
    print(f"1위 피쳐 종류(split마다 겹치는지 -- 겹치면 진짜신호, 안겹치면 노이즈): "
          f"{best_feats_raw} ({len(best_feats_raw)}종/{len(results)}칸)", flush=True)

    # ⚠️ WEEK_BARS=2016 롤링평균 차감으로 "추세제거"를 시도했다가 폐기 -- 이 상수가 하필
    # dual_momentum의 lookback(features/engineering.py:958, shift(2016))과 완전히 같아서
    # 기계적 아티팩트였음이 직접확인됨(dual_momentum vs 그 롤링평균 rho=0.84~0.86, vs raw
    # 타겟은 rho~0) -- "발견"이 아니라 내가 만든 순환논리였다. 대신 이 세션 전체가 써온
    # 정확한 기준(TRAIN/VAL/OOS 독립 split간 부호일관성)으로 판정한다.
    print("\n[교차-split 부호일관성 진단] 각 horizon마다 158개 피쳐 중 TRAIN·VAL·OOS 3개 split "
          "전부에서 같은 부호이면서 |IC|>=0.02인 피쳐:", flush=True)
    consistency_report = {}
    for h in HORIZONS:
        ic_train = results[("TRAIN", h)]["real_ic_all"]
        ic_val = results[("VAL", h)]["real_ic_all"]
        ic_oos = results[("OOS", h)]["real_ic_all"]
        consistent = []
        for i, feat in enumerate(BASE_158):
            vals = [ic_train[i], ic_val[i], ic_oos[i]]
            if any(np.isnan(vals)):
                continue
            same_sign = (vals[0] > 0) == (vals[1] > 0) == (vals[2] > 0)
            all_meaningful = all(abs(v) >= 0.02 for v in vals)
            if same_sign and all_meaningful:
                consistent.append({"feature": feat, "train_ic": float(vals[0]), "val_ic": float(vals[1]), "oos_ic": float(vals[2])})
        consistency_report[h] = consistent
        print(f"  {h}: {len(consistent)}개 피쳐 -- {[c['feature'] for c in consistent]}", flush=True)

    out = {
        "ic_scan": {f"{k[0]}|{k[1]}": {"n": v["n"], "best_feature": BASE_158[v["best_idx"]],
                                         "best_abs_ic": v["best_abs_ic"], "null_p95": v["null_p95"],
                                         "empirical_p": v["empirical_p"]}
                    for k, v in results.items()},
        "best_feature_overlap_across_splits": best_feats_raw,
        "cross_split_sign_consistent_features": consistency_report,
        "abandoned_check_note": "week-rolling-mean detrend abandoned: mechanically correlated with dual_momentum (shares WEEK_BARS=2016 window), see script comments",
    }
    out_path = ROOT / "tmp/eth_distributional_regression_feature_information_content_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
