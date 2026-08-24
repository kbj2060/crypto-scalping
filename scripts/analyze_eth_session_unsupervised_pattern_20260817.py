#!/usr/bin/env python3
"""ETH 5m: 세션별 비지도 패턴 탐색 (매매 결정 없음, 구조 존재 여부만).

질문: 바(sample)들을 비지도로 묶으면 실제로 군집 구조가 존재하는가? 그리고 그 구조가
세션(us / europe / asia / none / late2023)과 관련이 있는가?

**핵심 방법론**: k-means 는 구조가 없어도 항상 k 개를 뱉는다. 따라서 세 가지 귀무 대조를 건다.
  1) Gap statistic (Tibshirani) — PCA 바운딩박스 균등분포 기준분포와 비교.
     gap 이 k 에 대해 단조증가하면 "군집 없음(단일 덩어리)"이 정답이다.
  2) 분할-반쪽 안정성 — 하루 블록 A/B 로 나눠 각각 군집 → 공통 바에 대한 ARI.
     구조가 없으면 임의 분할이라 ARI 가 0 근처로 떨어진다.
  3) 세션-군집 연관성은 rotation null 로 검정 — 세션 라벨을 1~23시간 원형 회전시켜
     NMI 귀무분포를 만든다.

피처 처리
  - 시각 인코딩 5개 제외: 세션이 시각으로 정의되므로 포함하면 동어반복
    (2026-08-17 군집 분석에서 전역 ARI gap 의 대부분이 이 동어반복이었음)
  - close 레벨과 |Spearman| >= 0.5 인 16개 제외: 포함하면 군집이 시장 상태가 아니라
    '몇 월인가'(가격 에폭)로 수렴한다
  - `smart_money_flow` 제외: `oi_change_rate` 와 완전 동일 컬럼
  - rank-gauss 변환 후 PCA (분산 90% 유지)

읽기 전용 연구 스크립트. 진입/청산 규칙이나 수익 계산을 일절 하지 않는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_eth_session_feature_clustering_20260817 as C  # noqa: E402
import analyze_eth_session_split_feature_price_correlation_20260817 as A  # noqa: E402

OUTDIR = ROOT / "tmp/session_split_20260817"
SEED = 20260817
KS = list(range(2, 13))
N_REF = 10          # gap statistic 기준분포 개수
SIL_SUBSAMPLE = 8000

MECHANICAL = ["hour_sin", "hour_cos", "minute_sin", "minute_cos", "is_hour_open"]
CONTAMINATED = [
    "close", "low", "high", "open", "close_btc", "sum_open_interest_value",
    "sum_toptrader_long_short_ratio", "whale_retail_ratio", "funding_pressure",
    "squeeze_power", "last_funding_rate", "long_squeeze_risk", "short_squeeze_risk",
    "garman_klass_vol", "parkinson_vol", "rogers_satchell_vol",
]
DUPLICATE = ["smart_money_flow"]


def rank_gauss(train: np.ndarray, apply_to: list[np.ndarray]) -> list[np.ndarray]:
    """train 으로 적합한 rank-gauss 변환을 apply_to 각각에 적용 (두꺼운 꼬리 완화)."""
    out = [np.empty_like(x) for x in apply_to]
    for j in range(train.shape[1]):
        ref = np.sort(train[:, j])
        for xi, x in enumerate(apply_to):
            p = np.searchsorted(ref, x[:, j], side="right") / (len(ref) + 1.0)
            out[xi][:, j] = ndtri(np.clip(p, 1e-6, 1 - 1e-6))
    return out


def gap_statistic(X: np.ndarray, ks: list[int], n_ref: int, rng: np.random.Generator) -> pd.DataFrame:
    """Tibshirani gap statistic. 기준분포는 데이터의 바운딩박스 안 균등분포."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(X)
        log_wk = np.log(km.inertia_)
        ref_logs = []
        for b in range(n_ref):
            ref = rng.uniform(lo, hi, size=X.shape)
            kmr = KMeans(n_clusters=k, n_init=2, random_state=SEED + b).fit(ref)
            ref_logs.append(np.log(kmr.inertia_))
        ref_logs = np.asarray(ref_logs)
        gap = ref_logs.mean() - log_wk
        sk = ref_logs.std() * np.sqrt(1 + 1 / n_ref)
        rows.append({"k": k, "log_Wk": log_wk, "ref_mean": ref_logs.mean(), "gap": gap, "s_k": sk})
    df = pd.DataFrame(rows)
    # Tibshirani 기준: gap(k) >= gap(k+1) - s(k+1) 인 가장 작은 k
    df["gap_next_minus_s"] = df["gap"].shift(-1) - df["s_k"].shift(-1)
    df["is_elbow"] = df["gap"] >= df["gap_next_minus_s"]
    return df


def split_half_stability(X: np.ndarray, ts: pd.Series, ks: list[int],
                         rng: np.random.Generator) -> pd.DataFrame:
    """하루 블록 A/B 로 나눠 각각 학습 -> 전체 바에 예측 -> ARI.

    같은 데이터에 두 독립 모델을 씌워 라벨 일치도를 본다. 구조가 없으면 0 근처.
    """
    codes = pd.factorize(ts.dt.floor("D"), sort=True)[0]
    perm = rng.permutation(codes.max() + 1)
    rank = np.empty_like(perm)
    rank[perm] = np.arange(len(perm))
    a = rank[codes] % 2 == 0
    rows = []
    for k in ks:
        ka = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(X[a])
        kb = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(X[~a])
        rows.append({"k": k, "split_half_ari": adjusted_rand_score(ka.predict(X), kb.predict(X))})
    return pd.DataFrame(rows)


def session_association(labels: np.ndarray, sess: pd.Series, ts: pd.Series,
                        n_rot: int = 23) -> dict:
    """세션-군집 연관성(NMI)을 rotation null 과 비교."""
    obs = normalized_mutual_info_score(sess.to_numpy(), labels)
    null = []
    vals = sess.to_numpy()
    for h in range(1, n_rot + 1):
        null.append(normalized_mutual_info_score(np.roll(vals, 12 * h), labels))
    null = np.asarray(null)
    return {"nmi": obs, "null_mean": null.mean(), "null_max": null.max(),
            "z": (obs - null.mean()) / (null.std() + 1e-12)}


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR
    frames = A.load_frames(cache)
    disc = pd.concat([frames["train_raw"], frames["val_raw"]], ignore_index=True)
    disc = disc.sort_values("timestamp").reset_index(drop=True)
    oos = frames["oos_raw"].sort_values("timestamp").reset_index(drop=True)

    drop = set(MECHANICAL + CONTAMINATED + DUPLICATE)
    cols = [c for c in disc.columns if c != "timestamp" and c not in drop]
    print(f"사용 피처 {len(cols)}개 (102 - 시각 {len(MECHANICAL)} - 오염 {len(CONTAMINATED)} "
          f"- 중복 {len(DUPLICATE)})")

    rng = np.random.default_rng(SEED)
    Xd_raw = disc[cols].to_numpy(float)
    Xo_raw = oos[cols].to_numpy(float)
    Xd, Xo = rank_gauss(Xd_raw, [Xd_raw, Xo_raw])

    pca = PCA(n_components=0.90, random_state=SEED).fit(Xd)
    Zd, Zo = pca.transform(Xd), pca.transform(Xo)
    print(f"PCA: {Zd.shape[1]}개 주성분으로 분산 90% (원 {len(cols)}차원)")
    print(f"  PC1-5 설명력: {np.round(pca.explained_variance_ratio_[:5], 3)}")

    # ---------------------------------------------------------------- 1) 군집이 존재하는가
    print(f"\n{'='*78}\n### 1. 군집 구조가 존재하는가 (TRAIN+VAL, n={len(Zd)})\n{'='*78}")
    gap = gap_statistic(Zd, KS, N_REF, rng)
    stab = split_half_stability(Zd, disc["timestamp"], KS, rng)
    sil = []
    idx = rng.choice(len(Zd), SIL_SUBSAMPLE, replace=False)
    for k in KS:
        km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(Zd)
        sil.append({"k": k, "silhouette": silhouette_score(Zd[idx], km.labels_[idx])})
    tab = gap.merge(stab, on="k").merge(pd.DataFrame(sil), on="k")
    print(tab[["k", "gap", "s_k", "is_elbow", "split_half_ari", "silhouette"]].round(4).to_string(index=False))

    elbows = tab.loc[tab.is_elbow, "k"].tolist()
    print(f"\n  Gap elbow (Tibshirani 기준 최소 k): {elbows[0] if elbows else '없음 (gap 단조증가)'}")
    print(f"  gap 최대 k: {int(tab.loc[tab.gap.idxmax(), 'k'])}")
    print(f"  silhouette 최대: k={int(tab.loc[tab.silhouette.idxmax(), 'k'])} "
          f"({tab.silhouette.max():.4f})")

    # ---------------------------------------------------------------- 2) 세션과 관련이 있는가
    print(f"\n{'='*78}\n### 2. 군집이 세션과 관련이 있는가 (rotation null 대비)\n{'='*78}")
    masks = C.build_masks(disc)
    sess = pd.Series(np.where(masks["us"], "us",
                     np.where(masks["europe"], "europe",
                     np.where(masks["asia"], "asia", "none"))))
    for k in [3, 5, 8]:
        km = KMeans(n_clusters=k, n_init=8, random_state=SEED).fit(Zd)
        assoc = session_association(km.labels_, sess, disc["timestamp"])
        print(f"  k={k:2d}  NMI={assoc['nmi']:.4f}  null_mean={assoc['null_mean']:.4f}  "
              f"null_max={assoc['null_max']:.4f}  z={assoc['z']:+.2f}")

    # ---------------------------------------------------------------- 3) OOS 재현
    print(f"\n{'='*78}\n### 3. OOS 재현 (TRAIN+VAL 군집을 OOS 에 적용, n={len(Zo)})\n{'='*78}")
    k_best = int(tab.loc[tab.silhouette.idxmax(), "k"])
    km = KMeans(n_clusters=k_best, n_init=8, random_state=SEED).fit(Zd)
    lab_o = km.predict(Zo)
    km_o = KMeans(n_clusters=k_best, n_init=8, random_state=SEED).fit(Zo)
    print(f"  k={k_best}: TRAIN+VAL 모델을 OOS 에 적용 vs OOS 자체 군집  "
          f"ARI={adjusted_rand_score(lab_o, km_o.labels_):.4f}")
    share_d = pd.Series(km.labels_).value_counts(normalize=True).sort_index()
    share_o = pd.Series(lab_o).value_counts(normalize=True).sort_index()
    comp = pd.DataFrame({"TRAIN+VAL": share_d, "OOS": share_o}).fillna(0.0)
    comp["차이"] = comp["OOS"] - comp["TRAIN+VAL"]
    print("\n  군집 점유율:")
    print(comp.round(4).to_string())

    tab.to_csv(OUTDIR / "unsupervised_cluster_validity.csv", index=False)
    print(f"\nWROTE {OUTDIR / 'unsupervised_cluster_validity.csv'}")


if __name__ == "__main__":
    main()
