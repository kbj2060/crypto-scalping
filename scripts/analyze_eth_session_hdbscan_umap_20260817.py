#!/usr/bin/env python3
"""ETH 5m: HDBSCAN / UMAP 기반 비지도 패턴 탐색 (매매 결정 없음).

결과 6(k-means/PCA)은 "선형 부분공간의 구형 군집"만 배제했다. 여기서는 밀도 기반(HDBSCAN)과
비선형 임베딩(UMAP)으로 남은 가능성을 검증한다.

**귀무 대조가 이 분석의 핵심이다.** UMAP 은 구조가 전혀 없는 데이터에서도 시각적으로 선명한
덩어리를 만들어낸다 (well-documented artifact). 따라서 실제 데이터의 결과는 반드시 동일
파이프라인을 통과한 귀무 데이터의 결과와 비교해야 한다. 귀무는 두 가지를 쓴다.

  NULL-A (iid 가우시안): 공분산만 일치. 구조 없음의 하한선.
  NULL-B (위상 무작위화): 각 주성분의 파워 스펙트럼을 보존해 **자기상관까지 일치**시킨다.
    5m 바는 연속 바가 거의 중복이라 밀도가 인위적으로 부풀려진다. NULL-A 는 이 효과를
    재현하지 못하므로 NULL-B 가 더 엄격하고 공정한 기준이다.

판정 지표
  - n_clusters, noise 비율
  - DBCV (hdbscan 의 relative_validity_) — 밀도 기반 군집의 내부 타당도
  - split-half 안정성 (하루 블록 A/B)

읽기 전용. 진입/청산 규칙이나 수익 계산 없음.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_eth_session_feature_clustering_20260817 as C  # noqa: E402
import analyze_eth_session_split_feature_price_correlation_20260817 as A  # noqa: E402
import analyze_eth_session_unsupervised_pattern_20260817 as U  # noqa: E402

SEED = 20260817
N_SUB = 30000
MIN_CLUSTER_SIZES = [150, 500, 1500]
UMAP_KW = dict(n_neighbors=50, min_dist=0.0, n_components=2, random_state=SEED)


def phase_randomize(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """열별 위상 무작위화. 파워 스펙트럼(=자기상관)은 보존, 비선형/비가우시안 구조는 파괴.

    주성분은 서로 무상관이므로 열별로 독립 수행해도 공분산 구조가 유지된다.
    """
    n = X.shape[0]
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        f = np.fft.rfft(X[:, j])
        ph = rng.uniform(0, 2 * np.pi, len(f))
        ph[0] = 0.0
        if n % 2 == 0:
            ph[-1] = 0.0
        out[:, j] = np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=n)
    return out


def run_hdbscan(X: np.ndarray, mcs: int) -> dict:
    cl = hdbscan.HDBSCAN(min_cluster_size=mcs, gen_min_span_tree=True, core_dist_n_jobs=4)
    lab = cl.fit_predict(X)
    k = len(set(lab) - {-1})
    try:
        dbcv = float(cl.relative_validity_)
    except Exception:
        dbcv = np.nan
    return {"labels": lab, "n_clusters": k, "noise": float((lab == -1).mean()), "dbcv": dbcv}


def describe(tag: str, res: dict) -> None:
    print(f"    {tag:26s} clusters={res['n_clusters']:3d}  noise={res['noise']:.3f}  "
          f"DBCV={res['dbcv']:+.4f}")


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp/session_split_20260817"
    frames = A.load_frames(cache)
    disc = pd.concat([frames["train_raw"], frames["val_raw"]], ignore_index=True)
    disc = disc.sort_values("timestamp").reset_index(drop=True)

    drop = set(U.MECHANICAL + U.CONTAMINATED + U.DUPLICATE)
    cols = [c for c in disc.columns if c != "timestamp" and c not in drop]
    Xr = disc[cols].to_numpy(float)
    X, = U.rank_gauss(Xr, [Xr])
    Z = PCA(n_components=0.90, random_state=SEED).fit_transform(X)
    print(f"피처 {len(cols)}개 -> PCA {Z.shape[1]}차원 (분산 90%), 전체 n={len(Z)}")

    rng = np.random.default_rng(SEED)
    # 귀무는 전체 길이에서 만들고(자기상관 보존을 위해), 그 다음 동일 인덱스로 부분추출
    null_a = rng.multivariate_normal(Z.mean(0), np.cov(Z, rowvar=False), size=len(Z))
    null_b = phase_randomize(Z, rng)
    sub = np.sort(rng.choice(len(Z), N_SUB, replace=False))
    datasets = {"REAL": Z[sub], "NULL-A(iid가우시안)": null_a[sub], "NULL-B(위상무작위)": null_b[sub]}
    print(f"부분추출 n={N_SUB}\n")

    # ------------------------------------------------------------------ A) PCA 공간 직접
    print("=" * 84)
    print("### A. HDBSCAN on PCA 공간 (33차원)")
    print("=" * 84)
    for mcs in MIN_CLUSTER_SIZES:
        print(f"  min_cluster_size={mcs}")
        for name, D in datasets.items():
            describe(name, run_hdbscan(D, mcs))

    # ------------------------------------------------------------------ B) UMAP -> HDBSCAN
    print("\n" + "=" * 84)
    print(f"### B. UMAP{UMAP_KW} -> HDBSCAN")
    print("=" * 84)
    emb = {}
    for name, D in datasets.items():
        emb[name] = umap.UMAP(**UMAP_KW).fit_transform(D)
        print(f"  {name} 임베딩 완료")
    print()
    umap_res = {}
    for mcs in MIN_CLUSTER_SIZES:
        print(f"  min_cluster_size={mcs}")
        for name in datasets:
            r = run_hdbscan(emb[name], mcs)
            umap_res[(name, mcs)] = r
            describe(name, r)

    # ------------------------------------------------------------------ C) 안정성 + 세션 연관
    print("\n" + "=" * 84)
    print("### C. 실제 데이터 군집의 안정성과 세션 연관 (UMAP->HDBSCAN, mcs=500)")
    print("=" * 84)
    lab = umap_res[("REAL", 500)]["labels"]
    ts_sub = disc["timestamp"].iloc[sub].reset_index(drop=True)

    # split-half: 하루 블록 A/B 로 나눠 각각 UMAP+HDBSCAN -> 공통 바 라벨 일치도는
    # UMAP 이 out-of-sample transform 을 지원하므로 A 모델로 전체를 임베딩해 비교한다.
    codes = pd.factorize(ts_sub.dt.floor("D"), sort=True)[0]
    perm = rng.permutation(codes.max() + 1)
    rank = np.empty_like(perm)
    rank[perm] = np.arange(len(perm))
    half_a = rank[codes] % 2 == 0
    Zs = datasets["REAL"]
    labs = []
    for m in [half_a, ~half_a]:
        um = umap.UMAP(**UMAP_KW).fit(Zs[m])
        labs.append(run_hdbscan(um.transform(Zs), 500)["labels"])
    print(f"  split-half ARI = {adjusted_rand_score(labs[0], labs[1]):.4f}")

    masks = C.build_masks(disc.iloc[sub].reset_index(drop=True))
    sess = pd.Series(np.where(masks["us"], "us",
                     np.where(masks["europe"], "europe",
                     np.where(masks["asia"], "asia", "none"))))
    obs = normalized_mutual_info_score(sess.to_numpy(), lab)
    # rotation null 은 시간 연속성이 필요하므로 부분추출 인덱스 상에서 회전시킨다
    null = [normalized_mutual_info_score(np.roll(sess.to_numpy(), s), lab)
            for s in rng.integers(1000, len(sess) - 1000, size=23)]
    null = np.asarray(null)
    print(f"  세션-군집 NMI={obs:.4f}  null={null.mean():.4f}±{null.std():.4f}  "
          f"z={(obs - null.mean()) / (null.std() + 1e-12):+.2f}")

    ct = pd.crosstab(sess, lab, normalize="columns")
    base = sess.value_counts(normalize=True)
    print("\n  군집별 세션 lift (base rate 대비):")
    print(ct.div(base, axis=0).round(2).to_string())
    print("\n  군집 크기:")
    print(pd.Series(lab).value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
