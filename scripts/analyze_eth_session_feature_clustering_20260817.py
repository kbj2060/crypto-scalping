#!/usr/bin/env python3
"""ETH 5m: 세션별 비지도 피처 군집 구조 분석.

질문: 세션(us / europe / asia / none / late20-23)으로 나눈 뒤 피처들을 비지도로 묶으면,
세션마다 다르게 분류되는 피처가 있는가?

방법
  - 거리 = 1 - |Spearman(feature_i, feature_j)| (세션별 상관행렬로 각각 계산)
  - average-linkage 계층군집, 고정 k 로 컷
  - 세션 간 군집 일치도 = Adjusted Rand Index (ARI)
  - **귀무 대조**: 같은 세션의 바를 하루 단위 블록으로 A/B 반쪽 분할 후 각각 군집 →
    within-session ARI. 표본 노이즈만으로도 군집은 흔들리므로, cross-session ARI 가
    within-session ARI 보다 유의하게 낮아야 "세션마다 구조가 다르다"고 말할 수 있다.
  - 피처 단위: 각 피처의 '상관 프로파일'(다른 101개 피처와의 상관 벡터)이 세션 간에 얼마나
    바뀌는지 = profile_agreement. 낮을수록 세션 의존적인 피처.
    같은 귀무 대조(within-session 반쪽)를 피처별로도 계산해 비교한다.

읽기 전용 연구 스크립트.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_eth_session_split_feature_price_correlation_20260817 as A  # noqa: E402

OUTDIR = ROOT / "tmp/session_split_20260817"
K_CLUSTERS = 10
SESSIONS = ["us", "europe", "asia", "none", "late2023"]


def zrank_matrix(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """열별 z-rank 행렬 + 유효 열 마스크 (상수열/비유한 열 제외)."""
    n, k = feats.shape
    z = np.zeros((n, k))
    valid = np.zeros(k, dtype=bool)
    for j in range(k):
        col = feats[:, j]
        if not np.all(np.isfinite(col)):
            continue
        r = rankdata(col)
        sd = r.std()
        if sd < 1e-12:
            continue
        z[:, j] = (r - r.mean()) / sd
        valid[j] = True
    return z, valid


def corr_matrix(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman 상관행렬 (유효 열만)."""
    z, valid = zrank_matrix(feats)
    zv = z[:, valid]
    c = (zv.T @ zv) / len(zv)
    np.fill_diagonal(c, 1.0)
    return np.clip(c, -1.0, 1.0), valid


def cluster_labels(corr: np.ndarray, k: int) -> np.ndarray:
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    link = linkage(squareform(dist, checks=False), method="average")
    return fcluster(link, t=k, criterion="maxclust")


def equal_size_halves(ts: pd.Series, n_target: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """하루 단위 블록을 섞어 A/B 두 반쪽으로 나누되, 각 반쪽이 정확히 n_target//2 바가 되도록 자른다.

    within(같은 세션 A vs B)과 cross(세션1 A vs 세션2 A) 비교가 **동일한 표본 크기**에서
    이뤄져야 ARI 를 공정하게 비교할 수 있다. 전체 n 으로 cross 를, n/2 로 within 을 재면
    baseline 이 인위적으로 낮아져 '세션 구조가 다르다'는 결론이 거짓으로 나온다.
    """
    codes = pd.factorize(ts.dt.floor("D"), sort=True)[0]
    order = rng.permutation(codes.max() + 1)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    shuffled_day = rank[codes]
    half = n_target // 2
    a = np.zeros(len(codes), dtype=bool)
    b = np.zeros(len(codes), dtype=bool)
    # 섞인 날짜 순서대로 A 를 채우고, 그 다음 B 를 채운다 (블록 구조 보존)
    pos = np.argsort(shuffled_day, kind="stable")
    a[pos[:half]] = True
    b[pos[half:2 * half]] = True
    return a, b


def build_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    sess = A.assign_sessions(df)
    ts = df["timestamp"]
    hr = ts.dt.hour.to_numpy()
    wknd = (ts.dt.dayofweek >= 5).to_numpy()
    masks = {s: (sess == s).to_numpy() for s in ["us", "europe", "asia", "none"]}
    masks["late2023"] = (sess == "none").to_numpy() & ~wknd & (hr >= 20)
    return masks


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTDIR
    frames = A.load_frames(cache)
    # 군집 구조는 표본이 클수록 안정적이므로 TRAIN+VAL 을 붙여 발견하고, OOS 로 재확인한다.
    disc = pd.concat([frames["train_raw"], frames["val_raw"]], ignore_index=True)
    disc = disc.sort_values("timestamp").reset_index(drop=True)
    oos = frames["oos_raw"].sort_values("timestamp").reset_index(drop=True)
    feature_cols = [c for c in disc.columns if c != "timestamp"]

    for tag, df in [("TRAIN+VAL", disc), ("OOS", oos)]:
        print(f"\n{'='*70}\n### {tag}  (n={len(df)})\n{'='*70}")
        masks = build_masks(df)
        feats_all = df[feature_cols].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(20260817)

        masks = {s: m for s, m in masks.items() if m.sum() >= 500}
        n_target = min(int(m.sum()) for m in masks.values())
        print(f"공통 표본 크기 n_target={n_target} (반쪽당 {n_target // 2})")

        halves = {}
        for s, m in masks.items():
            sub = feats_all[m]
            a, b = equal_size_halves(df.loc[m, "timestamp"], n_target, rng)
            ca, va = corr_matrix(sub[a])
            cb, vb = corr_matrix(sub[b])
            halves[s] = (ca, va, cb, vb)
            print(f"  {s}: n={int(m.sum())} -> A={int(a.sum())} B={int(b.sum())}")

        common = np.ones(len(feature_cols), dtype=bool)
        for _s, (_ca, va, _cb, vb) in halves.items():
            common &= va & vb
        cidx = np.where(common)[0]
        print(f"\n모든 세션·반쪽에서 유효한 공통 피처: {len(cidx)}개")

        def sub_corr(c: np.ndarray, valid: np.ndarray) -> np.ndarray:
            pos = np.cumsum(valid) - 1
            take = pos[cidx]
            return c[np.ix_(take, take)]

        sess_list = [s for s in SESSIONS if s in halves]
        # A/B 반쪽 각각의 군집 라벨. within 과 cross 모두 이 동일 크기 반쪽만 쓴다.
        labA = {s: cluster_labels(sub_corr(halves[s][0], halves[s][1]), K_CLUSTERS) for s in sess_list}
        labB = {s: cluster_labels(sub_corr(halves[s][2], halves[s][3]), K_CLUSTERS) for s in sess_list}

        print("\n=== within-session ARI (같은 세션 A vs B; 표본노이즈 baseline) ===")
        within = {}
        for s in sess_list:
            within[s] = adjusted_rand_score(labA[s], labB[s])
            print(f"  {s:10s} ARI={within[s]:.3f}")
        print(f"  -> baseline 평균 {np.mean(list(within.values())):.3f}")

        print("\n=== cross-session ARI (세션1 A vs 세션2 A, 동일 표본 크기) ===")
        cross = []
        for i, s1 in enumerate(sess_list):
            for s2 in sess_list[i + 1:]:
                ari = adjusted_rand_score(labA[s1], labA[s2])
                cross.append(ari)
                print(f"  {s1:10s} vs {s2:10s} ARI={ari:.3f}")
        print(f"  -> cross 평균 {np.mean(cross):.3f}")
        print(f"  -> 판정: cross({np.mean(cross):.3f}) vs within({np.mean(list(within.values())):.3f}) "
              f"= {'세션별 구조 차이 있음' if np.mean(cross) < np.mean(list(within.values())) - 0.05 else '표본노이즈와 구분 불가'}")

        corrs = {s: halves[s][0] for s in sess_list}
        valids = {s: halves[s][1] for s in sess_list}
        lab_c = labA

        # ---------------- 피처별 상관프로파일 안정성
        print("\n=== 세션 의존적 피처 (상관 프로파일이 세션 간에 가장 많이 바뀌는 것) ===")
        names = [feature_cols[i] for i in cidx]
        prof = {s: sub_corr(corrs[s], valids[s]) for s in sess_list}

        def profile_agreement(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
            """피처별로 '다른 피처들과의 상관 벡터' 두 개의 Pearson 상관."""
            out = np.zeros(m1.shape[0])
            for j in range(m1.shape[0]):
                v1 = np.delete(m1[j], j)
                v2 = np.delete(m2[j], j)
                out[j] = np.corrcoef(v1, v2)[0, 1]
            return out

        # 귀무: 각 세션의 A/B 반쪽 간 profile agreement
        null_rows = []
        for s, (ca, va, cb, vb) in halves.items():
            null_rows.append(profile_agreement(sub_corr(ca, va), sub_corr(cb, vb)))
        null_agree = np.mean(null_rows, axis=0)

        cross_rows = []
        for i, s1 in enumerate(sess_list):
            for s2 in sess_list[i + 1:]:
                cross_rows.append(profile_agreement(prof[s1], prof[s2]))
        cross_agree = np.mean(cross_rows, axis=0)

        # 세션이 시각으로 정의되므로 시각 인코딩 피처는 정의상 세션마다 프로파일이 달라진다.
        # 발견이 아니라 동어반복이므로 별도 표시한다.
        MECHANICAL = {"hour_sin", "hour_cos", "minute_sin", "minute_cos", "is_hour_open"}
        res = pd.DataFrame({
            "feature": names,
            "cross_session_agree": cross_agree,
            "within_session_null": null_agree,
            "gap": null_agree - cross_agree,
        })
        res["mechanical"] = res.feature.isin(MECHANICAL)
        res = res.sort_values("gap", ascending=False)
        print(res.head(20).round(3).to_string(index=False))
        print(f"\n  전체 평균: cross={cross_agree.mean():.3f}  null={null_agree.mean():.3f}  "
              f"gap={(null_agree - cross_agree).mean():.3f}")
        res.to_csv(OUTDIR / f"feature_profile_stability_{tag.replace('+','_')}.csv", index=False)

        # ---------------- 군집 멤버십이 세션마다 바뀌는 피처
        lab_df = pd.DataFrame({s: lab_c[s] for s in sess_list}, index=names)
        print("\n=== 세션별 군집 멤버십 (가장 큰 군집 3개에 속하지 않는 피처만) ===")
        big = set(pd.Series(lab_c[sess_list[0]]).value_counts().head(3).index)
        odd = lab_df[~lab_df[sess_list[0]].isin(big)]
        print(odd.to_string())
        lab_df.to_csv(OUTDIR / f"feature_cluster_labels_{tag.replace('+','_')}.csv")


if __name__ == "__main__":
    main()
