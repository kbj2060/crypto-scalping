"""
Clusters Volume Profile (K-Means) 피처 생성기
================================================================================
TradingView LuxAlgo의 Clusters Volume Profile 지표를 Python으로 구현.
K-Means 클러스터링으로 가격대별 거래량 분포를 분석하여 파생 피처 생성.

생성 피처 (5개):
  1. cvp_poc_dist         — POC(Point of Control) 대비 현재가 거리 (정규화)
  2. cvp_vah_val_width    — Value Area 폭 (변동성/유동성 프록시)
  3. cvp_cluster_position — 현재가가 속한 클러스터 내 위치 (0=하단, 1=상단)
  4. cvp_volume_imbalance — 현재가 위/아래 거래량 불균형 (-1 ~ +1)
  5. cvp_regime           — 클러스터 분포 기반 시장 레짐 (추세/횡보 강도)

사용:
  from cvp import add_cvp_features
  df = add_cvp_features(df, lookback=200, n_clusters=4, n_bins=50)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


# ════════════════════════════════════════════════════════════════
# 1. K-Means 클러스터링 (순수 NumPy, sklearn 불필요)
# ════════════════════════════════════════════════════════════════
def _kmeans_1d(prices: np.ndarray, volumes: np.ndarray,
               n_clusters: int = 4, n_iter: int = 10) -> np.ndarray:
    """
    1D K-Means: 가격을 클러스터링하되, centroid 업데이트 시 거래량 가중 평균 사용.

    Args:
        prices: HLC3 (= (high + low + close) / 3) 배열
        volumes: 거래량 배열
        n_clusters: 클러스터 수
        n_iter: 반복 횟수

    Returns:
        labels: 각 봉의 클러스터 라벨 [0, 1, ..., n_clusters-1]
    """
    n = len(prices)
    if n < n_clusters:
        return np.zeros(n, dtype=int)

    # 초기 centroid: 가격 범위를 균등 분할
    p_min, p_max = prices.min(), prices.max()
    if p_max - p_min < 1e-10:
        return np.zeros(n, dtype=int)

    centroids = np.linspace(p_min, p_max, n_clusters)

    for _ in range(n_iter):
        # Assign: 각 봉을 가장 가까운 centroid에 할당
        dists = np.abs(prices[:, None] - centroids[None, :])  # [N, K]
        labels = dists.argmin(axis=1)

        # Update: 거래량 가중 평균으로 centroid 재계산
        new_centroids = np.copy(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                vol_k = volumes[mask]
                total_vol = vol_k.sum()
                if total_vol > 0:
                    new_centroids[k] = np.average(prices[mask], weights=vol_k)
                else:
                    new_centroids[k] = prices[mask].mean()

        centroids = new_centroids

    # 최종 할당
    dists = np.abs(prices[:, None] - centroids[None, :])
    labels = dists.argmin(axis=1)

    return labels


# ════════════════════════════════════════════════════════════════
# 2. Volume Profile 계산
# ════════════════════════════════════════════════════════════════
def _compute_volume_profile(prices: np.ndarray, volumes: np.ndarray,
                            n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    거래량 프로필 계산.

    Returns:
        bin_centers: 각 빈의 가격 중심
        bin_volumes: 각 빈의 누적 거래량
        poc: Point of Control (최대 거래량 가격)
        vah: Value Area High (거래량 70% 구간 상한)
        val: Value Area Low (거래량 70% 구간 하한)
    """
    if len(prices) == 0 or prices.max() - prices.min() < 1e-10:
        mid = prices.mean() if len(prices) > 0 else 0
        return np.array([mid]), np.array([0.0]), mid, mid, mid

    bins = np.linspace(prices.min(), prices.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_volumes = np.zeros(n_bins)

    # 각 봉을 해당 빈에 배분
    indices = np.clip(
        np.digitize(prices, bins) - 1, 0, n_bins - 1
    )
    for i, idx in enumerate(indices):
        bin_volumes[idx] += volumes[i]

    # POC: 최대 거래량 빈
    poc_idx = bin_volumes.argmax()
    poc = bin_centers[poc_idx]

    # Value Area: POC에서 양쪽으로 확장하며 거래량 70% 달성
    total_vol = bin_volumes.sum()
    if total_vol == 0:
        return bin_centers, bin_volumes, poc, poc, poc

    va_vol = 0
    va_low_idx, va_high_idx = poc_idx, poc_idx
    va_vol += bin_volumes[poc_idx]

    while va_vol / total_vol < 0.70:
        vol_below = bin_volumes[va_low_idx - 1] if va_low_idx > 0 else 0
        vol_above = bin_volumes[va_high_idx + 1] if va_high_idx < n_bins - 1 else 0

        if vol_below == 0 and vol_above == 0:
            break

        if vol_above >= vol_below:
            va_high_idx = min(va_high_idx + 1, n_bins - 1)
            va_vol += vol_above
        else:
            va_low_idx = max(va_low_idx - 1, 0)
            va_vol += vol_below

    vah = bin_centers[va_high_idx]
    val_price = bin_centers[va_low_idx]

    return bin_centers, bin_volumes, poc, vah, val_price


# ════════════════════════════════════════════════════════════════
# 3. 피처 생성
# ════════════════════════════════════════════════════════════════
def _compute_cvp_features_at(
    hlc3: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, volume: np.ndarray,
    current_price: float,
    n_clusters: int = 4, n_bins: int = 50
) -> dict:
    """단일 윈도우에서 CVP 피처 5개 계산"""

    if len(hlc3) < n_clusters * 2:
        return {
            'cvp_poc_dist': 0.0,
            'cvp_vah_val_width': 0.0,
            'cvp_cluster_position': 0.5,
            'cvp_volume_imbalance': 0.0,
            'cvp_regime': 0.0,
        }

    # ── K-Means 클러스터링 ──
    labels = _kmeans_1d(hlc3, volume, n_clusters=n_clusters)

    # ── 전체 Volume Profile ──
    _, bin_volumes, poc, vah, val = _compute_volume_profile(hlc3, volume, n_bins)

    price_range = hlc3.max() - hlc3.min()
    if price_range < 1e-10:
        price_range = 1e-10

    # 1. POC 거리 (정규화: 가격 범위 대비)
    cvp_poc_dist = (current_price - poc) / price_range

    # 2. Value Area 폭 (정규화)
    cvp_vah_val_width = (vah - val) / price_range

    # 3. 현재가의 클러스터 내 위치
    # 현재가가 속한 클러스터를 찾고, 그 클러스터 내에서의 상대 위치
    current_cluster = labels[-1]  # 마지막 봉의 클러스터
    cluster_mask = labels == current_cluster
    cluster_prices = hlc3[cluster_mask]
    c_min, c_max = cluster_prices.min(), cluster_prices.max()
    if c_max - c_min > 1e-10:
        cvp_cluster_position = (current_price - c_min) / (c_max - c_min)
    else:
        cvp_cluster_position = 0.5
    cvp_cluster_position = np.clip(cvp_cluster_position, 0, 1)

    # 4. 거래량 불균형: 현재가 위 vs 아래 거래량
    above_mask = hlc3 > current_price
    below_mask = hlc3 <= current_price
    vol_above = volume[above_mask].sum()
    vol_below = volume[below_mask].sum()
    total = vol_above + vol_below
    if total > 0:
        cvp_volume_imbalance = (vol_below - vol_above) / total  # 양수=아래 거래량 많음(지지)
    else:
        cvp_volume_imbalance = 0.0

    # 5. 시장 레짐: 클러스터 간 겹침 정도
    #    겹침 많음 = 횡보(accumulation), 겹침 없음 = 추세
    cluster_ranges = []
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_ranges.append((hlc3[mask].min(), hlc3[mask].max()))

    if len(cluster_ranges) >= 2:
        # 클러스터 간 gap 비율 계산
        sorted_ranges = sorted(cluster_ranges, key=lambda x: x[0])
        total_gap = 0
        for i in range(len(sorted_ranges) - 1):
            gap = sorted_ranges[i + 1][0] - sorted_ranges[i][1]
            total_gap += max(0, gap)

        # gap이 크면 추세(+1), gap이 없으면 횡보(0), 겹침이 크면 (-1)
        total_overlap = 0
        for i in range(len(sorted_ranges) - 1):
            overlap = sorted_ranges[i][1] - sorted_ranges[i + 1][0]
            total_overlap += max(0, overlap)

        net_separation = (total_gap - total_overlap) / price_range
        cvp_regime = np.clip(net_separation * 2, -1, 1)  # [-1, 1]
    else:
        cvp_regime = 0.0

    return {
        'cvp_poc_dist': float(cvp_poc_dist),
        'cvp_vah_val_width': float(cvp_vah_val_width),
        'cvp_cluster_position': float(cvp_cluster_position),
        'cvp_volume_imbalance': float(cvp_volume_imbalance),
        'cvp_regime': float(cvp_regime),
    }


# ════════════════════════════════════════════════════════════════
# 4. 메인 함수: DataFrame에 피처 추가
# ════════════════════════════════════════════════════════════════
CVP_FEATURE_COLS = [
    'cvp_poc_dist',
    'cvp_vah_val_width',
    'cvp_cluster_position',
    'cvp_volume_imbalance',
    'cvp_regime',
]


def add_cvp_features(
    df: pd.DataFrame,
    lookback: int = 200,
    n_clusters: int = 4,
    n_bins: int = 50,
) -> pd.DataFrame:
    """
    DataFrame에 Clusters Volume Profile 피처 5개를 추가합니다.

    Args:
        df: OHLCV 데이터 (columns: open, high, low, close, volume 필수)
        lookback: K-Means에 사용할 과거 봉 수 (기본 200 = ~16시간)
        n_clusters: K-Means 클러스터 수 (기본 4)
        n_bins: Volume Profile 빈 수 (기본 50)

    Returns:
        CVP 피처가 추가된 DataFrame
    """
    required = ['high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    hlc3 = (high + low + close) / 3.0

    # 결과 저장
    results = {col: np.zeros(n) for col in CVP_FEATURE_COLS}

    for i in range(n):
        start = max(0, i - lookback + 1)
        window_hlc3 = hlc3[start:i + 1]
        window_high = high[start:i + 1]
        window_low = low[start:i + 1]
        window_close = close[start:i + 1]
        window_vol = volume[start:i + 1]
        current_price = close[i]

        feats = _compute_cvp_features_at(
            window_hlc3, window_high, window_low,
            window_close, window_vol, current_price,
            n_clusters=n_clusters, n_bins=n_bins,
        )

        for col in CVP_FEATURE_COLS:
            results[col][i] = feats[col]

    # DataFrame에 추가
    for col in CVP_FEATURE_COLS:
        df[col] = results[col]

    return df


# ════════════════════════════════════════════════════════════════
# 5. CLI / 테스트
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("📊 Clusters Volume Profile 피처 생성기 테스트")
    print("=" * 70)

    # 가상 OHLCV 데이터 생성
    np.random.seed(42)
    n = 500
    close = 3000 + np.cumsum(np.random.randn(n) * 5)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=n, freq='5min'),
        'open': close + np.random.randn(n) * 2,
        'high': close + np.abs(np.random.randn(n) * 5),
        'low': close - np.abs(np.random.randn(n) * 5),
        'close': close,
        'volume': np.random.exponential(1000, n),
    })

    print(f"\n변환 전: {df.shape[1]}개 컬럼")

    df = add_cvp_features(df, lookback=200, n_clusters=4)

    print(f"\n변환 후: {df.shape[1]}개 컬럼")
    print(f"  CVP 피처: {[c for c in CVP_FEATURE_COLS if c in df.columns]}")
    print(f"\n마지막 5행 CVP 피처:")
    print(df[CVP_FEATURE_COLS].tail().to_string())

    print(f"\n📈 CVP 피처 통계:")
    print(df[CVP_FEATURE_COLS].describe().round(4).to_string())