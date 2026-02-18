"""
자동 피처 선택 모듈 — TFT 학습 전 사전 필터링

3단계 파이프라인:
    Stage 1: 통계 기반 사전 필터링 (학습 전 1회)
        - 분산 필터: 정규화 후 거의 상수인 피처 제거
        - Lagged MI: 시간차를 고려한 Mutual Information (선행 지표 포착)
        - Correlation Dedup: 서로 중복인 피처 제거 (r > 0.85)
    Stage 2: 도메인 지식 기반 필수 포함 (must_include)
    Stage 3: TFT의 VSN이 나머지에서 동적 선택 (학습 중 자동)

핵심 설계:
    - MI는 동시적 상관만 측정 → 온체인/고래 지표 같은 선행 신호를 과소평가
    - Lagged MI: 피처를 1~6봉 시프트하여 MI 계산 → max(MI_lag0, MI_lag1, ..., MI_lag6)
    - must_include: 도메인 지식으로 반드시 포함할 피처 지정 (MI 순위와 무관)

사용법:
    from core.feature_selector import auto_select_features

    selected = auto_select_features(
        train_df, feature_cols,
        target_col='target_cumret_6',
        max_features=15,
        must_include=['whale_conviction', 'funding_pressure', 'net_taker_ratio'],
    )
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# 피처에서 항상 제외할 컬럼 (타겟과 직접 관련)
ALWAYS_EXCLUDE = {'log_return'}


class FeatureSelector:
    """
    TFT 학습 전 자동 피처 선택.

    4단계 필터:
        1. 분산 필터: 정규화 후 변동이 거의 없는 피처 제거
        2. Lagged MI 스코어링: 시간차 고려한 비선형 상관 측정
        3. 상관관계 중복 제거: 피처 간 r > threshold면 MI 낮은 쪽 제거
        4. must_include 보장: 도메인 지식 기반 필수 피처 포함
    """

    def __init__(self, target_col: str = 'target_cumret_6',
                 static_cols: List[str] = None):
        self.target_col = target_col
        self.static_cols = static_cols or ['session_asia', 'session_europe', 'session_us']
        self.mi_scores_ = None
        self.selected_features_ = None
        self.report_ = {}

    def fit_select(self, df: pd.DataFrame, feature_cols: List[str],
                   max_features: int = 15,
                   mi_threshold: float = 0.0,
                   corr_threshold: float = 0.85,
                   variance_threshold: float = 0.01,
                   must_include: List[str] = None,
                   mi_lags: List[int] = None) -> List[str]:
        """
        피처 선택 수행.

        Args:
            df: 학습 데이터
            feature_cols: 후보 피처 목록
            max_features: 최종 선택할 최대 피처 수 (static, must_include 제외)
            mi_threshold: MI 점수 최소 기준 (0이면 자동)
            corr_threshold: 피처 간 상관관계 제거 기준
            variance_threshold: 정규화 후 분산 최소 기준
            must_include: 반드시 포함할 피처 리스트 (MI 순위 무관)
            mi_lags: MI 계산 시 시프트할 lag 리스트 (기본: [0,1,2,3,6])

        Returns:
            선택된 피처 리스트 (static + must_include 포함)
        """
        must_include = must_include or []
        mi_lags = mi_lags or [0, 1, 2, 3, 6]

        # static, 타겟, 항상 제외 컬럼 필터링
        exclude = set(self.static_cols) | {self.target_col} | ALWAYS_EXCLUDE
        temporal_cols = [c for c in feature_cols if c not in exclude]

        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 자동 피처 선택 시작: {len(temporal_cols)}개 후보")
        if ALWAYS_EXCLUDE & set(feature_cols):
            logger.info(f"   자동 제외: {ALWAYS_EXCLUDE & set(feature_cols)}")
        if must_include:
            logger.info(f"   필수 포함: {must_include}")
        logger.info(f"{'='*60}")

        # 타겟 존재 확인
        if self.target_col not in df.columns:
            logger.warning(f"타겟 '{self.target_col}' 없음 — 필터링 스킵")
            return [c for c in feature_cols if c not in ALWAYS_EXCLUDE]

        # NaN 제거 (MI 계산용)
        valid_df = df[temporal_cols + [self.target_col]].dropna()
        if len(valid_df) < 1000:
            logger.warning(f"유효 데이터 {len(valid_df)}행 — 필터링 스킵")
            return [c for c in feature_cols if c not in ALWAYS_EXCLUDE]

        # ── Stage 1: 분산 필터 ──
        survivors = self._variance_filter(valid_df, temporal_cols, variance_threshold)

        # ── Stage 2: Lagged MI 스코어링 ──
        survivors, mi_scores = self._lagged_mi_scoring(valid_df, survivors, mi_threshold, mi_lags)

        # ── Stage 3: 상관관계 중복 제거 ──
        survivors = self._correlation_dedup(valid_df, survivors, mi_scores, corr_threshold)

        # ── Stage 4: must_include 보장 + Top N 선택 ──
        # must_include에서 유효한 것만
        valid_must = [c for c in must_include
                      if c in temporal_cols and c not in ALWAYS_EXCLUDE]

        # must_include를 survivors에서 빼고, 나머지에서 top N 채움
        remaining = [c for c in survivors if c not in valid_must]
        n_auto = max_features - len(valid_must)

        if len(remaining) > n_auto:
            sorted_by_mi = sorted(remaining, key=lambda c: mi_scores.get(c, 0), reverse=True)
            remaining = sorted_by_mi[:n_auto]

        # 합치기: must_include 먼저 + MI top
        final_temporal = list(dict.fromkeys(valid_must + remaining))

        # static 추가
        available_static = [c for c in self.static_cols if c in feature_cols]
        selected = final_temporal + available_static

        self.selected_features_ = selected
        self.report_['final_count'] = len(selected)
        self.report_['must_included'] = valid_must
        self.report_['auto_selected'] = remaining

        logger.info(f"\n✅ 최종 선택: {len(final_temporal)}개 temporal + {len(available_static)}개 static = {len(selected)}개")
        logger.info(f"   필수 포함 ({len(valid_must)}개): {valid_must}")
        logger.info(f"   자동 선택 ({len(remaining)}개): {remaining}")

        return selected

    def _variance_filter(self, df: pd.DataFrame, cols: List[str],
                        threshold: float) -> List[str]:
        """
        정규화 후 분산 체크 — 스케일이 작은 피처가 잘못 제거되는 문제 방지.
        
        원본 스케일로 체크하면 garman_klass_vol(var=0.000002)처럼
        값이 작지만 정보량이 많은 피처가 제거됨.
        Z-score 정규화 후에는 모든 피처가 동일 스케일이므로
        진짜 상수 컬럼만 걸러냄.
        """
        # Z-score 정규화 후 분산 체크
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(df[cols]),
            columns=cols, index=df.index
        )
        variances_norm = normalized.var()
        
        # 원본 분산도 참고용으로 기록
        variances_raw = df[cols].var()
        
        low_var = variances_norm[variances_norm < threshold].index.tolist()
        survivors = [c for c in cols if c not in low_var]

        if low_var:
            logger.info(f"\n  [분산 필터] {len(low_var)}개 제거 (정규화 후 var < {threshold})")
            for c in low_var:
                logger.info(f"    ✗ {c} (norm_var={variances_norm[c]:.6f}, raw_var={variances_raw[c]:.6f})")
        else:
            logger.info(f"\n  [분산 필터] 제거 없음 (정규화 후 기준)")

        self.report_['variance_removed'] = low_var
        return survivors

    def _lagged_mi_scoring(self, df: pd.DataFrame, cols: List[str],
                          threshold: float, lags: List[int]) -> Tuple[List[str], dict]:
        """
        Lagged Mutual Information 스코어링.
        
        일반 MI는 "현재 피처값 ↔ 현재 타겟"만 봐서
        고래/온체인 같은 선행 지표를 과소평가함.
        
        Lagged MI: 피처를 1~6봉 shift하여 "과거 피처값 ↔ 현재 타겟" MI를 계산.
        각 lag 중 최대 MI를 해당 피처의 점수로 사용.
        
        예: whale_conviction의 lag=0 MI가 0.001이지만
            lag=3 MI가 0.008이면 → 고래가 3봉(15분) 선행한다는 의미
            → 최종 MI = 0.008, best_lag = 3
        """
        logger.info(f"\n  [Lagged MI 스코어링] {len(cols)}개 피처, lags={lags}")

        y = df[self.target_col].values
        scaler = StandardScaler()

        # 샘플링
        sample_size = min(len(df), 50000)
        if sample_size < len(df):
            idx = np.sort(np.random.choice(len(df), sample_size, replace=False))
        else:
            idx = np.arange(len(df))

        y_sample = y[idx]

        best_mi = {}
        best_lag = {}

        for lag in lags:
            # lag만큼 피처를 shift (과거 피처 → 현재 타겟)
            if lag == 0:
                X = df[cols].values
            else:
                X = df[cols].shift(lag).values

            # shift로 생긴 NaN 행 제거
            valid_mask = ~np.isnan(X).any(axis=1)
            valid_idx = idx[valid_mask[idx]]

            if len(valid_idx) < 1000:
                continue

            X_valid = X[valid_idx]
            y_valid = y[valid_idx]

            X_scaled = scaler.fit_transform(X_valid)

            mi = mutual_info_regression(X_scaled, y_valid, n_neighbors=10, random_state=42)

            for j, col in enumerate(cols):
                if col not in best_mi or mi[j] > best_mi[col]:
                    best_mi[col] = mi[j]
                    best_lag[col] = lag

        mi_scores = best_mi
        self.mi_scores_ = mi_scores

        # 정렬 출력
        sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"\n  Lagged MI 순위 (max across lags):")
        for i, (col, score) in enumerate(sorted_mi):
            lag = best_lag.get(col, 0)
            lag_str = f"lag={lag}" if lag > 0 else "lag=0"
            bar = '█' * int(score * 500)
            marker = '✓' if score > threshold else '✗'
            logger.info(f"    {marker} {i+1:2d}. {col:30s} MI={score:.4f} ({lag_str}) {bar}")

        # 임계값 적용
        if threshold > 0:
            survivors = [c for c in cols if mi_scores.get(c, 0) > threshold]
            removed = [c for c in cols if mi_scores.get(c, 0) <= threshold]
            if removed:
                logger.info(f"\n  MI 필터: {len(removed)}개 제거 (MI ≤ {threshold})")
        else:
            survivors = cols
            logger.info(f"\n  MI 필터: 임계값 0 — 제거 없음 (상관관계 중복제거에서 처리)")

        self.report_['mi_scores'] = sorted_mi
        self.report_['best_lags'] = best_lag
        return survivors, mi_scores

    def _correlation_dedup(self, df: pd.DataFrame, cols: List[str],
                          mi_scores: dict, threshold: float) -> List[str]:
        """상관관계가 높은 피처 쌍에서 MI 낮은 쪽 제거."""
        if len(cols) <= 1:
            return cols

        corr_matrix = df[cols].corr().abs()
        to_remove = set()

        logger.info(f"\n  [상관관계 중복 제거] threshold={threshold}")

        for i in range(len(cols)):
            if cols[i] in to_remove:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove:
                    continue
                if corr_matrix.iloc[i, j] > threshold:
                    # MI 낮은 쪽 제거
                    mi_i = mi_scores.get(cols[i], 0)
                    mi_j = mi_scores.get(cols[j], 0)
                    if mi_i >= mi_j:
                        victim = cols[j]
                        keeper = cols[i]
                    else:
                        victim = cols[i]
                        keeper = cols[j]
                    to_remove.add(victim)
                    logger.info(
                        f"    ✗ {victim} (r={corr_matrix.iloc[i,j]:.3f} with {keeper}, "
                        f"MI {mi_scores.get(victim,0):.4f} < {mi_scores.get(keeper,0):.4f})")

        survivors = [c for c in cols if c not in to_remove]
        logger.info(f"\n  상관관계 필터: {len(to_remove)}개 제거, {len(survivors)}개 생존")

        self.report_['corr_removed'] = list(to_remove)
        return survivors

    def get_report(self) -> dict:
        """선택 과정 리포트 반환."""
        return self.report_

    def get_mi_scores(self) -> dict:
        """MI 스코어 반환."""
        return self.mi_scores_ or {}


# ════════════════════════════════════════════════════════════════
# 편의 함수
# ════════════════════════════════════════════════════════════════

def auto_select_features(train_df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'target_cumret_6',
                         max_features: int = 15,
                         corr_threshold: float = 0.85,
                         variance_threshold: float = 0.01,
                         must_include: List[str] = None) -> List[str]:
    """
    원라인 피처 선택.

    Args:
        must_include: 도메인 지식 기반 필수 포함 피처.
            추천 (온체인/고래/오더북 선행 지표):
            - 'whale_conviction': 고래 포지션 방향
            - 'funding_pressure': 펀딩비 압력 (롱/숏 쏠림)
            - 'net_taker_ratio': 테이커 매수/매도 비율
            - 'oi_change_rate': OI 변화율 (포지션 빌드업)
            - 'smart_money_flow': 스마트머니 흐름

    사용법:
        selected = auto_select_features(
            train_df, ULTIMATE_FEATURE_COLS,
            target_col='target_cumret_6',
            max_features=15,
            must_include=['whale_conviction', 'funding_pressure',
                          'net_taker_ratio', 'oi_change_rate'],
        )
    """
    selector = FeatureSelector(target_col=target_col)
    return selector.fit_select(
        train_df, feature_cols,
        max_features=max_features,
        corr_threshold=corr_threshold,
        variance_threshold=variance_threshold,
        must_include=must_include,
    )