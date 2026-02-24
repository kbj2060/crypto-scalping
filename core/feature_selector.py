"""
자동 피처 선택 모듈 — TFT 학습 전 사전 필터링

[IDEA 3] Granger Causality 결합: MI × Granger 점수 사용
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

from statsmodels.tsa.stattools import grangercausalitytests
_HAS_STATSMODELS = False

class FeatureSelector:
    """
    TFT 학습 전 자동 피처 선택.

    4단계 필터 + Granger 결합:
        1. 분산 필터
        2. Lagged MI + Granger 결합 스코어링
        3. 상관관계 중복 제거
        4. must_include 보장
    """

    def __init__(self, target_col: str = 'target_ret_3',
                 static_cols: List[str] = None):
        self.target_col = target_col
        self.static_cols = static_cols or ['session_asia', 'session_europe', 'session_us']
        self.mi_scores_ = None
        self.granger_scores_ = None
        self.combined_scores_ = None
        self.selected_features_ = None
        self.report_ = {}
  
    def fit_select(self, df: pd.DataFrame, feature_cols: List[str],
                   max_features: int = 30,
                   mi_threshold: float = 0.0,
                   corr_threshold: float = 0.85,
                   variance_threshold: float = 0.01,
                   must_include: List[str] = None,
                   mi_lags: List[int] = None,
                   use_granger: bool = True) -> List[str]:
        """
        피처 선택 수행 (Granger 옵션 추가).
        """
        must_include = must_include or []
        mi_lags = mi_lags or [0, 1, 2, 3, 6]

        exclude = set(self.static_cols) | {self.target_col}
        temporal_cols = [c for c in feature_cols if c not in exclude]

        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 자동 피처 선택 시작: {len(temporal_cols)}개 후보")
        if must_include:
            logger.info(f"   필수 포함: {must_include}")
        logger.info(f"{'='*60}")

        if self.target_col not in df.columns:
            logger.warning(f"타겟 '{self.target_col}' 없음 — 필터링 스킵")
            return [c for c in feature_cols]

        valid_df = df[temporal_cols + [self.target_col]].dropna()
        if len(valid_df) < 1000:
            logger.warning(f"유효 데이터 {len(valid_df)}행 — 필터링 스킵")
            return [c for c in feature_cols]

        # ── Stage 1: 분산 필터 ──
        survivors = self._variance_filter(valid_df, temporal_cols, variance_threshold)

        # ── Stage 2: Lagged MI 스코어링 ──
        survivors, mi_scores = self._lagged_mi_scoring(valid_df, survivors, mi_threshold, mi_lags)

        # [IDEA 3] Granger 스코어 계산 및 결합
        if use_granger and _HAS_STATSMODELS:
            granger_scores = self._granger_scoring(valid_df, survivors, self.target_col, max_lag=6)
            combined_scores = {c: mi_scores.get(c, 0) * granger_scores.get(c, 0) for c in survivors}
            self.granger_scores_ = granger_scores
            self.combined_scores_ = combined_scores
            logger.info("\n  [Granger 결합] MI × Granger 점수 사용")
        else:
            combined_scores = mi_scores
            if use_granger and not _HAS_STATSMODELS:
                logger.warning("Granger 요청되었으나 statsmodels 없음 — MI만 사용")

        # ── Stage 3: 상관관계 중복 제거 (combined_scores 사용) ──
        survivors = self._correlation_dedup(valid_df, survivors, combined_scores, corr_threshold)

        # ── Stage 4: must_include 보장 + Top N 선택 ──
        valid_must = [c for c in must_include
                      if c in temporal_cols]
        remaining = [c for c in survivors if c not in valid_must]
        n_auto = max_features - len(valid_must)

        if len(remaining) > n_auto:
            sorted_by_score = sorted(remaining, key=lambda c: combined_scores.get(c, 0), reverse=True)
            remaining = sorted_by_score[:n_auto]

        final_temporal = list(dict.fromkeys(valid_must + remaining))
        available_static = [c for c in self.static_cols if c in feature_cols]
        selected = final_temporal + available_static

        self.selected_features_ = selected
        self.report_['final_count'] = len(selected)
        self.report_['must_included'] = valid_must
        self.report_['auto_selected'] = remaining

        logger.info(f"\n✅ 최종 선택: {len(final_temporal)}개 temporal + {len(available_static)}개 static = {len(selected)}개")
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

    # [IDEA 3] Granger causality scoring
    def _granger_scoring(self, df: pd.DataFrame, cols: List[str],
                        target_col: str, max_lag: int = 6) -> dict:
        """
        안정화된 Granger Causality 스코어링
        - 키 접근 오류 방지 (다중 테스트 대체)
        - 샘플 수/다중공선성 예외 처리
        - 로깅으로 디버깅 지원
        """
        scores = {}
        failed_features = []
        
        for col in cols:
            try:
                # 1. 데이터 준비 (최소 200 샘플 요구)
                test_data = df[[target_col, col]].dropna()
                if len(test_data) < 200:
                    scores[col] = 0.0
                    continue
                
                # 2. Granger 테스트 실행 (상세 로그 비활성화)
                result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                
                # 3. p-value 추출 (다중 테스트 대체 전략)
                p_vals = []
                for lag in range(1, max_lag + 1):
                    if lag not in result:
                        continue
                    
                    # statsmodels 버전에 따른 키 접근
                    test_dict = result[lag][0]  # 첫 번째 요소는 테스트 딕셔너리
                    
                    # 우선순위: ssr_ftest > ssr_chi2test > lrtest
                    if 'ssr_ftest' in test_dict:
                        p_val = test_dict['ssr_ftest'][1]
                    elif 'ssr_chi2test' in test_dict:
                        p_val = test_dict['ssr_chi2test'][1]
                    elif 'lrtest' in test_dict:
                        p_val = test_dict['lrtest'][1]
                    elif 'params_ftest' in test_dict:
                        p_val = test_dict['params_ftest'][1]
                    else:
                        continue  # 사용 가능한 테스트 없음
                    
                    p_vals.append(p_val)
                
                # 4. 스코어 계산 (낮은 p-value = 높은 스코어)
                if p_vals:
                    avg_p = np.mean(p_vals)
                    # p-value 0.05 이하만 유의미한 인과관계로 간주
                    if avg_p < 0.05:
                        scores[col] = 1.0 - avg_p  # 0.95~1.0
                    else:
                        scores[col] = 0.0  # 유의미하지 않음
                else:
                    scores[col] = 0.0
                    
            except Exception as e:
                failed_features.append((col, str(e)[:50]))
                scores[col] = 0.0
        
        # 5. 실패 피처 로깅 (디버깅용)
        if failed_features:
            logger.warning(f"⚠️ Granger test failed for {len(failed_features)}/{len(cols)} features:")
            for col, err in failed_features[:5]:  # 상위 5개만 표시
                logger.warning(f"   - {col}: {err}")
            if len(failed_features) > 5:
                logger.warning(f"   ... and {len(failed_features)-5} more")
        
        return scores

    def _correlation_dedup(self, df: pd.DataFrame, cols: List[str],
                           mi_scores: dict, threshold: float) -> List[str]:
        """상관관계가 높은 피처 쌍에서 MI(또는 Combined) 낮은 쪽 제거."""
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
                    # Score(MI or Combined) 낮은 쪽 제거
                    score_i = mi_scores.get(cols[i], 0)
                    score_j = mi_scores.get(cols[j], 0)
                    if score_i >= score_j:
                        victim = cols[j]
                        keeper = cols[i]
                    else:
                        victim = cols[i]
                        keeper = cols[j]
                    to_remove.add(victim)
                    logger.info(
                        f"    ✗ {victim} (r={corr_matrix.iloc[i,j]:.3f} with {keeper}, "
                        f"Score {mi_scores.get(victim,0):.4f} < {mi_scores.get(keeper,0):.4f})")

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
                         target_col: str = 'target_ret_3',
                         max_features: int = 30,
                         corr_threshold: float = 0.85,
                         variance_threshold: float = 0.01,
                         must_include: List[str] = None,
                         use_granger: bool = True) -> List[str]:
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
        use_granger: Granger Causality 점수 결합 여부 (statsmodels 필요)

    사용법:
        selected = auto_select_features(
            train_df, ULTIMATE_FEATURE_COLS,
            target_col='target_ret_3',
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
        use_granger=use_granger,
    )