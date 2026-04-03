"""
환 피쳐 선택 모듈 — TFT/MacroHFT 학습 전 사전 필터링
================================================================================
빠르고 안정적인 5단계 비선형 필터링:
    1. 제외 피쳐 사전 제거  (메타/절대값 컴럼 사전 딥)
    2. 분산 필터         (상수 피쳐 제거)
    3. Lagged Mutual Information (시차를 둔 비선형 정보량 스코어링)
    4. 상관관계 중복 제거  (다중공선성 방지)
    5. 도메인 필수 피쳐(must_include) 강제 보장
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from features.engineering import EXCLUDE_FEATURE_COLS, MUST_INCLUDE_FEATURES

logger = logging.getLogger(__name__)


def _ensure_required_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    must_include: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    feature_cols = list(feature_cols)

    def _register(col: str, values: pd.Series) -> None:
        if col not in df.columns:
            df[col] = values.astype(np.float32).fillna(0.0)
        if col not in feature_cols:
            feature_cols.append(col)

    def _ensure_mtf(col: str, ema_col: str, span: int) -> None:
        if col not in must_include:
            return
        # If the column already exists in df, still ensure it is selectable.
        if col in df.columns and df[col].notna().any():
            if col not in feature_cols:
                feature_cols.append(col)
                logger.info("auto-added existing feature to candidate list: %s", col)
            return
        if "close" not in df.columns:
            return
        if ema_col in df.columns and df[ema_col].notna().any():
            values = (df["close"] / df[ema_col].replace(0, np.nan)) - 1.0
        else:
            values = (df["close"] / df["close"].ewm(span=span, adjust=False).mean().replace(0, np.nan)) - 1.0
        _register(col, values)
        logger.info("auto-created missing feature: %s", col)

    _ensure_mtf("mtf_trend_1h", "ema_1h", 12)
    _ensure_mtf("mtf_trend_4h", "ema_4h", 48)

    return df, feature_cols

class FeatureSelector:
    def __init__(self, target_col: str = 'target_ret_1'):
        self.target_col = target_col
        self.mi_scores_ = None
        self.selected_features_ = None
        self.report_ = {}
  
    def fit_select(self, df: pd.DataFrame, feature_cols: List[str],
                   max_features: int = 30,
                   mi_threshold: float = 0.0,
                   corr_threshold: float = 0.85,
                   variance_threshold: float = 0.01,
                   must_include: List[str] = None,
                   mi_lags: List[int] = None) -> List[str]:

        must_include = must_include or []
        mi_lags = mi_lags or [0, 1, 2, 3, 6]
        df, feature_cols = _ensure_required_features(df, feature_cols, must_include)

        _exclude = {self.target_col} | set(EXCLUDE_FEATURE_COLS) | set(must_include)
        candidates = [c for c in feature_cols if c not in _exclude]

        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 자동 피처 선택: {len(candidates)}개 후보 (VIP 및 메타 {len(_exclude)}개 필터링 패스)")
        if must_include:
            logger.info(f"   필수 포함: {must_include}")
        logger.info(f"{'='*60}")

        if self.target_col not in df.columns:
            logger.warning(f"타겟 '{self.target_col}' 없음 — 필터링 스킵")
            return candidates

        valid_df = df[candidates + [self.target_col]].dropna()
        if len(valid_df) < 1000:
            logger.warning(f"유효 데이터 부족 ({len(valid_df)}행) — 필터링 스킵")
            return candidates

        survivors = self._variance_filter(valid_df, candidates, variance_threshold)
        survivors, mi_scores = self._lagged_mi_scoring(valid_df, survivors, mi_threshold, mi_lags)
        survivors = self._correlation_dedup(valid_df, survivors, mi_scores, corr_threshold)

        # [수정사항 3] must_include 리스트 무결성 검증 알럿
        missing_must = [c for c in must_include if c not in feature_cols]
        if missing_must:
            logger.warning(f"⚠️ must_include 중 데이터에 없는 피처 (무시됨): {missing_must}")

        valid_must = [c for c in must_include if c in feature_cols]
        remaining = [c for c in survivors if c not in valid_must]
        n_auto = max_features - len(valid_must)

        if len(remaining) > n_auto:
            sorted_by_score = sorted(remaining, key=lambda c: mi_scores.get(c, 0), reverse=True)
            remaining = sorted_by_score[:n_auto]

        selected = list(dict.fromkeys(valid_must + remaining))

        self.selected_features_ = selected
        self.report_['final_count'] = len(selected)
        self.report_['must_included'] = valid_must
        self.report_['auto_selected'] = remaining

        logger.info(f"\n {selected}")
        logger.info(f"\n✅ 최종 선택 완료: {len(selected)}개")
        return selected

    def _variance_filter(self, df: pd.DataFrame, cols: List[str], threshold: float) -> List[str]:
        scaler = StandardScaler()
        normalized = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols, index=df.index)
        variances_norm = normalized.var()
        
        low_var = variances_norm[variances_norm < threshold].index.tolist()
        survivors = [c for c in cols if c not in low_var]

        if low_var:
            logger.info(f"\n  [분산 필터] {len(low_var)}개 제거 (정규화 분산 < {threshold})")
        return survivors

    def _lagged_mi_scoring(self, df: pd.DataFrame, cols: List[str], threshold: float, lags: List[int]) -> Tuple[List[str], dict]:
        logger.info(f"\n  [Lagged MI 스코어링] {len(cols)}개 피처 검사 중...")
        y = df[self.target_col].values
        scaler = StandardScaler()

        # [수정사항 1, 2] 랜덤 시드 고정 및 불필요한 데드코드(y_sample) 제거/정리
        rng = np.random.RandomState(42)
        sample_size = min(len(df), 20000)
        idx = np.sort(rng.choice(len(df), sample_size, replace=False)) if sample_size < len(df) else np.arange(len(df))

        best_mi = {}
        for lag in lags:
            X = df[cols].values if lag == 0 else df[cols].shift(lag).values
            valid_mask = ~np.isnan(X).any(axis=1)
            valid_idx = idx[valid_mask[idx]]

            if len(valid_idx) < 1000: continue

            X_valid = scaler.fit_transform(X[valid_idx])
            mi = mutual_info_classif(X_valid, y[valid_idx], n_neighbors=5, random_state=42)

            for j, col in enumerate(cols):
                if col not in best_mi or mi[j] > best_mi[col]:
                    best_mi[col] = mi[j]

        self.mi_scores_ = best_mi
        survivors = [c for c in cols if best_mi.get(c, 0) > threshold] if threshold > 0 else cols
        return survivors, best_mi

    def _correlation_dedup(self, df: pd.DataFrame, cols: List[str], mi_scores: dict, threshold: float) -> List[str]:
        if len(cols) <= 1: return cols

        corr_matrix = df[cols].corr().abs()
        to_remove = set()

        for i in range(len(cols)):
            if cols[i] in to_remove: continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_remove: continue
                
                if corr_matrix.iloc[i, j] > threshold:
                    victim = cols[j] if mi_scores.get(cols[i], 0) >= mi_scores.get(cols[j], 0) else cols[i]
                    to_remove.add(victim)

        survivors = [c for c in cols if c not in to_remove]
        if to_remove:
            logger.info(f"\n  제거된 피처: {to_remove}")
            logger.info(f"\n  [다중공선성 필터] {len(to_remove)}개 제거 (상관계수 > {threshold})")
        return survivors


# ════════════════════════════════════════════════════════════════
# 편의 함수
# ════════════════════════════════════════════════════════════════
def auto_select_features(train_df: pd.DataFrame,
                         feature_cols: List[str],
                         target_col: str = 'target_ret_1',
                         max_features: int = 35,
                         corr_threshold: float = 0.85,
                         variance_threshold: float = 0.01,
                         must_include: List[str] = None) -> List[str]:
    """
    원라인 피처 선택 (Lagged MI 기반).

    - 제외 목록: feature_engineering.EXCLUDE_FEATURE_COLS 자동 적용
    - 핵심 피처: must_include 미지정 시 feature_engineering.MUST_INCLUDE_FEATURES 사용
    """
    if must_include is None:
        must_include = MUST_INCLUDE_FEATURES
    selector = FeatureSelector(target_col=target_col)
    return selector.fit_select(
        train_df, feature_cols,
        max_features=max_features,
        corr_threshold=corr_threshold,
        variance_threshold=variance_threshold,
        must_include=must_include,
    )
