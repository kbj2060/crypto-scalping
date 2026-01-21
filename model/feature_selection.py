"""
XGBoost 기반 피처 선택 모듈
데이터 전체를 분석하여 미래 변동성을 가장 잘 예측하는 핵심 피처를 선정
"""
import xgboost as xgb
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# matplotlib은 선택적 (시각화용)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class FeatureSelector:
    def __init__(self, top_k=8):
        """
        Args:
            top_k (int): 선택할 상위 피처 개수 (기본 8개)
        """
        self.top_k = top_k
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

    def select_features(self, df, feature_columns, target_horizon=20):
        """
        XGBoost로 피처 중요도를 계산하고 상위 k개를 선정
        
        Args:
            df (pd.DataFrame): 전체 데이터
            feature_columns (list): 분석할 피처 후보군 (전체 리스트)
            target_horizon (int): 예측할 미래 시점 (20 = 약 1시간 뒤 가격 변화)
            
        Returns:
            selected_features (list): 선정된 상위 피처 리스트
        """
        logger.info(f"🔍 XGBoost 피처 선택 시작 (후보 {len(feature_columns)}개 -> 목표 {self.top_k}개)")
        
        # 1. 데이터 준비
        # 입력(X): 현재의 피처 값들
        X = df[feature_columns].copy()
        
        # 목표(y): 미래의 절대 수익률 (변동성 예측)
        # "이 지표가 높을 때 미래에 가격이 크게 움직이는가?"를 봅니다.
        future_return = df['close'].shift(-target_horizon) / df['close'] - 1
        y = future_return.abs()  # 방향 상관없이 '변동성'이 큰 구간을 맞추도록 유도
        
        # NaN 제거 (미래 데이터가 없는 끝부분)
        valid_idx = ~y.isna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        # Inf/NaN 처리
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 2. XGBoost 학습
        logger.info("⚡ XGBoost 학습 중...")
        try:
            self.model.fit(X, y)
        except Exception as e:
            logger.error(f"XGBoost 학습 실패: {e}")
            # 실패 시 상위 k개를 그대로 반환
            return feature_columns[:self.top_k] if len(feature_columns) >= self.top_k else feature_columns
        
        # 3. 중요도 추출
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # 4. 상위 k개 선정
        top_features = feature_importance_df.head(self.top_k)['Feature'].tolist()
        
        # 5. 결과 리포트
        logger.info("=" * 40)
        logger.info("🏆 XGBoost 선정 핵심 피처 Top 10")
        logger.info("-" * 40)
        for idx, row in feature_importance_df.head(10).iterrows():
            logger.info(f"{row['Feature']:<30} : {row['Importance']:.4f}")
        logger.info("=" * 40)
        
        # 시각화 (선택적)
        self._plot_importance(feature_importance_df)
        
        return top_features

    def _plot_importance(self, df):
        """중요도 그래프 저장"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            os.makedirs('logs', exist_ok=True)
            plt.figure(figsize=(10, 6))
            top_15 = df.head(15)
            plt.barh(top_15['Feature'][::-1], top_15['Importance'][::-1])
            plt.title('XGBoost Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.savefig('logs/feature_importance.png', dpi=150)
            plt.close()
            logger.info("📊 피처 중요도 그래프 저장: logs/feature_importance.png")
        except Exception as e:
            logger.debug(f"그래프 저장 실패: {e}")
