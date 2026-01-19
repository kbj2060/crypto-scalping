"""
데이터 전처리 모듈
웨이블릿 변환을 이용한 노이즈 제거 및 정규화
"""
import numpy as np
import logging
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logging.warning("pywavelets 미설치: 웨이블릿 변환 비활성화")

try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn 미설치: MinMaxScaler 비활성화")

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """데이터 전처리 클래스: 웨이블릿 노이즈 제거 + Min-Max Scaling"""
    def __init__(self, feature_range=(-1, 1)):
        """
        Args:
            feature_range: 정규화 범위 (xLSTM은 -1~1 사이 입력 선호)
        """
        if SKLEARN_AVAILABLE:
            self.scaler = MinMaxScaler(feature_range=feature_range)
        else:
            self.scaler = None
            logger.warning("MinMaxScaler 사용 불가: 수동 정규화 사용")
        
        self.wavelet = 'db4'  # Daubechies 4 웨이블릿 (시계열에 흔히 사용)
        self.level = 1  # 분해 레벨
        self.feature_range = feature_range

    def wavelet_denoising(self, data):
        """
        웨이블릿 변환을 이용한 노이즈 제거
        
        Args:
            data: 1D 배열 (시계열 데이터)
        Returns:
            denoised_data: 노이즈 제거된 데이터
        """
        if not PYWT_AVAILABLE:
            logger.warning("웨이블릿 변환 비활성화: 원본 데이터 반환")
            return data
        
        try:
            # 웨이블릿 분해
            coeff = pywt.wavedec(data, self.wavelet, mode="per")
            
            # 고주파 성분(노이즈)에 대한 임계값 처리
            if len(coeff) > self.level:
                sigma = (1 / 0.6745) * self.madev(coeff[-self.level])
                uthresh = sigma * np.sqrt(2 * np.log(len(data)))
                
                # 고주파 계수에 임계값 적용 (하드 임계값)
                coeff[1:] = [pywt.threshold(i, value=uthresh, mode='hard') 
                            for i in coeff[1:]]
            
            # 재구성된 신호 반환
            denoised = pywt.waverec(coeff, self.wavelet, mode='per')
            
            # 원본 길이로 맞춤
            if len(denoised) > len(data):
                return denoised[:len(data)]
            elif len(denoised) < len(data):
                # 부족한 경우 패딩
                padding = np.zeros(len(data) - len(denoised))
                return np.concatenate([denoised, padding])
            else:
                return denoised
                
        except Exception as e:
            logger.error(f"웨이블릿 변환 실패: {e}, 원본 데이터 반환")
            return data

    def madev(self, d, axis=None):
        """
        평균 절대 편차 계산 (Median Absolute Deviation의 근사)
        
        Args:
            d: 데이터 배열
            axis: 축 (None이면 전체)
        Returns:
            평균 절대 편차
        """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def fit_transform(self, data):
        """
        데이터 정규화 (Min-Max Scaling)
        
        Args:
            data: (seq_len, feature_dim) 형태의 배열
        Returns:
            정규화된 데이터
        """
        if self.scaler is not None:
            return self.scaler.fit_transform(data)
        else:
            # 수동 정규화 (sklearn 없을 때)
            data = np.array(data, dtype=np.float32)
            if data.size == 0:
                return data
            
            # 각 피처별로 정규화
            data_min = data.min(axis=0, keepdims=True)
            data_max = data.max(axis=0, keepdims=True)
            
            # 0으로 나누기 방지
            range_val = data_max - data_min
            range_val[range_val == 0] = 1.0
            
            normalized = (data - data_min) / range_val
            
            # feature_range로 스케일링
            min_val, max_val = self.feature_range
            normalized = normalized * (max_val - min_val) + min_val
            
            return normalized

    def transform(self, data):
        """
        학습된 스케일러로 데이터 변환
        
        Args:
            data: (seq_len, feature_dim) 형태의 배열
        Returns:
            정규화된 데이터
        """
        if self.scaler is not None:
            return self.scaler.transform(data)
        else:
            # fit_transform과 동일 (수동 정규화)
            return self.fit_transform(data)
