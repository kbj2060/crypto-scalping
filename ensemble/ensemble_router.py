"""
Crypto Price Meta-Router Ensemble Forecaster v4.2 (Offline & Bug-Fixed)
================================================================================
6개 모델 앙상블 (커스텀 2 + 파운데이션 4) + Meta Router 동적 가중치 결합
+ ⚡ 로컬 파운데이션 모델 즉각 로딩 (HTTP 통신 제거)
+ 🛠️ uni2ts 로컬 config 로딩 버그 완벽 우회 및 JAX Warning 제거
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from collections import namedtuple
PredictionOutput = namedtuple('PredictionOutput', ['median', 'confidence'])

# 💡 [NEW] PyTorch Lightning의 눈치 없는 반복 로그 및 불필요한 경고 완벽 차단!
import pytorch_lightning as pl
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys

# 💡 [NEW] PyTorch Lightning의 좀비 같은 프로그레스 바를 물리적으로 차단하는 클래스
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        
# ensemble_router.py 위치: crypto-scalping/ensemble/ensemble_router.py
# 실행 위치: crypto-scalping/ (루트)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)  # crypto-scalping/
_TFT_DIR = os.path.join(_THIS_DIR, 'TFT')
_MACROHFT_DIR = os.path.join(_THIS_DIR, 'macroHFT')

for _p in [_ROOT_DIR, _THIS_DIR, _TFT_DIR, _MACROHFT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from TFT_model import TFTSignalModel
from macroHFT_model import ForecastingMacroHFT, MacroHFTConfig

# ════════════════════════════════════════════════════════════════
# 쓸데없는 라이브러리 경고(Warning) 완벽 차단
# ════════════════════════════════════════════════════════════════
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*jax_cuda12_plugin.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # TensorFlow/JAX C++ 로그 차단

# ════════════════════════════════════════════════════════════════
# 모든 하위 모델 내부 tqdm 출력 전역 억제 패치
# ════════════════════════════════════════════════════════════════
try:
    import tqdm as _tqdm_module
    _orig_tqdm_init = _tqdm_module.tqdm.__init__
    def _silent_tqdm_init(self, *args, **kwargs):
        kwargs.setdefault('disable', True)   # 기본적으로 모두 비활성화
        _orig_tqdm_init(self, *args, **kwargs)
    _tqdm_module.tqdm.__init__ = _silent_tqdm_init
except Exception:
    pass


# ════════════════════════════════════════════════════════════════
# 경로 설정 및 지능형 로컬 모델 로더
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.getcwd()
KRONOS_REPO = os.path.join(BASE_DIR, "Kronos")

# 사용자 환경에 맞춰 models 또는 local_foundation_models 자동 선택
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models")

if os.path.exists(KRONOS_REPO) and KRONOS_REPO not in sys.path:
    sys.path.insert(0, KRONOS_REPO)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_path(model_name: str, hf_repo: str) -> str:
    """
    로컬 폴더에 모델이 존재하면 해당 경로를 반환하여 
    HuggingFace HTTP 통신을 완전히 차단(0초 로드)합니다. 
    """
    local_path = os.path.join(LOCAL_MODEL_DIR, model_name)
    if os.path.exists(local_path):
        return local_path
    return hf_repo

# ════════════════════════════════════════════════════════════════
# 1. 표준 출력 인터페이스
# ════════════════════════════════════════════════════════════════
STANDARD_QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]

@dataclass
class ForecastOutput:
    quantiles: np.ndarray      # [N, H, Q]
    median: np.ndarray         # [N, H]
    confidence: np.ndarray     # [N, H] (0~1)
    model_name: str

class BaseForecaster(ABC):
    name: str = "base"
    available: bool = True

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        pass

    @staticmethod
    def _compute_confidence(quantiles_array: np.ndarray) -> np.ndarray:
        q10, q90 = quantiles_array[..., 0], quantiles_array[..., -1]
        spread = np.clip(q90 - q10, 1e-8, None)
        return 1.0 / (1.0 + spread / (np.std(spread) + 1e-8))

    def _empty_output(self, horizon: int) -> ForecastOutput:
        n, Q = 1, len(STANDARD_QUANTILES)
        return ForecastOutput(
            np.zeros((n, horizon, Q)), np.zeros((n, horizon)),
            np.zeros((n, horizon)), self.name
        )

# ════════════════════════════════════════════════════════════════
# 2. 커스텀 모델 (TFT, MacroHFT)
# ════════════════════════════════════════════════════════════════
class TFTForecaster(BaseForecaster):
    def __init__(self, model_path: str = 'data/tft/tft_best.pt'):
        self.name = 'TFT'
        self.feature_cols = []
        self.scaler_params = {}
        try:
            from TFT_model import TFTSignalModel
            self.model_wrapper = TFTSignalModel.load(model_path)
            self.feature_cols = self.model_wrapper.feature_cols
            self.scaler_params = self.model_wrapper.scaler_params
            self.available = True
            logger.info(f"✅ [TFT] 로드 완료")
        except Exception as e:
            logger.warning(f"❌ [TFT] 로드 실패: {e}")
            self.available = False

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        if not self.available: return self._empty_output(horizon)
        try:
            cfg = self.model_wrapper.config
            
            # 💡 [핵심 수정 1] DataLoader를 버리고, 정확히 최신 input_window 만큼만 꼬리를 자름
            df_tail = df.tail(cfg.input_window).copy()
            if len(df_tail) < cfg.input_window:
                return self._empty_output(horizon)
                
            # 💡 [핵심 수정 2] 수동 정규화 후 Tensor로 직접 변환 (Shape: [1, 64, 40])
            df_tail[self.feature_cols] = (df_tail[self.feature_cols] - self.scaler_params['mean']) / self.scaler_params['std']
            tensor_input = torch.tensor(df_tail[self.feature_cols].values, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            
            # 💡 [핵심 수정 3] 모델에 직접 Forward Pass
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                    preds, _, _ = self.model_wrapper.model(tensor_input)
                    
            preds_arr = preds.cpu().numpy() # [1, Horizon, Quantiles]
            
            # 타겟 스케일(수익률 %) 복원
            if getattr(self.model_wrapper, 'target_scaler', None):
                preds_arr = (preds_arr * self.model_wrapper.target_scaler['std']) + self.model_wrapper.target_scaler['mean']
                
            mid_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2
            median_pred = preds_arr[..., mid_idx]
            
            spread = np.clip(preds_arr[..., -1] - preds_arr[..., 0], 1e-6, 10.0)
            confidence = 1.0 / (1.0 + spread / (np.std(spread) + 1e-6))
            
            return ForecastOutput(preds_arr, median_pred, confidence, self.name)
        except Exception as e:
            logger.error(f"TFT Predict Error: {e}")
            return self._empty_output(horizon)

class MacroHFTForecaster(BaseForecaster):
    def __init__(self, model_path: str = 'data/macroHFT/macrohft_best.pt',
                 meta_path: str = 'data/macroHFT/macrohft_best_meta.json'):
        self.name = 'MacroHFT'
        try:
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            cfg = MacroHFTConfig()
            for k, v in meta['config'].items():
                if hasattr(cfg, k): setattr(cfg, k, v)
            self.config = cfg
            self.feature_cols = meta['feature_cols']
            self.scaler_params = {k: np.array(v) if isinstance(v, list) else v for k, v in meta['scaler_params'].items()}
            self.model = ForecastingMacroHFT(cfg)
            self.model.load_state_dict(torch.load(model_path, map_location=cfg.device))
            self.model.to(cfg.device)
            self.model.eval()
            self.available = True
            logger.info(f"✅ [MacroHFT] 로드 완료")
        except Exception as e:
            logger.warning(f"❌ [MacroHFT] 로드 실패: {e}")
            self.available = False

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        if not self.available: return self._empty_output(horizon)
        try:
            cfg = self.config
            
            # 💡 [핵심 수정 1] 최신 데이터 직접 추출
            df_tail = df.tail(cfg.input_window).copy()
            if len(df_tail) < cfg.input_window:
                return self._empty_output(horizon)
                
            # 💡 [핵심 수정 2] 수동 정규화 및 Tensor 변환
            df_tail[self.feature_cols] = (df_tail[self.feature_cols] - self.scaler_params['mean']) / self.scaler_params['std']
            tensor_input = torch.tensor(df_tail[self.feature_cols].values, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            
            with torch.no_grad():
                preds = self.model(tensor_input)
                
            preds_arr = preds.cpu().numpy() # Shape: [1, Horizon, 1] (단일 로짓)
            median_pred = preds_arr[..., 0] # Shape: [1, Horizon]
            
            # MacroHFT v2.5는 단일 로짓을 출력하므로, 앙상블 규격에 맞게 Quantile 차원을 Mocking
            q_arr = np.repeat(preds_arr, len(STANDARD_QUANTILES), axis=-1)
            
            # 💡 확신도 산출: 로짓의 절대값 크기가 클수록 강한 확신을 가짐 (Sigmoid 근사)
            confidence = 1.0 / (1.0 + np.exp(-np.abs(median_pred))) 
            
            return ForecastOutput(q_arr, median_pred, confidence, self.name)
        except Exception as e:
            logger.error(f"MacroHFT Predict Error: {e}")
            return self._empty_output(horizon)

# ════════════════════════════════════════════════════════════════
# 3. 파운데이션 모델들 (로컬 오프라인 우선 로드)
# ════════════════════════════════════════════════════════════════
class ChronosForecaster(BaseForecaster):
    def __init__(self, device: str = "auto"):
        self.name = 'Chronos'
        self.pipeline, self.pipeline_type, self.available = None, None, False
        dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            from chronos import Chronos2Pipeline
            load_path = get_model_path("chronos-2", "amazon/chronos-2")
            logger.info(f"💾 [Chronos-2] 로드 경로: {load_path}")
            self.pipeline = Chronos2Pipeline.from_pretrained(load_path, device_map=dev)
            self.pipeline_type, self.name, self.available = 'chronos2', 'Chronos-2', True
        except ImportError:
            try:
                from chronos import ChronosPipeline
                load_path = get_model_path("chronos-bolt", "amazon/chronos-bolt-base")
                self.pipeline = ChronosPipeline.from_pretrained(load_path, device_map=dev, dtype=torch.float32)
                self.pipeline_type, self.name, self.available = 'bolt', 'Chronos-Bolt', True
            except Exception as e:
                logger.warning(f"❌ [Chronos] 로드 실패: {e}")

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        if not self.available: return self._empty_output(horizon)
        try:
            close_prices = df['close'].values
            if self.pipeline_type == 'chronos2':
                timestamps = pd.date_range(end=pd.Timestamp.now(), periods=len(close_prices), freq='5min')
                ctx_df = pd.DataFrame({'timestamp': timestamps, 'target': close_prices, 'id': 'ETH'})
                pred_df = self.pipeline.predict_df(ctx_df, prediction_length=horizon, quantile_levels=STANDARD_QUANTILES, id_column='id', timestamp_column='timestamp', target='target')
                q_values = [((pred_df[str(q)].values / close_prices[-1]) - 1.0) if str(q) in pred_df.columns else np.zeros(horizon) for q in STANDARD_QUANTILES]
                quantiles_arr = np.stack(q_values, axis=-1)[np.newaxis, :horizon, :]
            else:
                context = torch.tensor(close_prices, dtype=torch.float32)
                forecast = self.pipeline.predict(context, horizon, num_samples=100) if 'num_samples' in self.pipeline.predict.__code__.co_varnames else self.pipeline.predict(context, horizon)
                samples = (forecast[0].numpy() / close_prices[-1]) - 1.0
                if samples.ndim == 1: samples = samples[np.newaxis, :]
                quantiles_arr = np.quantile(samples, STANDARD_QUANTILES, axis=0).T[np.newaxis, :, :]
                
            return ForecastOutput(quantiles_arr, quantiles_arr[..., len(STANDARD_QUANTILES)//2], self._compute_confidence(quantiles_arr), self.name)
        except Exception:
            return self._empty_output(horizon)

class KronosForecaster(BaseForecaster):
    def __init__(self, device: str = "auto"):
        self.name = 'Kronos'
        self.available = False
        dev = device if device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
            m_path = get_model_path("kronos-small", "NeoQuasar/Kronos-small")
            t_path = get_model_path("kronos-tokenizer", "NeoQuasar/Kronos-Tokenizer-base")
            logger.info(f"💾 [Kronos] 로드 경로: {m_path}")
            
            tokenizer = KronosTokenizer.from_pretrained(t_path)
            model = Kronos.from_pretrained(m_path)
            self.predictor = KronosPredictor(model, tokenizer, device=dev, max_context=512)
            self.available = True
        except Exception as e:
            logger.warning(f"❌ [Kronos] 로드 실패: {e}")

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        if not self.available: return self._empty_output(horizon)
        try:
            close_prices = df['close'].values
            input_df = df[['open', 'high', 'low', 'close']].copy()
            if 'volume' in df.columns: input_df['volume'] = df['volume'].values
            
            now = pd.Timestamp.now()
            x_ts = pd.Series(pd.date_range(end=now, periods=len(input_df), freq='5min'))
            y_ts = pd.Series(pd.date_range(start=now + pd.Timedelta('5min'), periods=horizon, freq='5min'))
            
            samples = np.stack([self.predictor.predict(
                df=input_df.reset_index(drop=True),
                x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=horizon, T=1.0, top_p=0.9, sample_count=1
            )['close'].values for _ in range(20)])
            return_samples = (samples / close_prices[-1]) - 1.0
            quantiles_arr = np.quantile(return_samples, STANDARD_QUANTILES, axis=0).T[np.newaxis, :, :]
            return ForecastOutput(quantiles_arr, quantiles_arr[..., len(STANDARD_QUANTILES)//2], self._compute_confidence(quantiles_arr), self.name)
        except Exception:
            return self._empty_output(horizon)

class TimesFMForecaster(BaseForecaster):
    def __init__(self, device: str = "auto"):
        self.name = 'TimesFM'
        self.available = False
        self._use_transformers = False
        dev_map = "cuda" if (device == "auto" and torch.cuda.is_available()) else device if device != "auto" else None
        
        try:
            load_path = get_model_path("timesfm-2.0", "google/timesfm-2.0-500m-pytorch")
            logger.info(f"💾 [TimesFM] 로드 경로: {load_path}")
            from transformers import TimesFmModelForPrediction
            self._model = TimesFmModelForPrediction.from_pretrained(load_path, dtype=torch.float32, device_map=dev_map)
            self._use_transformers = True
            self.available = True
        except ImportError:
            try:
                import timesfm
                backend = "gpu" if torch.cuda.is_available() else "cpu"
                self._model = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend=backend, horizon_len=128), checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-2.5-200m-pytorch"))
                self._model.load_from_checkpoint("google/timesfm-2.5-200m-pytorch")
                self.available = True
            except Exception as e:
                logger.warning(f"❌ [TimesFM] 로드 실패: {e}")
        except Exception as e:
            logger.warning(f"❌ [TimesFM] 로드 실패: {e}")

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        if not self.available: return self._empty_output(horizon)
        try:
            close_prices = df['close'].values
            if self._use_transformers:
                device = next(self._model.parameters()).device
                input_tensor = [torch.tensor(close_prices, dtype=torch.float32).to(device)]
                freq_tensor = torch.tensor([0], dtype=torch.long).to(device)
                with torch.no_grad():
                    outputs = self._model(past_values=input_tensor, freq=freq_tensor, return_dict=True)
                point = outputs.mean_predictions[0, :horizon].float().cpu().numpy()
                point_return = (point / close_prices[-1]) - 1.0
                
                if hasattr(outputs, 'quantile_predictions') and outputs.quantile_predictions is not None:
                    q_returns = (outputs.quantile_predictions[0, :horizon].float().cpu().numpy() / close_prices[-1]) - 1.0
                    selected = q_returns[:, [0, 2, 4, 6, 8]] if q_returns.shape[-1] >= 9 else np.column_stack([np.quantile(q_returns, q, axis=-1) for q in STANDARD_QUANTILES])
                    quantiles_arr = selected[np.newaxis, :, :]
                else:
                    spread = np.abs(point_return) * 0.5 + 1e-6
                    quantiles_arr = np.stack([point_return + (q - 0.5) * 2 * spread for q in STANDARD_QUANTILES], axis=-1)[np.newaxis, :, :]
            else:
                point_f, quant_f = self._model.forecast([close_prices.tolist()], freq=[0])
                point_return = (np.array(point_f[0][:horizon]) / close_prices[-1]) - 1.0
                spread = np.abs(point_return) * 0.5 + 1e-6
                quantiles_arr = np.stack([point_return + (q - 0.5) * 2 * spread for q in STANDARD_QUANTILES], axis=-1)[np.newaxis, :, :]

            return ForecastOutput(quantiles_arr, quantiles_arr[..., len(STANDARD_QUANTILES)//2], self._compute_confidence(quantiles_arr), self.name)
        except Exception:
            return self._empty_output(horizon)

class MoiraiForecaster(BaseForecaster):
    def __init__(self, size: str = "small", device: str = "auto"):
        self.name, self.available = 'MOIRAI-2', False
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            from uni2ts.distribution import MixtureOutput, StudentTOutput, NormalOutput, LogNormalOutput
            
            model_name = f"moirai-2.0-{size}"
            load_path = get_model_path(model_name, f"Salesforce/moirai-2.0-R-{size}")
            logger.info(f"💾 [MOIRAI-2] 로드 경로: {load_path}")
            
            # 🛠️ 로컬 config.json 파싱 버그 우회 (필수 파라미터 강제 주입)
            kwargs = {}
            distr_output = MixtureOutput([StudentTOutput(), NormalOutput(), LogNormalOutput()])
            kwargs["distr_output"] = distr_output
            
            # 🚨 핵심 버그 픽스: MOIRAI 표준 패치 사이즈 강제 할당
            kwargs["patch_sizes"] = [8, 16, 32, 64, 128]
            
            config_path = os.path.join(load_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                
                # patch_sizes가 config에 명시되어 있다면 그것으로 덮어씀
                if "patch_sizes" in cfg:
                    kwargs["patch_sizes"] = cfg["patch_sizes"]
                    
                for key in ["d_model", "num_layers", "max_seq_len", "attn_dropout_p", "dropout_p", "scaling"]:
                    if key in cfg:
                        kwargs[key] = cfg[key]
            
            self._module = MoiraiModule.from_pretrained(load_path, **kwargs)
            self._MoiraiForecast = MoiraiForecast
            self.available = True
            logger.info(f"✅ [{self.name}] 로드 성공")
            
        except Exception as e:
            logger.warning(f"❌ [MOIRAI-2] 로드 실패: {e}")

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> ForecastOutput:
        if not self.available: return self._empty_output(horizon)
        try:
            from gluonts.dataset.pandas import PandasDataset
            from gluonts.dataset.split import split
            from itertools import islice
            
            close_prices = df['close'].values
            ds = PandasDataset({'ETH': pd.DataFrame({'target': close_prices}, index=pd.date_range(end=pd.Timestamp.now(), periods=len(close_prices), freq='5min'))})
            train, test_template = split(ds, offset=-horizon)
            predictor = self._MoiraiForecast(module=self._module, prediction_length=horizon, context_length=200, patch_size="auto", num_samples=100, target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).create_predictor(batch_size=1)
            
            forecasts = list(islice(predictor.predict(test_template.generate_instances(horizon).input), 1))
            if not forecasts: raise ValueError("Empty output")
            
            samples = (forecasts[0].samples / close_prices[-1]) - 1.0
            quantiles_arr = np.quantile(samples, STANDARD_QUANTILES, axis=0).T[np.newaxis, :, :]
            return ForecastOutput(quantiles_arr, quantiles_arr[..., len(STANDARD_QUANTILES)//2], self._compute_confidence(quantiles_arr), self.name)
        except Exception:
            return self._empty_output(horizon)


# ════════════════════════════════════════════════════════════════
# 4. MetaRouter 신경망 & Ensemble 
# ════════════════════════════════════════════════════════════════
class MetaRouter(nn.Module):
    """
    고도화된 상황 인지형 라우터 (Context-Aware MoE Router v2)
    - Feature Attention, Residual Block, Learnable Temperature 적용
    """
    def __init__(self, input_dim: int, num_models: int, hidden_dim: int = 128):
        super().__init__()
        
        # 1. Feature Attention (Squeeze-and-Excitation 스타일)
        # 시장 상황에 따라 40개 피처 중 어떤 피처를 눈여겨볼지 스스로 결정
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # 2. Residual Block 1
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 3. Residual Block 2
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 4. Output Head & Temperature
        self.head = nn.Linear(hidden_dim, num_models)
        
        # [핵심] 1/N 뇌사 상태를 방지하는 학습 가능한 온도 파라미터
        # 초기값을 낮게 주어 처음부터 날카로운(극단적인) 선택을 유도함
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        # Step 1: Feature Attention 적용
        attn_weights = self.feature_attention(x)
        x_attended = x * attn_weights
        
        # Step 2: Residual Network 통과
        h1 = self.layer1(x_attended)
        h2 = self.layer2(h1) + h1 
        
        # Step 3: 원본 Logits 추출
        logits = self.head(h2)
        
        # 💡 [핵심 트릭 1] Noisy Top-K Gating (학습 중 강제 탐색 유도)
        # 훈련(Training) 중에만 로짓에 가우시안 노이즈를 주입하여 편식을 막고 골고루 써보게 함
        if self.training:
            noise = torch.randn_like(logits) * 0.5  # 노이즈 스케일
            logits = logits + noise
            
        # Temperature Scaling
        scaled_logits = logits / (self.temperature + 1e-8)
        
        # 💡 [핵심 트릭 2] K=3으로 확장 (상위 3개 모델 선택)
        k = 3
        topk_logits, topk_indices = torch.topk(scaled_logits, k, dim=-1)
        
        # 선택받지 못한 모델은 -inf로 밀어버림
        sparse_logits = torch.full_like(scaled_logits, float('-inf'))
        sparse_logits.scatter_(1, topk_indices, topk_logits)
        
        return F.softmax(sparse_logits, dim=-1)

class MetaRouterEnsembleForecaster:
    def __init__(self, models: List[BaseForecaster], tft_forecaster: Optional[TFTForecaster] = None, router_path: Optional[str] = None):
        self.models = [m for m in models if m.available]
        self.num_models = len(self.models)
        self.feature_cols = tft_forecaster.feature_cols if tft_forecaster else []
        self.scaler_params = tft_forecaster.scaler_params if tft_forecaster else {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        state_dim = len(self.feature_cols)
        if state_dim > 0:
            self.router = MetaRouter(input_dim=state_dim, num_models=self.num_models).to(self.device)
            if router_path and os.path.exists(router_path):
                self.router.load_state_dict(torch.load(router_path, map_location=self.device))
                logger.info("✅ [MetaRouter] 가중치 로드 완료")
            self.router.eval()
        else:
            self.router = None
            
        logger.info(f"✅ 통합 앙상블 구성 완료: {[m.name for m in self.models]}")

    def _get_router_weights(self, df: pd.DataFrame) -> np.ndarray:
        if not self.router or not self.feature_cols:
            return np.ones(self.num_models) / self.num_models
        temporal_cols = self.feature_cols
        for col in temporal_cols:
            if col not in df.columns: df[col] = 0.0
        
        last_state = df[temporal_cols].iloc[-1].values.astype(np.float32)
        norm_state = (last_state - self.scaler_params.get('mean', 0)) / self.scaler_params.get('std', 1)
        
        with torch.no_grad():
            weights = self.router(torch.tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        return weights

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> Dict:
        outputs, model_names = [], []
        for model in self.models:
            out = model.predict(df, horizon)
            if out.median is not None and not np.all(out.median == 0):
                outputs.append(out)
                model_names.append(model.name)
            else:
                logger.warning(f"  ✗ {model.name}: 빈 출력 반환")

        if not outputs: raise ValueError("모든 모델이 추론에 실패했습니다.")

        medians = np.stack([o.median[-1][:horizon] for o in outputs])           
        confidences = np.stack([o.confidence[-1][:horizon] for o in outputs])   
        quantiles = np.stack([o.quantiles[-1][:horizon] for o in outputs])      
        M, H = medians.shape
        
        router_weights_raw = self._get_router_weights(df)
        success_indices = [i for i, m in enumerate(self.models) if m.name in model_names]
        router_weights = router_weights_raw[success_indices]
        router_weights = router_weights / (router_weights.sum() + 1e-8)
        
        hybrid_weights = router_weights[:, np.newaxis] * confidences
        hybrid_weights = hybrid_weights / (hybrid_weights.sum(axis=0, keepdims=True) + 1e-8)

        ensemble_median = (medians * hybrid_weights).sum(axis=0)
        ensemble_quantiles = (quantiles * hybrid_weights[..., np.newaxis]).sum(axis=0)

        directions = np.sign(medians)
        bull_ratio = (directions > 0).mean(axis=0)
        bear_ratio = (directions < 0).mean(axis=0)
        direction_consensus = np.where(bull_ratio >= 0.6, 1.0, np.where(bear_ratio >= 0.6, -1.0, 0.0))

        contributions = {name: {'router_alloc': float(router_weights[i]), 'hybrid_weight': float(hybrid_weights[i].mean()), 'median_pred': float(medians[i, 0])} for i, name in enumerate(model_names)}

        return {
            'ensemble_median': ensemble_median,
            'ensemble_quantiles': ensemble_quantiles,
            'direction_consensus': direction_consensus,
            'model_contributions': contributions,
            'num_models': M
        }

# -----------------------------------------------------------------------------
# [REAL] IBM Tiny Time Mixers (Granite TTM) 실전 추론 엔진
# -----------------------------------------------------------------------------
class TTMForecaster:
    """IBM Granite TTM: 초고속 Zero-shot 파운데이션 모델"""
    def __init__(self):
        self.available = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # 💡 [핵심 패치] transformers가 아니라 IBM의 공식 패키지에서 꺼내옵니다!
            from tsfm_public import TinyTimeMixerForPrediction
            
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                "ibm-granite/granite-timeseries-ttm-r1" 
            ).to(self.device)
            self.model.eval()
            self.context_length = self.model.config.context_length
            self.available = True
            logger.info("✅ IBM TTM (Granite-R1) 실전 로드 완료 (Ultra-Fast)")
        except ImportError:
            logger.warning("⚠️ TTM 대기 중: 'pip install \"git+https://github.com/ibm-granite/granite-tsfm.git\"' 실행이 필요합니다.")
        except Exception as e:
            logger.warning(f"⚠️ TTM 로드 실패 상세 원인: {e}")

    def predict(self, df, horizon=6):
        if not self.available or len(df) < 2:
            return PredictionOutput(median=np.zeros((1, horizon)), confidence=np.ones((1, horizon))*0.5)
        try:
            seq = df['close'].values[-self.context_length:]
            if len(seq) < self.context_length:
                seq = np.pad(seq, (self.context_length - len(seq), 0), 'edge')
            
            seq_mean, seq_std = np.mean(seq), np.std(seq) + 1e-8
            seq_scaled = (seq - seq_mean) / seq_std

            # TTM-R1 입력 텐서 형태: [batch_size, context_length, num_features]
            inputs = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

            with torch.no_grad():
                outputs = self.model(past_values=inputs)
                pred_scaled = outputs.prediction_outputs.squeeze(0).squeeze(-1).cpu().numpy()
            
            pred = pred_scaled * seq_std + seq_mean
            pred = pred[:horizon] if len(pred) >= horizon else np.pad(pred, (0, horizon - len(pred)), 'edge')
            
            return PredictionOutput(median=np.array([pred]), confidence=np.ones((1, horizon)) * 0.6)
        except Exception as e:
            logger.debug(f"TTM 추론 에러: {e}")
            return PredictionOutput(median=np.zeros((1, horizon)), confidence=np.ones((1, horizon))*0.5)
# -----------------------------------------------------------------------------
# [REAL] NeuralForecast 기반 실전 엔진 (4종 통합 싱글톤 래퍼)
# -----------------------------------------------------------------------------
class UnifiedNFForecaster:
    """단변량/다변량 4종 통합 엔진 (메모리 최적화 및 파이프라인 충돌 버그 수정)"""
    _nf_model = None  # 싱글톤 패턴: 메모리에 딱 한 번만 로드하여 VRAM 낭비 방지!
    _available = False

    def __init__(self, model_type):
        self.model_type = model_type
        # 단변량/다변량 상관없이 7대 알파를 모두 선언
        self.exog_cols = [
            'session_us', 'hour_cos', 'cvp_poc_dist', 
            'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate'
        ]
        
        # 최초 1회만 모델을 로드하여 _nf_model에 공유
        if UnifiedNFForecaster._nf_model is None:
            try:
                from neuralforecast import NeuralForecast
                model_dir = os.path.join(os.getcwd(), 'data', 'nf')
                if os.path.exists(model_dir):
                    UnifiedNFForecaster._nf_model = NeuralForecast.load(path=model_dir)
                    UnifiedNFForecaster._available = True
                    logger.info("✅ NeuralForecast 4종 통합 팩 로드 완료")
                else:
                    logger.warning(f"⚠️ NF 모델 폴더({model_dir})가 없습니다.")
            except Exception as e:
                logger.error(f"⚠️ NF 초기화 에러: {e}")
        
        self.nf = UnifiedNFForecaster._nf_model
        self.available = UnifiedNFForecaster._available

    def predict(self, df, horizon=6):
        if not self.available or len(df) < 256:
            return PredictionOutput(median=np.zeros((1, horizon)), confidence=np.ones((1, horizon))*0.5)
        
        try:
            # 💡 [핵심 버그 수정] 어떤 모델이 호출하든 무조건 7대 알파를 모두 넣어서 파이프라인 충돌을 막습니다!
            # (PatchTST 같은 단변량 모델은 알파 피처가 들어와도 알아서 무시하고 가격만 봅니다.)
            # 💡 [핵심 버그 수정] 어떤 모델이 호출하든 무조건 7대 알파를 모두 넣어서 파이프라인 충돌을 막습니다!
            df_nf = df[['close'] + self.exog_cols].tail(256).copy() 
            
            # 💡 [Pandas 버전 호환성 패치] fillna(method) 대신 ffill() 전용 메서드 사용
            df_nf.ffill(inplace=True)
            df_nf.fillna(0.0, inplace=True)
            
            df_nf['ds'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_nf), freq='5min')
            df_nf['unique_id'] = 'ETH'
            df_nf.rename(columns={'close': 'y'}, inplace=True)
            
            # 💡 [핵심] 예측을 수행하는 이 순간에만 터미널 출력을 완벽히 물리적으로 차단!
            with SuppressOutput():
                pred_df = self.nf.predict(df=df_nf)
                
            # 예측 로짓 추출
            pred = pred_df[self.model_type].values[:horizon]
            return PredictionOutput(median=np.array([pred]), confidence=np.ones((1, horizon)) * 0.5)
            
        except Exception as e:
            # 숨겨져 있던 에러를 보이도록 ERROR 레벨로 출력
            logger.error(f"🔥 {self.model_type} 추론 실패 상세 에러: {e}")
            return PredictionOutput(median=np.zeros((1, horizon)), confidence=np.ones((1, horizon))*0.5)

# ── 각 모델별 클래스 매핑 ──
class PatchTSTForecaster(UnifiedNFForecaster):
    def __init__(self): super().__init__(model_type="PatchTST")
class ITransformerForecaster(UnifiedNFForecaster):
    def __init__(self): super().__init__(model_type="iTransformer")
class NHITSForecaster(UnifiedNFForecaster):
    def __init__(self): super().__init__(model_type="NHITS")
class TiDEForecaster(UnifiedNFForecaster):
    def __init__(self): super().__init__(model_type="TiDE")

if __name__ == "__main__":
    # ── 유저 최적화 3대 파운데이션 모델 ──
    timesfm = TimesFMForecaster()
    chronos = ChronosForecaster()
    ttm = TTMForecaster() 
    
    # ── 단변량(Price) 집중 2대 모델 ──
    patchtst = PatchTSTForecaster()
    itransformer = ITransformerForecaster()
    
    # ── 다변량(Price + 7대 알파) 집중 2대 모델 ──
    nhits = NHITSForecaster()
    tide = TiDEForecaster()

    # Meta Router에 군더더기 없는 7개의 브레인만 완벽하게 주입! (TFT는 삭제되었으므로 None 처리)
    ensemble = MetaRouterEnsembleForecaster(
        models=[timesfm, chronos, ttm, patchtst, itransformer, nhits, tide], 
        tft_forecaster=None 
    )

    n = 200
    close = 3000 + np.cumsum(np.random.randn(n) * 10)
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n, freq='5min'),
        'open': close + np.random.randn(n) * 2,
        'high': close + np.abs(np.random.randn(n) * 5),
        'low': close - np.abs(np.random.randn(n) * 5),
        'close': close,
        'volume': np.random.exponential(100, n),
    })
    
    if tft.available:
        for col in tft.feature_cols:
            if col not in df.columns: df[col] = np.random.randn(n) * 0.1

    print("\n▶ 앙상블 추론 시작...")
    result = ensemble.predict(df, horizon=6)
    print(f"\n📊 앙상블 결과 요약: {[f'{x:+.4f}' for x in result['ensemble_median']]}")