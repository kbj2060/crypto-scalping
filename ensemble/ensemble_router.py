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
            result = self.model_wrapper.predict(df)
            if not result: raise ValueError("TFT Empty Result")
            return ForecastOutput(result['quantiles'], result['median_pred'], result['confidence'], self.name)
        except Exception:
            return self._empty_output(horizon)

class MacroHFTForecaster(BaseForecaster):
    def __init__(self, model_path: str = 'data/macrohft/macrohft_best.pt',
                 meta_path: str = 'data/macrohft/macrohft_best_meta.json'):
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
            from train_macroHFT import MacroHFTDataset
            from torch.utils.data import DataLoader
            cfg = self.config
            
            # 🌟 수정: static_cols 강제 분리 로직 완전 제거
            temporal_cols = self.feature_cols
            df_norm = df.copy()
            df_norm[temporal_cols] = (df_norm[temporal_cols] - self.scaler_params['mean']) / self.scaler_params['std']
            
            # DataLoader에서 static_cols 인자 제거
            loader = DataLoader(MacroHFTDataset(cfg, df_norm, temporal_cols), batch_size=cfg.batch_size, shuffle=False)
            all_preds = []
            with torch.no_grad():
                # 🌟 수정: 언패킹 2개(seq, tgt)로 변경 및 info 파라미터 제외
                for seq, _ in loader:
                    all_preds.append(self.model(seq.to(cfg.device)).cpu().numpy())
            if not all_preds: raise ValueError("MacroHFT Empty Result")
            
            preds_arr = np.concatenate(all_preds, axis=0)
            mid_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else len(cfg.quantiles) // 2
            return ForecastOutput(preds_arr, preds_arr[..., mid_idx], self._compute_confidence(preds_arr), self.name)
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
            self._model = TimesFmModelForPrediction.from_pretrained(load_path, torch_dtype=torch.float32, device_map=dev_map)
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
    def __init__(self, input_dim: int, num_models: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, num_models)
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return F.softmax(self.net(x) / 1.2, dim=-1)

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

if __name__ == "__main__":
    tft = TFTForecaster()
    macrohft = MacroHFTForecaster()
    chronos = ChronosForecaster()
    kronos = KronosForecaster()
    timesfm = TimesFMForecaster()
    moirai = MoiraiForecaster()

    ensemble = MetaRouterEnsembleForecaster(models=[tft, macrohft, chronos, kronos, timesfm, moirai], tft_forecaster=tft if tft.available else None)

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