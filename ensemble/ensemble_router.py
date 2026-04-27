"""NeuralForecast-based high-performance batch and live inference router with internal bug fixes."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections import namedtuple

# NVRTC 및 CUDA 라이브러리 충돌 해결을 위한 환경 변수 설정
os.environ["TORCH_COMPILE_DISABLE"] = "1"
OLLAMA_CUDA_PATH = "/usr/local/lib/ollama/mlx_cuda_v13"
if os.path.exists(OLLAMA_CUDA_PATH):
    os.environ["LD_LIBRARY_PATH"] = OLLAMA_CUDA_PATH + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import pandas as pd

# --- PyTorch Lightning Monkeypatch (Fixes 'val_monitor' error) ---
try:
    import pytorch_lightning as pl
    _orig_trainer_init = pl.Trainer.__init__
    def _patched_trainer_init(self, *args, **kwargs):
        if 'val_monitor' in kwargs:
            kwargs.pop('val_monitor')
        return _orig_trainer_init(self, *args, **kwargs)
    pl.Trainer.__init__ = _patched_trainer_init
    # logger.info("Successfully patched PyTorch Lightning Trainer to ignore 'val_monitor'")
except Exception:
    pass
# -----------------------------------------------------------------

PredictionOutput = namedtuple("PredictionOutput", ["median", "confidence"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for _log_name in ["pytorch_lightning", "lightning", "neuralforecast", "nixtla"]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning)


class SuppressOutput:
    def __enter__(self):
        self._original_stdout, self._original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, "w")
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout, sys.stderr = self._original_stdout, self._original_stderr


class PatchTSTForecaster:
    """NeuralForecast 기반 추론기 (Monkeypatched)."""
    _nf_model = None
    _available = False
    name = "PatchTST"

    def __init__(self, device: str | None = None):
        self.model_type = self.name
        self.lookback = 256
        self.device = device # None means use default (NF handles it)
        self.exog_cols = [
            "session_us", "hour_cos", "cvp_poc_dist", "cvp_volume_imbalance",
            "fvg_dist", "breakout_strength", "oi_change_rate", "ofti", "kel",
            "mta_funding", "svps"
        ]
        self._load_model_pack()
        self.nf = self.__class__._nf_model
        self.available = self.__class__._available
        
        # CPU 강제 모드일 경우 모델 이동
        if self.available and self.device == "cpu" and self.nf is not None:
            try:
                for m in self.nf.models:
                    m.to("cpu")
            except Exception:
                pass

    @classmethod
    def _load_model_pack(cls) -> None:
        if cls._nf_model is not None: return
        try:
            from neuralforecast import NeuralForecast
            # 프로젝트 루트 기준으로 절대 경로 구성
            _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(_base, "data", f"nf_{cls.name.lower()}")
            
            if not os.path.exists(model_dir):
                cls._available = False
                logger.warning("❌ %s 모델 디렉토리 없음: %s", cls.name, model_dir)
                return
            
            with SuppressOutput():
                cls._nf_model = NeuralForecast.load(path=model_dir)
            cls._available = True
            logger.info("✅ %s NeuralForecast pack loaded", cls.name)
        except Exception as e:
            cls._available = False
            logger.error("❌ %s load failed at %s: %s", cls.name, model_dir if 'model_dir' in locals() else 'unknown', e)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        
        # 안전한 컬럼 추출 도우미
        def get_s(name, default=0.0):
            val = work.get(name)
            if val is None or (not isinstance(val, pd.Series)):
                return pd.Series([float(default)] * len(work), index=work.index)
            return pd.to_numeric(val, errors="coerce").fillna(float(default))

        # 공통 피처 계산
        smf = get_s("smart_money_flow")
        wc = get_s("whale_conviction")
        aiz = get_s("amihud_illiquidity_z")
        work["ofti"] = np.tanh(smf * wc * (aiz.abs() + 1.0) * 3.0)

        fr = get_s("last_funding_rate")
        for w in (12, 48, 288):
            col = f"funding_roc_{w}"
            if col not in work.columns:
                shifted = fr.shift(w).fillna(0)
                work[col] = ((fr - shifted) / (shifted.abs().clip(lower=1e-4) + 1e-8)).clip(-10, 10)

        oic = get_s("oi_change_rate")
        gkv = get_s("garman_klass_vol", 0.0001)
        fp = get_s("funding_pressure")
        
        kel_raw = oic / (gkv + 1e-6) * np.sign(fp)
        work["kel"] = np.tanh((kel_raw - kel_raw.rolling(288, min_periods=1).mean()) / (kel_raw.rolling(288, min_periods=1).std().fillna(1.0) + 1e-8) * 0.5)

        fr12, fr48, fr288 = get_s("funding_roc_12"), get_s("funding_roc_48"), get_s("funding_roc_288")
        fabs = get_s("funding_abs", 1e-8).clip(lower=1e-8)
        sqp = get_s("squeeze_power")
        sq_z = (sqp - sqp.rolling(288, min_periods=1).mean()) / (sqp.rolling(288, min_periods=1).std().fillna(1.0) + 1e-8)
        work["mta_funding"] = (( (0.5*fr12 + 0.3*fr48 + 0.2*fr288) / fabs ) * np.tanh(sq_z)).clip(-3, 3) / 3.0

        cpd = get_s("cvp_poc_dist")
        cvi = get_s("cvp_volume_imbalance")
        cvw = get_s("cvp_vah_val_width")
        work["svps"] = np.tanh(2.0 * cpd * cvi * np.exp(-cvw.clip(0, 5)))

        cols = ["close"] + self.exog_cols
        for c in cols:
            if c not in work.columns: work[c] = 0.0
        return work[cols].ffill().fillna(0.0)

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> PredictionOutput:
        if not self.available: return PredictionOutput(np.full((1, horizon), np.nan), np.full((1, horizon), np.nan))
        prep = self._prepare_data(df).tail(self.lookback)
        if len(prep) < self.lookback: return PredictionOutput(np.full((1, horizon), np.nan), np.full((1, horizon), np.nan))
        
        prep_nf = prep.copy()
        prep_nf["ds"] = pd.date_range(end=pd.Timestamp.now(), periods=len(prep_nf), freq="5min")
        prep_nf["unique_id"] = "ETH"
        prep_nf.rename(columns={"close": "y"}, inplace=True)
        
        try:
            with SuppressOutput():
                pred_df = self.nf.predict(df=prep_nf)
            pred = pred_df[self.model_type].values[:horizon]
            return PredictionOutput(np.array([pred], dtype=np.float32), np.full((1, horizon), 0.5, dtype=np.float32))
        except Exception as e:
            logger.warning(
                "❌ %s predict failed (len=%d, lookback=%d, horizon=%d): %s",
                self.name,
                len(prep_nf),
                self.lookback,
                horizon,
                e,
            )
            return PredictionOutput(np.full((1, horizon), np.nan), np.full((1, horizon), np.nan))

    def get_refined_features(self, df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        pred = self.predict(df, horizon=1)
        if pred.median.size == 0 or np.isnan(pred.median).all():
            # Keep schema stable even when model output is unavailable.
            nan_df = pd.DataFrame(index=[0])
            tmp = self._apply_refined_batch_logic(nan_df, np.array([np.nan], dtype=np.float32))
            for c in tmp.columns:
                out[c] = 0.0
            return out

        val = float(pred.median[0, 0])
        one = pd.DataFrame(index=[0])
        tmp = self._apply_refined_batch_logic(one, np.array([val], dtype=np.float32))
        for c in tmp.columns:
            v = tmp.iloc[-1][c]
            out[c] = float(v) if np.isfinite(v) else 0.0
        return out

    def predict_batch(self, df: pd.DataFrame, chunk_size: int = 500) -> pd.DataFrame:
        """Monkeypatched Trainer를 사용하여 안정적인 대용량 배치 추론 수행."""
        if not self.available: return pd.DataFrame(index=df.index)
        
        prep = self._prepare_data(df)
        n = len(prep)
        if n < self.lookback: return pd.DataFrame(index=df.index)

        all_preds = np.full(n, np.nan)
        start_idx = self.lookback
        
        for i in range(start_idx, n, chunk_size):
            end_idx = min(i + chunk_size, n)
            chunk_indices = range(i, end_idx)
            
            chunk_windows = []
            for idx in chunk_indices:
                window = prep.iloc[idx - self.lookback : idx].copy()
                window.insert(0, "unique_id", f"w_{idx}")
                window.insert(1, "ds", pd.date_range(start="2024-01-01", periods=self.lookback, freq="5min"))
                chunk_windows.append(window)
            
            if not chunk_windows: continue
            
            long_df = pd.concat(chunk_windows).rename(columns={"close": "y"})
            
            try:
                with SuppressOutput():
                    # Monkeypatch 덕분에 이제 에러 없이 작동함
                    chunk_pred_df = self.nf.predict(df=long_df)
                
                # 결과값 매핑 (최신 NF는 unique_id가 컬럼인 경우가 많음)
                if "unique_id" in chunk_pred_df.columns:
                    chunk_pred_df = chunk_pred_df.set_index("unique_id")
                
                # 모델 타입과 일치하는 컬럼 찾기 (대소문자 무시 혹은 포함 관계)
                target_col = None
                for c in chunk_pred_df.columns:
                    if self.model_type.lower() in c.lower():
                        target_col = c
                        break
                
                if target_col is None:
                    logger.warning("❌ %s 컬럼을 찾을 수 없음. Columns: %s", self.model_type, list(chunk_pred_df.columns))
                    continue

                filled_count = 0
                for idx in chunk_indices:
                    uid = f"w_{idx}"
                    if uid in chunk_pred_df.index:
                        val = chunk_pred_df.loc[uid, target_col]
                        # horizon이 여러 개일 경우 첫 번째 값 사용
                        if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray)):
                            val = val.iloc[0] if hasattr(val, "iloc") else val[0]
                        
                        all_preds[idx] = float(val)
                        filled_count += 1
                
                if i % (chunk_size * 10) == 0:
                    logger.info("  ... %s 배치 진행 중 (%d/%d), 이번 청크 채워진 행: %d", self.name, i, n, filled_count)
            except Exception as e:
                logger.warning("❌ %s 청크 추론 실패 (%d~%d): %s", self.name, i, end_idx, e)

        # Warmup 구간(lookback 이전) 처리:
        # - bfill을 쓰면 미래 시점 정보가 과거 행으로 전파되어 look-ahead가 생길 수 있음
        # - 따라서 과거->미래 방향 ffill만 허용
        # - warmup 구간은 NaN 유지 후 최종적으로 0.0으로만 치환 (보수적)
        pred_s = pd.Series(all_preds, index=df.index, dtype="float64")
        if pred_s.notna().any():
            pred_s = pred_s.ffill()
        else:
            pred_s = pred_s.fillna(0.0)
        pred_s = pred_s.fillna(0.0)

        res_df = pd.DataFrame(index=df.index)
        return self._apply_refined_batch_logic(res_df, pred_s.to_numpy(dtype=np.float32))

    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        s = pd.Series(preds, dtype="float64")
        res_df[f"{self.name.lower()}_median"] = s.values

        # Dynamic similarity proxy (not constant):
        # high when current prediction is close to its local trend, lower on abrupt deviations.
        ema = s.ewm(alpha=0.1, adjust=False).mean()
        dev = (s - ema).abs()
        denom = dev.rolling(128, min_periods=8).std(ddof=0).fillna(dev.std(ddof=0))
        denom = denom.replace(0.0, np.nan).fillna(1e-6)
        sim = 1.0 - np.tanh(dev / (3.0 * denom))
        res_df[f"{self.name.lower()}_regime_sim"] = np.clip(sim.fillna(0.0), 0.0, 1.0).values
        return res_df

class TiDEVolatilityForecaster(PatchTSTForecaster):
    _nf_model = None
    _available = False
    name = "TiDE"
    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        res_df["tide_vol_raw"] = preds
        s = pd.Series(preds)
        res_df["tide_vol_zscore"] = (s - s.rolling(200, min_periods=20).mean()) / (s.rolling(200, min_periods=20).std() + 1e-8)
        return res_df

class TimesNetCycleForecaster(PatchTSTForecaster):
    _nf_model = None
    _available = False
    name = "TimesNet"
    def __init__(self):
        # TimesNet은 NVRTC 라이브러리 충돌 이슈(CUDA 13.0 요구)로 인해 CPU 강제 모드 사용
        super().__init__(device="cpu")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """TimesNet 전용: 60봉 롤링 VWAP 이격도를 계산하여 'close' 컬럼에 주입."""
        work = df.copy()
        
        # 필수 컬럼 확인
        if all(c in work.columns for c in ['high', 'low', 'close', 'volume']):
            window = 60
            # Rolling VWAP 계산
            tp = (work['high'] + work['low'] + work['close']) / 3
            v = work['volume']
            pv_sum = (tp * v).rolling(window=window).sum()
            v_sum = v.rolling(window=window).sum().replace(0, 1)
            vwap = pv_sum / v_sum
            
            # 이격도 (%)를 'close' 컬럼에 덮어씀
            work['close'] = (work['close'] / vwap - 1) * 100
        
        # 부모의 나머지 피처 계산 로직 호출
        return super()._prepare_data(work)

    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        # 이제 preds는 VWAP 이격도 (%) 임
        s = pd.Series(preds)
        
        # 이격도 자체가 이미 순환적인 특징을 가짐
        res_df["timesnet_cycle_sin"] = np.tanh(s / 0.5) # 0.5% 편차를 기준으로 스케일링
        res_df["timesnet_cycle_cos"] = np.sin(2 * np.pi * s / 1.0) # 1.0% 주기의 사이클 표현
        res_df["timesnet_cycle_delta"] = s.diff().fillna(0.0).values # 이격도의 변화율
        return res_df

class DLinearOFIForecaster(PatchTSTForecaster):
    _nf_model = None
    _available = False
    name = "DLinear"
    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
        s = pd.Series(preds)
        ema = s.ewm(alpha=0.1, adjust=False).mean()
        res_df["dlinear_smf_ema"] = ema.values
        res_df["dlinear_smf_slope"] = ema.diff().fillna(0.0).values
        return res_df

class EnsembleRouter:
    def __init__(self):
        self.models = {
            "PatchTST": PatchTSTForecaster(),
            "TiDE": TiDEVolatilityForecaster(),
            "TimesNet": TimesNetCycleForecaster(),
            "DLinear": DLinearOFIForecaster(),
        }

    def get_refined_features(self, df: pd.DataFrame) -> pd.DataFrame:
        res_df = pd.DataFrame(index=df.index)
        if len(df) >= 1000:
            logger.info("EnsembleRouter: 대량 데이터 패치 배치 추론 시작 (%d 행)", len(df))
            for name, model in self.models.items():
                if model.available:
                    batch_res = model.predict_batch(df)
                    if not batch_res.empty:
                        for col in batch_res.columns: res_df[col] = batch_res[col]
                        logger.info("✅ %s 배치 추론 완료", name)
        else:
            for name, model in self.models.items():
                if model.available:
                    try:
                        feat_dict = model.get_refined_features(df)
                        for col, val in feat_dict.items():
                            if col not in res_df.columns: res_df[col] = np.nan
                            res_df.loc[df.index[-1], col] = val
                    except Exception as e:
                        logger.warning("⚠️ %s live refined feature generation failed: %s", name, e)
        return res_df
