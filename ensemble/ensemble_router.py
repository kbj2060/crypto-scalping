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

try:
    _orig_string_dtype_init = pd.StringDtype.__init__

    def _patched_string_dtype_init(self, storage=None, na_value=None):
        return _orig_string_dtype_init(self, storage=storage)

    pd.StringDtype.__init__ = _patched_string_dtype_init
except Exception:
    pass

# --- PyTorch Lightning Monkeypatch (Fixes 'val_monitor' error) ---
if os.getenv("ENSEMBLE_ROUTER_PATCH_LIGHTNING", "0").strip().lower() in {"1", "true", "yes", "on"}:
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


def _router_chunk_size(default: int = 4000) -> int:
    raw = os.getenv("ENSEMBLE_ROUTER_CHUNK_SIZE", "").strip()
    if not raw:
        return int(default)
    try:
        return max(64, int(raw))
    except Exception:
        return int(default)


def _num_s(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _rolling_vwap_dev(df: pd.DataFrame, window: int = 60) -> pd.Series:
    if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
        return pd.Series(0.0, index=df.index, dtype="float64")
    high, low, close = _num_s(df, "high"), _num_s(df, "low"), _num_s(df, "close").clip(lower=1e-12)
    volume = _num_s(df, "volume").clip(lower=0.0)
    tp = (high + low + close) / 3.0
    pv_sum = (tp * volume).rolling(window=window, min_periods=window).sum()
    v_sum = volume.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)
    vwap = pv_sum / (v_sum + 1e-12)
    return ((close / (vwap + 1e-12)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _flow_pressure(df: pd.DataFrame) -> pd.Series:
    smf = _num_s(df, "smart_money_flow")
    ofi = _num_s(df, "ofi_acceleration")
    ntr = _num_s(df, "net_taker_ratio")
    taker = _num_s(df, "taker_acceleration")
    cvp = _num_s(df, "cvp_volume_imbalance")
    return pd.Series(np.tanh(1.2 * smf + 0.8 * ofi + 0.6 * ntr + 0.6 * taker + 0.4 * cvp), index=df.index)


def _past_barrier_proxy(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    close = _num_s(df, "close").clip(lower=1e-12)
    ret = close.pct_change().fillna(0.0)
    rv = ret.rolling(36, min_periods=6).std().fillna(ret.abs().rolling(36, min_periods=1).mean()).fillna(0.001)
    move = close.pct_change(horizon).fillna(0.0)
    edge = move / (1.2 * rv * np.sqrt(horizon) + 1e-8)
    return pd.Series(np.tanh(edge), index=df.index).fillna(0.0)


def _past_adverse_risk(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    close = _num_s(df, "close").clip(lower=1e-12)
    low = _num_s(df, "low", close) if "low" in df.columns else close
    rolling_low = low.rolling(horizon, min_periods=1).min()
    return (1.0 - rolling_low / close).clip(lower=0.0).fillna(0.0)


def _past_reward_proxy(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    close = _num_s(df, "close").clip(lower=1e-12)
    high = _num_s(df, "high", close) if "high" in df.columns else close
    rolling_high = high.rolling(horizon, min_periods=1).max()
    return (rolling_high / close - 1.0).clip(lower=0.0).fillna(0.0)


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

        # 기본 타겟(PatchTST 전공): 과거 barrier-edge proxy
        work["y"] = _past_barrier_proxy(work, horizon=6)

        cols_with_y = ["y"] + cols
        return work[cols_with_y].ffill().fillna(0.0)

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> PredictionOutput:
        if not self.available: return PredictionOutput(np.full((1, horizon), np.nan), np.full((1, horizon), np.nan))
        prep = self._prepare_data(df).tail(self.lookback)
        if len(prep) < self.lookback: return PredictionOutput(np.full((1, horizon), np.nan), np.full((1, horizon), np.nan))

        prep_nf = prep.copy()
        prep_nf["ds"] = pd.date_range(end=pd.Timestamp.now(), periods=len(prep_nf), freq="5min")
        prep_nf["unique_id"] = "ETH"

        try:
            with SuppressOutput():
                pred_df = self.nf.predict(df=prep_nf)
            pred = pred_df[self.model_type].values[:horizon]
            return PredictionOutput(np.array([pred], dtype=np.float32), np.full((1, horizon), 0.5, dtype=np.float32))
        except Exception as e:
            err_msg = str(e)
            if "CUDA" in err_msg or "cuda" in err_msg or "NVRTC" in err_msg:
                logger.warning("🚨 CUDA error in %s. Switching to CPU and retrying...", self.name)
                try:
                    self.device = "cpu"
                    for m in self.nf.models: m.to("cpu")
                    with SuppressOutput():
                        pred_df = self.nf.predict(df=prep_nf)
                    pred = pred_df[self.model_type].values[:horizon]
                    return PredictionOutput(np.array([pred], dtype=np.float32), np.full((1, horizon), 0.5, dtype=np.float32))
                except Exception as e2:
                    logger.error("❌ %s CPU fallback also failed: %s", self.name, e2)

            logger.warning(
                "❌ %s predict failed (len=%d, lookback=%d, horizon=%d): %s",
                self.name, len(prep_nf), self.lookback, horizon, e
            )
            return PredictionOutput(np.full((1, horizon), np.nan), np.full((1, horizon), np.nan))

    def get_refined_features(self, df: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        pred = self.predict(df, horizon=1)
        if pred.median.size == 0 or np.isnan(pred.median).all():
            # Keep schema stable even when model output is unavailable.
            nan_df = pd.DataFrame(index=[0])
            tmp = self._apply_refined_batch_logic(nan_df, np.array([np.nan], dtype=np.float32), df)
            for c in tmp.columns:
                out[c] = 0.0
            return out

        val = float(pred.median[0, 0])
        one = pd.DataFrame(index=[0])
        tmp = self._apply_refined_batch_logic(one, np.array([val], dtype=np.float32), df)
        for c in tmp.columns:
            v = tmp.iloc[-1][c]
            out[c] = float(v) if np.isfinite(v) else 0.0
        return out

    def predict_batch(self, df: pd.DataFrame, chunk_size: int = 4000) -> pd.DataFrame:
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

            long_df = pd.concat(chunk_windows)

            try:
                with SuppressOutput():
                    chunk_pred_df = self.nf.predict(df=long_df)
            except Exception as e:
                err_msg = str(e)
                if ("CUDA" in err_msg or "cuda" in err_msg or "NVRTC" in err_msg) and self.device != "cpu":
                    logger.warning("🚨 Batch CUDA error in %s. Switching to CPU...", self.name)
                    self.device = "cpu"
                    for m in self.nf.models: m.to("cpu")
                    with SuppressOutput():
                        chunk_pred_df = self.nf.predict(df=long_df)
                else:
                    logger.error("❌ %s predict_batch chunk failed: %s", self.name, e)
                    continue

            try:
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
        return self._apply_refined_batch_logic(res_df, pred_s.to_numpy(dtype=np.float32), df)

    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
        edge = pd.Series(preds, dtype="float64").clip(-1.0, 1.0).fillna(0.0)
        p_up = pd.Series(1.0 / (1.0 + np.exp(-4.0 * edge)), index=edge.index)
        p_down = pd.Series(1.0 / (1.0 + np.exp(4.0 * edge)), index=edge.index)
        p_flat = (1.0 - edge.abs()).clip(0.0, 1.0)
        denom = (p_up + p_down + p_flat).replace(0.0, 1.0)
        p_up, p_down, p_flat = p_up / denom, p_down / denom, p_flat / denom
        probs = np.column_stack([p_down, p_flat, p_up])
        entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1) / np.log(3.0)

        res_df["ai_dir_edge"] = edge.values
        res_df["ai_dir_p_up"] = p_up.values
        res_df["ai_dir_p_down"] = p_down.values
        res_df["ai_dir_p_flat"] = p_flat.values
        res_df["ai_dir_entropy"] = entropy
        res_df["patchtst_median"] = edge.values
        res_df["patchtst_regime_sim"] = (1.0 - pd.Series(entropy)).clip(0.0, 1.0).values
        return res_df

class TiDEVolatilityForecaster(PatchTSTForecaster):
    _nf_model = None
    _available = False
    name = "TiDE"

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        work = super()._prepare_data(df)
        work["y"] = _past_adverse_risk(df, horizon=6).reindex(work.index).fillna(0.0)
        return work

    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
        risk = pd.Series(preds, dtype="float64").clip(lower=0.0).fillna(0.0)
        if source_df is not None and len(source_df) == len(risk):
            reward_proxy = _past_reward_proxy(source_df, horizon=6).reset_index(drop=True)
        else:
            reward_proxy = pd.Series(0.0025, index=risk.index, dtype="float64")
        reward_risk = (reward_proxy / (risk + 5e-4)).clip(0.0, 8.0).fillna(0.0)
        z = (risk - risk.rolling(200, min_periods=20).mean()) / (risk.rolling(200, min_periods=20).std() + 1e-8)
        res_df["ai_adverse_risk"] = risk.values
        res_df["ai_reward_risk"] = reward_risk.values
        res_df["ai_vol_regime_pct"] = (1.0 / (1.0 + np.exp(-z.fillna(0.0)))).values
        res_df["tide_vol_raw"] = risk.values
        res_df["tide_vol_zscore"] = z.values
        return res_df

class TimesNetCycleForecaster(PatchTSTForecaster):
    _nf_model = None
    _available = False
    name = "TimesNet"
    def __init__(self):
        # TimesNet은 NVRTC 라이브러리 충돌 이슈(CUDA 13.0 요구)로 인해 CPU 강제 모드 사용
        super().__init__(device="cpu")

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """TimesNet 전용: 60봉 롤링 VWAP 이격도를 계산하여 'y' 컬럼에 주입."""
        work = super()._prepare_data(df)

        # 필수 컬럼 확인 (원본 df에서 가져옴)
        if all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
            window = 60
            # Rolling VWAP 계산
            tp = (df['high'] + df['low'] + df['close']) / 3
            v = df['volume']
            pv_sum = (tp * v).rolling(window=window).sum()
            v_sum = v.rolling(window=window).sum().replace(0, 1)
            vwap = pv_sum / v_sum

            # 이격도 (%)를 'y' 컬럼에 덮어씀
            work['y'] = ((df['close'] / vwap) - 1.0) * 100.0
            work['y'] = pd.to_numeric(work['y'], errors="coerce").ffill().fillna(0.0)

        return work

    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
        # 이제 preds는 미래 VWAP 이격도 (%) 임
        s = pd.Series(preds, dtype="float64").fillna(0.0)
        if source_df is not None and len(source_df) == len(s):
            cur = (_rolling_vwap_dev(source_df, window=60).reset_index(drop=True) * 100.0).fillna(0.0)
        else:
            cur = s.shift(1).fillna(0.0)
        reversion = (cur.abs() - s.abs()).clip(-5.0, 5.0)
        res_df["ai_anchor_revert_prob"] = (1.0 / (1.0 + np.exp(-2.0 * reversion))).values
        res_df["ai_anchor_overheat"] = np.tanh(cur / 0.5).values
        res_df["ai_anchor_trend_escape_prob"] = (1.0 / (1.0 + np.exp(-(s.abs() - cur.abs())))).values

        # 이격도 자체가 이미 순환적인 특징을 가짐
        res_df["timesnet_cycle_sin"] = np.tanh(s / 0.5) # 0.5% 편차를 기준으로 스케일링
        res_df["timesnet_cycle_cos"] = np.sin(2 * np.pi * s / 1.0) # 1.0% 주기의 사이클 표현
        res_df["timesnet_cycle_delta"] = s.diff().fillna(0.0).values # 이격도의 변화율
        return res_df

class DLinearOFIForecaster(PatchTSTForecaster):
    _nf_model = None
    _available = False
    name = "DLinear"
    flow_cols = [
        "smart_money_flow", "ofi_acceleration", "cvp_volume_imbalance",
        "whale_retail_ratio", "net_taker_ratio", "taker_acceleration",
        "volume", "quote_volume", "taker_buy_base", "taker_buy_quote",
    ]

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()

        out = pd.DataFrame(index=work.index)
        out["y"] = _flow_pressure(work).fillna(0.0)
        for c in self.flow_cols:
            if c in work.columns:
                out[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
        return out.ffill().fillna(0.0)

    def _apply_refined_batch_logic(self, res_df: pd.DataFrame, preds: np.ndarray, source_df: pd.DataFrame | None = None) -> pd.DataFrame:
        s = pd.Series(preds, dtype="float64").clip(-1.0, 1.0).fillna(0.0)
        if source_df is not None and len(source_df) == len(s):
            cur = _flow_pressure(source_df).reset_index(drop=True).clip(-1.0, 1.0).fillna(0.0)
        else:
            cur = s.shift(1).fillna(0.0)
        exhaustion = (cur.abs() - s.abs()).clip(-1.0, 1.0)
        flip_prob = 1.0 / (1.0 + np.exp(8.0 * cur * s))
        ema = s.ewm(alpha=0.1, adjust=False).mean()
        res_df["ai_flow_pressure"] = s.values
        res_df["ai_flow_exhaustion"] = exhaustion.values
        res_df["ai_flow_flip_prob"] = flip_prob.values
        res_df["ai_flow_slope"] = ema.diff().fillna(0.0).values
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
            chunk_size = _router_chunk_size()
            logger.info("EnsembleRouter: 대량 데이터 패치 배치 추론 시작 (%d 행, chunk=%d)", len(df), chunk_size)
            for name, model in self.models.items():
                if model.available:
                    batch_res = model.predict_batch(df, chunk_size=chunk_size)
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
