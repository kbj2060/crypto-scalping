"""PatchTST-only live forecaster for the selected M7 + DSAC pipeline."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

PredictionOutput = namedtuple("PredictionOutput", ["median", "confidence"])

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for _log_name in [
    "pytorch_lightning",
    "pytorch_lightning.utilities.rank_zero",
    "lightning",
    "lightning.pytorch",
    "lightning.fabric",
    "lightning_fabric",
    "neuralforecast",
    "nixtla",
]:
    logging.getLogger(_log_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


@dataclass
class ForecastOutput:
    quantiles: np.ndarray
    median: np.ndarray
    confidence: np.ndarray
    model_name: str


class PatchTSTForecaster:
    """Runtime PatchTST wrapper backed by the local NeuralForecast model pack."""

    _nf_model = None
    _available = False
    name = "PatchTST"

    def __init__(self):
        self.model_type = "PatchTST"
        self.exog_cols = [
            "session_us",
            "hour_cos",
            "cvp_poc_dist",
            "cvp_volume_imbalance",
            "fvg_dist",
            "breakout_strength",
            "oi_change_rate",
            "ofti",
            "kel",
            "mta_funding",
            "svps",
        ]
        if PatchTSTForecaster._nf_model is None:
            self._load_model_pack()
        self.nf = PatchTSTForecaster._nf_model
        self.available = PatchTSTForecaster._available

    @classmethod
    def _load_model_pack(cls) -> None:
        try:
            from neuralforecast import NeuralForecast

            model_dir = os.path.join(os.getcwd(), "data", "nf")
            if not os.path.exists(model_dir):
                logger.warning("PatchTST model folder not found: %s", model_dir)
                cls._available = False
                return
            logging.disable(logging.INFO)
            try:
                with SuppressOutput():
                    cls._nf_model = NeuralForecast.load(path=model_dir)
            finally:
                logging.disable(logging.NOTSET)
            cls._available = True
            logger.info("PatchTST NeuralForecast pack loaded")
        except Exception as e:
            cls._available = False
            cls._nf_model = None
            logger.warning("PatchTST load failed: %s", e)

    @staticmethod
    def _empty_output(horizon: int) -> PredictionOutput:
        return PredictionOutput(
            median=np.zeros((1, horizon), dtype=np.float32),
            confidence=np.zeros((1, horizon), dtype=np.float32),
        )

    def predict(self, df: pd.DataFrame, horizon: int = 6) -> PredictionOutput:
        if not self.available or self.nf is None:
            raise RuntimeError("PatchTST unavailable: NeuralForecast model pack is not loaded")
        if len(df) < 256:
            raise RuntimeError(f"PatchTST requires at least 256 rows, got {len(df)}")

        work = df.copy()
        try:
            smf = pd.to_numeric(work["smart_money_flow"], errors="coerce")
            wc = pd.to_numeric(work["whale_conviction"], errors="coerce")
            aiz = pd.to_numeric(work["amihud_illiquidity_z"], errors="coerce")
            work["ofti"] = np.tanh(smf * wc * (aiz.abs() + 1.0) * 3.0)

            if "last_funding_rate" in work.columns:
                fr_base = pd.to_numeric(work["last_funding_rate"], errors="coerce")
                for _win, _col in ((12, "funding_roc_12"), (48, "funding_roc_48"), (288, "funding_roc_288")):
                    if _col not in work.columns:
                        shifted = fr_base.shift(_win)
                        work[_col] = ((fr_base - shifted) / (shifted.abs().clip(lower=1e-4) + 1e-8)).clip(-10.0, 10.0)

            oic = pd.to_numeric(work["oi_change_rate"], errors="coerce")
            gkv = pd.to_numeric(work["garman_klass_vol"], errors="coerce")
            fp = pd.to_numeric(work["funding_pressure"], errors="coerce")
            kel_raw = oic / (gkv + 1e-6) * np.sign(fp)
            rm = kel_raw.rolling(288, min_periods=1).mean()
            rs = kel_raw.rolling(288, min_periods=1).std().fillna(1e-8) + 1e-8
            work["kel"] = np.tanh((kel_raw - rm) / rs * 0.5)

            fr12 = pd.to_numeric(work["funding_roc_12"], errors="coerce")
            fr48 = pd.to_numeric(work["funding_roc_48"], errors="coerce")
            fr288 = pd.to_numeric(work["funding_roc_288"], errors="coerce")
            fabs = pd.to_numeric(work["funding_abs"], errors="coerce").clip(lower=1e-8)
            sqp = pd.to_numeric(work["squeeze_power"], errors="coerce")
            sq_mean = sqp.rolling(288, min_periods=1).mean()
            sq_std = sqp.rolling(288, min_periods=1).std().fillna(1e-8) + 1e-8
            sq_z = (sqp - sq_mean) / sq_std
            w_roc = 0.5 * fr12 + 0.3 * fr48 + 0.2 * fr288
            work["mta_funding"] = ((w_roc / fabs) * np.tanh(sq_z)).clip(-3.0, 3.0) / 3.0

            cpd = pd.to_numeric(work["cvp_poc_dist"], errors="coerce")
            cvi = pd.to_numeric(work["cvp_volume_imbalance"], errors="coerce")
            cvw = pd.to_numeric(work["cvp_vah_val_width"], errors="coerce")
            work["svps"] = np.tanh(2.0 * cpd * cvi * np.exp(-cvw.clip(0.0, 5.0)))

            df_nf = work[["close"] + self.exog_cols].tail(256).copy()
            df_nf.ffill(inplace=True)
            nan_cols = [c for c in df_nf.columns if bool(df_nf[c].isna().any())]
            if nan_cols:
                raise ValueError(f"PatchTST NaN remains after ffill: {','.join(nan_cols)}")

            df_nf["ds"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_nf), freq="5min")
            df_nf["unique_id"] = "ETH"
            df_nf.rename(columns={"close": "y"}, inplace=True)

            with SuppressOutput():
                pred_df = self.nf.predict(df=df_nf)

            pred = np.asarray(pred_df[self.model_type].values[:horizon], dtype=np.float32)
            if len(pred) < horizon:
                pred = np.pad(pred, (0, horizon - len(pred)), mode="edge")
            conf = np.ones((1, horizon), dtype=np.float32) * 0.5
            return PredictionOutput(median=np.array([pred], dtype=np.float32), confidence=conf)
        except Exception as e:
            logger.warning("PatchTST predict failed: %s", e)
            raise
