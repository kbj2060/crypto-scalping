"""Live computation of `swing_transition_prob` for the BTC h48qual swingtransition candidate.

The promoted BTC parent bundle (btc_omega4_3head_parent72_loose_entry_quality_swingtransition_
20260806) requires one feature the shared FeatureEngineer pipeline does not produce:
`swing_transition_prob`, the Layer A LightGBM "zigzag pivot imminent within 24 bars" probability
from the 2026-08-06 session. This module reproduces that feature live, per-bar, for the whole
decision frame (the adapter requires every base_col to be finite over the full history window).

Feature recipe (must stay byte-identical to the offline chain that built the candidate's
training feature -- see scripts/train_btc_5m_layerA_swing_transition_save_model_20260807.py,
which verified the saved model regenerates the offline parquet bit-exactly):
- 96 of the 110 Layer A inputs are already in the live frame (causalfix panel cols).
- 10 mtf1h_* cols: 1h resample of the raw 5m buffer -> compute_features + causal trend-scan
  (exact functions imported from the offline builders), made available at bar_close (+1h),
  merge_asof backward onto the 5m frame.
- 4 dvol_btc* cols: Deribit DVOL hourly index (public REST, no credentials -- same deliberate
  choice as scripts/download_deribit_dvol_20260804.py), features computed on the hourly series,
  made available at +1h, merge_asof backward. Hourly cache is refreshed incrementally; only
  fully-closed hourly bars can ever be selected by the availability shift.

Fail-fast: any missing input column, non-finite output, insufficient DVOL history, or Deribit
fetch failure raises RuntimeError -- the shadow loop's per-asset exception handling logs it and
skips the refresh rather than trading on a silently degraded feature.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_1h_trendscan_dataset_btc_full_20260801 import (  # noqa: E402
    compute_features as _compute_1h_features,
    resample_1h as _resample_1h,
)
from build_btc_1h_trendscan_causal_fix_20260804 import (  # noqa: E402
    TS_WINDOWS as _TS_WINDOWS,
    _trend_scan_causal,
)

FEATURE_NAME = "swing_transition_prob"
DEFAULT_MODEL_PATH = ROOT / "data/ensemble/supervised/btc_swing_transition_layerA_20260807/layerA_lgbm.pkl"

_MTF1H_COLS = ["ts_t_value", "ts_opt_L", "rsi_14", "rvol_6", "rvol_12", "rvol_24", "rvol_48",
               "atr_pct", "bb_width", "bb_pos"]
_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
_DVOL_RESOLUTION = "3600"
# dvol_btc_pctrank_720h needs 720h of history, roc_168h needs 168h; keep a cushion on top of the
# frame span (<=1200 5m bars = 100h). 1100h of hourly bars covers everything with margin.
_DVOL_MIN_HOURS = 1100
_RAW_5M_COLS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]
# The bot's live kline buffer is causally TRIMMED to ancillary-complete rows (~500 bars right
# after boot), far short of what the 1h overlay needs: 48h of mature 1h windows + the frame span
# + EWM (rsi_14) settling. Below this row count the provider self-fetches raw 5m klines from the
# Binance futures public REST instead (same public data the training CSVs were built from).
_MIN_RAW_5M_BARS = 3000
_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
_KLINES_TARGET_BARS = 4032  # 14 days of 5m -- puts the rsi EWM truncation error below ~1e-7


class BtcSwingTransitionLiveFeature:
    """Appends `swing_transition_prob` to a BTC 5m decision frame, per-bar and causally."""

    def __init__(self, *, model_path: str | Path = DEFAULT_MODEL_PATH,
                 dvol_fetcher=None) -> None:
        with Path(model_path).open("rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.feature_columns: list[str] = list(payload["feature_columns"])
        if len(self.feature_columns) != 110:
            raise RuntimeError(
                f"swing_transition layerA: expected 110 feature columns, got {len(self.feature_columns)}")
        # injectable for offline parity tests; defaults to the real REST fetch
        self._dvol_fetcher = dvol_fetcher or self._fetch_dvol_rows
        self._dvol_cache: pd.DataFrame | None = None
        self._klines_cache: pd.DataFrame | None = None

    # ---------------- raw 5m klines (self-fetched) ----------------
    def _fetch_klines(self, start_ms: int, end_ms: int) -> list[list]:
        """Forward pagination: with startTime set, fapi returns the EARLIEST bars of the range
        ascending, so walk the start cursor forward until the range is covered."""
        rows: list[list] = []
        cursor_start = start_ms
        for _ in range(10):
            if cursor_start >= end_ms:
                break
            resp = requests.get(_KLINES_URL, params={
                "symbol": "BTCUSDT", "interval": "5m", "limit": 1500,
                "startTime": cursor_start}, timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(f"Binance klines fetch failed: HTTP {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            if not data:
                break
            rows.extend(data)
            last_open = int(data[-1][0])
            if len(data) < 1500 or last_open >= end_ms:
                break
            cursor_start = last_open + 300_000
            time.sleep(0.1)
        if not rows:
            raise RuntimeError("Binance klines fetch returned no rows")
        return rows

    def _refresh_klines_cache(self) -> pd.DataFrame:
        now_ms = int(time.time() * 1000)
        if self._klines_cache is None:
            start = now_ms - (_KLINES_TARGET_BARS + 24) * 300_000
        else:
            start = int(self._klines_cache["timestamp"].max().value // 10**6) - 600_000
        rows = self._fetch_klines(start, now_ms)
        fresh = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                              for r in rows], columns=["open_ms", "open", "high", "low", "close", "volume"])
        fresh["timestamp"] = pd.to_datetime(fresh["open_ms"], unit="ms")
        fresh = fresh.drop(columns=["open_ms"])
        if self._klines_cache is not None:
            keep = self._klines_cache[self._klines_cache["timestamp"] < fresh["timestamp"].min()]
            fresh = pd.concat([keep, fresh], ignore_index=True)
        fresh = fresh.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        # drop the still-forming 5m bar so only closed bars feed the 1h resample
        cutoff = pd.to_datetime((now_ms // 300_000) * 300_000, unit="ms")
        fresh = fresh[fresh["timestamp"] < cutoff]
        self._klines_cache = fresh.tail(_KLINES_TARGET_BARS + 288).reset_index(drop=True)
        if len(self._klines_cache) < _MIN_RAW_5M_BARS:
            raise RuntimeError(f"self-fetched klines too short: {len(self._klines_cache)}")
        return self._klines_cache

    # ---------------- DVOL ----------------
    def _fetch_dvol_rows(self, start_ms: int, end_ms: int) -> list[list]:
        """Backward-paginating fetch (Deribit fills from the END of the range; 'continuation' is
        the next EARLIER end_timestamp -- same empirical behavior as the 2026-08-04 downloader)."""
        rows: list[list] = []
        cursor_end = end_ms
        for _ in range(20):
            if cursor_end <= start_ms:
                break
            resp = requests.get(_DVOL_URL, params={
                "currency": "BTC", "start_timestamp": start_ms, "end_timestamp": cursor_end,
                "resolution": _DVOL_RESOLUTION}, timeout=15)
            if resp.status_code != 200:
                raise RuntimeError(f"Deribit DVOL fetch failed: HTTP {resp.status_code} {resp.text[:200]}")
            result = resp.json().get("result") or {}
            data = result.get("data") or []
            if not data:
                break
            rows.extend(data)
            continuation = result.get("continuation")
            if continuation is None or int(continuation) >= cursor_end:
                break
            cursor_end = int(continuation)
            time.sleep(0.1)
        if not rows:
            raise RuntimeError("Deribit DVOL fetch returned no rows")
        return rows

    def _refresh_dvol_cache(self, now_utc: pd.Timestamp) -> pd.DataFrame:
        end_ms = int(now_utc.value // 10**6) + 3_600_000
        if self._dvol_cache is None:
            start = now_utc - pd.Timedelta(hours=_DVOL_MIN_HOURS + 8)
        else:
            # refetch the last two cached hours too so a previously-partial bar gets finalized
            start = self._dvol_cache["timestamp"].max() - pd.Timedelta(hours=2)
        rows = self._dvol_fetcher(int(start.value // 10**6), end_ms)
        fresh = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close"])
        fresh["timestamp"] = pd.to_datetime(fresh["timestamp_ms"], unit="ms")
        fresh = fresh[["timestamp", "close"]].sort_values("timestamp")
        if self._dvol_cache is not None:
            keep = self._dvol_cache[self._dvol_cache["timestamp"] < fresh["timestamp"].min()]
            fresh = pd.concat([keep, fresh], ignore_index=True)
        fresh = fresh.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
        cutoff = now_utc - pd.Timedelta(hours=_DVOL_MIN_HOURS + 48)
        self._dvol_cache = fresh[fresh["timestamp"] >= cutoff].reset_index(drop=True)
        return self._dvol_cache

    def _dvol_features(self, now_utc: pd.Timestamp) -> pd.DataFrame:
        """Exact mirror of eval_btc_5m_layerA_layerB_20260806.build_dvol_features(), on the live
        hourly cache: availability shift first, then rocs/pctrank on the shifted series."""
        hourly = self._refresh_dvol_cache(now_utc)
        if len(hourly) < _DVOL_MIN_HOURS:
            raise RuntimeError(f"insufficient DVOL history: {len(hourly)} < {_DVOL_MIN_HOURS} hours")
        df = hourly.copy()
        df["available_at"] = df["timestamp"] + pd.Timedelta(hours=1)
        df = df[["available_at", "close"]].rename(columns={"available_at": "timestamp", "close": "dvol_btc"}).sort_values("timestamp")
        df["dvol_btc_roc_24h"] = df["dvol_btc"].pct_change(24)
        df["dvol_btc_roc_168h"] = df["dvol_btc"].pct_change(168)
        df["dvol_btc_pctrank_720h"] = df["dvol_btc"].rolling(720, min_periods=180).apply(
            lambda x: (x.iloc[-1] >= x).mean(), raw=False)
        return df

    # ---------------- mtf1h ----------------
    def _mtf1h_overlay(self, raw_5m: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in _RAW_5M_COLS if c in raw_5m.columns]
        missing = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c not in cols]
        if missing:
            raise RuntimeError(f"swing_transition mtf1h: raw 5m buffer missing {missing}")
        src = raw_5m[cols].copy()
        src["timestamp"] = pd.to_datetime(src["timestamp"]).astype("datetime64[ns]")
        r1h = _resample_1h(src)
        feats = _compute_1h_features(r1h)
        logc = np.log(np.maximum(r1h["close"].to_numpy(dtype=np.float64), 1e-12))
        t_vals, opt_l, _betas = _trend_scan_causal(logc, np.array(sorted(_TS_WINDOWS), dtype=np.int32))
        feats["ts_t_value"] = t_vals.astype(np.float32)
        feats["ts_opt_L"] = opt_l.astype(np.int16)
        overlay = feats[["timestamp"] + _MTF1H_COLS].copy()
        overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
        overlay["available_at"] = overlay["timestamp"] + pd.Timedelta(hours=1)
        overlay = overlay.rename(columns={c: f"mtf1h_{c}" for c in _MTF1H_COLS}).drop(columns=["timestamp"])
        return overlay.sort_values("available_at").reset_index(drop=True)

    # ---------------- main ----------------
    def append(self, frame: pd.DataFrame, *, raw_5m: pd.DataFrame | None = None) -> pd.DataFrame:
        if frame.empty:
            raise RuntimeError("swing_transition: empty decision frame")
        if raw_5m is None or len(raw_5m) < _MIN_RAW_5M_BARS:
            # bot buffers are causally trimmed right after boot (~500 bars) -- too short for the
            # 1h overlay's windows; fall back to the provider's own full-depth kline cache
            raw_5m = self._refresh_klines_cache()
        out = frame.copy()
        # live frames arrive as datetime64[us]; merge_asof needs one dtype everywhere, so pin ns
        ts = pd.to_datetime(out["timestamp"]).astype("datetime64[ns]").reset_index(drop=True)
        work = pd.DataFrame({"timestamp": ts})

        in_frame_cols = [c for c in self.feature_columns
                         if not c.startswith("mtf1h_") and not c.startswith("dvol_btc")]
        missing = [c for c in in_frame_cols if c not in out.columns]
        if missing:
            raise RuntimeError(f"swing_transition: frame missing layerA inputs {missing[:10]}")
        for c in in_frame_cols:
            work[c] = pd.to_numeric(out[c], errors="coerce").to_numpy()

        overlay = self._mtf1h_overlay(raw_5m)
        work = pd.merge_asof(work.sort_values("timestamp"), overlay,
                             left_on="timestamp", right_on="available_at", direction="backward")
        if work["available_at"].isna().any():
            raise RuntimeError("swing_transition: mtf1h overlay does not cover the frame start")
        work = work.drop(columns=["available_at"])

        dvol = self._dvol_features(ts.iloc[-1])
        work = pd.merge_asof(work.sort_values("timestamp"), dvol, on="timestamp", direction="backward")

        x = work[self.feature_columns]
        prob = self.model.predict_proba(x)[:, 1]
        if not np.all(np.isfinite(prob)):
            raise RuntimeError("swing_transition: non-finite layerA probability")
        out[FEATURE_NAME] = prob
        return out
