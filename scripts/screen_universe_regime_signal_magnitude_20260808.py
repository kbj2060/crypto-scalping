"""Universe screen: which of the ~60 downloaded symbols carry ETH-like WITHIN-REGIME direction
signal magnitude? (2026-08-08, analysis-only, train+VAL, no OOS.)

Discriminator (from the ETH-vs-SOL comparison): per-D2-regime (bear/bull) signed VAL AUC edge of
the regime's train-selected top-K features. ETH ≈ bear +0.048 / bull +0.072 on the full panel;
SOL ≈ +0.014 / +0.049. Here every symbol -- including ETH and SOL as calibration anchors -- is
measured with the SAME reduced 15-feature set (klines-derived flow/positioning/vwap/structure
features + daily toptrader long-short metrics), so ranks are comparable even though absolute
numbers differ slightly from the full-panel run.

Per symbol: 5m klines 2024-06.., corrected TB trade-outcome labels (identical constants),
D2 regime (288-bar return ±4%), splits train ≤2025-08-31 (288-bar purge) / VAL 2025-09..12-31.
Symbols with <80k train rows or incomplete VAL coverage are skipped.
Output: ranked table (json + printed) sorted by mean(bear,bull) signed VAL edge.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_sol_5m_tripbarrier_tradeoutcome_labels_20260807 import (  # noqa: E402
    _triple_barrier_race, CUMRET_BARS, VOL_LOOKBACK, TP_MULT, SL_MULT, HORIZON_BARS,
)

KLINES_DIR = ROOT / "binance_data/klines"
METRICS_DIR = ROOT / "binance_data/metrics"
OUT_PATH = ROOT / "tmp/sol_dl_rl_survey_20260807/universe_regime_signal_screen.json"
TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
START = pd.Timestamp("2024-06-01")
TOP_K = 5
MIN_TRAIN_ROWS = 80_000


def auc_binary(x, y):
    m = np.isfinite(x)
    x, y = x[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 50 or n0 < 50:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_metrics(symbol: str) -> pd.DataFrame | None:
    files = sorted(METRICS_DIR.glob(f"{symbol}-metrics-*.zip"))
    if len(files) < 300:
        return None
    frames = []
    for p in files:
        try:
            with zipfile.ZipFile(p) as z:
                with z.open(z.namelist()[0]) as f:
                    frames.append(pd.read_csv(f, usecols=["create_time", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]))
        except Exception:
            continue
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["create_time"])
    return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)[
        ["timestamp", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]]


def build_features(df: pd.DataFrame, metrics: pd.DataFrame | None) -> tuple[np.ndarray, list[str]]:
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    vol = df["volume"].to_numpy(dtype=np.float64)
    qvol = df["quote_volume"].to_numpy(dtype=np.float64)
    tb = df["taker_buy_base"].to_numpy(dtype=np.float64)
    trades = df["trades"].to_numpy(dtype=np.float64)
    s = pd.Series
    feats = {}
    net = 2.0 * tb - vol
    for w, nm in ((48, "cvd_48"), (288, "cvd_288")):
        feats[nm] = (s(net).rolling(w).sum() / s(vol).rolling(w).sum().replace(0, np.nan)).to_numpy()
    feats["net_taker_12"] = (s(net).rolling(12).sum() / s(vol).rolling(12).sum().replace(0, np.nan)).to_numpy()
    ats = np.where(trades > 0, qvol / np.clip(trades, 1, None), np.nan)
    feats["avg_trade_size_z"] = ((s(ats) - s(ats).rolling(288).mean()) / s(ats).rolling(288).std()).to_numpy()
    for w in (24, 96, 288):
        vwap = (s(qvol).rolling(w).sum() / s(vol).rolling(w).sum().replace(0, np.nan)).to_numpy()
        feats[f"vwap_dist_{w}"] = close / vwap - 1.0
    ma96 = s(close).rolling(96).mean()
    sd96 = s(close).rolling(96).std()
    feats["mean_reversion_z"] = ((s(close) - ma96) / sd96).to_numpy()
    mx = s(high).rolling(288).max().to_numpy()
    mn = s(low).rolling(288).min().to_numpy()
    feats["donchian_pos_288"] = (close - mn) / np.clip(mx - mn, 1e-9, None)
    logc = np.log(np.clip(close, 1e-9, None))
    r72 = np.full(len(close), np.nan)
    r72[72:] = logc[72:] - logc[:-72]
    feats["r72"] = r72
    lr = np.diff(logc, prepend=logc[0])
    feats["mom12_z"] = ((s(lr).rolling(12).sum() - s(lr).rolling(12).sum().rolling(288).mean()) / s(lr).rolling(12).sum().rolling(288).std()).to_numpy()
    feats["vol_ratio"] = (s(lr).rolling(12).std() / s(lr).rolling(288).std()).to_numpy()
    if metrics is not None:
        m = pd.merge_asof(df[["timestamp"]], metrics, on="timestamp", direction="backward")
        for col, nm in (("sum_toptrader_long_short_ratio", "toptrader_ls"), ("count_long_short_ratio", "count_ls")):
            v = pd.to_numeric(m[col], errors="coerce")
            feats[f"{nm}_z"] = ((v - v.rolling(2880, min_periods=288).mean()) / v.rolling(2880, min_periods=288).std()).to_numpy()
            feats[f"{nm}_chg3d"] = (v - v.shift(864)).to_numpy()
    names = list(feats)
    return np.column_stack([feats[k] for k in names]).astype(np.float64), names


def screen_symbol(symbol: str) -> dict | None:
    path = KLINES_DIR / symbol / f"{symbol}-5m-api.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                    "quote_volume", "trades", "taker_buy_base"], low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["timestamp"] >= START].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(df) < MIN_TRAIN_ROWS:
        return None
    ts = df["timestamp"]
    if ts.iloc[-1] < VAL_END or ts.iloc[0] > pd.Timestamp("2024-09-01"):
        return None
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(np.clip(close, 1e-9, None)), prepend=np.log(close[0]))
    cumret = pd.Series(log_ret).rolling(CUMRET_BARS).sum().to_numpy()
    volb = pd.Series(cumret).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std().to_numpy()
    label, _, _ = _triple_barrier_race(open_, high, low, TP_MULT * volb, SL_MULT * volb, HORIZON_BARS)

    r288 = np.full(len(close), np.nan)
    r288[288:] = close[288:] / close[:-288] - 1.0
    regime = np.full(len(close), 1, dtype=np.int8)
    regime[r288 > 0.04] = 2
    regime[r288 < -0.04] = 0

    x, names = build_features(df, load_metrics(symbol))
    train_mask = (ts <= TRAIN_END).to_numpy()
    tr_all = np.flatnonzero(train_mask)
    if len(tr_all) < MIN_TRAIN_ROWS:
        return None
    train_mask[tr_all[-HORIZON_BARS:]] = False
    train_mask &= np.isfinite(volb)
    val_mask = ((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()
    tr_idx = np.flatnonzero(train_mask)
    v_idx = np.flatnonzero(val_mask)

    out = {"symbol": symbol, "n_rows": int(len(df)), "n_feats": len(names)}
    edges = []
    for r, rname in ((0, "bear"), (2, "bull")):
        auc_tr = np.full(len(names), np.nan)
        auc_v = np.full(len(names), np.nan)
        for w_idx, target in ((tr_idx, auc_tr), (v_idx, auc_v)):
            sub = w_idx[regime[w_idx] == r]
            a = label[sub]
            nz = a != 0
            if nz.sum() > 300:
                y = (a[nz] == 1).astype(int)
                for f in range(len(names)):
                    target[f] = auc_binary(x[sub, f][nz], y)
        dev = np.abs(np.nan_to_num(auc_tr, nan=0.5) - 0.5)
        top = np.argsort(-dev)[:TOP_K]
        s_tr = np.sign(auc_tr[top] - 0.5)
        signed = float(np.nanmean((np.nan_to_num(auc_v[top], nan=0.5) - 0.5) * s_tr))
        agree = float(np.mean(s_tr == np.sign(np.nan_to_num(auc_v[top], nan=0.5) - 0.5)))
        out[f"{rname}_signed_val_edge"] = round(signed, 4)
        out[f"{rname}_sign_agreement"] = round(agree, 2)
        out[f"{rname}_top_feature"] = names[top[0]]
        edges.append(signed)
    out["mean_signed_val_edge"] = round(float(np.mean(edges)), 4)
    return out


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    symbols = sorted(p.name for p in KLINES_DIR.iterdir() if p.is_dir())
    results, skipped = [], []
    for sym in symbols:
        try:
            rec = screen_symbol(sym)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"symbol": sym, "reason": f"error: {exc}"})
            continue
        if rec is None:
            skipped.append({"symbol": sym, "reason": "insufficient data"})
            continue
        results.append(rec)
        print(json.dumps(rec), flush=True)
    results.sort(key=lambda r: -r["mean_signed_val_edge"])
    OUT_PATH.write_text(json.dumps({"ranked": results, "skipped": skipped}, indent=2))
    print(f"\nranked {len(results)} symbols, skipped {len(skipped)}; wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
