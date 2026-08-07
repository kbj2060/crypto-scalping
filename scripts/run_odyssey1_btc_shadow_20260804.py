"""
Odyssey1 -- BTC shadow-monitoring bot (Layer1 CUSUM gate -> Layer2/3 LightGBM
quality/direction -> Layer4 RL exit-timing -> Layer5 quantile-uncertainty
sizing/leverage), first live shadow test of this session's new BTC
architecture. Modeled directly on Tau1's non-executing shadow pattern
(project memory project-tau1-name-spec-20260801) and this project's
SCALP_REUSE_MODES dashboard contract (dashboard/server.py).

SAFETY: this process NEVER submits orders. It only computes signals from
public market data and writes a state file for dashboard display, exactly
like Tau1 (activation_allowed=False, order_submission_supported=False,
enforced in the written state and checked by dashboard/server.py's
_require_scalp_contract).

Live data: no BTC-native live feature stream exists yet (unlike ETH's
decision_feature_snapshot.jsonl), so this script fetches raw market data
directly from Binance USDM futures public REST endpoints every poll,
reusing the SAME fetch functions/conventions as
scripts/build_eth_micro_scalp_v3_feature_stream_20260718.py, and reuses
the identical FeatureEngineer/1h-overlay/regime3 HMM code this session's
offline pipeline was built and audited with -- NOT reimplemented, to avoid
live/backtest parity drift.

STATUS: first-pass live wiring, NOT YET live/backtest-parity validated
against the offline Fresh-Forward walk-forward result (see
docs/backtest_runtime_native_parity_contract.md convention) -- treat like
Tau1: "not promotion-ready", shadow-monitoring only.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from features.engineering import FeatureEngineer  # noqa: E402
from build_1h_trendscan_dataset_btc_full_20260801 import resample_1h as resample_1h_raw  # noqa: E402
from build_1h_trendscan_dataset_btc_full_20260801 import compute_features as compute_1h_overlay  # noqa: E402
from experiment_regime3_current_hmm_wide24_20260529 import _transform as regime3_transform  # noqa: E402
from build_omega1_2_triple_barrier_labels_btc_20260708 import _atr_price_move  # noqa: E402
from build_btc_cusum_trailing_final_model_20260803 import HARD_SL_MULT, HARD_SL_MIN, ENTRY_THRESHOLD  # noqa: E402
from train_btc_exit_stopping_rl_20260803 import trail_threshold, STATE_DIM  # noqa: E402
from ensemble.train_rl_dsac_unified_2025 import DSACAgent  # noqa: E402
import joblib  # noqa: E402

MODEL_ID = "btc_odyssey1_shadow_v1_20260804"
STATE_SCHEMA = "micro_scalp_reuse.shadow_bot_step.v1"
SUMMARY_SCHEMA = "micro_scalp_reuse.shadow_bot.v1"
ASSET, SYMBOL, CROSS_SYMBOL = "btc", "BTCUSDT", "ETHUSDT"
MODE = "btc_odyssey1"

LIVE_DIR = ROOT / "data/live"
STATE_PATH = LIVE_DIR / "btc_odyssey1_shadow_state.json"
POLL_SECONDS = 300  # matches 5m bar cadence

BASE_URL = "https://fapi.binance.com"
PUBLIC_ENDPOINTS = {
    "klines": "/fapi/v1/klines",
    "funding": "/fapi/v1/fundingRate",
    "open_interest": "/futures/data/openInterestHist",
    "top_position": "/futures/data/topLongShortPositionRatio",
    "global_account": "/futures/data/globalLongShortAccountRatio",
}
KLINE_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "close_time",
                  "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore")
HISTORY_DAYS = 12  # enough for all rolling windows (max 288 5m-bars = 1 day) plus warmup slack
CAUSAL_CONTEXT = pd.Timedelta(hours=12)

MODEL_DIR = ROOT / "data/ensemble/supervised"
CKPT_DIR = ROOT / "data/ensemble/ckpt"
REGIME3_JOBLIB = MODEL_DIR / "btc_regime3_current_hmm_sensitive_wide24_20260708/regime3_current_sensitive_hmm_wide24_2024.joblib"

# Layer 5 calibration (final MDD<=10% design, see this session's leverage sizing work)
LEV_MIN, LEV_MAX, A_LEV = 2.0, 16.0, 2.8
MARGIN_MIN, MARGIN_MAX, MARGIN_BASE, A_MARGIN = 0.05, 0.25, 0.12, 2.0


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _request_json(session, endpoint, params, retries=4):
    if endpoint not in PUBLIC_ENDPOINTS.values():
        raise RuntimeError(f"endpoint outside allowlist: {endpoint}")
    for attempt in range(retries):
        r = session.get(BASE_URL + endpoint, params=params, timeout=30)
        if r.status_code in (418, 429):
            time.sleep(max(float(r.headers.get("Retry-After", 1.0)), 1.0) * (attempt + 1))
            continue
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected response: {payload}")
        return payload
    raise RuntimeError(f"exhausted retries: {endpoint}")


def fetch_klines(session, symbol, interval, start, end):
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    rows, cursor = [], start_ms
    while cursor <= end_ms:
        batch = _request_json(session, PUBLIC_ENDPOINTS["klines"],
                               {"symbol": symbol, "interval": interval, "startTime": cursor,
                                "endTime": end_ms, "limit": 1500})
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 1
        if nxt <= cursor:
            raise RuntimeError("kline pagination stalled")
        cursor = nxt
        if len(batch) < 1500:
            break
        time.sleep(0.05)
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        raise RuntimeError(f"no {symbol} {interval} klines returned")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["trades"] = pd.to_numeric(frame["trades"], errors="coerce")
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
    return frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def _fetch_series_backward(session, endpoint, symbol, start, end):
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    cursor_end = int(end.tz_localize("UTC").timestamp() * 1000)
    rows = []
    while cursor_end >= start_ms:
        batch = _request_json(session, endpoint, {"symbol": symbol, "period": "5m",
                               "startTime": start_ms, "endTime": cursor_end, "limit": 500})
        if not batch:
            break
        rows.extend(batch)
        first_ts = min(int(row["timestamp"]) for row in batch)
        if first_ts <= start_ms:
            break
        nxt = first_ts - 1
        if nxt >= cursor_end:
            raise RuntimeError(f"metric pagination stalled: {endpoint}")
        cursor_end = nxt
        time.sleep(0.05)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    uniq = {int(r["timestamp"]): r for r in rows if start_ms <= int(r["timestamp"]) <= end_ms}
    return [uniq[k] for k in sorted(uniq)]


def fetch_metrics(session, start, end):
    oi = pd.DataFrame(_fetch_series_backward(session, PUBLIC_ENDPOINTS["open_interest"], SYMBOL, start, end))
    if oi.empty:
        raise RuntimeError("open-interest history empty")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(oi["timestamp"], unit="ms"),
                           "sum_open_interest_value": pd.to_numeric(oi["sumOpenInterestValue"])}).sort_values("timestamp")
    for endpoint, target in [("top_position", "sum_toptrader_long_short_ratio"),
                              ("global_account", "count_long_short_ratio")]:
        raw = pd.DataFrame(_fetch_series_backward(session, PUBLIC_ENDPOINTS[endpoint], SYMBOL, start, end))
        aligned = pd.DataFrame({"timestamp": pd.to_datetime(raw["timestamp"], unit="ms"),
                                 target: pd.to_numeric(raw["longShortRatio"])}).sort_values("timestamp")
        frame = pd.merge_asof(frame, aligned, on="timestamp", direction="backward", tolerance=pd.Timedelta(minutes=5))
    required = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio"]
    if frame[required].isna().any().any():
        raise RuntimeError("causal metric join has missing values")
    return frame


def fetch_funding(session, start, end):
    start_ms = int(start.tz_localize("UTC").timestamp() * 1000)
    end_ms = int(end.tz_localize("UTC").timestamp() * 1000)
    rows, cursor = [], start_ms
    while cursor <= end_ms:
        batch = _request_json(session, PUBLIC_ENDPOINTS["funding"],
                               {"symbol": SYMBOL, "startTime": cursor, "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            raise RuntimeError("funding pagination stalled")
        cursor = nxt
        if len(batch) < 1000:
            break
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("funding history empty")
    out = pd.DataFrame({"timestamp": pd.to_datetime(frame["fundingTime"], unit="ms"),
                         "last_funding_rate": pd.to_numeric(frame["fundingRate"])})
    return out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")


def build_engineered_frame(session, end):
    start = (end - pd.Timedelta(days=HISTORY_DAYS)).floor("5min")
    btc = fetch_klines(session, SYMBOL, "5m", start, end)
    eth = fetch_klines(session, CROSS_SYMBOL, "5m", start, end)
    ctx_start = start - CAUSAL_CONTEXT
    metrics = fetch_metrics(session, ctx_start, end)
    funding = fetch_funding(session, ctx_start, end)
    btc = pd.merge_asof(btc.sort_values("timestamp"), metrics, on="timestamp", direction="backward", tolerance=pd.Timedelta(hours=9))
    btc = pd.merge_asof(btc.sort_values("timestamp"), funding, on="timestamp", direction="backward")
    req = ["sum_open_interest_value", "sum_toptrader_long_short_ratio", "count_long_short_ratio", "last_funding_rate"]
    if btc[req].isna().any().any():
        raise RuntimeError("causal metric/funding join has missing values on primary frame")
    fe = FeatureEngineer()
    engineered = fe.process(btc, eth[["timestamp", "close", "volume", "quote_volume"]].rename(
        columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"}))
    engineered["timestamp"] = pd.to_datetime(engineered["timestamp"])
    return engineered.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True), btc


def build_1h_overlay(btc_raw: pd.DataFrame) -> pd.DataFrame:
    hourly = resample_1h_raw(btc_raw)
    overlay = compute_1h_overlay(hourly)
    cols = ["ts_action", "ts_t_value", "ts_opt_L", "rsi_14", "rvol_6", "rvol_12", "rvol_24", "rvol_48",
            "atr_pct", "bb_width", "bb_pos"]
    overlay = overlay[["timestamp"] + cols].rename(columns={c: f"mtf1h_{c}" for c in cols})
    overlay["available_at"] = overlay["timestamp"] + pd.Timedelta(hours=1)
    return overlay.drop(columns=["timestamp"]).sort_values("available_at")


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"position": None, "cusum": {"s_pos": 0.0, "s_neg": 0.0}, "closed_trades": [], "equity": 1.0}


def main():
    print(f"Odyssey1 BTC shadow starting, model_id={MODEL_ID}", flush=True)
    with open(MODEL_DIR / "btc_cusum_trailing_final_long.pkl", "rb") as f:
        long_model = pickle.load(f)
    with open(MODEL_DIR / "btc_cusum_trailing_final_short.pkl", "rb") as f:
        short_model = pickle.load(f)
    feat_cols = long_model.feature_name_
    quantile_models = {}
    for name in ("long_q10", "long_q50", "long_q90", "short_q10", "short_q50", "short_q90"):
        with open(MODEL_DIR / f"btc_layer5_sizing_{name}.pkl", "rb") as f:
            quantile_models[name] = pickle.load(f)
    med, iqr = np.load(ROOT / "tmp/btc_layer5_train_score_stats.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DSACAgent(state_dim=STATE_DIM, hidden_dim=128, n_quantiles=25, device=device)
    agent.actor.load_state_dict(torch.load(CKPT_DIR / "btc_exit_stopping_rl_actor_seed270705_20260803.pth", map_location=device))
    agent.actor.eval()

    regime3_payload = joblib.load(REGIME3_JOBLIB)
    session = requests.Session()
    state = load_state()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    while True:
        try:
            now = pd.Timestamp.utcnow().tz_localize(None).floor("5min")
            engineered, btc_raw = build_engineered_frame(session, now)
            overlay = build_1h_overlay(btc_raw)
            regime3, _ = regime3_transform(regime3_payload, engineered)
            frame = pd.merge_asof(engineered.sort_values("timestamp"), overlay,
                                   left_on="timestamp", right_on="available_at", direction="backward").drop(columns=["available_at"])
            frame = frame.merge(regime3[["timestamp", "regime3_current_sensitive_wide24_bull_prob",
                                          "regime3_current_sensitive_wide24_bear_prob",
                                          "regime3_current_sensitive_wide24_confidence"]], on="timestamp", how="left")
            frame = frame.dropna(subset=feat_cols).reset_index(drop=True)
            if len(frame) < 2:
                raise RuntimeError("insufficient rows after feature assembly")

            close = frame["close"].to_numpy(dtype=np.float64)
            high = frame["high"].to_numpy(dtype=np.float64)
            low = frame["low"].to_numpy(dtype=np.float64)
            atr = _atr_price_move(frame)
            i = len(frame) - 2  # last fully-closed decision bar (leave 1 bar margin for any late revision)
            logret_i = float(np.log(close[i]) - np.log(close[i - 1])) if i > 0 else 0.0

            pos = state.get("position")
            step_log = {}
            if pos is not None:
                bar_idx = pos["entry_frame_idx"] + pos["bars_held"] + 1
                if bar_idx < len(frame):
                    a = agent.act(np.array(pos["last_state"], dtype=np.float32), deterministic=True)
                    mult = 0.5 + (float(np.clip(a.reshape(-1)[0], -1, 1)) + 1.0) * 0.75
                    entry_price = pos["entry_price"]
                    side = pos["side"]
                    sl_move = pos["sl_move"]
                    hi, lo, close_px = high[bar_idx], low[bar_idx], close[bar_idx]
                    sl_hit = (lo <= entry_price * (1 - sl_move)) if side == 1 else (hi >= entry_price * (1 + sl_move))
                    extreme = max(pos["extreme"], hi) if side == 1 else min(pos["extreme"], lo)
                    trail = trail_threshold(float(atr[bar_idx]), mult)
                    trail_hit = (close_px <= extreme * (1 - trail)) if side == 1 else (close_px >= extreme * (1 + trail))
                    pnl = (close_px / entry_price - 1.0) if side == 1 else (1.0 - close_px / entry_price)
                    pos["bars_held"] += 1
                    pos["extreme"] = extreme
                    terminal = sl_hit or trail_hit or pos["bars_held"] >= 288
                    if terminal:
                        net = pnl - 0.0014
                        state["equity"] *= (1 + max(net * pos["notional"], -pos["margin_fraction"]))
                        state["closed_trades"].append({"entry_ts": pos["entry_ts"], "exit_ts": str(frame["timestamp"].iloc[bar_idx]),
                                                        "side": side, "net": net, "notional": pos["notional"]})
                        state["position"] = None
                        step_log["exit"] = True
                    else:
                        pos["last_state"] = [pnl, pos["bars_held"] / 288, max(0.0, pnl), min(0.0, pnl),
                                              pos["conviction"], pos.get("ts_t", 0.0) / 5.0, pos.get("ts_action", 0.0),
                                              *pos.get("regime", [0.0, 0.0, 0.0]), 0.0, 0.0, 0.0]
                        state["position"] = pos
            else:
                s_pos = state["cusum"]["s_pos"] + max(0.0, logret_i)
                s_neg = min(0.0, state["cusum"]["s_neg"] + logret_i)
                thresh = max(float(atr[i]), 0.001) * 2.0
                fired = (state["cusum"]["s_pos"] + max(0.0, logret_i)) > thresh or s_neg < -thresh
                s_pos_acc = max(0.0, state["cusum"]["s_pos"] + logret_i)
                s_neg_acc = min(0.0, state["cusum"]["s_neg"] + logret_i)
                if s_pos_acc > thresh or s_neg_acc < -thresh:
                    x = frame.loc[[i], feat_cols]
                    pl = float(long_model.predict(x)[0])
                    ps = float(short_model.predict(x)[0])
                    side, conv = (0, 0.0)
                    if pl >= ENTRY_THRESHOLD and pl >= ps:
                        side, conv = 1, pl
                    elif ps >= ENTRY_THRESHOLD:
                        side, conv = 2, ps
                    if side != 0:
                        q10 = quantile_models[f"{'long' if side==1 else 'short'}_q10"].predict(x)[0]
                        q50 = quantile_models[f"{'long' if side==1 else 'short'}_q50"].predict(x)[0]
                        q90 = quantile_models[f"{'long' if side==1 else 'short'}_q90"].predict(x)[0]
                        z = (q50 / (q90 - q10 + 1e-6) - med) / iqr
                        margin_fraction = float(np.clip(MARGIN_BASE + (MARGIN_MAX - MARGIN_MIN) * (sigmoid(A_MARGIN * z) - 0.5), MARGIN_MIN, MARGIN_MAX))
                        leverage = float(LEV_MIN + (LEV_MAX - LEV_MIN) * sigmoid(A_LEV * z))
                        entry_i = i + 1
                        if entry_i < len(frame):
                            entry_price = float(frame["open"].iloc[entry_i])
                            sl_move = max(HARD_SL_MIN, HARD_SL_MULT * float(atr[i]))
                            state["position"] = {
                                "side": side, "entry_price": entry_price, "entry_frame_idx": i,
                                "bars_held": 0, "extreme": entry_price, "sl_move": sl_move,
                                "conviction": conv, "margin_fraction": margin_fraction, "leverage": leverage,
                                "notional": margin_fraction * leverage,
                                "entry_ts": str(frame["timestamp"].iloc[entry_i]),
                                "ts_t": float(frame["mtf1h_ts_t_value"].iloc[i]),
                                "ts_action": float(frame["mtf1h_ts_action"].iloc[i]),
                                "regime": [float(frame["regime3_current_sensitive_wide24_bull_prob"].iloc[i]),
                                           float(frame["regime3_current_sensitive_wide24_bear_prob"].iloc[i]),
                                           float(frame["regime3_current_sensitive_wide24_confidence"].iloc[i])],
                                "last_state": [0.0, 0.0, 0.0, 0.0, conv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            }
                            step_log["entry"] = side
                    state["cusum"] = {"s_pos": 0.0, "s_neg": 0.0}
                else:
                    state["cusum"] = {"s_pos": s_pos_acc, "s_neg": s_neg_acc}

            summary = {
                "schema_version": SUMMARY_SCHEMA, "model_id": MODEL_ID, "asset": ASSET, "symbol": SYMBOL,
                "mode": MODE, "activation_allowed": False, "order_submission_supported": False,
                "equity_multiple": state["equity"], "n_closed_trades": len(state["closed_trades"]),
                "position_open": state["position"] is not None,
                "last_bar_ts": str(frame["timestamp"].iloc[i]),
            }
            out_state = {
                "schema_version": STATE_SCHEMA, "model_id": MODEL_ID, "asset": ASSET, "symbol": SYMBOL,
                "mode": MODE, "activation_allowed": False, "order_submission_supported": False,
                "updated_at": pd.Timestamp.utcnow().isoformat(), "summary": summary,
                "position": state["position"], "cusum": state["cusum"],
                "recent_closed_trades": state["closed_trades"][-50:], "equity": state["equity"],
                "step_log": step_log,
            }
            _write_json_atomic(STATE_PATH, out_state)
            print(f"[{pd.Timestamp.utcnow()}] bar={frame['timestamp'].iloc[i]} "
                  f"position_open={state['position'] is not None} equity={state['equity']:.4f}x "
                  f"n_trades={len(state['closed_trades'])} step={step_log}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{pd.Timestamp.utcnow()}] ERROR: {exc}", flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
