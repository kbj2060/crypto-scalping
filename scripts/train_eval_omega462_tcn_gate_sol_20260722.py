#!/usr/bin/env python3
"""TCN Sequence Entry Gate ported to SOL, layered on the fresh-retrain SOL Omega4.6.1
zig075-only parent (tmp/causal_regen_20260516/sol_omega4_6_1_fresh_retrain_20260722/).

Why this is a standalone script instead of reusing
scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py directly: that
script's `make_parent()` / `Omega462SourceParentLiveAdapter` lineage is ETH-only. The
production SOL live adapter (trading_bot_modules/omega4_6_1_live.py:Omega461LiveAdapter)
also can't be reused as-is for THIS parent: it hard-asserts
risk_feature_mode=="parent_outputs" + side_split_model + dynamic_leverage, but the
fresh-retrain SOL zig075 risk sidecar
(tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_fresh_retrain_20260722/risk_sidecar.pkl)
uses risk_feature_mode="all", side_split_model=False, dynamic_leverage=False -- a
different, incompatible sidecar contract (verified empirically, see report.json in this
script's out_dir under "sidecar_contract").

Instead this script does its own self-contained causal bar-by-bar replay, reusing the
already-vetted low-level SOL/ETH-shared helper modules (TabM inference, ATR-safety
SL/TP, regime3 routing, the risk-sidecar's `_replay_with_risk`-style exit-head loop,
execution/fee model) plus the TCN gate architecture and training routine imported
verbatim from the ETH sequence-gate script (SequenceEntryTCN / train_tcn / select_threshold
are asset-agnostic -- no ETH-specific data is baked into them).

Frozen fresh-retrain SOL zig075 config (from
tmp/causal_regen_20260516/sol_omega4_6_1_fresh_retrain_20260722/final_report.json):
quality_threshold=0.70, duration_gate_threshold=0.00552845229,
final_scale_map={long_scale:1.0, short_scale:3.0}, exit_threshold=0.95.

Fresh-forward contract: fixed VAL 2025-09-01..2025-12-31 / OOS 2026-01-01..2026-03-31,
causal bar-by-bar walk-forward. fresh_forward_bar_by_bar=true,
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega_eth  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_sol_20260707 as omega_sol  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_2_risk_sidecar_sol_20260707 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
from train_eval_omega462_live_native_sequence_entry_gate_20260703 import (  # noqa: E402
    SequenceEntryTCN,
    SequenceGateArtifact,
    apply_norm,
    predict_one,
    save_artifact,
    select_threshold,
    train_tcn,
)
from trading_bot_modules.omega4_6_2_source_parent_live import Regime3CurrentLiveFeatures  # noqa: E402

MODEL_ID = "omega462_tcn_sequence_entry_gate_sol_20260722"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega462_tcn_gate_sol_20260722"
FEATURES_PATH = ROOT / "data/splits/year_oos/sol_features_2024_2026.csv"

PARENT_DIR = ROOT / "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_fresh_retrain_20260722"
BUNDLE_PATH = PARENT_DIR / "true_3head_tabm_bundle.pt"
SIDECAR_PATH = ROOT / "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_fresh_retrain_20260722/risk_sidecar.pkl"
REGIME3_PATH = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/regime3_current_sensitive_hmm_wide24_2024.joblib"

QUALITY_THRESHOLD = 0.70
DURATION_THRESHOLD = 0.00552845229
LONG_SCALE = 1.0
SHORT_SCALE = 3.0
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
EXIT_THRESHOLD = 0.95
ATR_WINDOW = 192
TP_MULT, SL_MULT = 12.0, 6.0
MIN_TP, MIN_SL, MAX_TP, MAX_SL = 0.075, 0.040, 0.22, 0.12
COST_MULT = 3.0
ROUNDTRIP_COST_DEFAULT = 0.000612
LABEL_HOLD_BARS = 72  # matches BASE_TEMPLATE["max_hold"] (6h @ 5m bars)
LOOKBACK = 48

FRAME_START = "2024-09-01 00:00:00"  # buffer for ATR192/regime warm-up before TRAIN_START
TRAIN_START = "2025-01-01 00:00:00"
TRAIN_END = "2025-09-01 00:00:00"
GATE_TRAIN_END = "2025-06-15 00:00:00"
VAL_START = "2025-09-01 00:00:00"
VAL_END = "2026-01-01 00:00:00"
OOS_START = "2026-01-01 00:00:00"
OOS_END = "2026-04-01 00:00:00"
FRAME_END = "2026-04-01 00:00:00"


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_bundle_and_sidecar(device: torch.device) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    bundle = torch.load(BUNDLE_PATH, map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    models = dict(bundle["models"])
    missing = sorted(set(hard.EXPERT_NAMES) - set(models))
    if missing:
        raise RuntimeError(f"bundle missing experts: {missing}")
    loaded = parent._load_payloads(models, device=device)
    with open(SIDECAR_PATH, "rb") as f:
        pkl = pickle.load(f)
    return base_cols, loaded, pkl


def predict_direction_quality(loaded: dict[str, Any], base_x: pd.DataFrame, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    n = len(base_x)
    direction: dict[str, np.ndarray] = {}
    quality: dict[str, np.ndarray] = {}
    for expert, (model, scaler) in loaded.items():
        x_np = parent._standardize_apply(base_x, scaler)
        d_chunks: list[np.ndarray] = []
        q_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, n, 8192):
                xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
                out = model(xb)
                d_chunks.append(torch.softmax(out["direction"], dim=-1).mean(dim=1).cpu().numpy())
                q_chunks.append(torch.softmax(out["quality"], dim=-1).mean(dim=1).cpu().numpy())
        direction[expert] = np.concatenate(d_chunks, axis=0)
        quality[expert] = np.concatenate(q_chunks, axis=0)
    n_rows = len(base_x)
    direction_arr = np.zeros((n_rows, 3), dtype=np.float64)
    quality_arr = np.zeros((n_rows, 3), dtype=np.float64)
    return direction, quality, direction_arr, quality_arr


def prepare_frame(device: torch.device) -> dict[str, Any]:
    print("stage=load_frame", flush=True)
    cols = list(pd.read_csv(FEATURES_PATH, nrows=0).columns)
    df = pd.read_csv(FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= pd.Timestamp(FRAME_START)) & (df["timestamp"] < pd.Timestamp(FRAME_END))].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("empty SOL frame after slicing")

    print("stage=append_regime3", flush=True)
    regime3 = Regime3CurrentLiveFeatures(current_path=REGIME3_PATH)
    frame = regime3.append(df.copy())

    print("stage=load_bundle_sidecar", flush=True)
    base_cols, loaded, pkl = load_bundle_and_sidecar(device)
    if pkl.get("risk_feature_mode") != "all":
        raise RuntimeError(f"unexpected sidecar risk_feature_mode: {pkl.get('risk_feature_mode')}")
    missing_cols = [c for c in base_cols if c not in frame.columns]
    if missing_cols:
        raise RuntimeError(f"frame missing base_cols: {missing_cols[:20]}")

    base_x = parent._base_input(frame, base_cols)

    print("stage=route", flush=True)
    route = hard._route_id(frame)

    print("stage=tabm_inference", flush=True)
    direction_by_expert, quality_by_expert, direction_arr, quality_arr = predict_direction_quality(loaded, base_x, device)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        direction_arr[mask] = direction_by_expert[expert][mask]
        quality_arr[mask] = quality_by_expert[expert][mask]

    dir_action = direction_arr.argmax(axis=1)
    n = len(frame)
    qual_for_action = np.where(dir_action > 0, quality_arr[np.arange(n), dir_action], quality_arr[:, 0])
    final_action = np.where((dir_action != 0) & (qual_for_action >= QUALITY_THRESHOLD), dir_action, 0).astype(np.int64)
    side = np.where(final_action == 1, 1, np.where(final_action == 2, -1, 0)).astype(np.int64)
    active = final_action != 0
    router_expert_raw = np.asarray(hard.EXPERT_NAMES, dtype=object)[route]
    router_expert_scale_key = np.where(router_expert_raw == "chop", "chop_expert", router_expert_raw)

    dec = pd.DataFrame(
        {
            "action": final_action,
            "side": side,
            "notional_exposure": np.where(active, float(omega_eth.BASE_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(omega_eth.BASE_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(omega_eth.BASE_TEMPLATE["notional"]), 0.0),
            "take_profit": np.where(active, float(omega_eth.BASE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(omega_eth.BASE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(omega_eth.BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(omega_eth.BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": qual_for_action,
            "confidence": direction_arr.max(axis=1),
            "router_expert": router_expert_scale_key,
        }
    )
    for expert, scale in omega_eth.EXPERT_SCALES.items():
        m = active & dec["router_expert"].eq(expert)
        dec.loc[m, "notional_exposure"] = dec.loc[m, "notional_exposure"].astype(float) * float(scale)
        dec.loc[m, "position_fraction"] = dec.loc[m, "position_fraction"].astype(float) * float(scale)

    print("stage=atr_safety_sltp", flush=True)
    dec_atr, _ = atr_eval._apply_atr_safety_sltp(
        dec, frame, atr_window=ATR_WINDOW, tp_mult=TP_MULT, sl_mult=SL_MULT, min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL
    )
    atr = atr_eval._atr_pct(frame, ATR_WINDOW)

    print("stage=risk_sidecar_score", flush=True)
    route_probs = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    router_confidence = route_probs.max(axis=1)
    router_margin = pd.to_numeric(frame["regime3_current_sensitive_wide24_margin"], errors="raise").to_numpy(dtype=np.float64)
    src = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "omega1_regime3_expertdq_router_expert": router_expert_raw,
            "omega1_regime3_expertdq_router_confidence": router_confidence,
            "omega1_regime3_expertdq_router_margin": router_margin,
            "omega1_regime3_expertdq_dir_p_cash": direction_arr[:, 0],
            "omega1_regime3_expertdq_dir_p_long": direction_arr[:, 1],
            "omega1_regime3_expertdq_dir_p_short": direction_arr[:, 2],
            "omega1_regime3_expertdq_dir_confidence": direction_arr.max(axis=1),
            "omega1_regime3_expertdq_dir_side_edge": direction_arr[:, 1] - direction_arr[:, 2],
            "omega1_regime3_expertdq_dir_trade_prob": direction_arr[:, 1] + direction_arr[:, 2],
            "omega1_regime3_expertdq_dir_action": dir_action,
            "omega1_regime3_expertdq_quality_p_cash": quality_arr[:, 0],
            "omega1_regime3_expertdq_quality_p_long": quality_arr[:, 1],
            "omega1_regime3_expertdq_quality_p_short": quality_arr[:, 2],
            "omega1_regime3_expertdq_quality_for_action": qual_for_action,
            "omega1_regime3_expertdq_quality_threshold": np.full(n, QUALITY_THRESHOLD),
            "omega1_regime3_expertdq_final_action": final_action,
        }
    )
    risk_features = sidecar._risk_feature_frame(frame, src, dec_atr, base_cols, atr_pct=atr, feature_mode="all")
    x_risk, _ = sidecar._feature_matrix(risk_features, pkl["feature_columns"])
    if pkl.get("side_split_model"):
        raise RuntimeError("unexpected side_split_model=True for fresh-retrain sidecar")
    score = np.asarray(pkl["model"].predict(x_risk), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    base_margin = sidecar._risk_margins(
        dec_atr, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS}
    )
    base_leverage = np.ones(n, dtype=np.float64) if not pkl.get("dynamic_leverage") else sidecar._risk_leverage(
        dec_atr, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}
    )

    print("stage=final_scale_map_and_duration_gate", flush=True)
    side_arr = dec_atr["side"].to_numpy(dtype=np.int64)
    scale = np.where(side_arr > 0, LONG_SCALE, np.where(side_arr < 0, SHORT_SCALE, 1.0))
    leverage_scaled = np.minimum(base_leverage * scale, LEVERAGE_CAP)
    notional_scaled = np.minimum(base_margin * leverage_scaled, NOTIONAL_CAP)
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage_scaled = np.where(base_margin > 0.0, notional_scaled / np.maximum(base_margin, 1e-12), leverage_scaled)
    margin_final = base_margin.copy()
    ou_halflife = pd.to_numeric(frame["ou_halflife"], errors="raise").to_numpy(dtype=np.float64)
    duration_ok = ou_halflife > DURATION_THRESHOLD
    margin_final = np.where(duration_ok, margin_final, 0.0)

    print("stage=static_tape", flush=True)
    static_tape, feature_names = build_static_tape(frame, dec_atr, atr, margin_final, leverage_scaled)

    return {
        "frame": frame,
        "base_x": base_x,
        "dec_atr": dec_atr,
        "loaded": loaded,
        "margin": margin_final,
        "leverage": leverage_scaled,
        "static_tape": static_tape,
        "feature_names": feature_names,
        "fee_slip": omega_sol._load_fee_slip(),
        "sidecar_contract": {
            "risk_feature_mode": pkl.get("risk_feature_mode"),
            "side_split_model": pkl.get("side_split_model"),
            "dynamic_leverage": pkl.get("dynamic_leverage"),
        },
    }


def build_static_tape(frame: pd.DataFrame, dec_atr: pd.DataFrame, atr: np.ndarray, margin: np.ndarray, leverage: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = len(frame)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    ts = pd.to_datetime(frame["timestamp"])
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()

    def ret(bars: int) -> np.ndarray:
        prev = np.concatenate([np.full(bars, np.nan), close[:-bars]])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = close / np.where(prev > 0, prev, np.nan) - 1.0
        return np.nan_to_num(r, nan=0.0)

    rv_24h = pd.Series(close).pct_change().rolling(288, min_periods=1).std().fillna(0.0).to_numpy()

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in frame.columns:
            return np.full(n, default, dtype=np.float64)
        return pd.to_numeric(frame[name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)

    cols = {
        "side": dec_atr["side"].to_numpy(dtype=np.float64),
        "notional": margin * leverage,
        "quality_score": dec_atr["quality_score"].to_numpy(dtype=np.float64),
        "confidence": dec_atr["confidence"].to_numpy(dtype=np.float64),
        "atr_pct": np.nan_to_num(atr, nan=0.0),
        "rsi": col("rsi"),
        "bb_width": col("bb_width"),
        "ou_halflife": col("ou_halflife"),
        "regime3_bull": col("regime3_current_sensitive_wide24_bull_prob"),
        "regime3_bear": col("regime3_current_sensitive_wide24_bear_prob"),
        "regime3_chop": col("regime3_current_sensitive_wide24_chop_prob"),
        "regime3_confidence": col("regime3_current_sensitive_wide24_confidence"),
        "regime3_margin": col("regime3_current_sensitive_wide24_margin"),
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
        "ret_1h": ret(12),
        "ret_6h": ret(72),
        "ret_24h": ret(288),
        "rv_24h": rv_24h,
    }
    names = list(cols.keys())
    arr = np.stack([cols[k] for k in names], axis=1).astype(np.float32)
    return arr, names + ["overlay_loss_streak"]


def slice_bundle(bundle: dict[str, Any], start: str, end_exclusive: str) -> dict[str, Any]:
    frame = bundle["frame"]
    ts = frame["timestamp"].to_numpy()
    start_idx = np.flatnonzero(ts >= pd.Timestamp(start).to_datetime64())
    end_idx = np.flatnonzero(ts < pd.Timestamp(end_exclusive).to_datetime64())
    if len(start_idx) == 0 or len(end_idx) == 0:
        raise RuntimeError(f"empty slice {start}..{end_exclusive}")
    start_i = int(start_idx[0])
    end_i = int(end_idx[-1])
    if end_i < start_i:
        raise RuntimeError(f"empty slice range {start}..{end_exclusive}")
    sl = slice(start_i, end_i + 1)
    return {
        "frame": frame.iloc[sl].reset_index(drop=True),
        "base_x": bundle["base_x"].iloc[sl].reset_index(drop=True),
        "dec_atr": bundle["dec_atr"].iloc[sl].reset_index(drop=True),
        "margin": bundle["margin"][sl],
        "leverage": bundle["leverage"][sl],
        "static_tape": bundle["static_tape"][sl],
    }


def counterfactual_label(
    i: int, high: np.ndarray, low: np.ndarray, close: np.ndarray, side: int, tp: float, sl: float, notional: float, max_hold_bars: int, roundtrip_cost: float
) -> float | None:
    n = len(close)
    forced_exit_i = i + max_hold_bars
    if max_hold_bars <= 0 or forced_exit_i >= n:
        return None
    entry_price = float(close[i])
    highs = high[i + 1 : forced_exit_i + 1]
    lows = low[i + 1 : forced_exit_i + 1]
    if side > 0:
        sl_hits = (lows / entry_price - 1.0) <= -sl
        tp_hits = (highs / entry_price - 1.0) >= tp
    else:
        sl_hits = (entry_price / highs - 1.0) <= -sl
        tp_hits = (entry_price / lows - 1.0) >= tp
    hit_any = sl_hits | tp_hits
    if bool(hit_any.any()):
        offset = int(np.argmax(hit_any))
        raw_move = -sl if bool(sl_hits[offset]) else tp
    else:
        exit_i = forced_exit_i
        raw_move = close[exit_i] / entry_price - 1.0 if side > 0 else entry_price / close[exit_i] - 1.0
    net = raw_move - roundtrip_cost
    return float(net * notional)


def compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl_pct": 0.0, "mdd_pct": 0.0, "trades": 0, "wr": 0.0}
    returns = ledger["trade_return"].astype(float).to_numpy(dtype=np.float64)
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {
        "pnl_pct": float((curve[-1] - 1.0) * 100.0),
        "mdd_pct": float(dd.min() * 100.0),
        "trades": int(len(ledger)),
        "wr": float((returns > 0.0).mean()),
        "long_trades": int((ledger["side"].astype(int) > 0).sum()),
        "short_trades": int((ledger["side"].astype(int) < 0).sum()),
        "reason_counts": dict(Counter(ledger["reason"].astype(str))),
    }


def replay_with_gate(
    *,
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded: dict[str, Any],
    margin: np.ndarray,
    leverage: np.ndarray,
    static_tape: np.ndarray,
    fee: float,
    slip: float,
    device: torch.device,
    gate_artifact: Any | None = None,
    collect_labels: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega_sol._active(dec)
    fee_eff = float(fee) * float(COST_MULT)
    slip_eff = float(slip) * float(COST_MULT)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage_pos = 1.0
    margin_pos = 0.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    loss_streak = 0
    rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    gate_counts: Counter[str] = Counter()
    route = hard._route_id(frame)
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(base_x, loaded)
    history: deque[np.ndarray] = deque(maxlen=LOOKBACK)
    n = len(frame)
    close_arr, high_arr, low_arr = arrays["close"], arrays["high"], arrays["low"]

    for i in range(0, n - 2):
        feat = np.concatenate([static_tape[i], [float(min(loss_streak, 3))]]).astype(np.float32)
        history.append(feat)

        if (i - 0) % 20000 == 0:
            print(json.dumps({"i": int(i), "n": int(n), "trades": int(len(rows)), "pos": pos}), flush=True)

        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = sidecar._predict_exit_prob_one(
                    base_np,
                    exit_runtime,
                    pos_idx,
                    row_i=int(i),
                    expert=expert,
                    pos_values=[
                        float(pos),
                        float(hold),
                        float(move),
                        float(mfe),
                        float(mae),
                        float(np.clip(giveback, 0.0, 10.0)),
                        float(take_profit - move),
                        float(move + abs(stop_loss)),
                        float(notional),
                        float(leverage_pos),
                        float(notional * leverage_pos),
                        float(take_profit),
                        float(stop_loss),
                    ],
                    device=device,
                )
                exit_prob = float(prob)
                if prob >= EXIT_THRESHOLD:
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _r = omega_sol._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1e-12) - 1.0
                loss_streak = loss_streak + 1 if trade_return <= 0.0 else 0
                rows.append(
                    {
                        "entry_i": int(entry_signal_i),
                        "exit_i": int(i),
                        "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                        "side": int(pos),
                        "reason": reason,
                        "trade_return": float(trade_return),
                        "notional": float(notional),
                        "leverage": float(leverage_pos),
                        "margin_fraction": float(margin_pos),
                        "exit_prob": float(exit_prob),
                    }
                )
                pos = 0
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue

        base_notional_i = float(margin[i]) * float(leverage[i])

        if collect_labels and base_notional_i > 0.0 and len(history) == LOOKBACK:
            lbl = counterfactual_label(
                i, high_arr, low_arr, close_arr, side, float(row["take_profit"]), float(row["stop_loss"]), base_notional_i, LABEL_HOLD_BARS, ROUNDTRIP_COST_DEFAULT
            )
            if lbl is not None:
                label_rows.append(
                    {
                        "entry_i": int(i),
                        "timestamp": str(frame["timestamp"].iloc[int(i)]),
                        "side": int(side),
                        "trade_return": float(lbl),
                        "seq": np.stack(list(history)).astype(np.float32),
                    }
                )

        if base_notional_i <= 0.0:
            continue  # risk-sidecar / duration-gate vetoed this candidate

        if gate_artifact is not None:
            if len(history) < gate_artifact.lookback:
                gate_counts["sequence_gate_warmup_veto"] += 1
                continue
            score = predict_one(gate_artifact, history)
            if score is None or score < gate_artifact.threshold:
                gate_counts["sequence_gate_veto"] += 1
                continue
            gate_counts["sequence_gate_allow"] += 1

        filled, px, fee_paid, _r = omega_sol._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        row_leverage = float(leverage[int(i)])
        row_margin = float(margin[int(i)])
        row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, n - 1)
        entry_signal_i = int(i)
        leverage_pos = row_leverage
        margin_pos = row_margin
        notional = row_notional
        take_profit = float(row["take_profit"])
        stop_loss = float(row["stop_loss"])
        cash -= cash * float(fee_paid) * notional
        mfe = 0.0
        mae = 0.0

    if pos != 0:
        exit_px = omega_sol._fill_price(arrays, n - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1e-12) - 1.0
        rows.append(
            {
                "entry_i": int(entry_signal_i),
                "exit_i": int(n - 1),
                "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                "exit_timestamp": str(frame["timestamp"].iloc[-1]),
                "side": int(pos),
                "reason": "forced_end",
                "trade_return": float(trade_return),
                "notional": float(notional),
                "leverage": float(leverage_pos),
                "margin_fraction": float(margin_pos),
                "exit_prob": 0.0,
            }
        )

    ledger_df = pd.DataFrame(rows)
    metrics = compound_metrics(ledger_df)
    metrics["sequence_gate_counts"] = dict(gate_counts)
    return metrics, ledger_df, label_rows


def run() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    bundle = prepare_frame(device)
    fee, slip = bundle["fee_slip"]

    print("stage=train_slice_and_labels", flush=True)
    train_slice = slice_bundle(bundle, TRAIN_START, TRAIN_END)
    _, _, label_rows = replay_with_gate(
        frame=train_slice["frame"],
        base_x=train_slice["base_x"],
        dec=train_slice["dec_atr"],
        loaded=bundle["loaded"],
        margin=train_slice["margin"],
        leverage=train_slice["leverage"],
        static_tape=train_slice["static_tape"],
        fee=fee,
        slip=slip,
        device=device,
        gate_artifact=None,
        collect_labels=True,
    )
    if not label_rows:
        raise RuntimeError("no counterfactual entry labels collected for TCN gate training")

    train_rows = [r for r in label_rows if r["timestamp"] < GATE_TRAIN_END]
    calib_rows = [r for r in label_rows if r["timestamp"] >= GATE_TRAIN_END]
    if not train_rows or not calib_rows:
        raise RuntimeError(f"empty chronological gate split: train={len(train_rows)} calib={len(calib_rows)}")

    train_seq = np.stack([r["seq"] for r in train_rows]).astype(np.float32)
    train_y = np.asarray([r["trade_return"] for r in train_rows], dtype=np.float32)
    calib_seq = np.stack([r["seq"] for r in calib_rows]).astype(np.float32)
    calib_labels_df = pd.DataFrame({"trade_return": [r["trade_return"] for r in calib_rows]})

    print(f"stage=fit_tcn train_rows={len(train_rows)} calib_rows={len(calib_rows)}", flush=True)
    model, norm, train_report = train_tcn(
        train_seq=train_seq,
        train_y=train_y,
        calib_seq=calib_seq,
        calib_labels=calib_labels_df,
        epochs=8,
        batch_size=128,
        lr=8.0e-4,
        seed=260722,
        device=device,
    )
    threshold = float(train_report["threshold"]["selected"]["threshold"])
    artifact = SequenceGateArtifact(
        name="sol_tcn_seq_gate_L48_flat_20260722",
        lookback=LOOKBACK,
        sample_mode="flat",
        feature_cols=bundle["feature_names"],
        mean=norm["mean"],
        std=norm["std"],
        threshold=threshold,
        threshold_payload=train_report["threshold"],
        model=model,
        train_report=train_report,
        path="",
    )
    artifact.path = save_artifact(artifact, OUT_DIR)

    results: dict[str, Any] = {}
    for split, start, end in (("validation", VAL_START, VAL_END), ("oos", OOS_START, OOS_END)):
        print(f"stage=eval_{split}", flush=True)
        sl = slice_bundle(bundle, start, end)
        parent_metrics, parent_ledger, _ = replay_with_gate(
            frame=sl["frame"],
            base_x=sl["base_x"],
            dec=sl["dec_atr"],
            loaded=bundle["loaded"],
            margin=sl["margin"],
            leverage=sl["leverage"],
            static_tape=sl["static_tape"],
            fee=fee,
            slip=slip,
            device=device,
            gate_artifact=None,
            collect_labels=False,
        )
        gated_metrics, gated_ledger, _ = replay_with_gate(
            frame=sl["frame"],
            base_x=sl["base_x"],
            dec=sl["dec_atr"],
            loaded=bundle["loaded"],
            margin=sl["margin"],
            leverage=sl["leverage"],
            static_tape=sl["static_tape"],
            fee=fee,
            slip=slip,
            device=device,
            gate_artifact=artifact,
            collect_labels=False,
        )
        parent_ledger.to_csv(OUT_DIR / f"{split}_parent_alone_ledger.csv", index=False)
        gated_ledger.to_csv(OUT_DIR / f"{split}_parent_plus_tcn_gate_ledger.csv", index=False)
        results[split] = {
            "start": start,
            "end_exclusive": end,
            "parent_alone": parent_metrics,
            "parent_plus_tcn_gate": gated_metrics,
        }

    report = {
        "schema_version": "omega462.tcn_sequence_entry_gate.sol.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "parent": "sol_omega4_6_1_fresh_retrain_20260722 (zig075-only)",
        "parent_config": {
            "quality_threshold": QUALITY_THRESHOLD,
            "duration_gate_threshold": DURATION_THRESHOLD,
            "final_scale_map": {"long_scale": LONG_SCALE, "short_scale": SHORT_SCALE},
            "exit_threshold": EXIT_THRESHOLD,
            "leverage_cap": LEVERAGE_CAP,
            "notional_cap": NOTIONAL_CAP,
            "cost_mult": COST_MULT,
        },
        "sidecar_contract": bundle["sidecar_contract"],
        "why_not_omega461_live_adapter": (
            "Omega461LiveAdapter asserts risk_feature_mode=='parent_outputs' + side_split_model + "
            "dynamic_leverage; this fresh-retrain SOL sidecar uses risk_feature_mode='all', "
            "side_split_model=False, dynamic_leverage=False -- a different, incompatible contract. "
            "Verified empirically (RuntimeError: sidecar risk_feature_mode contract mismatch)."
        ),
        "tcn_gate": {
            "architecture": "SequenceEntryTCN (dilated causal Conv1d stack), imported verbatim from "
            "scripts/train_eval_omega462_live_native_sequence_entry_gate_20260703.py",
            "lookback": LOOKBACK,
            "sample_mode": "flat (only-when-flat candidates, matches how labels/candidates are collected)",
            "feature_count": len(bundle["feature_names"]),
            "train_window": [TRAIN_START, TRAIN_END],
            "gate_train_end": GATE_TRAIN_END,
            "train_rows": len(train_rows),
            "calibration_rows": len(calib_rows),
            "threshold": threshold,
            "artifact_path": artifact.path,
        },
        "results": results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "validation_window_canonical": [VAL_START, VAL_END],
        "oos_window_canonical": [OOS_START, OOS_END],
        "caveat": (
            "Parent-alone numbers here are this script's OWN independent bar-by-bar reproduction of the "
            "frozen fresh-retrain config on the canonical VAL/OOS windows, not a byte-identical replay of "
            "tmp/causal_regen_20260516/sol_omega4_6_1_fresh_retrain_20260722/final_report.json (that report's "
            "own VAL/OOS pipeline and this script's precomputed-prediction-file boundaries differ; this "
            "script instead runs its own TabM inference end-to-end on a single continuous frame so VAL/OOS "
            "windows land exactly on the canonical 09-01/01-01 boundaries). The parent-alone vs "
            "parent+TCN-gate DELTA is internally consistent since both use the identical replay harness."
        ),
        "artifacts": {"out_dir": str(OUT_DIR), "report": str(OUT_DIR / "report.json")},
    }
    write_json(OUT_DIR / "report.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps(report["results"], ensure_ascii=False, indent=2, default=json_default), flush=True)
