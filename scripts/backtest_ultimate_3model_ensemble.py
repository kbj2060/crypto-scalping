#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")


@dataclass
class Metrics:
    pnl_pct: float
    mdd_pct: float
    trades: int
    wr_pct: float
    avg_pnl_per_trade_pct: float
    profit_factor: float
    long_entries: int
    short_entries: int


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def _load_frame(csv_path: str, start: str | None, end: str | None, max_rows: int) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise KeyError("timestamp column missing")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if start:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end:
        df = df[df["timestamp"] <= pd.Timestamp(end)]

    if "close" not in df.columns:
        raise KeyError("close column missing")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"]).reset_index(drop=True)

    base_defaults = {
        "open": None,
        "high": None,
        "low": None,
        "volume": 1.0,
        "smart_money_flow": 0.0,
        "amihud_illiquidity_z": 0.0,
        "mtf_trend_1h": 0.0,
        "mtf_trend_4h": 0.0,
        "log_return": 0.0,
        "garch_vol_z": 0.0,
        "oi_change_rate": 0.0,
        "jump_z": 0.0,
        "evt_excess_z": 0.0,
        "jump_flag": 0.0,
        "evt_tail_flag": 0.0,
        "taker_acceleration": 0.0,
        "rogers_satchell_vol": 0.0,
    }
    try:
        from features.schema import STATE_ALPHA as _STATE_ALPHA, STATE_SYNTH as _STATE_SYNTH

        for c in list(_STATE_ALPHA) + list(_STATE_SYNTH):
            base_defaults.setdefault(str(c), 0.0)
    except Exception:
        pass

    for col, default in base_defaults.items():
        if col not in df.columns:
            if default is None:
                df[col] = df["close"]
            else:
                df[col] = default
    for col in ("open", "high", "low"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["close"])

    if max_rows > 0:
        df = df.head(int(max_rows)).reset_index(drop=True)
    return df


def _m7_signal_from_row(row_dict: dict[str, Any]) -> dict:
    from features.m7 import trend_signal_from_m7

    # aliases
    if "m7_prob_dn" not in row_dict:
        row_dict["m7_prob_dn"] = _safe_float(row_dict.get("prob_dn", row_dict.get("m7_trend_xgb_dn", 0.0)))
    if "m7_prob_fl" not in row_dict:
        row_dict["m7_prob_fl"] = _safe_float(row_dict.get("prob_flat", row_dict.get("m7_trend_xgb_fl", 0.0)))
    if "m7_prob_up" not in row_dict:
        row_dict["m7_prob_up"] = _safe_float(row_dict.get("prob_up", row_dict.get("m7_trend_xgb_up", 0.0)))
    row_dict.setdefault("m7_trend_xgb_dn", row_dict["m7_prob_dn"])
    row_dict.setdefault("m7_trend_xgb_fl", row_dict["m7_prob_fl"])
    row_dict.setdefault("m7_trend_xgb_up", row_dict["m7_prob_up"])

    m7_defaults = {
        "m7_confidence": 0.0,
        "m7_action": 0.0,
        "m7_size": 0.0,
        "m7_gate_block": 0.0,
        "m7_hdb_label": -1.0,
        "m7_hdb_prob": 0.0,
        "m7_iso_score": 0.0,
        "m7_iso_pred": 1.0,
        "m7_iso_anom": 0.0,
        "m7_vae_error": 0.0,
        "m7_vae_threshold": 0.0,
        "m7_vae_anom": 0.0,
        "m7_q10": 0.0,
        "m7_q50": 0.0,
        "m7_q90": 0.0,
        "m7_qwidth": 0.0,
        "m7_quality_pred": 0.0,
        "m7_hold_pred": 0.0,
        "m7_target_hold": 0.0,
        "m7_entry_long_offset": 0.0,
        "m7_entry_short_offset": 0.0,
        "m7_entry_long_price": 0.0,
        "m7_entry_short_price": 0.0,
        "m7_tp_offset": 0.0,
        "m7_sl_offset": 0.0,
        "m7_tp_price": 0.0,
        "m7_sl_price": 0.0,
        "m7_gmm_cluster": -1.0,
        "m7_gmm_conf": 0.0,
        "m7_gmm_vol_rank": 0.5,
        "m7_expected_ret": 0.0,
        "m7_tail_risk": 0.0,
        "m7_composite_score": 0.0,
    }
    for k, v in m7_defaults.items():
        row_dict.setdefault(k, v)

    return trend_signal_from_m7(row_dict)


def _nf_preds_from_row(row_dict: dict[str, Any]) -> dict:
    from features.schema import STATE_PRED as DSAC_STATE_PRED, STATE_CONF as DSAC_STATE_CONF

    nf_preds = dict(row_dict)
    pred_fallback = _safe_float(nf_preds.get("pred_patchtst", 0.0))
    conf_fallback = float(np.clip(_safe_float(nf_preds.get("conf_patchtst", 0.5), 0.5), 0.0, 1.0))

    for c in DSAC_STATE_PRED:
        if c not in nf_preds:
            nf_preds[c] = pred_fallback
    for c in DSAC_STATE_CONF:
        if c not in nf_preds:
            nf_preds[c] = conf_fallback
    return nf_preds


class PrimaryModelAdapter:
    def __init__(self, ckpt_path: str):
        from trading_bot import DSACSignalRouter

        self.router = DSACSignalRouter(model_path=ckpt_path)

    def decide(self, processed_df: pd.DataFrame, nf_preds: dict, trend_signal: dict, pos: dict) -> tuple[int, float, float, dict]:
        # sync router internal position state
        self.router.pos = pos.get("type")
        self.router.entry_price = _safe_float(pos.get("entry_price", 0.0), 0.0)
        self.router.hold_count = int(pos.get("hold_count", 0))
        self.router.current_leverage = _safe_float(pos.get("margin_usage", 0.0), 0.0)

        _, lev, info, _, _ = self.router.decide(processed_df, nf_preds, m7_signal=trend_signal)
        raw = float(info.get("raw_action", info.get("primary_raw", 0.0)))

        # pure_rl mapping
        pos_th = float(os.getenv("DSAC_PURE_RL_POS_TH", "0.12"))
        close_th = float(os.getenv("DSAC_PURE_RL_CLOSE_TH", "0.03"))
        flip_th = float(os.getenv("DSAC_PURE_RL_FLIP_TH", str(max(pos_th * 1.5, pos_th))))
        flip_kelly_mult = float(os.getenv("DSAC_PURE_RL_FLIP_KELLY_MULT", "0.85"))
        max_kelly = float(os.getenv("DSAC_PURE_RL_MAX_KELLY", "1.0"))
        abs_raw = abs(raw)

        action = 0
        kelly = 0.0
        cur_pos = pos.get("type")
        if cur_pos is None:
            if raw > pos_th:
                action, kelly = 1, min(abs_raw, max_kelly)
            elif raw < -pos_th:
                action, kelly = 2, min(abs_raw, max_kelly)
        elif cur_pos == "LONG":
            if abs_raw < close_th:
                action, kelly = 0, 0.0
            elif raw < -flip_th:
                action, kelly = 2, min(abs_raw, max_kelly) * flip_kelly_mult
            else:
                action, kelly = 1, min(abs_raw, max_kelly)
        else:
            if abs_raw < close_th:
                action, kelly = 0, 0.0
            elif raw > flip_th:
                action, kelly = 1, min(abs_raw, max_kelly) * flip_kelly_mult
            else:
                action, kelly = 2, min(abs_raw, max_kelly)

        return int(action), float(np.clip(kelly, 0.0, 1.0)), float(raw), info


class LongSpecialistAdapter:
    def __init__(self, ckpt_path: str):
        from ensemble.train_rl_dsac_long_agent import DSACLongRouter, SigmoidActor, STATE_DIM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dim = int(ckpt.get("state_dim", STATE_DIM) or STATE_DIM)
        actor = SigmoidActor(state_dim=state_dim).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        self.router = DSACLongRouter(actor=actor, device=device)

    def decide(self, features: dict, pos: dict) -> tuple[int, float, float, dict]:
        action, lev, info = self.router.decide(features, pos)
        raw = float(info.get("raw_action", 0.0))
        return int(action), float(np.clip(lev, 0.0, 1.0)), float(raw), info


class ShortSpecialistAdapter:
    def __init__(self, ckpt_path: str):
        from ensemble.train_rl_dsac_short_agent import DSACShortRouter, SigmoidActor, STATE_DIM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dim = int(ckpt.get("state_dim", STATE_DIM) or STATE_DIM)
        actor = SigmoidActor(state_dim=state_dim).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        self.router = DSACShortRouter(actor=actor, device=device)

    def decide(self, features: dict, pos: dict) -> tuple[int, float, float, dict]:
        action, lev, info = self.router.decide(features, pos)
        raw = float(info.get("raw_action", 0.0))
        return int(action), float(np.clip(lev, 0.0, 1.0)), float(raw), info


class StrategyRunner:
    def __init__(self, name: str):
        from trading_bot import DSACTrendRouter

        self.name = name
        self.meta = DSACTrendRouter()
        self.meta._save_live_state = lambda *args, **kwargs: None

        self.balance = 1.0
        self.trades: list[dict] = []
        self.long_entries = 0
        self.short_entries = 0

    def pos_dict(self, current_price: float) -> dict:
        cur_pos = self.meta.pos
        if cur_pos is None:
            return {
                "type": None,
                "entry_price": 0.0,
                "unrealized": 0.0,
                "mdd": 0.0,
                "hold_count": 0.0,
                "hold_norm": 0.0,
                "margin_usage": 0.0,
            }

        unr = float(self.meta._net_pnl_frac(current_price))
        cur_equity = 1.0 + unr
        peak = max(float(self.meta.peak_equity), cur_equity, 1e-8)
        mdd = min(cur_equity / peak - 1.0, 0.0)
        return {
            "type": cur_pos,
            "entry_price": float(self.meta.entry_price),
            "unrealized": float(unr),
            "mdd": float(mdd),
            "hold_count": float(self.meta.hold_count + 1),
            "hold_norm": float(min((self.meta.hold_count + 1) / 96.0, 1.0)),
            "margin_usage": float(np.clip(self.meta.current_leverage, 0.0, 1.0)),
        }

    def apply(self, action: int, kelly: float, next_price: float, trend_signal: dict, ts: str) -> None:
        prev_pos = self.meta.pos
        prev_hold = int(self.meta.hold_count)
        prev_entry = float(self.meta.entry_price)
        prev_lev = float(self.meta.current_leverage)

        self.meta._update_pos(int(action), float(next_price), float(np.clip(kelly, 0.0, 1.0)), trend_signal)

        if prev_pos is None and self.meta.pos == "LONG":
            self.long_entries += 1
        elif prev_pos is None and self.meta.pos == "SHORT":
            self.short_entries += 1

        if prev_pos is not None and self.meta.pos != prev_pos:
            realized = float(self.meta.last_realized_pnl or 0.0)
            self.balance *= (1.0 + realized)
            self.trades.append(
                {
                    "ts": ts,
                    "side": prev_pos,
                    "entry_price": prev_entry,
                    "exit_price": float(next_price),
                    "lev": prev_lev,
                    "hold_bars": prev_hold,
                    "pnl_frac": realized,
                }
            )

    def close_terminal(self, terminal_price: float, ts: str) -> None:
        if self.meta.pos is None:
            return
        realized = float(self.meta._net_pnl_frac(float(terminal_price)))
        self.balance *= (1.0 + realized)
        self.trades.append(
            {
                "ts": ts,
                "side": str(self.meta.pos),
                "entry_price": float(self.meta.entry_price),
                "exit_price": float(terminal_price),
                "lev": float(self.meta.current_leverage),
                "hold_bars": int(self.meta.hold_count),
                "pnl_frac": realized,
                "terminal": True,
            }
        )

    def metrics(self) -> Metrics:
        trades = len(self.trades)
        pnl_pct = float((self.balance - 1.0) * 100.0)
        wins = sum(1 for t in self.trades if float(t.get("pnl_frac", 0.0)) > 0.0)
        wr = (100.0 * wins / trades) if trades > 0 else 0.0
        avg = (100.0 * sum(float(t.get("pnl_frac", 0.0)) for t in self.trades) / trades) if trades > 0 else 0.0

        gp = sum(max(float(t.get("pnl_frac", 0.0)), 0.0) for t in self.trades)
        gl = -sum(min(float(t.get("pnl_frac", 0.0)), 0.0) for t in self.trades)
        pf = float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else 0.0)

        eq = [1.0]
        b = 1.0
        for t in self.trades:
            b *= (1.0 + float(t.get("pnl_frac", 0.0)))
            eq.append(b)
        eq_arr = np.asarray(eq, dtype=np.float64)
        peak = np.maximum.accumulate(np.maximum(eq_arr, 1e-12))
        dd = eq_arr / peak - 1.0
        mdd_pct = float(np.min(dd) * 100.0) if eq_arr.size else 0.0

        return Metrics(
            pnl_pct=pnl_pct,
            mdd_pct=mdd_pct,
            trades=int(trades),
            wr_pct=float(wr),
            avg_pnl_per_trade_pct=float(avg),
            profit_factor=float(pf if np.isfinite(pf) else 9999.0),
            long_entries=int(self.long_entries),
            short_entries=int(self.short_entries),
        )


def _decide_ultimate_ensemble(
    primary_raw: float,
    primary_lev: float,
    long_raw: float,
    long_lev: float,
    short_raw: float,
    short_lev: float,
    cur_pos: str | None,
    weights: dict[str, float] | None = None,
) -> tuple[int, float, dict]:
    if isinstance(weights, dict):
        w_p = float(weights.get("primary", 0.50))
        w_l = float(weights.get("long", 0.25))
        w_s = float(weights.get("short", 0.25))
    else:
        w_p = float(os.getenv("ULT_W_PRIMARY", "0.50"))
        w_l = float(os.getenv("ULT_W_LONG", "0.25"))
        w_s = float(os.getenv("ULT_W_SHORT", "0.25"))

    long_score = (w_p * max(primary_raw, 0.0)) + (w_l * max(long_raw, 0.0))
    short_score = (w_p * max(-primary_raw, 0.0)) + (w_s * max(short_raw, 0.0))

    avg_kelly = (
        (w_p * np.clip(primary_lev, 0.0, 1.0))
        + (w_l * np.clip(long_lev, 0.0, 1.0))
        + (w_s * np.clip(short_lev, 0.0, 1.0))
    ) / max(w_p + w_l + w_s, 1e-8)

    entry_th = float(os.getenv("ULT_ENTRY_TH", "0.10"))
    close_th = float(os.getenv("ULT_CLOSE_TH", "0.03"))
    flip_th = float(os.getenv("ULT_FLIP_TH", "0.16"))
    max_kelly = float(os.getenv("ULT_MAX_KELLY", "1.0"))
    kelly_scale = float(os.getenv("ULT_KELLY_SCALE", "0.85"))

    net = long_score - short_score
    action = 0
    if cur_pos is None:
        if long_score >= entry_th and long_score > short_score:
            action = 1
        elif short_score >= entry_th and short_score > long_score:
            action = 2
    elif cur_pos == "LONG":
        if long_score < close_th and short_score < close_th:
            action = 0
        elif net <= -flip_th:
            action = 2
        else:
            action = 1
    else:
        if long_score < close_th and short_score < close_th:
            action = 0
        elif net >= flip_th:
            action = 1
        else:
            action = 2

    kelly = 0.0
    if action != 0:
        side_score = long_score if action == 1 else short_score
        conf = side_score / max(entry_th, 1e-6)
        kelly = float(np.clip(avg_kelly * kelly_scale * conf, 0.0, max_kelly))

    diag = {
        "long_score": float(long_score),
        "short_score": float(short_score),
        "net": float(net),
        "avg_kelly": float(avg_kelly),
        "weights": {"primary": w_p, "long": w_l, "short": w_s},
    }
    return int(action), float(kelly), diag


def _detect_regime(row: dict[str, Any]) -> str:
    if _safe_float(row.get("regime_bull", 0.0), 0.0) >= 0.5:
        return "bull"
    if _safe_float(row.get("regime_bear", 0.0), 0.0) >= 0.5:
        return "bear"
    if _safe_float(row.get("regime_chop", 0.0), 0.0) >= 0.5:
        return "chop"
    if _safe_float(row.get("regime_whipsaw", 0.0), 0.0) >= 0.5:
        return "whipsaw"
    return "normal"


def _weights_for_regime(regime: str) -> dict[str, float]:
    r = (regime or "normal").lower()
    if r == "bull":
        return {
            "primary": float(os.getenv("ULT_BULL_W_PRIMARY", "0.20")),
            "long": float(os.getenv("ULT_BULL_W_LONG", "0.70")),
            "short": float(os.getenv("ULT_BULL_W_SHORT", "0.10")),
        }
    if r == "bear":
        return {
            "primary": float(os.getenv("ULT_BEAR_W_PRIMARY", "0.25")),
            "long": float(os.getenv("ULT_BEAR_W_LONG", "0.10")),
            "short": float(os.getenv("ULT_BEAR_W_SHORT", "0.65")),
        }
    if r in ("chop", "whipsaw"):
        return {
            "primary": float(os.getenv("ULT_CHOP_W_PRIMARY", "0.60")),
            "long": float(os.getenv("ULT_CHOP_W_LONG", "0.20")),
            "short": float(os.getenv("ULT_CHOP_W_SHORT", "0.20")),
        }
    return {
        "primary": float(os.getenv("ULT_NORMAL_W_PRIMARY", os.getenv("ULT_W_PRIMARY", "0.50"))),
        "long": float(os.getenv("ULT_NORMAL_W_LONG", os.getenv("ULT_W_LONG", "0.25"))),
        "short": float(os.getenv("ULT_NORMAL_W_SHORT", os.getenv("ULT_W_SHORT", "0.25"))),
    }


def _run_compare(
    df: pd.DataFrame,
    primary_ckpt: str,
    long_ckpt: str,
    short_ckpt: str,
    ensemble_only: bool = False,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="ultimate3m_bt_") as tmpdir:
        os.environ["DSAC_LIVE_STATE_PATH"] = os.path.join(tmpdir, "live_state.json")
        os.environ["FUSE_ADAPT_STATE_PATH"] = os.path.join(tmpdir, "adapt_state.json")

        primary_model = PrimaryModelAdapter(primary_ckpt)
        long_model = LongSpecialistAdapter(long_ckpt)
        short_model = ShortSpecialistAdapter(short_ckpt)

        primary_runner = StrategyRunner("primary_only") if not ensemble_only else None
        long_runner = StrategyRunner("long_specialist_only") if not ensemble_only else None
        short_runner = StrategyRunner("short_specialist_only") if not ensemble_only else None
        ens_runner = StrategyRunner("ultimate_ensemble_3m")
        ens_regime_runner = StrategyRunner("ultimate_ensemble_regime_weighted")

        for i in range(60, len(df) - 1):
            start_i = max(0, i - 300)
            processed_df = df.iloc[start_i : i + 1].copy()
            current_price = float(processed_df.iloc[-1]["close"])
            next_price = float(df.iloc[i + 1]["close"])
            ts = str(df.iloc[i + 1]["timestamp"])

            row = processed_df.iloc[-1].to_dict()
            trend_signal = _m7_signal_from_row(dict(row))
            nf_preds = _nf_preds_from_row(dict(row))
            feature_row = dict(row)
            feature_row.update(trend_signal or {})
            _h = _safe_float(feature_row.get("high", feature_row.get("close", 0.0)), 0.0)
            _l = _safe_float(feature_row.get("low", feature_row.get("close", 0.0)), 0.0)
            _c = max(_safe_float(feature_row.get("close", 0.0), 0.0), 1e-8)
            feature_row["current_spread"] = float(np.clip((_h - _l) / _c, 0.0, 0.05))

            if not ensemble_only:
                # primary only
                p_pos = primary_runner.pos_dict(current_price)
                p_act, p_kel, p_raw, _ = primary_model.decide(processed_df, nf_preds, trend_signal, p_pos)
                primary_runner.apply(p_act, p_kel, next_price, trend_signal, ts)

                # long specialist only
                l_pos = long_runner.pos_dict(current_price)
                l_act, l_kel, l_raw, _ = long_model.decide(feature_row, l_pos)
                long_runner.apply(l_act, l_kel, next_price, trend_signal, ts)

                # short specialist only
                s_pos = short_runner.pos_dict(current_price)
                s_act, s_kel, s_raw, _ = short_model.decide(feature_row, s_pos)
                short_runner.apply(s_act, s_kel, next_price, trend_signal, ts)

            # static ensemble (same current ensemble position fed to all models)
            e_pos = ens_runner.pos_dict(current_price)
            _, p_lev_e, p_raw_e, _ = primary_model.decide(processed_df, nf_preds, trend_signal, e_pos)
            _, l_lev_e, l_raw_e, _ = long_model.decide(feature_row, e_pos)
            _, s_lev_e, s_raw_e, _ = short_model.decide(feature_row, e_pos)
            e_act, e_kel, _ = _decide_ultimate_ensemble(
                primary_raw=p_raw_e,
                primary_lev=p_lev_e,
                long_raw=l_raw_e,
                long_lev=l_lev_e,
                short_raw=s_raw_e,
                short_lev=s_lev_e,
                cur_pos=e_pos.get("type"),
            )
            ens_runner.apply(e_act, e_kel, next_price, trend_signal, ts)

            # regime-weighted ensemble
            er_pos = ens_regime_runner.pos_dict(current_price)
            _, p_lev_r, p_raw_r, _ = primary_model.decide(processed_df, nf_preds, trend_signal, er_pos)
            _, l_lev_r, l_raw_r, _ = long_model.decide(feature_row, er_pos)
            _, s_lev_r, s_raw_r, _ = short_model.decide(feature_row, er_pos)
            regime = _detect_regime(row)
            regime_weights = _weights_for_regime(regime)
            er_act, er_kel, _ = _decide_ultimate_ensemble(
                primary_raw=p_raw_r,
                primary_lev=p_lev_r,
                long_raw=l_raw_r,
                long_lev=l_lev_r,
                short_raw=s_raw_r,
                short_lev=s_lev_r,
                cur_pos=er_pos.get("type"),
                weights=regime_weights,
            )
            ens_regime_runner.apply(er_act, er_kel, next_price, trend_signal, ts)

        ts_last = str(df.iloc[-1]["timestamp"])
        px_last = float(df.iloc[-1]["close"])
        ens_runner.close_terminal(px_last, ts_last)
        ens_regime_runner.close_terminal(px_last, ts_last)
        if not ensemble_only:
            for r in (primary_runner, long_runner, short_runner):
                r.close_terminal(px_last, ts_last)

        results = {
            "ultimate_ensemble_3m": asdict(ens_runner.metrics()),
            "ultimate_ensemble_regime_weighted": asdict(ens_regime_runner.metrics()),
        }
        trade_samples = {
            "ultimate_ensemble_3m": ens_runner.trades[-5:],
            "ultimate_ensemble_regime_weighted": ens_regime_runner.trades[-5:],
        }
        if not ensemble_only:
            results.update(
                {
                    "primary_only": asdict(primary_runner.metrics()),
                    "long_specialist_only": asdict(long_runner.metrics()),
                    "short_specialist_only": asdict(short_runner.metrics()),
                }
            )
            trade_samples.update(
                {
                    "primary_only": primary_runner.trades[-5:],
                    "long_specialist_only": long_runner.trades[-5:],
                    "short_specialist_only": short_runner.trades[-5:],
                }
            )
        return {"results": results, "trade_samples": trade_samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="data/rl_training_data_latest.csv")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--primary-ckpt", default="data/ensemble/ckpt/best_dsac_agents.pth")
    ap.add_argument("--long-ckpt", default="data/ensemble/ckpt/best_dsac_long_agents.pth")
    ap.add_argument("--short-ckpt", default="data/ensemble/ckpt/best_dsac_short_agents.pth")
    ap.add_argument("--ensemble-only", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    df = _load_frame(args.csv_path, args.start, args.end, args.max_rows)

    payload = {
        "csv_path": args.csv_path,
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "rows": int(len(df)),
        "primary_ckpt": args.primary_ckpt,
        "long_ckpt": args.long_ckpt,
        "short_ckpt": args.short_ckpt,
        "comparison": _run_compare(
            df,
            args.primary_ckpt,
            args.long_ckpt,
            args.short_ckpt,
            ensemble_only=bool(args.ensemble_only),
        ),
    }

    rank = sorted(
        payload["comparison"]["results"].items(),
        key=lambda kv: (kv[1]["pnl_pct"], -abs(kv[1]["mdd_pct"]), kv[1]["wr_pct"]),
        reverse=True,
    )
    payload["ranking"] = [k for k, _ in rank]

    out_json = args.out_json
    if not out_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = os.path.join("data/ensemble/metrics", f"ultimate_3model_compare_{ts}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(json.dumps(payload["comparison"]["results"], indent=2, ensure_ascii=False))
    print("ranking:", " > ".join(payload["ranking"]))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
