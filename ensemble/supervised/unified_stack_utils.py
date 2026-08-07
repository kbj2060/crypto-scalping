from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEE = 0.0005
SLIP = 0.0002
REGIME_COLS = ["regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"]


@dataclass
class BacktestResult:
    pnl_pct: float
    trades: int
    wr_pct: float
    mdd_pct: float
    longs: int
    shorts: int


def regime_name(df: pd.DataFrame) -> np.ndarray:
    return (
        df[REGIME_COLS]
        .idxmax(axis=1)
        .str.replace("regime_", "", regex=False)
        .to_numpy()
    )


def infer_side_from_probs(
    df: pd.DataFrame,
    long_col: str,
    flat_col: str,
    short_col: str,
) -> np.ndarray:
    lp = pd.to_numeric(df[long_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df[short_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    fp = pd.to_numeric(df[flat_col], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.where((lp >= sp) & (lp >= fp), 1, np.where((sp > lp) & (sp >= fp), -1, 0)).astype(np.int8)


def build_sparse_candidates(
    df: pd.DataFrame,
    long_col: str,
    flat_col: str,
    short_col: str,
    params: dict[str, Any],
    prefix: str = "ud_stack",
) -> pd.DataFrame:
    out = df.copy()
    raw_side = np.sign(pd.to_numeric(out["m7_action"], errors="coerce").fillna(0.0).to_numpy(np.float64)).astype(np.int8)
    sup_side = infer_side_from_probs(out, long_col=long_col, flat_col=flat_col, short_col=short_col)
    q = pd.to_numeric(out["m7_target_quality"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    raw_edge = (
        pd.to_numeric(out["m7_prob_up"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["m7_prob_dn"], errors="coerce").fillna(0.0)
    ).abs().to_numpy(np.float64)
    sup_prob_max = (
        out[[long_col, short_col, flat_col]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .to_numpy(np.float64)
    )
    hold_pred = (
        pd.to_numeric(out["m7_hold_pred"], errors="coerce")
        .fillna(6.0)
        .clip(4.0, 8.0)
        .round()
        .astype(np.int32)
        .to_numpy()
    )
    raw_change = np.r_[True, raw_side[1:] != raw_side[:-1]]
    agree = (raw_side == sup_side) & (sup_side != 0)
    regimes = regime_name(out)

    candidate = np.zeros(len(out), dtype=np.int8)
    side = np.zeros(len(out), dtype=np.int8)
    last_idx = -10**9
    for i in range(len(out)):
        if sup_side[i] == 0:
            continue
        if q[i] < float(params["quality_min"]):
            continue
        if raw_edge[i] < float(params["raw_edge_min"]):
            continue
        if sup_prob_max[i] < float(params["sup_prob_min"]):
            continue
        if bool(params.get("require_agreement", False)) and not agree[i]:
            continue
        if bool(params.get("sign_change_only", False)) and not raw_change[i]:
            continue
        if i - last_idx < int(params["debounce_bars"]):
            continue
        candidate[i] = 1
        side[i] = sup_side[i]
        last_idx = i

    out[f"{prefix}_flag"] = candidate
    out[f"{prefix}_side"] = side
    out[f"{prefix}_hold"] = hold_pred
    out[f"{prefix}_quality"] = q
    out[f"{prefix}_raw_edge"] = raw_edge
    out[f"{prefix}_prob_max"] = sup_prob_max
    out[f"{prefix}_regime"] = regimes
    return out


def apply_regime_veto(
    df: pd.DataFrame,
    veto_rule: dict[str, str],
    prefix: str = "ud_stack",
) -> np.ndarray:
    cand = pd.to_numeric(df[f"{prefix}_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df[f"{prefix}_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    regimes = df[f"{prefix}_regime"].astype(str).to_numpy()
    out = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if cand[i] != 1 or side[i] == 0:
            continue
        mode = str(veto_rule.get(regimes[i], "skip"))
        out[i] = (
            mode == "both"
            or (mode == "long" and side[i] == 1)
            or (mode == "short" and side[i] == -1)
        )
    return out


def signed_return(entry_fill: float, exit_fill: float, side: int) -> float:
    return ((exit_fill - entry_fill) / max(entry_fill, 1e-8)) if side == 1 else ((entry_fill - exit_fill) / max(entry_fill, 1e-8))


def simulate_trades(
    df: pd.DataFrame,
    take_mask: np.ndarray,
    prefix: str = "ud_stack",
    hold_scale: float = 1.0,
    close_on_opp: bool = False,
) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df[f"{prefix}_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df[f"{prefix}_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df[f"{prefix}_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()

    rows: list[dict[str, Any]] = []
    pos = 0
    entry_idx = -1
    entry_fill = 0.0
    hold = 0
    target_hold = 0
    for i in range(len(df)):
        allowed = bool(take_mask[i]) and cand[i] == 1 and side[i] != 0
        if pos == 0:
            if allowed:
                pos = int(side[i])
                entry_idx = i
                entry_fill = close[i] * (1.0 + SLIP) if pos == 1 else close[i] * (1.0 - SLIP)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * hold_scale)))
        else:
            hold += 1
            reverse = allowed and side[i] == -pos and close_on_opp
            if reverse or hold >= target_hold:
                exit_fill = close[i] * (1.0 - SLIP) if pos == 1 else close[i] * (1.0 + SLIP)
                pnl = signed_return(entry_fill, exit_fill, pos)
                rows.append(
                    {
                        "entry_idx": int(entry_idx),
                        "exit_idx": int(i),
                        "side": int(pos),
                        "entry_fill": float(entry_fill),
                        "exit_fill": float(exit_fill),
                        "target_hold": int(target_hold),
                        "pnl": float(pnl),
                    }
                )
                pos = 0
                entry_idx = -1
                entry_fill = 0.0
                hold = 0
                target_hold = 0
    if pos != 0:
        exit_fill = close[-1] * (1.0 - SLIP) if pos == 1 else close[-1] * (1.0 + SLIP)
        pnl = signed_return(entry_fill, exit_fill, pos)
        rows.append(
            {
                "entry_idx": int(entry_idx),
                "exit_idx": int(len(df) - 1),
                "side": int(pos),
                "entry_fill": float(entry_fill),
                "exit_fill": float(exit_fill),
                "target_hold": int(target_hold),
                "pnl": float(pnl),
            }
        )
    return pd.DataFrame(rows)


def build_hazard_rows(
    df: pd.DataFrame,
    trades: pd.DataFrame,
    prob_cols: dict[str, str],
    min_hold_bars: int = 2,
    improve_margin: float = 0.0015,
    adverse_gap: float = 0.0035,
) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    regimes = regime_name(df)
    rows: list[dict[str, Any]] = []
    for t in trades.itertuples(index=False):
        entry_idx = int(t.entry_idx)
        exit_idx = int(t.exit_idx)
        side = int(t.side)
        entry_fill = float(t.entry_fill)
        if exit_idx - entry_idx <= min_hold_bars:
            continue

        signed_path = []
        for i in range(entry_idx, exit_idx + 1):
            fill = close[i] * (1.0 - SLIP) if side == 1 else close[i] * (1.0 + SLIP)
            signed_path.append(signed_return(entry_fill, fill, side))
        signed_path = np.asarray(signed_path, dtype=np.float64)
        final_pnl = float(signed_path[-1])

        for rel_i in range(min_hold_bars, len(signed_path) - 1):
            i = entry_idx + rel_i
            current_pnl = float(signed_path[rel_i])
            future_slice = signed_path[rel_i:]
            future_min = float(np.min(future_slice))
            bars_held = rel_i
            remaining = exit_idx - i
            label = int((current_pnl >= final_pnl + improve_margin) or ((current_pnl - future_min) >= adverse_gap))
            rows.append(
                {
                    "timestamp": df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                    "haz_side": side,
                    "haz_bars_held": bars_held,
                    "haz_bars_held_norm": float(bars_held / max(int(t.target_hold), 1)),
                    "haz_remaining_norm": float(remaining / max(int(t.target_hold), 1)),
                    "haz_current_pnl": current_pnl,
                    "haz_mfe_sofar": float(np.max(signed_path[: rel_i + 1])),
                    "haz_mae_sofar": float(np.min(signed_path[: rel_i + 1])),
                    "haz_final_pnl": final_pnl,
                    "haz_future_min": future_min,
                    "haz_exit_label": label,
                    "haz_regime": regimes[i],
                    "haz_long_prob": float(pd.to_numeric(df.iloc[i][prob_cols["long"]], errors="coerce")),
                    "haz_flat_prob": float(pd.to_numeric(df.iloc[i][prob_cols["flat"]], errors="coerce")),
                    "haz_short_prob": float(pd.to_numeric(df.iloc[i][prob_cols["short"]], errors="coerce")),
                    "haz_prob_max": float(
                        max(
                            pd.to_numeric(df.iloc[i][prob_cols["long"]], errors="coerce"),
                            pd.to_numeric(df.iloc[i][prob_cols["flat"]], errors="coerce"),
                            pd.to_numeric(df.iloc[i][prob_cols["short"]], errors="coerce"),
                        )
                    ),
                    "m7_target_quality": float(pd.to_numeric(df.iloc[i]["m7_target_quality"], errors="coerce")),
                    "smart_money_flow": float(pd.to_numeric(df.iloc[i]["smart_money_flow"], errors="coerce")),
                    "taker_acceleration": float(pd.to_numeric(df.iloc[i]["taker_acceleration"], errors="coerce")),
                    "trade_intensity": float(pd.to_numeric(df.iloc[i]["trade_intensity"], errors="coerce")),
                    "garch_vol_z": float(pd.to_numeric(df.iloc[i]["garch_vol_z"], errors="coerce")),
                    "rogers_satchell_vol": float(pd.to_numeric(df.iloc[i]["rogers_satchell_vol"], errors="coerce")),
                    "amihud_illiquidity_z": float(pd.to_numeric(df.iloc[i]["amihud_illiquidity_z"], errors="coerce")),
                    "patchtst_regime_sim": float(pd.to_numeric(df.iloc[i].get("patchtst_regime_sim", 0.0), errors="coerce")),
                    "timesnet_cycle_delta": float(pd.to_numeric(df.iloc[i].get("timesnet_cycle_delta", 0.0), errors="coerce")),
                    "dlinear_smf_slope": float(pd.to_numeric(df.iloc[i].get("dlinear_smf_slope", 0.0), errors="coerce")),
                    "regime_bull": float(pd.to_numeric(df.iloc[i]["regime_bull"], errors="coerce")),
                    "regime_bear": float(pd.to_numeric(df.iloc[i]["regime_bear"], errors="coerce")),
                    "regime_chop": float(pd.to_numeric(df.iloc[i]["regime_chop"], errors="coerce")),
                    "regime_whipsaw": float(pd.to_numeric(df.iloc[i]["regime_whipsaw"], errors="coerce")),
                    "regime_normal": float(pd.to_numeric(df.iloc[i]["regime_normal"], errors="coerce")),
                }
            )
    return pd.DataFrame(rows)


def run_backtest_with_hazard(
    df: pd.DataFrame,
    take_mask: np.ndarray,
    prefix: str = "ud_stack",
    hold_scale: float = 1.0,
    close_on_opp: bool = False,
    hazard_payload: dict[str, Any] | None = None,
    hazard_threshold: float = 0.65,
    min_hold_bars: int = 2,
) -> BacktestResult:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df[f"{prefix}_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side_arr = pd.to_numeric(df[f"{prefix}_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df[f"{prefix}_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    balance = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    longs = 0
    shorts = 0

    hazard_model = None
    hazard_features: list[str] = []
    if hazard_payload is not None:
        hazard_model = hazard_payload["model"]
        hazard_features = list(hazard_payload["feature_cols"])

    pos = 0
    entry_idx = -1
    entry_fill = 0.0
    hold = 0
    target_hold = 0
    for i in range(len(df)):
        allowed = bool(take_mask[i]) and cand[i] == 1 and side_arr[i] != 0
        if pos == 0:
            if allowed:
                pos = int(side_arr[i])
                entry_idx = i
                entry_fill = close[i] * (1.0 + SLIP) if pos == 1 else close[i] * (1.0 - SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * hold_scale)))
                longs += int(pos == 1)
                shorts += int(pos == -1)
        else:
            hold += 1
            reverse = allowed and side_arr[i] == -pos and close_on_opp
            hazard_exit = False
            if hazard_model is not None and hold >= min_hold_bars:
                current_fill = close[i] * (1.0 - SLIP) if pos == 1 else close[i] * (1.0 + SLIP)
                current_pnl = signed_return(entry_fill, current_fill, pos)
                start_fill = close[entry_idx:i + 1]
                if pos == 1:
                    sofar = (start_fill * (1.0 - SLIP) - entry_fill) / max(entry_fill, 1e-8)
                else:
                    sofar = (entry_fill - start_fill * (1.0 + SLIP)) / max(entry_fill, 1e-8)
                row = {
                    "haz_side": pos,
                    "haz_bars_held": hold,
                    "haz_bars_held_norm": float(hold / max(target_hold, 1)),
                    "haz_remaining_norm": float(max(target_hold - hold, 0) / max(target_hold, 1)),
                    "haz_current_pnl": float(current_pnl),
                    "haz_mfe_sofar": float(np.max(sofar)),
                    "haz_mae_sofar": float(np.min(sofar)),
                    "haz_long_prob": float(pd.to_numeric(df.iloc[i].get("ud_tsfm_long_prob", 0.0), errors="coerce")),
                    "haz_flat_prob": float(pd.to_numeric(df.iloc[i].get("ud_tsfm_flat_prob", 0.0), errors="coerce")),
                    "haz_short_prob": float(pd.to_numeric(df.iloc[i].get("ud_tsfm_short_prob", 0.0), errors="coerce")),
                    "haz_prob_max": float(pd.to_numeric(df.iloc[i].get("ud_tsfm_prob_max", 0.0), errors="coerce")),
                    "m7_target_quality": float(pd.to_numeric(df.iloc[i]["m7_target_quality"], errors="coerce")),
                    "smart_money_flow": float(pd.to_numeric(df.iloc[i]["smart_money_flow"], errors="coerce")),
                    "taker_acceleration": float(pd.to_numeric(df.iloc[i]["taker_acceleration"], errors="coerce")),
                    "trade_intensity": float(pd.to_numeric(df.iloc[i]["trade_intensity"], errors="coerce")),
                    "garch_vol_z": float(pd.to_numeric(df.iloc[i]["garch_vol_z"], errors="coerce")),
                    "rogers_satchell_vol": float(pd.to_numeric(df.iloc[i]["rogers_satchell_vol"], errors="coerce")),
                    "amihud_illiquidity_z": float(pd.to_numeric(df.iloc[i]["amihud_illiquidity_z"], errors="coerce")),
                    "patchtst_regime_sim": float(pd.to_numeric(df.iloc[i].get("patchtst_regime_sim", 0.0), errors="coerce")),
                    "timesnet_cycle_delta": float(pd.to_numeric(df.iloc[i].get("timesnet_cycle_delta", 0.0), errors="coerce")),
                    "dlinear_smf_slope": float(pd.to_numeric(df.iloc[i].get("dlinear_smf_slope", 0.0), errors="coerce")),
                    "regime_bull": float(pd.to_numeric(df.iloc[i]["regime_bull"], errors="coerce")),
                    "regime_bear": float(pd.to_numeric(df.iloc[i]["regime_bear"], errors="coerce")),
                    "regime_chop": float(pd.to_numeric(df.iloc[i]["regime_chop"], errors="coerce")),
                    "regime_whipsaw": float(pd.to_numeric(df.iloc[i]["regime_whipsaw"], errors="coerce")),
                    "regime_normal": float(pd.to_numeric(df.iloc[i]["regime_normal"], errors="coerce")),
                }
                x = pd.DataFrame([{k: row.get(k, 0.0) for k in hazard_features}])
                prob = float(hazard_model.predict_proba(x)[0, 1])
                hazard_exit = prob >= hazard_threshold

            if reverse or hold >= target_hold or hazard_exit:
                exit_fill = close[i] * (1.0 - SLIP) if pos == 1 else close[i] * (1.0 + SLIP)
                pnl = signed_return(entry_fill, exit_fill, pos)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                wins += int(pnl > 0)
                pos = 0
                entry_idx = -1
                entry_fill = 0.0
                hold = 0
                target_hold = 0
        peak = max(peak, balance)
        mdd = min(mdd, balance / max(peak, 1e-8) - 1.0)

    if pos != 0:
        exit_fill = close[-1] * (1.0 - SLIP) if pos == 1 else close[-1] * (1.0 + SLIP)
        pnl = signed_return(entry_fill, exit_fill, pos)
        balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
        trades += 1
        wins += int(pnl > 0)

    return BacktestResult(
        pnl_pct=float((balance - 1.0) * 100.0),
        trades=int(trades),
        wr_pct=float(wins / max(trades, 1) * 100.0),
        mdd_pct=float(mdd * 100.0),
        longs=int(longs),
        shorts=int(shorts),
    )
