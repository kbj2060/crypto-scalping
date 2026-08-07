#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as ex  # noqa: E402


MODEL_ID = "omega1_2_1_exit_q_market_features_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_POS_FEATURES = ex._pos_features

MARKET_COLS = (
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "sum_open_interest_value",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "last_funding_rate",
    "close_btc",
    "volume_btc",
    "quote_volume_btc",
    "whale_retail_ratio",
    "whale_conviction",
    "smart_money_flow",
    "squeeze_power",
    "oi_change_rate",
    "net_taker_ratio",
    "taker_acceleration",
    "trade_intensity",
    "big_trade_ratio",
    "log_return",
    "volatility_z",
    "rsi",
    "macd_hist",
    "bb_width",
    "bb_width_z",
    "hma_slope",
    "wick_ratio",
    "garman_klass_vol",
    "realized_vol_ratio",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rogers_satchell_vol",
    "parkinson_vol",
    "amihud_illiquidity_z",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "fvg_dist",
    "chop_index",
    "cvp_poc_dist",
    "cvp_vah_val_width",
    "cvp_cluster_position",
    "cvp_volume_imbalance",
    "turtle_signal",
    "dual_momentum",
    "mean_reversion_z",
    "breakout_strength",
    "volume_profile_signal",
    "vwap_dist",
    "regime_break",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_z_score",
    "long_squeeze_risk",
    "short_squeeze_risk",
    "funding_price_divergence",
    "hurst_48",
    "hurst_288",
    "hurst_change",
    "regime_trending",
    "ofi_acceleration",
    "kalman_velocity",
    "realized_skewness",
    "ofti",
    "garch_vol_z",
    "liquidity_vacuum",
    "execution_quality",
    "jump_z",
    "jump_flag",
    "evt_tail_flag",
)


def _json_default(obj: Any) -> Any:
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


def _safe_series(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _market_feature_block(frame: pd.DataFrame) -> pd.DataFrame:
    out = ex._rolling_features(frame)
    close = pd.to_numeric(frame["close"], errors="raise").replace([np.inf, -np.inf], np.nan).ffill()
    out["mkt_close"] = close
    out["mkt_close_z48"] = ((close - close.rolling(48, min_periods=12).mean()) / close.rolling(48, min_periods=12).std()).replace([np.inf, -np.inf], np.nan)
    out["mkt_close_z288"] = ((close - close.rolling(288, min_periods=48).mean()) / close.rolling(288, min_periods=48).std()).replace([np.inf, -np.inf], np.nan)
    for col in MARKET_COLS:
        if col not in frame.columns:
            continue
        s = _safe_series(frame, col)
        out[f"mkt_{col}"] = s
        if col in {"volume", "quote_volume", "trades", "sum_open_interest_value", "close_btc", "volume_btc", "quote_volume_btc"}:
            denom = s.rolling(288, min_periods=24).median().replace(0.0, np.nan)
            out[f"mkt_{col}_rel288"] = (s / denom).replace([np.inf, -np.inf], np.nan)
        if col in {"smart_money_flow", "net_taker_ratio", "taker_acceleration", "oi_change_rate", "vwap_dist", "macd_hist", "hma_slope", "kalman_velocity"}:
            out[f"mkt_{col}_diff3"] = s.diff(3)
            out[f"mkt_{col}_mean12"] = s.rolling(12, min_periods=3).mean()
    out = out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return out


def _state_base_market(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = _market_feature_block(frame)
    cols = [
        "router_confidence",
        "router_margin",
        "dir_p_cash",
        "dir_p_long",
        "dir_p_short",
        "dir_confidence",
        "dir_side_edge",
        "dir_trade_prob",
        "quality_p_cash",
        "quality_p_long",
        "quality_p_short",
        "quality_for_action",
    ]
    for col in cols:
        out[f"tabm_{col}"] = pd.to_numeric(src[f"{prefix}{col}"], errors="raise").to_numpy(dtype=np.float64)
    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    for name in ("bull", "bear", "chop_expert"):
        out[f"tabm_router_{name}"] = expert.eq(name).astype(float).to_numpy()
    for col in ("action", "side", "quality_score", "confidence", "notional_exposure", "position_fraction", "leverage", "take_profit", "stop_loss"):
        out[f"dec_{col}"] = pd.to_numeric(dec[col], errors="raise").to_numpy(dtype=np.float64)
    out["dec_rr"] = out["dec_take_profit"] / np.maximum(np.abs(out["dec_stop_loss"]), 1e-8)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ex._reject_forbidden(list(out.columns), "exit_q_market_state_base")
    return out


def _pos_features_market(base_state: pd.DataFrame, pos: ex.Position, unreal: float, i: int) -> pd.DataFrame:
    row = BASE_POS_FEATURES(base_state, pos, unreal, i)
    cur_close = float(row.get("mkt_close", pd.Series([pos.entry_price])).iloc[0])
    side = float(pos.side)
    raw_since_entry = (cur_close - float(pos.entry_price)) / max(float(pos.entry_price), 1e-12)
    hold = max(int(i) - int(pos.entry_i), 0)
    mfe = max(float(pos.mfe), float(unreal))
    mae = min(float(pos.mae), float(unreal))
    vals = {
        "path_side_ret_since_entry": float(side * raw_since_entry),
        "path_unreal_per_bar": float(unreal / max(hold, 1)),
        "path_mfe_per_bar": float(mfe / max(hold, 1)),
        "path_mae_per_bar": float(mae / max(hold, 1)),
        "path_tp_progress_per_bar": float((unreal / max(float(pos.take_profit), 1e-8)) / max(hold, 1)),
        "path_mfe_to_mae": float(mfe / max(abs(mae), 1e-8)),
        "path_hold_log1p": float(np.log1p(hold)),
        "path_stale_profit": float(unreal > 0.0 and hold >= 96 and unreal < max(float(pos.take_profit), 1e-8) * 0.35),
        "path_deep_progress": float(unreal > max(float(pos.take_profit), 1e-8) * 0.65),
    }
    for lag in (3, 6, 12, 24, 48):
        col = f"ret_{lag}"
        if col in row.columns:
            vals[f"path_side_ret_{lag}"] = float(row[col].iloc[0]) * side
    for k, v in vals.items():
        row[k] = v
    row = row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ex._reject_forbidden(list(row.columns), "exit_q_market_pos_features")
    return row


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_adapter_actions": metrics.get("adapter_actions", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-states", type=int, default=4200)
    ap.add_argument("--max-forward-bars", type=int, default=432)
    ap.add_argument("--cql-weight", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=260612)
    ap.add_argument("--generators", default="high")
    ap.add_argument("--min-advs", default="0,0.001,0.0025,0.005,0.01,0.02")
    ap.add_argument("--full-exit-modes", default="0,1")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ex._state_base = _state_base_market  # type: ignore[assignment]
    ex._pos_features = _pos_features_market  # type: ignore[assignment]

    fee, slip = ex.omega._load_fee_slip()
    splits = ex._build_splits()
    built: dict[str, dict[str, Any]] = {}
    threshold_sets = {"high": ex.HIGH_THRESHOLDS, "low": ex.LOW_THRESHOLDS}
    requested = tuple(x.strip() for x in str(args.generators).split(",") if x.strip())
    for threshold_name in requested:
        if threshold_name not in threshold_sets:
            raise RuntimeError(f"unknown generator: {threshold_name}")
        built[threshold_name] = {}
        for split, payload in splits.items():
            dec = ex._to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=threshold_sets[threshold_name])
            state = _state_base_market(payload["frame"], payload["src"], dec, payload["prefix"])
            built[threshold_name][split] = {"frame": payload["frame"], "dec": dec, "state": state}

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for threshold_name in requested:
        val = built[threshold_name]["validation"]
        oos = built[threshold_name]["oos"]
        x_train, rewards, data_diag = ex._collect_dataset(
            val["frame"],
            val["dec"],
            val["state"],
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            stride=int(args.stride),
            max_states=int(args.max_states),
            max_forward_bars=int(args.max_forward_bars),
        )
        model, train_diag = ex._train_q(x_train, rewards, epochs=int(args.epochs), seed=int(args.seed), cql_weight=float(args.cql_weight))
        model_path = OUT_DIR / f"{threshold_name}_exit_q_market_features.pt"
        import torch

        torch.save({"state_dict": model.state_dict(), "norm": model.norm, "actions": ex.ACTION_NAMES, "train_diag": train_diag}, model_path)  # type: ignore[attr-defined]
        x_train.to_csv(OUT_DIR / f"{threshold_name}_train_states.csv", index=False)
        pd.DataFrame(rewards, columns=[f"reward_{a}" for a in ex.ACTION_NAMES]).to_csv(OUT_DIR / f"{threshold_name}_train_rewards.csv", index=False)

        val_base, val_base_ledger = ex._simulate_policy(val["frame"], val["dec"], val["state"], model=None, min_adv=1.0, fee=fee, slip=slip, cost_mult=3.0, allowed_full_exit=False)
        oos_base, oos_base_ledger = ex._simulate_policy(oos["frame"], oos["dec"], oos["state"], model=None, min_adv=1.0, fee=fee, slip=slip, cost_mult=3.0, allowed_full_exit=False)
        val_base_ledger.to_csv(OUT_DIR / f"{threshold_name}_validation_baseline_ledger.csv", index=False)
        oos_base_ledger.to_csv(OUT_DIR / f"{threshold_name}_oos_baseline_ledger.csv", index=False)
        rows.append({"candidate_generator": threshold_name, "policy": "baseline_no_exit_q", "min_adv": None, **_row("val", val_base), **_row("oos", oos_base)})

        for allowed_full_exit in tuple(bool(int(x.strip())) for x in str(args.full_exit_modes).split(",") if x.strip()):
            for min_adv in tuple(float(x.strip()) for x in str(args.min_advs).split(",") if x.strip()):
                val_m, val_ledger = ex._simulate_policy(
                    val["frame"],
                    val["dec"],
                    val["state"],
                    model=model,
                    min_adv=float(min_adv),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                    allowed_full_exit=bool(allowed_full_exit),
                )
                oos_m, oos_ledger = ex._simulate_policy(
                    oos["frame"],
                    oos["dec"],
                    oos["state"],
                    model=model,
                    min_adv=float(min_adv),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                    allowed_full_exit=bool(allowed_full_exit),
                )
                tag = f"{threshold_name}_{'full' if allowed_full_exit else 'def'}_adv{str(min_adv).replace('.', 'p')}"
                val_ledger.to_csv(OUT_DIR / f"validation_{tag}_ledger.csv", index=False)
                oos_ledger.to_csv(OUT_DIR / f"oos_{tag}_ledger.csv", index=False)
                rows.append(
                    {
                        "candidate_generator": threshold_name,
                        "policy": "exit_q_market_full_exit" if allowed_full_exit else "exit_q_market_defensive",
                        "min_adv": float(min_adv),
                        **_row("val", val_m),
                        **_row("oos", oos_m),
                        "val_entry_audit": ex._entry_audit(val_base_ledger, val_ledger),
                        "oos_entry_audit": ex._entry_audit(oos_base_ledger, oos_ledger),
                    }
                )
        reports[threshold_name] = {
            "dataset": data_diag,
            "training": train_diag,
            "state_feature_count": int(x_train.shape[1]),
            "model_path": str(model_path),
        }

    ranking = pd.DataFrame(rows)
    high_base = ranking[(ranking["candidate_generator"] == requested[0]) & (ranking["policy"] == "baseline_no_exit_q")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(high_base["oos_pnl"])
    ranking["delta_val_pnl"] = ranking["val_pnl"] - float(high_base["val_pnl"])
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "exit_q_market_features_ranking.csv", index=False)
    promotable = ranking[
        (ranking["policy"] != "baseline_no_exit_q")
        & (ranking["oos_pnl"] > float(high_base["oos_pnl"]))
        & (ranking["val_pnl"] > float(high_base["val_pnl"]) * 0.80)
        & (ranking["oos_mdd"] >= float(high_base["oos_mdd"]) * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "exit_q_market_features_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "purpose": "Exit Q model with expanded market features plus position/path features. Entry layer is frozen.",
        "baseline": high_base.to_dict(),
        "reports": reports,
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "exit_q_market_features_ranking.csv"),
            "promotable": str(OUT_DIR / "exit_q_market_features_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
