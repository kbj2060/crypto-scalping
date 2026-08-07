#!/usr/bin/env python3
"""Native supervised portfolio ranker with 2024/2025/2026 time splits.

Splits:
- 2024: train
- 2025-01-01..2025-08-31: calibration / auxiliary validation
- 2025-09-01..2025-12-31: final validation
- 2026: OOS

The replay is native bar-by-bar. It generates current-bar candidates from the
frozen ETH/SOL/BTC Omega artifacts and closes positions with the corresponding
asset/component exit-head, TP/SL, fee, and slippage contracts.
"""
from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_omega4_6_1_greedy_router_20260706 as eth_greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as eth_valmod  # noqa: E402
import replay_omega4_6_1_two_component_router_assets_20260708 as asset_router  # noqa: E402
import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as eth_retest  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

MODEL_ID = "portfolio_supervised_ranker_native_split_20260709"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
DOC_PATH = ROOT / f"docs/model_contracts/{MODEL_ID}.md"
ASSETS = ("eth", "sol", "btc")

BASE_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
WIDE24_2024 = WIDE24_DIR / "training_features_2024_regime3_current_sensitive_hmm_wide24.csv"
WIDE24_2025 = WIDE24_DIR / "training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
ASSET_2024 = {
    "sol": ROOT / "data/splits/year_oos/sol_features_2024.csv",
    "btc": ROOT / "data/splits/year_oos/btc_features_2024.csv",
}
DIAGNOSTICS: dict[str, Any] = {}

ETH_PRED = {
    "train": {
        "h48qual": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/train_predictions_q050.csv",
        "zig075": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/train_predictions_q075.csv",
    },
    "validation": eth_valmod.VAL_PRED,
    "oos": {
        name: eth_greedy.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
        for name, cfg in eth_retest.COMPONENTS.items()
    },
}

FEATURE_COLS = [
    "asset_eth",
    "asset_sol",
    "asset_btc",
    "side_long",
    "side_short",
    "notional",
    "margin_fraction",
    "leverage",
    "take_profit",
    "stop_loss",
    "ou_halflife",
    "asset_score",
    "hour_sin",
    "hour_cos",
    "month_norm",
]


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


def _window(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if start is not None:
        out = out[out["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out["timestamp"] <= pd.Timestamp(end)]
    return out.reset_index(drop=True)


def _load_eth_base(year: int, start: str | None, end: str | None) -> pd.DataFrame:
    if year == 2024:
        base_path, overlay_path = BASE_2024, WIDE24_2024
    elif year == 2025:
        base_path, overlay_path = BASE_2025, WIDE24_2025
    else:
        return eth_retest.load_frame_current(start or "2026-01-01", end or "2026-06-30")
    frame = pd.read_csv(base_path, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(overlay_path, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    return _window(frame, start, end)


@torch.no_grad()
def _prediction_frame(frame: pd.DataFrame, bundle_path: Path, threshold: float, prefix: str, device: torch.device) -> pd.DataFrame:
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    missing = sorted(set(base_cols) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{bundle_path}: frame missing base columns: {missing[:20]}")
    x = parent._base_input(frame, base_cols)
    models = dict(bundle["models"])
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    return parent._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)


def _prepare_eth_components(split: str, frame: pd.DataFrame, device: torch.device) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for name, cfg in eth_retest.COMPONENTS.items():
        if split == "train_2024":
            pred = _prediction_frame(frame, Path(cfg["bundle"]), float(cfg["q_tag"].replace("q", "")) / 100.0, "omega1_regime3_expertdq", device)
        else:
            pred = pd.read_csv(ETH_PRED[split][name])
        pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[pred["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        frame_c = frame[frame["timestamp"].isin(pred["timestamp"])].reset_index(drop=True)
        if len(frame_c) != len(frame):
            raise RuntimeError(f"ETH {split} {name}: frame/prediction timestamp intersection changed row count")
        tmp_pred = OUT_DIR / f"_eth_{split}_{name}_aligned.csv"
        tmp_pred.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(tmp_pred, index=False)
        components[name] = eth_greedy.prepare_component(frame_c, tmp_pred, cfg, device)
        components[name]["sidecar"] = eth_greedy.sidecar
        components[name]["long_scale"] = eth_greedy.SCALE_MAP[f"{name}_L"]
        components[name]["short_scale"] = eth_greedy.SCALE_MAP[f"{name}_S"]
    return components


def _concat_components(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise RuntimeError("cannot concatenate empty component list")
    out = dict(parts[0])
    out["dec"] = pd.concat([p["dec"] for p in parts], ignore_index=True)
    for key in ("margin", "leverage", "base_np", "route"):
        out[key] = np.concatenate([np.asarray(p[key]) for p in parts], axis=0)
    return out


def _slice_component(comp: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    out = dict(comp)
    out["dec"] = comp["dec"].iloc[idx].reset_index(drop=True)
    for key in ("margin", "leverage", "base_np", "route"):
        out[key] = np.asarray(comp[key])[idx]
    return out


def _load_asset_2024(asset: str) -> pd.DataFrame:
    frame = pd.read_csv(ASSET_2024[asset], low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24_2024, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    missing = frame[cols].isna().any(axis=1)
    if missing.any():
        DIAGNOSTICS[f"{asset}_2024_overlay_missing_rows_dropped"] = int(missing.sum())
        print(f"stage=drop_overlay_gaps asset={asset} rows={int(missing.sum())}", flush=True)
        frame = frame.loc[~missing].reset_index(drop=True)
    return frame


def _asset_prediction_frame(asset: str, frame: pd.DataFrame, cfg: dict[str, Any], device: torch.device) -> pd.DataFrame:
    tag = str(cfg["tag"])
    threshold = float(tag.replace("q", "")) / 100.0
    return _prediction_frame(frame, ROOT / cfg["parent_dir"] / "true_3head_tabm_bundle.pt", threshold, "omega1_regime3_expertdq_oof", device)


def _prepare_asset_component_from_pred(asset: str, frame: pd.DataFrame, pred: pd.DataFrame, cfg: dict[str, Any], *, oof: bool, device: torch.device) -> dict[str, Any]:
    date = asset_router.ASSET_DATES[asset]
    sidecar = importlib.import_module(f"train_eval_omega4_2_risk_sidecar_{asset}_{date}")
    bundle = torch.load(ROOT / cfg["parent_dir"] / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(dict(bundle["models"]), device=device)
    pred = pred.copy().reset_index(drop=True)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"])
    frame_ts = pd.to_datetime(frame["timestamp"]).reset_index(drop=True)
    pred_ts = pd.to_datetime(pred["timestamp"]).reset_index(drop=True)
    if len(pred_ts) != len(frame_ts) or not pred_ts.equals(frame_ts):
        raise RuntimeError(f"{asset}: custom prediction timestamps do not match frame")
    x = parent._base_input(frame, base_cols)
    dec_base = parent._to_decisions(pred, oof=bool(oof))
    dec, _ = atr_eval._apply_atr_safety_sltp(
        dec_base,
        frame,
        atr_window=192,
        tp_mult=12.0,
        sl_mult=6.0,
        min_tp=0.075,
        min_sl=0.040,
        max_tp=0.22,
        max_sl=0.12,
    )
    atr = atr_eval._atr_pct(frame, 192)
    with open(ROOT / cfg["risk_dir"] / "risk_sidecar.pkl", "rb") as f:
        pkl = pickle.load(f)
    features = sidecar._risk_feature_frame(frame, pred, dec, base_cols, atr_pct=atr, feature_mode=pkl["risk_feature_mode"])
    x_all, _ = sidecar._feature_matrix(features, pkl["feature_columns"])
    side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    score = sidecar._predict_side_split_models(pkl["model"], x_all, side_all) if pkl["side_split_model"] else np.asarray(pkl["model"].predict(x_all), dtype=np.float64)
    mapping = pkl["selected_mapping"]
    margin = sidecar._risk_margins(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.MARGIN_CFG_KEYS})
    leverage = sidecar._risk_leverage(dec, score, train_q50=pkl["train_score_q50"], train_iqr=pkl["train_score_iqr"], **{k: mapping[k] for k in sidecar.LEVERAGE_CFG_KEYS}) if pkl["dynamic_leverage"] else np.ones(len(dec))
    base_np, exit_runtime, pos_idx = sidecar._prepare_exit_runtime(x, loaded)
    return {
        "dec": dec,
        "margin": margin,
        "leverage": leverage,
        "base_np": base_np,
        "exit_runtime": exit_runtime,
        "pos_idx": pos_idx,
        "route": hard._route_id(frame),
        "exit_threshold": 0.95,
        "long_scale": float(cfg["long_scale"]),
        "short_scale": float(cfg["short_scale"]),
        "sidecar": sidecar,
    }


def _asset_component(asset: str, split: str, start: str | None, end: str | None, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], tuple[float, float]]:
    cfg = native.SOL_CFG if asset == "sol" else native.BTC_CFG
    if split == "train_2024":
        frame = _load_asset_2024(asset)
        pred = _asset_prediction_frame(asset, frame, cfg, device)
        full_comp = _prepare_asset_component_from_pred(asset, frame, pred, cfg, oof=True, device=device)
    else:
        frames = asset_router._load_frames(asset)
        if split == "validation_2025":
            train_frame = frames["train_raw"].copy()
            val_frame = frames["val_raw"].copy()
            train_frame["timestamp"] = pd.to_datetime(train_frame["timestamp"])
            val_frame["timestamp"] = pd.to_datetime(val_frame["timestamp"])
            train_comp = asset_router._prepare_component(asset, "train", train_frame, cfg, device=device)
            val_comp = asset_router._prepare_component(asset, "validation", val_frame, cfg, device=device)
            frame = pd.concat([train_frame, val_frame], ignore_index=True)
            full_comp = _concat_components([train_comp, val_comp])
        else:
            frame = frames["oos_raw"].copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            full_comp = asset_router._prepare_component(asset, "oos", frame, cfg, device=device)
    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= frame["timestamp"].to_numpy() >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        mask &= frame["timestamp"].to_numpy() <= np.datetime64(pd.Timestamp(end))
    idx = np.flatnonzero(mask)
    sliced_frame = frame.iloc[idx].reset_index(drop=True)
    comp_name = "zig075" if asset == "sol" else "h48qual"
    comp = _slice_component(full_comp, idx)
    omega = importlib.import_module(f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{asset_router.ASSET_DATES[asset]}")
    fee, slip = omega._load_fee_slip()
    return sliced_frame, {comp_name: comp}, (float(fee), float(slip))


def _eth_component(split: str, start: str | None, end: str | None, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], tuple[float, float]]:
    if split == "train_2024":
        frame = _load_eth_base(2024, start, end)
    elif split == "validation_2025":
        full_frame = _load_eth_base(2025, None, None)
        parts_frame: list[pd.DataFrame] = []
        parts_comp: dict[str, list[dict[str, Any]]] = {name: [] for name in eth_retest.COMPONENTS}
        for pred_split in ("train", "validation"):
            pred_any = next(iter(ETH_PRED[pred_split].values()))
            pred_ts = pd.to_datetime(pd.read_csv(pred_any, usecols=["timestamp"])["timestamp"])
            frame_part = full_frame[full_frame["timestamp"].isin(pred_ts)].reset_index(drop=True)
            comp_part = _prepare_eth_components(pred_split, frame_part, device)
            parts_frame.append(frame_part)
            for name in parts_comp:
                parts_comp[name].append(comp_part[name])
        frame = pd.concat(parts_frame, ignore_index=True)
        components = {name: _concat_components(parts) for name, parts in parts_comp.items()}
        idx = np.flatnonzero((frame["timestamp"] >= pd.Timestamp(start)) & (frame["timestamp"] <= pd.Timestamp(end)))
        frame = frame.iloc[idx].reset_index(drop=True)
        components = {name: _slice_component(comp, idx) for name, comp in components.items()}
        fee, slip = eth_greedy.omega._load_fee_slip()
        return frame, components, (float(fee), float(slip))
    else:
        frame = _load_eth_base(2026, start, end)
    components = _prepare_eth_components(split, frame, device)
    fee, slip = eth_greedy.omega._load_fee_slip()
    return frame, components, (float(fee), float(slip))


def _build_world(split: str, start: str | None, end: str | None, device: torch.device) -> dict[str, Any]:
    eth_frame, eth_comps, eth_fee = _eth_component(split, start, end, device)
    sol_frame, sol_comps, sol_fee = _asset_component("sol", split, start, end, device)
    btc_frame, btc_comps, btc_fee = _asset_component("btc", split, start, end, device)
    world = {
        "eth": {"frame": eth_frame, "components": eth_comps, "fee_slip": eth_fee, "arrays": native._arrays(eth_frame), "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(eth_frame["timestamp"])}},
        "sol": {"frame": sol_frame, "components": sol_comps, "fee_slip": sol_fee, "arrays": native._arrays(sol_frame), "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(sol_frame["timestamp"])}},
        "btc": {"frame": btc_frame, "components": btc_comps, "fee_slip": btc_fee, "arrays": native._arrays(btc_frame), "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(btc_frame["timestamp"])}},
    }
    common = set(world["eth"]["ts_to_i"]).intersection(world["sol"]["ts_to_i"]).intersection(world["btc"]["ts_to_i"])
    world["timestamps"] = sorted(common)
    return world


def _flat_decision_candidate_rows(world: dict[str, Any], device: torch.device) -> list[tuple[pd.Timestamp, native.Candidate]]:
    rows: list[tuple[pd.Timestamp, native.Candidate]] = []
    position: native.Position | None = None
    cash = 1.0
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, _closed, _mark = native._try_close(world, position, ts, cash, device)
            continue
        candidates = [native._candidate_for_asset(world, asset, ts) for asset in ASSETS]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            continue
        rows.extend((ts, c) for c in candidates)
        candidates.sort(key=lambda c: (native.ASSET_SCORES[c.asset], c.notional), reverse=True)
        position, cash = native._open_position(world, candidates[0], cash)
    return rows


def _features(world: dict[str, Any], c: native.Candidate, ts: pd.Timestamp) -> dict[str, float]:
    return {
        "asset_eth": float(c.asset == "eth"),
        "asset_sol": float(c.asset == "sol"),
        "asset_btc": float(c.asset == "btc"),
        "side_long": float(c.side > 0),
        "side_short": float(c.side < 0),
        "notional": float(c.notional),
        "margin_fraction": float(c.margin),
        "leverage": float(c.leverage),
        "take_profit": float(c.take_profit),
        "stop_loss": float(c.stop_loss),
        "ou_halflife": float(world[c.asset]["frame"]["ou_halflife"].iloc[c.local_i]),
        "asset_score": float(native.ASSET_SCORES[c.asset]),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24.0)),
        "month_norm": float((ts.month - 6.5) / 6.0),
    }


def _simulate_candidate(world: dict[str, Any], c: native.Candidate, device: torch.device) -> dict[str, Any]:
    pos, cash = native._open_position(world, c, 1.0)
    asset_frame = world[c.asset]["frame"]
    closed_row: dict[str, Any] | None = None
    for ts in asset_frame["timestamp"].iloc[c.local_i + 1 :]:
        pos, cash, closed, _mark = native._try_close(world, pos, pd.Timestamp(ts), cash, device)
        if closed is not None:
            closed_row = closed
            break
    if closed_row is None and pos is not None:
        cash, closed_row = native._force_close(world, pos, cash)
    return closed_row


def _build_dataset(world: dict[str, Any], device: torch.device) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = _flat_decision_candidate_rows(world, device)
    for idx, (ts, c) in enumerate(candidates):
        if idx % 50 == 0:
            print(f"stage=build_dataset idx={idx}/{len(candidates)}", flush=True)
        closed = _simulate_candidate(world, c, device)
        ret = float(closed["trade_return"])
        mae = float(closed.get("mae_price_move", 0.0) or 0.0)
        hold_bars = int(closed["exit_i"]) - int(closed["entry_i"])
        label = ret - 0.20 * max(0.0, -mae - 0.02) - 0.00001 * max(hold_bars, 0)
        rows.append(
            {
                "timestamp": ts,
                "asset": c.asset,
                "component": c.component,
                "label": float(label),
                "trade_return": ret,
                "mae_price_move": mae,
                "hold_bars": int(hold_bars),
                **_features(world, c, ts),
            }
        )
    return pd.DataFrame(rows)


def _train_model(train_df: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.04,
        num_leaves=7,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.2,
        reg_lambda=3.0,
        random_state=60709,
        verbose=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df["label"])
    return model


def _replay_ranker(world: dict[str, Any], model: lgb.LGBMRegressor, *, threshold: float, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: native.Position | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = native._try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                rows.append(closed)
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        candidates = [native._candidate_for_asset(world, asset, ts) for asset in ASSETS]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            continue
        feat_df = pd.DataFrame([_features(world, c, ts) for c in candidates])
        scores = model.predict(feat_df[FEATURE_COLS])
        best_i = int(np.argmax(scores))
        best_score = float(scores[best_i])
        decisions.append(
            {
                "timestamp": ts,
                "selected_asset": candidates[best_i].asset if best_score >= threshold else "cash",
                "selected_score": best_score,
                "threshold": float(threshold),
                **{f"score_{c.asset}": float(s) for c, s in zip(candidates, scores)},
            }
        )
        if best_score < threshold:
            continue
        position, cash = native._open_position(world, candidates[best_i], cash)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = native._force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = native._compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    metrics["decisions"] = int(len(decisions))
    metrics["cash_decisions"] = int(sum(d["selected_asset"] == "cash" for d in decisions))
    metrics["final_cash"] = float(cash)
    return metrics, ledger, pd.DataFrame(decisions)


def _score(metrics: dict[str, Any]) -> float:
    cash_ratio = metrics["cash_decisions"] / max(metrics["decisions"], 1)
    return float(metrics["pnl"]) - 0.35 * abs(float(metrics["mdd"])) - 8.0 * max(0.0, cash_ratio - 0.55)


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Supervised Ranker Native Split - 2026-07-09",
        "",
        "LightGBM candidate ranker trained on 2024 native counterfactual outcomes.",
        "",
        f"Selected threshold from 2025-01..2025-08 calibration: `{report['selected_threshold']}`",
        "",
        "| split | PnL | MDD | MTM MDD | trades | WR | decisions | cash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train_2024", "calibration_2025_01_08", "final_validation_2025_09_12", "oos_2026"):
        m = report["results"][split]
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} | {m.get('decisions', 0)} | {m.get('cash_decisions', 0)} |")
    lines.extend(
        [
            "",
            "Contract flags: `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false`.",
            "",
        ]
    )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = eth_retest.DEVICE
    splits = {
        "train_2024": ("train_2024", "2024-01-01", "2024-12-31 23:59:59"),
        "calibration_2025_01_08": ("validation_2025", "2025-01-01", "2025-08-31 23:59:59"),
        "final_validation_2025_09_12": ("validation_2025", "2025-09-01", "2025-12-31 23:59:59"),
        "oos_2026": ("oos", "2026-01-01", "2026-06-30 23:59:59"),
    }
    worlds: dict[str, dict[str, Any]] = {}
    for name, (split, start, end) in splits.items():
        print(f"stage=build_world name={name}", flush=True)
        worlds[name] = _build_world(split, start, end, device)
        print(f"stage=world_ready name={name} common_timestamps={len(worlds[name]['timestamps'])}", flush=True)

    print("stage=build_train_dataset", flush=True)
    train_df = _build_dataset(worlds["train_2024"], device)
    if len(train_df) < 20:
        raise RuntimeError(f"training dataset too thin: rows={len(train_df)}")
    train_df.to_csv(OUT_DIR / "train_2024_candidate_training_set.csv", index=False)
    print(f"stage=train_model rows={len(train_df)}", flush=True)
    model = _train_model(train_df)
    with open(OUT_DIR / "ranker_lgbm.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS}, f)

    threshold_grid = [-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.04, 0.08]
    grid_rows: list[dict[str, Any]] = []
    best_threshold: float | None = None
    best_score = -np.inf
    for th in threshold_grid:
        metrics, _ledger, _decisions = _replay_ranker(worlds["calibration_2025_01_08"], model, threshold=float(th), device=device)
        eligible = metrics["trades"] >= 15 and metrics["mdd"] >= -30.0
        score = _score(metrics) if eligible else -np.inf
        grid_rows.append({"threshold": float(th), "metrics": metrics, "eligible": bool(eligible), "score": float(score)})
        if eligible and score > best_score:
            best_score = float(score)
            best_threshold = float(th)
    if best_threshold is None:
        best_threshold = min(threshold_grid)
    pd.DataFrame(grid_rows).to_json(OUT_DIR / "calibration_threshold_grid.jsonl", orient="records", lines=True, force_ascii=False)

    results: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    decisions: dict[str, pd.DataFrame] = {}
    for name in ("train_2024", "calibration_2025_01_08", "final_validation_2025_09_12", "oos_2026"):
        print(f"stage=replay name={name} threshold={best_threshold}", flush=True)
        metrics, ledger, dec = _replay_ranker(worlds[name], model, threshold=best_threshold, device=device)
        results[name] = metrics
        ledgers[name] = ledger
        decisions[name] = dec
        ledger.to_csv(OUT_DIR / f"{name}_ledger.csv", index=False)
        dec.to_csv(OUT_DIR / f"{name}_decisions.csv", index=False)

    report = {
        "method": "portfolio_supervised_ranker_native_lgbm_time_split",
        "model_id": MODEL_ID,
        "split_contract": {
            "train": "2024-01-01..2024-12-31",
            "calibration_aux_validation": "2025-01-01..2025-08-31",
            "final_validation": "2025-09-01..2025-12-31",
            "oos": "2026-01-01..2026-06-30",
        },
        "training_data": "2024_native_counterfactual_candidate_outcomes",
        "parent_prediction_caveat": (
            "Existing train_predictions_qXXX.csv artifacts cover 2025 Jan-Sep, not 2024. "
            "For the requested 2024 train split this script scores the frozen parent bundles on 2024 features. "
            "This is a research split run, not a clean parent-model historical reproduction."
        ),
        "regime3_2024_overlay_caveat": (
            "SOL/BTC do not have asset-specific 2024 regime3_current wide24 sidecars in this workspace. "
            "The timestamp-only ETH 2024 wide24 sidecar supplies the six required regime3_current_sensitive_wide24 columns; "
            "rows with missing overlay timestamps are dropped and reported in diagnostics."
        ),
        "calibration_data": "2025-01-01_to_2025-08-31_threshold_selection_only",
        "final_validation_data": "2025-09-01_to_2025-12-31_not_used_for_training_or_threshold",
        "oos_usage": "reported_once_after_model_and_threshold_selection",
        "feature_cols": FEATURE_COLS,
        "training_rows": int(len(train_df)),
        "diagnostics": DIAGNOSTICS,
        "selected_threshold": best_threshold,
        "threshold_grid": grid_rows,
        "results": results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "selected_threshold": best_threshold, "results": results}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
