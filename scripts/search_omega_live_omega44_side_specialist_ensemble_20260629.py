#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega_live_omega44_1d_meta_barrier_20260629 as meta  # noqa: E402


MODEL_ID = "omega_live_omega44_side_specialist_ensemble_20260629"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"
BASE_META_DIR = ROOT / "tmp/causal_regen_20260516/omega_live_omega44_1d_meta_barrier_20260629"
BASE_REPORT = BASE_META_DIR / "report.json"

MAX_HOLD_BARS = 288
LEVERAGE_CAP = 5.0
TARGET_PNL = 100.0
TARGET_MDD = -20.0


@dataclass(frozen=True)
class SideSpec:
    variant: str
    side: int
    tp: float
    sl: float
    top_frac: float
    min_edge: float
    side_margin: float
    notional: float
    dd_governor: bool


@dataclass(frozen=True)
class EnsembleSpec:
    variant: str
    long_spec: SideSpec
    short_spec: SideSpec
    router: str


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


def _tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def _duration_days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    return max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1.0e-9)


def _load_base_report() -> dict[str, Any]:
    if not BASE_REPORT.exists():
        raise RuntimeError(f"missing base report: {BASE_REPORT}")
    return json.loads(BASE_REPORT.read_text(encoding="utf-8"))


def _load_models(report: dict[str, Any]) -> dict[tuple[float, float], tuple[Any, Any]]:
    out: dict[tuple[float, float], tuple[Any, Any]] = {}
    for pair_tag, paths in report["model_paths"].items():
        tp_text, sl_text = pair_tag.removeprefix("tp").split("_sl")
        tp = float(tp_text.replace("p", "."))
        sl = float(sl_text.replace("p", "."))
        with Path(paths["long"]).open("rb") as f:
            long_model = pickle.load(f)
        with Path(paths["short"]).open("rb") as f:
            short_model = pickle.load(f)
        out[(tp, sl)] = (long_model, short_model)
    return out


def _prepare_split_data(device: torch.device) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    omega44, _meta_info = meta._prepare_omega44_outputs(device)
    split_data: dict[str, dict[str, Any]] = {}
    align_diag: dict[str, Any] = {}
    for split, (frame, o44_src, o44_dec) in omega44.items():
        if split == "train":
            continue
        live_pred, live_prefix = meta._load_live_predictions(split)
        aligned_frame, aligned_o44_src, aligned_o44_dec, aligned_live, diag = meta._align_split(
            frame,
            o44_src,
            o44_dec,
            live_pred,
        )
        align_diag[split] = diag
        features = meta._feature_frame(aligned_frame, aligned_live, live_prefix, aligned_o44_src, aligned_o44_dec, split)
        split_data[split] = {"frame": aligned_frame, "features": features}

    feature_cols = list(split_data["validation"]["features"].columns)
    for split in ("validation", "oos"):
        missing = sorted(set(feature_cols) - set(split_data[split]["features"].columns))
        extra = sorted(set(split_data[split]["features"].columns) - set(feature_cols))
        if missing or extra:
            raise RuntimeError(f"{split} feature contract mismatch missing={missing[:10]} extra={extra[:10]}")
        split_data[split]["features"] = split_data[split]["features"].reindex(columns=feature_cols).astype(np.float32)
    return split_data, align_diag, feature_cols


def _predict_all(
    models: dict[tuple[float, float], tuple[Any, Any]],
    split_data: dict[str, dict[str, Any]],
) -> dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray]]]:
    out: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for pair, pair_models in models.items():
        out[pair] = {}
        for split in ("validation", "oos"):
            out[pair][split] = meta._predict_pair(pair_models, split_data[split]["features"])
    return out


def _side_signal(pred_long: np.ndarray, pred_short: np.ndarray, spec: SideSpec) -> tuple[np.ndarray, np.ndarray]:
    if spec.side > 0:
        own = pred_long
        other = pred_short
    else:
        own = pred_short
        other = pred_long
    score = own - float(spec.min_edge)
    active = (own >= float(spec.min_edge)) & ((own - other) >= float(spec.side_margin))
    return np.where(active, int(spec.side), 0).astype(np.int64), score.astype(np.float64)


def _candidate_side_specs(pair: tuple[float, float], side: int, pred_long: np.ndarray, pred_short: np.ndarray) -> list[SideSpec]:
    own = pred_long if side > 0 else pred_short
    finite = own[np.isfinite(own)]
    specs: list[SideSpec] = []
    side_name = "long" if side > 0 else "short"
    top_fracs = (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50)
    extras = (-0.001, 0.0, 0.0015, 0.003, 0.006)
    gaps = (0.0, 0.001, 0.0025, 0.005, 0.008)
    notionals = (0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
    for top_frac in top_fracs:
        q = 1.0 - float(top_frac)
        base_edge = float(np.quantile(finite, q)) if len(finite) else 0.0
        for extra in extras:
            min_edge = base_edge + float(extra)
            for gap in gaps:
                for notional in notionals:
                    for dd_governor in (False, True):
                        tp, sl = pair
                        specs.append(
                            SideSpec(
                                variant=(
                                    f"{side_name}_tp{_tag(tp)}_sl{_tag(sl)}_top{_tag(top_frac)}"
                                    f"_edge{_tag(min_edge)}_gap{_tag(gap)}_n{_tag(notional)}"
                                    f"{'_ddgov' if dd_governor else ''}"
                                ),
                                side=int(side),
                                tp=float(tp),
                                sl=float(sl),
                                top_frac=float(top_frac),
                                min_edge=float(min_edge),
                                side_margin=float(gap),
                                notional=float(notional),
                                dd_governor=bool(dd_governor),
                            )
                        )
    return specs


def _notional(spec: SideSpec, dd: float) -> float:
    n = float(spec.notional)
    if spec.dd_governor:
        if dd <= -0.16:
            n *= 0.35
        elif dd <= -0.12:
            n *= 0.55
        elif dd <= -0.08:
            n *= 0.75
    return float(np.clip(n, 0.0, LEVERAGE_CAP))


def _price_move_close(arrays: dict[str, np.ndarray], row_i: int, *, side: int, entry_price: float, slip_eff: float) -> float:
    px = float(arrays["close"][int(row_i)])
    if int(side) > 0:
        return float((px * (1.0 - slip_eff) - float(entry_price)) / max(float(entry_price), 1.0e-12))
    return float((float(entry_price) - px * (1.0 + slip_eff)) / max(float(entry_price), 1.0e-12))


def _replay_events(
    frame: pd.DataFrame,
    side: np.ndarray,
    spec_by_row: list[SideSpec | None],
    *,
    fee: float,
    slip: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = LEVERAGE_CAP
    margin_fraction = 0.0
    tp = 0.0
    sl = 0.0
    mfe = 0.0
    mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = margin_sum = leverage_sum = 0.0
    max_hold_seen = 0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = _price_move_close(arrays, int(i), side=pos, entry_price=entry_price, slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        dd = eq / max(peak, 1.0e-12) - 1.0
        mdd = min(mdd, dd)
        if pos != 0:
            hold = max(int(i) - int(entry_i), 0)
            max_hold_seen = max(max_hold_seen, hold)
            reason = ""
            if move >= tp:
                reason = "take_profit"
            elif move <= -abs(sl):
                reason = "stop_loss"
            elif hold >= MAX_HOLD_BARS:
                reason = "max_hold_1d"
            if reason:
                filled, exit_px, exit_fee, route = meta.omega._try_execution(
                    arrays,
                    int(i),
                    pos,
                    entry=False,
                    fee_base=fee_eff,
                    slip_base=slip_eff,
                )
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * float(exit_fee) * notional
                ret = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(
                    {
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                        "side": int(pos),
                        "reason": reason,
                        "win": int(win),
                        "raw_exit_price_move": float(raw_exit),
                        "mfe_price_move": float(mfe),
                        "mae_price_move": float(mae),
                        "trade_return": float(ret),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "take_profit": float(tp),
                        "stop_loss": float(sl),
                        "hold_bars": int(hold),
                        "cash_after": float(cash),
                        "exit_route": route,
                    }
                )
                pos = 0
                continue
        if pos != 0:
            continue
        row_side = int(side[int(i)]) if int(i) < len(side) else 0
        if row_side == 0:
            continue
        row_spec = spec_by_row[int(i)]
        if row_spec is None:
            continue
        filled, px, entry_fee, _route = meta.omega._try_execution(
            arrays,
            int(i),
            row_side,
            entry=True,
            fee_base=fee_eff,
            slip_base=slip_eff,
        )
        if not filled:
            continue
        entry_dd = cash / max(peak, 1.0e-12) - 1.0
        row_notional = _notional(row_spec, entry_dd)
        if row_notional <= 0.0:
            continue
        pos = row_side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        notional = row_notional
        margin_fraction = notional / max(leverage, 1.0e-12)
        tp = float(row_spec.tp)
        sl = float(row_spec.sl)
        cash -= cash * float(entry_fee) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        margin_sum += margin_fraction
        leverage_sum += leverage
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = meta.omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        hold = max(len(frame) - 1 - int(entry_i), 0)
        max_hold_seen = max(max_hold_seen, hold)
        ret = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(
            {
                "entry_signal_i": int(entry_signal_i),
                "entry_i": int(entry_i),
                "exit_i": int(len(frame) - 1),
                "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                "exit_timestamp": str(frame["timestamp"].iloc[-1]),
                "side": int(pos),
                "reason": "forced_end",
                "win": int(win),
                "raw_exit_price_move": float(raw_exit),
                "mfe_price_move": float(mfe),
                "mae_price_move": float(mae),
                "trade_return": float(ret),
                "notional": float(notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "take_profit": float(tp),
                "stop_loss": float(sl),
                "hold_bars": int(hold),
                "cash_after": float(cash),
                "exit_route": "forced_end",
            }
        )
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        max_hold_seen = int(max(max_hold_seen, int((pd.to_numeric(ledger["exit_i"]) - pd.to_numeric(ledger["entry_i"])).max())))
    n_entries = max(long_entries + short_entries, 1)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0,
            "trades_per_day": float(trades / _duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries),
            "avg_margin_fraction": float(margin_sum / n_entries),
            "avg_leverage": float(leverage_sum / n_entries),
            "max_leverage": LEVERAGE_CAP if trades else 0.0,
            "max_hold_bars": int(max_hold_seen),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "exit_reasons": reasons,
        },
        ledger,
    )


def _score_record(rec: dict[str, Any]) -> None:
    rec["pass_target"] = (
        rec["validation_pnl"] >= TARGET_PNL
        and rec["oos_pnl"] >= TARGET_PNL
        and rec["validation_mdd"] >= TARGET_MDD
        and rec["oos_mdd"] >= TARGET_MDD
        and rec["validation_max_hold_bars"] <= MAX_HOLD_BARS
        and rec["oos_max_hold_bars"] <= MAX_HOLD_BARS
        and rec["validation_max_leverage"] <= LEVERAGE_CAP
        and rec["oos_max_leverage"] <= LEVERAGE_CAP
    )
    rec["target_score"] = min(float(rec["validation_pnl"]), float(rec["oos_pnl"])) - 4.0 * max(0.0, TARGET_MDD - float(rec["validation_mdd"])) - 4.0 * max(0.0, TARGET_MDD - float(rec["oos_mdd"]))
    rec["validation_score"] = float(rec["validation_pnl"]) - 4.0 * max(0.0, TARGET_MDD - float(rec["validation_mdd"]))


def _eval_side_spec(
    spec: SideSpec,
    preds: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray]]],
    split_data: dict[str, dict[str, Any]],
    *,
    fee: float,
    slip: float,
    splits: tuple[str, ...] = ("validation", "oos"),
) -> dict[str, Any]:
    pair = (float(spec.tp), float(spec.sl))
    rec: dict[str, Any] = {
        "variant": spec.variant,
        "side": spec.side,
        "tp": spec.tp,
        "sl": spec.sl,
        "top_frac": spec.top_frac,
        "min_edge": spec.min_edge,
        "side_margin": spec.side_margin,
        "notional": spec.notional,
        "dd_governor": spec.dd_governor,
        "max_hold_contract_bars": MAX_HOLD_BARS,
        "leverage_cap": LEVERAGE_CAP,
    }
    for split in splits:
        pred_long, pred_short = preds[pair][split]
        signal, _score = _side_signal(pred_long, pred_short, spec)
        spec_by_row = [spec if int(x) != 0 else None for x in signal]
        metrics, _ledger = _replay_events(split_data[split]["frame"], signal, spec_by_row, fee=fee, slip=slip)
        for key, value in metrics.items():
            rec[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
    if "oos_pnl" in rec:
        _score_record(rec)
    else:
        rec["validation_score"] = float(rec["validation_pnl"]) - 4.0 * max(0.0, TARGET_MDD - float(rec["validation_mdd"]))
    return rec


def _ensemble_signal(
    spec: EnsembleSpec,
    preds: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray]]],
    split: str,
) -> tuple[np.ndarray, list[SideSpec | None]]:
    lp = (float(spec.long_spec.tp), float(spec.long_spec.sl))
    sp = (float(spec.short_spec.tp), float(spec.short_spec.sl))
    long_signal, long_score = _side_signal(*preds[lp][split], spec.long_spec)
    short_signal, short_score = _side_signal(*preds[sp][split], spec.short_spec)
    out_side = np.zeros(len(long_signal), dtype=np.int64)
    out_specs: list[SideSpec | None] = [None] * len(long_signal)
    for i in range(len(out_side)):
        l_active = int(long_signal[i]) > 0
        s_active = int(short_signal[i]) < 0
        if l_active and not s_active:
            out_side[i] = 1
            out_specs[i] = spec.long_spec
        elif s_active and not l_active:
            out_side[i] = -1
            out_specs[i] = spec.short_spec
        elif l_active and s_active:
            if spec.router == "none_on_conflict":
                continue
            if spec.router == "short_priority":
                out_side[i] = -1
                out_specs[i] = spec.short_spec
            elif spec.router == "long_priority":
                out_side[i] = 1
                out_specs[i] = spec.long_spec
            elif float(long_score[i]) >= float(short_score[i]):
                out_side[i] = 1
                out_specs[i] = spec.long_spec
            else:
                out_side[i] = -1
                out_specs[i] = spec.short_spec
    return out_side, out_specs


def _eval_ensemble_spec(
    spec: EnsembleSpec,
    preds: dict[tuple[float, float], dict[str, tuple[np.ndarray, np.ndarray]]],
    split_data: dict[str, dict[str, Any]],
    *,
    fee: float,
    slip: float,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    rec: dict[str, Any] = {
        "variant": spec.variant,
        "long_variant": spec.long_spec.variant,
        "short_variant": spec.short_spec.variant,
        "router": spec.router,
        "long_tp": spec.long_spec.tp,
        "long_sl": spec.long_spec.sl,
        "long_min_edge": spec.long_spec.min_edge,
        "long_side_margin": spec.long_spec.side_margin,
        "long_notional": spec.long_spec.notional,
        "long_dd_governor": spec.long_spec.dd_governor,
        "short_tp": spec.short_spec.tp,
        "short_sl": spec.short_spec.sl,
        "short_min_edge": spec.short_spec.min_edge,
        "short_side_margin": spec.short_spec.side_margin,
        "short_notional": spec.short_spec.notional,
        "short_dd_governor": spec.short_spec.dd_governor,
        "max_hold_contract_bars": MAX_HOLD_BARS,
        "leverage_cap": LEVERAGE_CAP,
    }
    ledgers: dict[str, pd.DataFrame] = {}
    for split in ("validation", "oos"):
        side, spec_by_row = _ensemble_signal(spec, preds, split)
        metrics, ledger = _replay_events(split_data[split]["frame"], side, spec_by_row, fee=fee, slip=slip)
        ledgers[split] = ledger
        for key, value in metrics.items():
            rec[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
    _score_record(rec)
    return rec, ledgers


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    report = _load_base_report()
    device = meta.parent._device("cuda")
    split_data, align_diag, feature_cols = _prepare_split_data(device)
    models = _load_models(report)
    preds = _predict_all(models, split_data)
    fee, slip = meta.omega._load_fee_slip()

    side_rows: list[dict[str, Any]] = []
    side_specs: dict[str, SideSpec] = {}
    for pair in sorted(models):
        val_long, val_short = preds[pair]["validation"]
        for side in (1, -1):
            specs = _candidate_side_specs(pair, side, val_long, val_short)
            print(json.dumps({"stage": "side_grid_start", "pair": pair, "side": side, "count": len(specs)}, ensure_ascii=False), flush=True)
            for idx, spec in enumerate(specs, start=1):
                rec = _eval_side_spec(spec, preds, split_data, fee=fee, slip=slip, splits=("validation",))
                side_rows.append(rec)
                side_specs[spec.variant] = spec
                if idx % 500 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "side_grid_progress",
                                "pair": pair,
                                "side": side,
                                "idx": idx,
                                "val": rec["validation_pnl"],
                                "mdd": rec["validation_mdd"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    side_grid = pd.DataFrame(side_rows).sort_values(["validation_score", "validation_pnl"], ascending=False).reset_index(drop=True)
    side_grid.to_csv(OUT_DIR / "side_specialist_validation_grid.csv", index=False)

    long_pool_df = side_grid[(side_grid["side"] == 1) & (side_grid["validation_trades"] >= 5)].sort_values(["validation_score", "validation_pnl"], ascending=False).head(20)
    short_pool_df = side_grid[(side_grid["side"] == -1) & (side_grid["validation_trades"] >= 5)].sort_values(["validation_score", "validation_pnl"], ascending=False).head(20)
    pool_readout_rows: list[dict[str, Any]] = []
    for variant in list(long_pool_df["variant"].astype(str)) + list(short_pool_df["variant"].astype(str)):
        pool_readout_rows.append(_eval_side_spec(side_specs[variant], preds, split_data, fee=fee, slip=slip))
    pool_readout = pd.DataFrame(pool_readout_rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    pool_readout.to_csv(OUT_DIR / "side_specialist_pool_readout.csv", index=False)
    routers = ("edge_score", "none_on_conflict", "short_priority", "long_priority")
    ensemble_rows: list[dict[str, Any]] = []
    ledger_cache: dict[str, dict[str, pd.DataFrame]] = {}
    total = int(len(long_pool_df) * len(short_pool_df) * len(routers))
    idx = 0
    for _, long_row in long_pool_df.iterrows():
        long_spec = side_specs[str(long_row.variant)]
        for _, short_row in short_pool_df.iterrows():
            short_spec = side_specs[str(short_row.variant)]
            for router in routers:
                idx += 1
                spec = EnsembleSpec(
                    variant=f"ens_{router}__{long_spec.variant}__{short_spec.variant}",
                    long_spec=long_spec,
                    short_spec=short_spec,
                    router=router,
                )
                rec, ledgers = _eval_ensemble_spec(spec, preds, split_data, fee=fee, slip=slip)
                ensemble_rows.append(rec)
                if idx <= 1 or idx % 200 == 0:
                    print(json.dumps({"stage": "ensemble_progress", "idx": idx, "total": total, "val": rec["validation_pnl"], "oos": rec["oos_pnl"]}, ensure_ascii=False), flush=True)
                ledger_cache[spec.variant] = ledgers
    ens_grid = pd.DataFrame(ensemble_rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    ens_grid.to_csv(OUT_DIR / "ensemble_grid.csv", index=False)
    ens_grid[ens_grid["pass_target"]].to_csv(OUT_DIR / "target_pass.csv", index=False)

    saved_ledgers: list[str] = []
    keep = set(ens_grid.head(12)["variant"].astype(str).tolist())
    for variant in keep:
        for split, ledger in ledger_cache.get(variant, {}).items():
            path = LEDGER_DIR / f"{split}_{variant[:180]}_ledger.csv"
            ledger.to_csv(path, index=False)
            saved_ledgers.append(str(path))

    out_report = {
        "model_id": MODEL_ID,
        "base_model_id": report["model_id"],
        "method": "Build long-only and short-only specialist policies from the existing train-only one-day meta-barrier HGB long/short edge regressors, then ensemble them with conflict routers selected on validation.",
        "contract": {
            "validation_pnl_min": TARGET_PNL,
            "oos_pnl_min": TARGET_PNL,
            "mdd_floor_pct": TARGET_MDD,
            "max_hold_bars": MAX_HOLD_BARS,
            "leverage_cap": LEVERAGE_CAP,
            "tp_sl_contract": "direct price-move barriers",
        },
        "alignment": align_diag,
        "feature_count": int(len(feature_cols)),
        "model_pairs": [f"tp{_tag(tp)}_sl{_tag(sl)}" for tp, sl in sorted(models)],
        "long_pool": long_pool_df.head(10).to_dict(orient="records"),
        "short_pool": short_pool_df.head(10).to_dict(orient="records"),
        "side_validation_top20": side_grid.head(20).to_dict(orient="records"),
        "side_pool_readout_top20": pool_readout.head(20).to_dict(orient="records"),
        "ensemble_top20": ens_grid.head(20).to_dict(orient="records"),
        "pass_count": int(ens_grid["pass_target"].sum()),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "side_validation_grid": str(OUT_DIR / "side_specialist_validation_grid.csv"),
            "side_pool_readout": str(OUT_DIR / "side_specialist_pool_readout.csv"),
            "ensemble_grid": str(OUT_DIR / "ensemble_grid.csv"),
            "target_pass": str(OUT_DIR / "target_pass.csv"),
            "ledgers": str(LEDGER_DIR),
            "saved_ledgers": saved_ledgers,
        },
        "notes": [
            "No parent retraining was performed in this script.",
            "The long and short edge regressors are the already trained side-specific HGB models from the base meta-barrier experiment.",
            "Side-specialist gates and ensemble routers are policy-layer searches.",
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(out_report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "pass_count": int(out_report["pass_count"]), "top": ens_grid.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
