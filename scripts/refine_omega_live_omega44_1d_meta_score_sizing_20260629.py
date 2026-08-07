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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega_live_omega44_1d_meta_barrier_20260629 as meta  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega_live_omega44_1d_meta_score_sizing_refine_20260629"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516/omega_live_omega44_1d_meta_barrier_20260629"


@dataclass(frozen=True)
class ScoreSpec:
    variant: str
    tp: float
    sl: float
    min_edge: float
    side_margin: float
    size_floor: float
    size_cap: float
    gamma: float
    side_filter: int
    dd_governor: bool


def _json_default(obj: Any) -> Any:
    return meta._json_default(obj)


def _score_signal(pred_long: np.ndarray, pred_short: np.ndarray, spec: ScoreSpec) -> tuple[np.ndarray, np.ndarray]:
    score = np.maximum(pred_long, pred_short)
    side = np.where(pred_long >= pred_short, 1, -1).astype(np.int64)
    edge_gap = np.abs(pred_long - pred_short)
    active = (score >= float(spec.min_edge)) & (edge_gap >= float(spec.side_margin))
    if int(spec.side_filter) != 0:
        active &= side == int(spec.side_filter)
    return np.where(active, side, 0).astype(np.int64), score.astype(np.float64)


def _entry_notional(score: float, spec: ScoreSpec, q95: float, dd: float) -> float:
    den = max(float(q95) - float(spec.min_edge), 1.0e-8)
    z = float(np.clip((float(score) - float(spec.min_edge)) / den, 0.0, 1.0))
    n = float(spec.size_floor) + (float(spec.size_cap) - float(spec.size_floor)) * (z ** float(spec.gamma))
    if spec.dd_governor:
        if dd <= -0.16:
            n *= 0.30
        elif dd <= -0.12:
            n *= 0.50
        elif dd <= -0.08:
            n *= 0.70
    return float(np.clip(n, 0.0, meta.LEVERAGE_CAP))


def _replay_scaled(frame: pd.DataFrame, signal: np.ndarray, score: np.ndarray, spec: ScoreSpec, q95: float, *, fee: float, slip: float) -> tuple[dict[str, Any], pd.DataFrame]:
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
    margin_fraction = 0.0
    mfe = 0.0
    mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = margin_sum = 0.0
    max_hold_seen = 0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = meta._price_move_close(arrays, int(i), side=pos, entry_price=entry_price, slip_eff=slip_eff)
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
            if move >= float(spec.tp):
                reason = "take_profit"
            elif move <= -abs(float(spec.sl)):
                reason = "stop_loss"
            elif hold >= meta.MAX_HOLD_BARS:
                reason = "max_hold_1d"
            if reason:
                filled, exit_px, exit_fee, route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
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
                        "leverage": float(meta.LEVERAGE_CAP),
                        "take_profit": float(spec.tp),
                        "stop_loss": float(spec.sl),
                        "hold_bars": int(hold),
                        "cash_after": float(cash),
                        "exit_route": route,
                    }
                )
                pos = 0
                continue
        if pos != 0:
            continue
        side = int(signal[int(i)]) if int(i) < len(signal) else 0
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        entry_dd = cash / max(peak, 1.0e-12) - 1.0
        row_notional = _entry_notional(float(score[int(i)]), spec, q95, entry_dd)
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        notional = row_notional
        margin_fraction = notional / max(meta.LEVERAGE_CAP, 1.0e-12)
        cash -= cash * float(entry_fee) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0
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
            "trades_per_day": float(trades / meta._duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries),
            "avg_margin_fraction": float(margin_sum / n_entries),
            "avg_leverage": meta.LEVERAGE_CAP if trades else 0.0,
            "max_leverage": meta.LEVERAGE_CAP if trades else 0.0,
            "max_hold_bars": int(max_hold_seen),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "exit_reasons": reasons,
        },
        ledger,
    )


def _load_split_data() -> dict[str, dict[str, Any]]:
    device = parent._device("cuda")
    omega44, _meta = meta._prepare_omega44_outputs(device)
    out: dict[str, dict[str, Any]] = {}
    for split, (frame, o44_src, o44_dec) in omega44.items():
        live_pred, live_prefix = meta._load_live_predictions(split)
        aligned_frame, aligned_o44_src, aligned_o44_dec, aligned_live, _diag = meta._align_split(frame, o44_src, o44_dec, live_pred)
        out[split] = {
            "frame": aligned_frame,
            "features": meta._feature_frame(aligned_frame, aligned_live, live_prefix, aligned_o44_src, aligned_o44_dec, split),
        }
    cols = list(out["train"]["features"].columns)
    for split in ("validation", "oos"):
        out[split]["features"] = out[split]["features"].reindex(columns=cols).astype(np.float32)
    return out


def _specs(val_score: np.ndarray, tp: float, sl: float) -> list[ScoreSpec]:
    specs: list[ScoreSpec] = []
    for top_frac in (0.18, 0.25, 0.35, 0.50):
        base_edge = float(np.quantile(val_score, 1.0 - float(top_frac)))
        for edge_add in (0.0, 0.001, 0.0025, 0.005):
            min_edge = base_edge + edge_add
            for side_margin in (0.0, 0.001, 0.0025, 0.005):
                for floor in (0.35, 0.50, 0.75, 1.00):
                    for cap in (1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.25):
                        if cap < floor:
                            continue
                        for gamma in (0.5, 1.0, 1.75, 2.5):
                            for side_filter in (-1, 0):
                                for dd_governor in (False, True):
                                    specs.append(
                                        ScoreSpec(
                                            variant=(
                                                f"tp{meta._tag(tp)}_sl{meta._tag(sl)}_top{meta._tag(top_frac)}"
                                                f"_edge{meta._tag(min_edge)}_gap{meta._tag(side_margin)}"
                                                f"_floor{meta._tag(floor)}_cap{meta._tag(cap)}_g{meta._tag(gamma)}"
                                                f"{'_shortonly' if side_filter < 0 else ''}"
                                                f"{'_ddgov' if dd_governor else ''}"
                                            ),
                                            tp=float(tp),
                                            sl=float(sl),
                                            min_edge=float(min_edge),
                                            side_margin=float(side_margin),
                                            size_floor=float(floor),
                                            size_cap=float(cap),
                                            gamma=float(gamma),
                                            side_filter=int(side_filter),
                                            dd_governor=bool(dd_governor),
                                        )
                                    )
    return specs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    split_data = _load_split_data()
    fee, slip = omega._load_fee_slip()
    tp, sl = 0.052, 0.028
    pair_tag = f"tp{meta._tag(tp)}_sl{meta._tag(sl)}"
    with (SOURCE_DIR / "models" / f"{pair_tag}_long_hgb.pkl").open("rb") as f:
        long_model = pickle.load(f)
    with (SOURCE_DIR / "models" / f"{pair_tag}_short_hgb.pkl").open("rb") as f:
        short_model = pickle.load(f)
    val_long, val_short = meta._predict_pair((long_model, short_model), split_data["validation"]["features"])
    oos_long, oos_short = meta._predict_pair((long_model, short_model), split_data["oos"]["features"])
    val_score = np.maximum(val_long, val_short)
    oos_score = np.maximum(oos_long, oos_short)
    rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    specs = _specs(val_score, tp, sl)
    for idx, spec in enumerate(specs, start=1):
        rec: dict[str, Any] = {
            "variant": spec.variant,
            "tp": spec.tp,
            "sl": spec.sl,
            "min_edge": spec.min_edge,
            "side_margin": spec.side_margin,
            "size_floor": spec.size_floor,
            "size_cap": spec.size_cap,
            "gamma": spec.gamma,
            "side_filter": spec.side_filter,
            "dd_governor": spec.dd_governor,
        }
        for split, pred_pair, score_arr in (
            ("validation", (val_long, val_short), val_score),
            ("oos", (oos_long, oos_short), oos_score),
        ):
            signal, score = _score_signal(pred_pair[0], pred_pair[1], spec)
            q95 = float(np.quantile(val_score, 0.95))
            metrics, ledger = _replay_scaled(split_data[split]["frame"], signal, score, spec, q95, fee=fee, slip=slip)
            for key, value in metrics.items():
                rec[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
            if idx <= 3:
                ledgers[(split, spec.variant)] = ledger
        rec["pass_target"] = (
            rec["validation_pnl"] >= meta.TARGET_PNL
            and rec["oos_pnl"] >= meta.TARGET_PNL
            and rec["validation_mdd"] >= meta.TARGET_MDD
            and rec["oos_mdd"] >= meta.TARGET_MDD
            and rec["validation_max_hold_bars"] <= meta.MAX_HOLD_BARS
            and rec["oos_max_hold_bars"] <= meta.MAX_HOLD_BARS
        )
        rec["target_score"] = min(float(rec["validation_pnl"]), float(rec["oos_pnl"])) - 4.0 * max(0.0, meta.TARGET_MDD - float(rec["validation_mdd"])) - 4.0 * max(0.0, meta.TARGET_MDD - float(rec["oos_mdd"]))
        rows.append(rec)
        if idx % 1000 == 0:
            print(json.dumps({"idx": idx, "total": len(specs), "val": rec["validation_pnl"], "oos": rec["oos_pnl"]}, ensure_ascii=False), flush=True)
    grid = pd.DataFrame(rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "score_sizing_grid.csv", index=False)
    passed = grid[grid["pass_target"]].copy()
    passed.to_csv(OUT_DIR / "target_pass.csv", index=False)
    for _, row in grid.head(10).iterrows():
        spec = ScoreSpec(
            variant=str(row.variant),
            tp=float(row.tp),
            sl=float(row.sl),
            min_edge=float(row.min_edge),
            side_margin=float(row.side_margin),
            size_floor=float(row.size_floor),
            size_cap=float(row.size_cap),
            gamma=float(row.gamma),
            side_filter=int(row.side_filter),
            dd_governor=bool(row.dd_governor),
        )
        for split, pred_pair, score_arr in (
            ("validation", (val_long, val_short), val_score),
            ("oos", (oos_long, oos_short), oos_score),
        ):
            signal, score = _score_signal(pred_pair[0], pred_pair[1], spec)
            metrics, ledger = _replay_scaled(split_data[split]["frame"], signal, score, spec, float(np.quantile(val_score, 0.95)), fee=fee, slip=slip)
            ledger.to_csv(LEDGER_DIR / f"{split}_{spec.variant}_ledger.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "source_model_id": meta.MODEL_ID,
        "method": "Validation-only score-scaled notional refinement over the trained 1d meta-barrier HGB outputs.",
        "pass_count": int(len(passed)),
        "top20": grid.head(20).to_dict(orient="records"),
        "passed": passed.head(20).to_dict(orient="records"),
        "artifacts": {
            "grid": str(OUT_DIR / "score_sizing_grid.csv"),
            "target_pass": str(OUT_DIR / "target_pass.csv"),
            "ledgers": str(LEDGER_DIR),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "pass_count": int(len(passed)), "top": grid.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
