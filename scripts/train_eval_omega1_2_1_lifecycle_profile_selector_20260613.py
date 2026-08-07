#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_lifecycle_profile_selector_20260613"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_BUNDLE_PATH = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


@dataclass(frozen=True)
class Profile:
    name: str
    tp_mult: float = 1.0
    sl_mult: float = 1.0
    notional_mult: float = 1.0
    runner: bool = True
    utility_penalty: float = 0.0


PROFILES = (
    Profile("wide_runner", 1.0, 1.0, 1.0, True, 0.0),
    Profile("wide_no_runner", 1.0, 1.0, 1.0, False, 0.0003),
    Profile("chop_short", 0.42, 0.57, 1.0, False, 0.0005),
    Profile("chop_tight", 0.31, 0.43, 1.0, False, 0.0008),
    Profile("reduced_wide", 1.0, 1.0, 0.80, True, 0.0004),
    Profile("reduced_short", 0.42, 0.57, 0.80, False, 0.0007),
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


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, holds: list[int], profile_counts: dict[str, int]) -> dict[str, Any]:
    eq = np.asarray(equity_curve if equity_curve else [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    arr = np.asarray(trades, dtype=np.float64)
    h = np.asarray(holds, dtype=np.float64)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(dd.min()),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_hold_bars": float(np.mean(h)) if len(h) else 0.0,
        "median_hold_bars": float(np.median(h)) if len(h) else 0.0,
        "max_hold_bars": int(np.max(h)) if len(h) else 0,
        "exit_reasons": dict(reasons),
        "profile_counts": dict(profile_counts),
    }


def _row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_long": int(m["long_entries"]),
        f"{prefix}_short": int(m["short_entries"]),
        f"{prefix}_avg_hold": float(m["avg_hold_bars"]),
        f"{prefix}_median_hold": float(m["median_hold_bars"]),
        f"{prefix}_max_hold": int(m["max_hold_bars"]),
        f"{prefix}_reasons": m["exit_reasons"],
        f"{prefix}_profile_counts": m["profile_counts"],
    }


def _apply_profile(dec: pd.DataFrame, i: int, profile: Profile) -> pd.DataFrame:
    out = dec.copy()
    out.loc[int(i), "take_profit"] = float(out.loc[int(i), "take_profit"]) * float(profile.tp_mult)
    out.loc[int(i), "stop_loss"] = float(out.loc[int(i), "stop_loss"]) * float(profile.sl_mult)
    out.loc[int(i), "notional_exposure"] = float(out.loc[int(i), "notional_exposure"]) * float(profile.notional_mult)
    out.loc[int(i), "position_fraction"] = float(out.loc[int(i), "position_fraction"]) * float(profile.notional_mult)
    return out


def _runner_extend_allowed(bundle: dict[str, Any] | None, frame: pd.DataFrame, state: pd.DataFrame, pos: base.Position, i: int, unreal: float) -> bool:
    if not bundle:
        return False
    template = meta.RunnerTemplate(**bundle["template"])
    return meta._selector_allowed(
        bundle.get("model"),
        list(bundle.get("feature_cols", [])),
        frame,
        state,
        pos,
        int(i),
        float(unreal),
        template=template,
        proba_min=float(bundle.get("proba_min", 0.55)),
    )


def _close_trade(cash: float, arrays: dict[str, np.ndarray], pos: base.Position, i: int, fee_eff: float, slip_eff: float) -> tuple[float, float]:
    before = float(cash)
    cash, _pos, _ = base._close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
    return cash, float((cash / max(before, 1e-12)) - 1.0)


def _simulate_single(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    i: int,
    *,
    fee_eff: float,
    slip_eff: float,
    tp_bundle: dict[str, Any],
    profile: Profile,
) -> dict[str, Any] | None:
    arrays = base._arrays(frame)
    dec_use = _apply_profile(dec, int(i), profile)
    cash, pos, entered = base._enter(1.0, arrays, dec_use, int(i), fee_eff, slip_eff)
    if not entered:
        return None
    template = meta.RunnerTemplate(**tp_bundle["template"])
    extensions = 0
    reason = "forced_end"
    exit_i = len(frame) - 1
    for j in range(int(pos.entry_i), len(frame) - 1):
        unreal = base._unreal(arrays, pos, j, slip_eff)
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        if pos.take_profit > 0.0 and unreal >= pos.take_profit:
            if profile.runner and extensions < int(template.max_extensions) and _runner_extend_allowed(tp_bundle, frame, state, pos, j, unreal):
                extensions += 1
                old_tp = float(pos.take_profit)
                pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                pos.take_profit = old_tp * float(template.extend_mult)
                continue
            reason = "take_profit"
            exit_i = int(j)
            break
        if pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
            reason = "meta_runner_profit_lock_exit"
            exit_i = int(j)
            break
        if pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
            reason = "stop_loss"
            exit_i = int(j)
            break
    cash, raw_return = _close_trade(cash, arrays, pos, exit_i, fee_eff, slip_eff)
    hold = max(int(exit_i) - int(pos.entry_i), 0)
    return {
        "profile": profile.name,
        "return": float(cash - 1.0),
        "raw_return": float(raw_return),
        "utility": float(cash - 1.0 - profile.utility_penalty - 0.000005 * hold),
        "hold": int(hold),
        "reason": reason,
        "mfe": float(pos.mfe),
        "mae": float(pos.mae),
    }


def _entry_features(frame: pd.DataFrame, dec: pd.DataFrame, state: pd.DataFrame, i: int) -> dict[str, float]:
    row = state.iloc[int(i)]
    drow = dec.iloc[int(i)]
    expert = str(drow.get("router_expert", ""))
    out = {
        "side": float(drow.get("side", 0.0)),
        "quality_score": float(drow.get("quality_score", 0.0)),
        "confidence": float(drow.get("confidence", 0.0)),
        "notional_exposure": float(drow.get("notional_exposure", 0.0)),
        "tp": float(drow.get("take_profit", 0.0)),
        "sl": float(drow.get("stop_loss", 0.0)),
        "rr": float(drow.get("take_profit", 0.0)) / max(abs(float(drow.get("stop_loss", 0.0))), 1e-8),
        "expert_bull": float(expert == "bull"),
        "expert_bear": float(expert == "bear"),
        "expert_chop": float(expert in {"chop", "chop_expert"}),
        "tabm_router_confidence": float(row.get("tabm_router_confidence", 0.0)),
        "tabm_router_margin": float(row.get("tabm_router_margin", 0.0)),
        "tabm_dir_confidence": float(row.get("tabm_dir_confidence", 0.0)),
        "tabm_dir_side_edge": float(row.get("tabm_dir_side_edge", 0.0)),
        "tabm_dir_trade_prob": float(row.get("tabm_dir_trade_prob", 0.0)),
        "tabm_quality_for_action": float(row.get("tabm_quality_for_action", 0.0)),
        "atr14_pct": float(row.get("atr14_pct", 0.0)),
        "bar_range_pct": float(row.get("bar_range_pct", 0.0)),
        "ema9_21_gap_side": float(row.get("ema9_21_gap", 0.0)) * float(drow.get("side", 0.0)),
        "tod_sin": float(row.get("tod_sin", 0.0)),
        "tod_cos": float(row.get("tod_cos", 0.0)),
    }
    for lag in (1, 3, 6, 12, 24):
        out[f"ret_{lag}_side"] = float(row.get(f"ret_{lag}", 0.0)) * float(drow.get("side", 0.0))
        out[f"ret_{lag}_abs"] = abs(float(row.get(f"ret_{lag}", 0.0)))
    return out


def _build_profile_dataset(payload: dict[str, Any], *, tp_bundle: dict[str, Any], cost_mult: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"]
    dec = payload["dec"]
    state = payload["state"]
    active_idx = np.flatnonzero(base.omega._active(dec))
    fee_eff = float(payload["fee"]) * float(cost_mult)
    slip_eff = float(payload["slip"]) * float(cost_mult)
    rows: list[dict[str, Any]] = []
    profile_counts = {p.name: 0 for p in PROFILES}
    for i in active_idx:
        feats = _entry_features(frame, dec, state, int(i))
        sims = [
            _simulate_single(frame, dec, state, int(i), fee_eff=fee_eff, slip_eff=slip_eff, tp_bundle=tp_bundle, profile=p)
            for p in PROFILES
        ]
        sims = [s for s in sims if s is not None]
        if not sims:
            continue
        best = max(sims, key=lambda x: float(x["utility"]))
        profile_counts[str(best["profile"])] += 1
        wide = next(s for s in sims if s["profile"] == "wide_runner")
        rows.append(
            {
                "entry_i": int(i),
                **feats,
                "label_profile": str(best["profile"]),
                "best_utility": float(best["utility"]),
                "wide_utility": float(wide["utility"]),
                "best_return": float(best["return"]),
                "wide_return": float(wide["return"]),
                "best_hold": int(best["hold"]),
                "wide_hold": int(wide["hold"]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("empty profile dataset")
    return df, {"rows": int(len(df)), "profile_counts": profile_counts}


def _train_selector(df: pd.DataFrame, *, kind: str, seed: int) -> tuple[Any, list[str], dict[str, Any]]:
    drop = {"entry_i", "label_profile", "best_utility", "wide_utility", "best_return", "wide_return", "best_hold", "wide_hold"}
    cols = [c for c in df.columns if c not in drop]
    x = df[cols].to_numpy(dtype=np.float64)
    y = df["label_profile"].astype(str).to_numpy()
    if kind == "hgb":
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=5, min_samples_leaf=4, l2_regularization=1.0, learning_rate=0.04, random_state=int(seed))
    elif kind == "et":
        model = ExtraTreesClassifier(n_estimators=240, max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=int(seed))
    else:
        raise RuntimeError(f"unknown kind {kind}")
    model.fit(x, y)
    return model, cols, {"kind": kind, "seed": int(seed), "classes": list(getattr(model, "classes_", [])), "feature_cols": cols}


def _choose_profile(model: Any | None, feature_cols: list[str], frame: pd.DataFrame, dec: pd.DataFrame, state: pd.DataFrame, i: int, fallback: str = "wide_runner") -> Profile:
    if model is None:
        name = fallback
    else:
        feats = _entry_features(frame, dec, state, int(i))
        x = np.asarray([[float(feats[c]) for c in feature_cols]], dtype=np.float64)
        name = str(model.predict(x)[0])
    return next((p for p in PROFILES if p.name == name), PROFILES[0])


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, extensions: int, profile: str) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, extensions)
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    row["profile"] = str(profile)
    return row


def _simulate_policy(payload: dict[str, Any], *, model: Any | None, feature_cols: list[str], tp_bundle: dict[str, Any], fallback: str = "wide_runner") -> tuple[dict[str, Any], pd.DataFrame]:
    frame = payload["frame"]
    dec = payload["dec"]
    state = payload["state"]
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    template = meta.RunnerTemplate(**tp_bundle["template"])
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    holds: list[int] = []
    reasons: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    pos = base.Position()
    extensions = 0
    long_entries = short_entries = 0
    cur_profile = PROFILES[0]
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                if cur_profile.runner and extensions < int(template.max_extensions) and _runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
                    extensions += 1
                    old_tp = float(pos.take_profit)
                    pos.floor_unreal = max(float(pos.floor_unreal), old_tp * float(template.floor_frac))
                    pos.take_profit = old_tp * float(template.extend_mult)
                else:
                    reason = "take_profit"
            elif pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
                reason = "meta_runner_profit_lock_exit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                holds.append(max(int(i) - int(close_pos.entry_i), 0))
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions, cur_profile.name))
                extensions = 0
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        cur_profile = _choose_profile(model, feature_cols, frame, dec, state, int(i), fallback=fallback)
        dec_use = _apply_profile(dec, int(i), cur_profile)
        side = int(dec_use.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec_use, i, fee_eff, slip_eff)
        if entered:
            profile_counts[cur_profile.name] = profile_counts.get(cur_profile.name, 0) + 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)
            extensions = 0
    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        holds.append(max(len(frame) - 1 - int(close_pos.entry_i), 0))
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions, cur_profile.name))
    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, holds, profile_counts), pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    tp_bundle = joblib.load(TP_BUNDLE_PATH)
    profile_df, dataset_diag = _build_profile_dataset(data["validation"], tp_bundle=tp_bundle, cost_mult=3.0)
    profile_df.to_csv(OUT_DIR / "validation_profile_selector_dataset.csv", index=False)
    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    candidates: list[tuple[str, Any | None, list[str], dict[str, Any]]] = [("baseline_wide_runner", None, [], {"kind": "fixed"})]
    for kind, seed in (("hgb", 260613), ("et", 260614)):
        model, cols, diag = _train_selector(profile_df, kind=kind, seed=seed)
        candidates.append((f"profile_selector_{kind}", model, cols, diag))
    for name, model, cols, diag in candidates:
        row: dict[str, Any] = {"candidate": name, **diag}
        ledgers[name] = {}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate_policy(data[split], model=model, feature_cols=cols, tp_bundle=tp_bundle)
            row.update(_row(split, metrics))
            ledgers[name][split] = ledger
        rows.append(row)
        print(json.dumps({"done": name, "oos_pnl": row["oos_pnl"], "oos_trades": row["oos_trades"], "oos_avg_hold": row["oos_avg_hold"], "profiles": row["oos_profile_counts"]}, ensure_ascii=False), flush=True)
    ranking = pd.DataFrame(rows)
    base_row = ranking[ranking["candidate"].eq("baseline_wide_runner")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.35 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.15 * ranking["validation_mdd"] - 0.02 * ranking["oos_avg_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "score"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "profile_selector_ranking.csv", index=False)
    for name in ranking["candidate"].astype(str).tolist():
        for split, ledger in ledgers[name].items():
            ledger.to_csv(OUT_DIR / f"{split}_{name}_ledger.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "purpose": "Entry-time lifecycle profile selector. Validation counterfactual labels choose among wide/short/reduced/no-runner profiles; OOS evaluates sequentially.",
        "dataset_diag": dataset_diag,
        "profiles": [p.__dict__ for p in PROFILES],
        "baseline": base_row.to_dict(),
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "dataset": str(OUT_DIR / "validation_profile_selector_dataset.csv"),
            "ranking": str(OUT_DIR / "profile_selector_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
