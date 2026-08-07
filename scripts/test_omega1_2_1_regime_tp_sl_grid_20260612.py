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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_tp_runner_meta_selector_20260610 as meta  # noqa: E402


MODEL_ID = "omega1_2_1_regime_tp_sl_grid_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_BUNDLE_PATH = ROOT / "data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib"


@dataclass(frozen=True)
class RegimeRiskCfg:
    name: str
    chop_tp: float
    chop_sl: float
    transition_tp: float | None = None
    transition_sl: float | None = None
    disable_chop_runner: bool = True
    chop_notional_mult: float = 1.0


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


def _metric(cash: float, equity_curve: list[float], trades: list[float], reasons: dict[str, int], long_entries: int, short_entries: int, holds: list[int], expert_counts: dict[str, int]) -> dict[str, Any]:
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
        "expert_entries": dict(expert_counts),
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
        f"{prefix}_expert_entries": m["expert_entries"],
    }


def _is_transition_risk(state: pd.DataFrame, i: int) -> bool:
    row = state.iloc[int(i)]
    churn = float(row.get("regime3_churn_h6_risk_score", 0.0))
    trans = float(row.get("regime3_transition_h6_risk_prob", 0.0))
    return bool(churn >= 0.55 or trans >= 0.60)


def _adjust_entry_risk(dec: pd.DataFrame, state: pd.DataFrame, i: int, cfg: RegimeRiskCfg) -> pd.DataFrame:
    out = dec
    expert = str(dec.iloc[int(i)].get("router_expert", ""))
    is_chop = expert in {"chop", "chop_expert"}
    is_transition = cfg.transition_tp is not None and _is_transition_risk(state, i)
    if not is_chop and not is_transition:
        return out

    out = dec.copy()
    base_tp = float(base.BASE_TP)
    base_sl = float(base.BASE_SL)
    if is_transition:
        tp = float(cfg.transition_tp)
        sl = float(cfg.transition_sl)
    else:
        tp = float(cfg.chop_tp)
        sl = float(cfg.chop_sl)

    old_tp = float(out.loc[int(i), "take_profit"])
    old_sl = float(out.loc[int(i), "stop_loss"])
    out.loc[int(i), "take_profit"] = old_tp * (tp / max(base_tp, 1e-12))
    out.loc[int(i), "stop_loss"] = old_sl * (sl / max(base_sl, 1e-12))
    if is_chop and float(cfg.chop_notional_mult) != 1.0:
        out.loc[int(i), "notional_exposure"] = float(out.loc[int(i), "notional_exposure"]) * float(cfg.chop_notional_mult)
        out.loc[int(i), "position_fraction"] = float(out.loc[int(i), "position_fraction"]) * float(cfg.chop_notional_mult)
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


def _ledger_row(frame: pd.DataFrame, arrays: dict[str, np.ndarray], pos: base.Position, exit_i: int, cash: float, net_pct: float, reason: str, extensions: int, entry_expert: str) -> dict[str, Any]:
    row = runner._ledger_row(frame, arrays, pos, exit_i, cash, net_pct, reason, extensions)
    row["hold_bars"] = int(exit_i) - int(pos.entry_i)
    row["entry_expert"] = str(entry_expert)
    return row


def _simulate(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    tp_bundle: dict[str, Any],
    cfg: RegimeRiskCfg,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    template = meta.RunnerTemplate(**tp_bundle["template"])
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    holds: list[int] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    expert_counts: dict[str, int] = {}
    pos = base.Position()
    extensions = 0
    long_entries = short_entries = 0
    entry_expert = ""

    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                runner_allowed = not (cfg.disable_chop_runner and entry_expert in {"chop", "chop_expert"})
                if runner_allowed and extensions < int(template.max_extensions) and _runner_extend_allowed(tp_bundle, frame, state, pos, i, unreal):
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
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, extensions, entry_expert))
                extensions = 0
                entry_expert = ""
            continue

        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        dec_use = _adjust_entry_risk(dec, state, i, cfg)
        before_side = int(dec_use.iloc[int(i)].get("side", 0) or 0)
        before_expert = str(dec_use.iloc[int(i)].get("router_expert", ""))
        cash, pos, entered = base._enter(cash, arrays, dec_use, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)
            expert_counts[before_expert] = expert_counts.get(before_expert, 0) + 1
            extensions = 0
            entry_expert = before_expert

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        holds.append(max(len(frame) - 1 - int(close_pos.entry_i), 0))
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", extensions, entry_expert))

    return _metric(cash, equity_curve, trades, reasons, long_entries, short_entries, holds, expert_counts), pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    tp_bundle = joblib.load(TP_BUNDLE_PATH)
    configs = [
        RegimeRiskCfg("baseline", chop_tp=float(base.BASE_TP), chop_sl=float(base.BASE_SL), disable_chop_runner=False),
        RegimeRiskCfg("chop_tp014_sl009", 0.014, 0.009),
        RegimeRiskCfg("chop_tp012_sl008", 0.012, 0.008),
        RegimeRiskCfg("chop_tp010_sl007", 0.010, 0.007),
        RegimeRiskCfg("chop_tp008_sl006", 0.008, 0.006),
        RegimeRiskCfg("chop_tp012_sl008_runner_on", 0.012, 0.008, disable_chop_runner=False),
        RegimeRiskCfg("chop_tp010_sl007_n080", 0.010, 0.007, chop_notional_mult=0.80),
        RegimeRiskCfg("chop_tp012_sl008_trans008_005", 0.012, 0.008, transition_tp=0.008, transition_sl=0.005),
    ]

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    for cfg in configs:
        row: dict[str, Any] = {"variant": cfg.name, **cfg.__dict__}
        ledgers[cfg.name] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = _simulate(
                payload["frame"],
                payload["dec"],
                payload["state"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                tp_bundle=tp_bundle,
                cfg=cfg,
            )
            row.update(_row(split, metrics))
            ledgers[cfg.name][split] = ledger
        rows.append(row)
        print(json.dumps({"done": cfg.name, "oos_pnl": row["oos_pnl"], "oos_trades": row["oos_trades"], "oos_avg_hold": row["oos_avg_hold"], "oos_experts": row["oos_expert_entries"]}, ensure_ascii=False), flush=True)

    ranking = pd.DataFrame(rows)
    base_row = ranking[ranking["variant"].eq("baseline")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.35 * ranking["validation_pnl"] + 0.25 * ranking["oos_mdd"] + 0.15 * ranking["validation_mdd"] - 0.02 * ranking["oos_avg_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "score"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "regime_tp_sl_grid_ranking.csv", index=False)

    keep = set(ranking["variant"].head(8).astype(str).tolist())
    for name in keep:
        for split, ledger in ledgers[name].items():
            ledger.to_csv(OUT_DIR / f"{split}_{name}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Regime-specific TP/SL grid: keep bull/bear wide, shorten chop and optional transition-risk entry barriers.",
        "baseline": base_row.to_dict(),
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "regime_tp_sl_grid_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
