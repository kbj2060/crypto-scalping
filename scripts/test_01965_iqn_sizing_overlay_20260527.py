#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import backtest_signal_limit_exit_guard  # noqa: E402
from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    _apply_decision_mods,
    _decision_sources,
    _default_limit_cfg,
    _guard,
    _load_frames,
    _load_stack,
    _overlay,
    _score,
    _sl_ratio,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results  # noqa: E402
from scripts.train_eval_alpha7_iqn_fallback_20260527 import _apply_scaler, _feature_matrix  # noqa: E402
from scripts.train_eval_alpha7_mamba_iqn_catboost_veto_20260527 import (  # noqa: E402
    MambaIQNNet,
    _mamba_iqn_scores,
    _side_veto_probs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "01965_iqn_sizing_overlay_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
PROMOTED_IQN_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_mamba_iqn_catboost_veto_20260527_promoted_v6_sideaware_notional300"
IQN_CKPT = PROMOTED_IQN_DIR / "mamba_iqn.pt"
CATBOOST_VETO = PROMOTED_IQN_DIR / "catboost_veto.cbm"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
AUDIT_OUT = OUT_DIR / "audit.json"
VAL_LEDGER_OUT = OUT_DIR / "val_best_cost3_ledger.csv"
OOS_LEDGER_OUT = OUT_DIR / "oos_best_cost3_ledger.csv"
ORACLE_VARIANTS = {
    "iqn_valrank_combo_cap2",
    "iqn_valrank_combo_cap3",
    "iqn_valrank_inverse_combo_cap2",
}


def _active(dec: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH) & (
        pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int) != 0
    )


def _load_iqn() -> tuple[MambaIQNNet, dict[str, Any], CatBoostClassifier, torch.device]:
    if not IQN_CKPT.exists():
        raise FileNotFoundError(f"IQN checkpoint not found: {IQN_CKPT}")
    if not CATBOOST_VETO.exists():
        raise FileNotFoundError(f"CatBoost veto model not found: {CATBOOST_VETO}")
    payload = torch.load(IQN_CKPT, map_location="cpu", weights_only=False)
    network = dict(payload["network"])
    model = MambaIQNNet(
        input_dim=int(network["input_dim"]),
        action_dim=int(network["action_dim"]),
        hidden_dim=int(network["hidden_dim"]),
        n_layers=int(network["n_layers"]),
        n_cos=int(network["n_cos"]),
    )
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    cat = CatBoostClassifier()
    cat.load_model(str(CATBOOST_VETO))
    return model, payload, cat, device


def _iqn_context(
    frame: pd.DataFrame,
    *,
    model: MambaIQNNet,
    payload: dict[str, Any],
    cat: CatBoostClassifier,
) -> dict[str, Any]:
    feature_cols = list(payload["feature_cols"])
    x_df = _feature_matrix(frame, feature_cols)
    x = _apply_scaler(x_df, payload["scaler"])
    network = dict(payload["network"])
    runtime = dict(payload["runtime"])
    scores = _mamba_iqn_scores(
        model,
        x,
        seq_len=int(network["seq_len"]),
        risk_tau=float(runtime["risk_tau"]),
        num_tau=32,
        batch_size=2048,
    )
    veto = _side_veto_probs(cat, x_df, scores)
    return {"scores": scores, "veto": veto, "runtime": runtime, "seq_len": int(network["seq_len"])}


def _strength(p: np.ndarray, edge: np.ndarray) -> np.ndarray:
    p_part = np.clip((p - 0.85) / 0.15, 0.0, 1.0)
    edge_part = np.clip((edge + 0.005) / 0.04, 0.0, 1.0)
    return np.clip(0.65 * p_part + 0.35 * edge_part, 0.0, 1.0)


def _rank_from_quantiles(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 1 or len(q) < 2:
        raise ValueError("rank quantiles must be a 1D array with at least 2 values")
    return np.clip(np.searchsorted(q, values, side="right") / float(len(q)), 0.0, 1.0)


def _active_iqn_values(dec: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    scores = np.asarray(ctx["scores"], dtype=np.float32)
    veto = np.asarray(ctx["veto"], dtype=np.float32)
    idx = np.arange(len(dec), dtype=np.int64)
    active = _active(dec).to_numpy(dtype=bool)
    active[: max(int(ctx["seq_len"]) - 1, 0)] = False
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    action = np.where(side > 0, 1, np.where(side < 0, 2, 0)).astype(np.int64)
    edge = scores[idx, action] - scores[:, 0]
    p = veto[idx, action]
    return {"active": active, "p": p.astype(np.float64), "edge": edge.astype(np.float64)}


def _fit_val_rank_calibration(dec: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    vals = _active_iqn_values(dec, ctx)
    active = vals["active"]
    if int(active.sum()) < 20:
        raise RuntimeError(f"not enough active rows to calibrate IQN rank sizing: {int(active.sum())}")
    grid = np.linspace(0.0, 1.0, 101)
    return {
        "p": np.quantile(vals["p"][active], grid),
        "edge": np.quantile(vals["edge"][active], grid),
    }


def _active_seed(dec: pd.DataFrame, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
    vals = _active_iqn_values(dec, ctx)
    active = vals["active"]
    return {"p": vals["p"][active].copy(), "edge": vals["edge"][active].copy()}


def _rolling_rank_strength(
    *,
    active: np.ndarray,
    p: np.ndarray,
    edge: np.ndarray,
    seed: dict[str, np.ndarray] | None,
    window: int = 240,
    min_history: int = 30,
) -> np.ndarray:
    p_hist = list(np.asarray(seed["p"], dtype=np.float64)[-int(window) :]) if seed is not None else []
    edge_hist = list(np.asarray(seed["edge"], dtype=np.float64)[-int(window) :]) if seed is not None else []
    out = np.zeros(len(active), dtype=np.float64)
    for i in range(len(active)):
        if not bool(active[i]):
            continue
        if len(p_hist) >= int(min_history) and len(edge_hist) >= int(min_history):
            p_rank = float(np.mean(np.asarray(p_hist, dtype=np.float64) <= float(p[i])))
            edge_rank = float(np.mean(np.asarray(edge_hist, dtype=np.float64) <= float(edge[i])))
            out[i] = float(np.clip(0.65 * p_rank + 0.35 * edge_rank, 0.0, 1.0))
        else:
            out[i] = 0.5
        p_hist.append(float(p[i]))
        edge_hist.append(float(edge[i]))
        if len(p_hist) > int(window):
            p_hist.pop(0)
            edge_hist.pop(0)
    return out


def _iqn_best_action(scores: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(scores, dtype=np.float32), axis=1).astype(np.int64)


def _apply_iqn_sizing(
    dec: pd.DataFrame,
    ctx: dict[str, Any],
    *,
    variant: str,
    rank_calibration: dict[str, np.ndarray] | None = None,
    rolling_seed: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = dec.copy().reset_index(drop=True)
    scores = np.asarray(ctx["scores"], dtype=np.float32)
    veto = np.asarray(ctx["veto"], dtype=np.float32)
    runtime = dict(ctx["runtime"])
    seq_len = int(ctx["seq_len"])
    idx = np.arange(len(out), dtype=np.int64)
    active = _active(out).to_numpy(dtype=bool)
    active[: max(seq_len - 1, 0)] = False
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    action = np.where(side > 0, 1, np.where(side < 0, 2, 0)).astype(np.int64)
    selected_q = scores[idx, action]
    cash_q = scores[:, 0]
    edge = selected_q - cash_q
    p = veto[idx, action]
    pass_mask = (
        active
        & (selected_q >= float(runtime["cvar_min"]))
        & (edge >= float(runtime["edge_min"]))
        & (p >= float(runtime["veto_threshold"]))
    )
    base_notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    strength = _strength(p.astype(np.float64), edge.astype(np.float64))

    if variant == "baseline_modded_01965":
        new_notional = base_notional
    elif variant == "iqn_scale_floor035_cap2":
        scale = 0.35 + 0.65 * strength
        new_notional = np.clip(base_notional * scale, 0.0, 2.0)
    elif variant == "iqn_scale_floor035_cap3":
        scale = 0.35 + 0.65 * strength
        new_notional = np.clip(base_notional * scale, 0.0, 3.0)
    elif variant == "iqn_gate_keep_cap2":
        out.loc[active & ~pass_mask, ["action", "side"]] = 0
        new_notional = np.clip(base_notional, 0.0, 2.0)
    elif variant == "iqn_gate_direct_cap3":
        out.loc[active & ~pass_mask, ["action", "side"]] = 0
        new_notional = np.where(pass_mask, 3.0, 0.0)
    elif variant == "iqn_direct_floor025_cap3":
        new_notional = np.where(active, 0.25 + 2.75 * strength, 0.0)
    elif variant in {"iqn_valrank_combo_cap2", "iqn_valrank_combo_cap3", "iqn_valrank_inverse_combo_cap2"}:
        if rank_calibration is None:
            raise RuntimeError(f"{variant} requires rank_calibration")
        p_rank = _rank_from_quantiles(p.astype(np.float64), rank_calibration["p"])
        edge_rank = _rank_from_quantiles(edge.astype(np.float64), rank_calibration["edge"])
        rank_strength = np.clip(0.65 * p_rank + 0.35 * edge_rank, 0.0, 1.0)
        if variant == "iqn_valrank_inverse_combo_cap2":
            rank_strength = 1.0 - rank_strength
        if variant.endswith("_cap3"):
            new_notional = np.where(active, 0.25 + 2.75 * rank_strength, 0.0)
        else:
            new_notional = np.where(active, 0.40 + 1.60 * rank_strength, 0.0)
    elif variant in {"iqn_rollrank_noseed_combo_cap3", "iqn_rollrank_seeded_combo_cap3"}:
        rank_strength = _rolling_rank_strength(
            active=active,
            p=p.astype(np.float64),
            edge=edge.astype(np.float64),
            seed=rolling_seed,
        )
        new_notional = np.where(active, 0.25 + 2.75 * rank_strength, 0.0)
    elif variant in {
        "iqn_downside_throttle_floor060",
        "iqn_downside_throttle_floor075",
        "iqn_exit_tighten_floor060",
        "iqn_conflict_throttle_floor050",
    }:
        rank_strength = _rolling_rank_strength(
            active=active,
            p=p.astype(np.float64),
            edge=edge.astype(np.float64),
            seed=rolling_seed,
        )
        if variant == "iqn_downside_throttle_floor075":
            scale = 0.75 + 0.25 * rank_strength
        elif variant == "iqn_conflict_throttle_floor050":
            best_action = _iqn_best_action(scores)
            conflict = active & (best_action != action) & (best_action != 0)
            cash_preferred = active & (best_action == 0)
            scale = np.ones(len(out), dtype=np.float64)
            scale[conflict] = 0.50 + 0.50 * rank_strength[conflict]
            scale[cash_preferred] = 0.65 + 0.35 * rank_strength[cash_preferred]
        else:
            scale = 0.60 + 0.40 * rank_strength
        new_notional = np.where(active, np.minimum(base_notional, base_notional * scale), 0.0)
        if variant == "iqn_exit_tighten_floor060":
            hold = pd.to_numeric(out["max_hold_bars"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            sl = pd.to_numeric(out["stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            bad = active & (rank_strength < 0.45)
            hold_scale = 0.45 + 0.55 * rank_strength
            sl_scale = 0.65 + 0.35 * rank_strength
            out["max_hold_bars"] = np.where(
                bad,
                np.maximum(3, np.rint(hold * hold_scale)).astype(int),
                np.maximum(1, np.rint(hold)).astype(int),
            )
            out["stop_loss"] = np.where(bad, np.maximum(0.001, sl * sl_scale), sl)
    else:
        raise ValueError(f"unknown IQN sizing variant: {variant}")

    out["notional_exposure"] = np.where(_active(out).to_numpy(dtype=bool), new_notional, 0.0)
    if "position_fraction" in out.columns:
        lev = pd.to_numeric(out.get("leverage", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        out["position_fraction"] = np.clip(out["notional_exposure"].to_numpy(dtype=np.float64) / np.maximum(lev, 1e-12), 0.0, 1.0)
    out["iqn_veto_prob"] = p.astype(float)
    out["iqn_edge_q"] = edge.astype(float)
    out["iqn_selected_q"] = selected_q.astype(float)

    after_active = _active(out).to_numpy(dtype=bool)
    audit = {
        "variant": variant,
        "rows": int(len(out)),
        "active_before": int(active.sum()),
        "active_after": int(after_active.sum()),
        "iqn_pass_rows": int(pass_mask.sum()),
        "iqn_pass_rate_on_active": float(pass_mask.sum() / max(active.sum(), 1)),
        "avg_veto_prob_active": float(np.mean(p[active])) if active.any() else 0.0,
        "avg_edge_active": float(np.mean(edge[active])) if active.any() else 0.0,
        "avg_notional_before": float(np.mean(base_notional[active])) if active.any() else 0.0,
        "avg_notional_after": float(pd.to_numeric(out.loc[after_active, "notional_exposure"], errors="coerce").mean()) if after_active.any() else 0.0,
    }
    return out, audit


def _eval_final_dec(
    *,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    split: str,
    variant: str,
    cost_mult: int,
    record: bool = False,
) -> dict[str, Any]:
    res = backtest_signal_limit_exit_guard(
        df.reset_index(drop=True),
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        q,
        dec.reset_index(drop=True),
        _overlay(stack["overlay"], cfg),
        _default_limit_cfg(),
        _guard(cfg),
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=float(cost_mult),
        record=record,
    )
    row = {
        "candidate": CANDIDATE,
        "variant": variant,
        "split": split,
        "cost": int(cost_mult),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "sl_ratio": float(_sl_ratio(res)),
        "score": float(_score(res)),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "avg_notional": float(res.get("avg_notional", 0.0)),
        "avg_leverage": float(res.get("avg_leverage", 0.0)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }
    if record:
        row["_records"] = res.get("trade_records", [])
    return row


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    df = pd.DataFrame(records)
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    winners = ret[ret > 0].sort_values(ascending=False)
    losers = ret[ret <= 0].sort_values()
    return {
        "rows": int(len(df)),
        "gross_trade_return_sum": float(ret.sum()),
        "gross_trade_return_mean": float(ret.mean()),
        "top5_trade_return_sum": float(winners.head(5).sum()) if len(winners) else 0.0,
        "bottom5_trade_return_sum": float(losers.head(5).sum()) if len(losers) else 0.0,
        "final_cash_after": float(pd.to_numeric(df["cash_after"], errors="coerce").iloc[-1]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    stack = _load_stack()
    val_df, eval_df = _load_frames()
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    model, payload, cat, device = _load_iqn()
    val_ctx = _iqn_context(val_df, model=model, payload=payload, cat=cat)
    eval_ctx = _iqn_context(eval_df, model=model, payload=payload, cat=cat)

    source = str(cfg["source"])
    base_val = _apply_decision_mods(sources[source][0], cfg)
    base_eval = _apply_decision_mods(sources[source][1], cfg)
    rank_calibration = _fit_val_rank_calibration(base_val, val_ctx)
    val_seed = _active_seed(base_val, val_ctx)
    variants = [
        "baseline_modded_01965",
        "iqn_scale_floor035_cap2",
        "iqn_scale_floor035_cap3",
        "iqn_gate_keep_cap2",
        "iqn_gate_direct_cap3",
        "iqn_direct_floor025_cap3",
        "iqn_valrank_combo_cap2",
        "iqn_valrank_combo_cap3",
        "iqn_valrank_inverse_combo_cap2",
        "iqn_rollrank_noseed_combo_cap3",
        "iqn_rollrank_seeded_combo_cap3",
        "iqn_downside_throttle_floor060",
        "iqn_downside_throttle_floor075",
        "iqn_exit_tighten_floor060",
        "iqn_conflict_throttle_floor050",
    ]
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {
        "candidate": CANDIDATE,
        "source": source,
        "iqn_checkpoint": str(IQN_CKPT),
        "catboost_veto": str(CATBOOST_VETO),
        "device": str(device),
        "iqn_runtime": dict(payload["runtime"]),
        "variants": {},
    }
    ledgers: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for variant in variants:
        eval_seed = val_seed if variant in {
            "iqn_rollrank_seeded_combo_cap3",
            "iqn_downside_throttle_floor060",
            "iqn_downside_throttle_floor075",
            "iqn_exit_tighten_floor060",
            "iqn_conflict_throttle_floor050",
        } else None
        val_dec, val_audit = _apply_iqn_sizing(
            base_val,
            val_ctx,
            variant=variant,
            rank_calibration=rank_calibration,
            rolling_seed=None,
        )
        eval_dec, eval_audit = _apply_iqn_sizing(
            base_eval,
            eval_ctx,
            variant=variant,
            rank_calibration=rank_calibration,
            rolling_seed=eval_seed,
        )
        audits["variants"][variant] = {"val": val_audit, "oos": eval_audit}
        for split, df, q, dec in (("val", val_df, val_q, val_dec), ("oos", eval_df, eval_q, eval_dec)):
            for cost in (1, 2, 3):
                record = cost == 3
                row = _eval_final_dec(
                    df=df,
                    q=q,
                    dec=dec,
                    stack=stack,
                    cfg=cfg,
                    split=split,
                    variant=variant,
                    cost_mult=cost,
                    record=record,
                )
                if record:
                    ledgers[(split, variant)] = list(row.pop("_records", []))
                rows.append(row)

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    AUDIT_OUT.write_text(json.dumps(audits, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    oos_c3 = grid[(grid["split"].eq("oos")) & (grid["cost"].eq(3))].sort_values(["pnl", "score"], ascending=False)
    val_c3 = grid[(grid["split"].eq("val")) & (grid["cost"].eq(3))].sort_values(["pnl", "score"], ascending=False)
    best_oos = str(oos_c3.iloc[0]["variant"])
    best_val = str(val_c3.iloc[0]["variant"])
    pd.DataFrame(ledgers.get(("oos", best_oos), [])).to_csv(OOS_LEDGER_OUT, index=False)
    pd.DataFrame(ledgers.get(("val", best_val), [])).to_csv(VAL_LEDGER_OUT, index=False)
    summary = {
        "candidate": CANDIDATE,
        "baseline_variant": "baseline_modded_01965",
        "oracle_diagnostic_variants": sorted(ORACLE_VARIANTS),
        "best_oos_cost3_variant": best_oos,
        "best_val_cost3_variant": best_val,
        "best_oos_cost3": oos_c3.iloc[0].to_dict(),
        "best_val_cost3": val_c3.iloc[0].to_dict(),
        "best_live_like_oos_cost3": oos_c3[~oos_c3["variant"].isin(ORACLE_VARIANTS)].iloc[0].to_dict(),
        "best_live_like_val_cost3": val_c3[~val_c3["variant"].isin(ORACLE_VARIANTS)].iloc[0].to_dict(),
        "baseline_oos_cost3": grid[(grid["variant"].eq("baseline_modded_01965")) & (grid["split"].eq("oos")) & (grid["cost"].eq(3))].iloc[0].to_dict(),
        "baseline_val_cost3": grid[(grid["variant"].eq("baseline_modded_01965")) & (grid["split"].eq("val")) & (grid["cost"].eq(3))].iloc[0].to_dict(),
        "oos_best_ledger_stats": _ledger_stats(ledgers.get(("oos", best_oos), [])),
        "val_best_ledger_stats": _ledger_stats(ledgers.get(("val", best_val), [])),
        "grid": str(GRID_OUT),
        "audit": str(AUDIT_OUT),
        "oos_best_ledger": str(OOS_LEDGER_OUT),
        "val_best_ledger": str(VAL_LEDGER_OUT),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "audit": str(AUDIT_OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
