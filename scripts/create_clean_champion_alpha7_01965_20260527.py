#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from scripts.test_01965_iqn_sizing_overlay_20260527 import (  # noqa: E402
    _active_seed,
    _apply_iqn_sizing,
    _fit_val_rank_calibration,
    _iqn_context,
    _load_iqn,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "clean_champion_alpha7_01965_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
TOP_GRID_OUT = OUT_DIR / "top_costs.csv"


@dataclass(frozen=True)
class FilterCfg:
    name: str
    quality_min: float | None = None
    confidence_min: float | None = None
    side_mode: str = "all"
    max_active_notional: float | None = None
    max_hold_bars: int | None = None


FILTERS = [
    FilterCfg("base"),
    FilterCfg("q002_conf050", quality_min=0.002, confidence_min=0.50),
    FilterCfg("q002_conf055", quality_min=0.002, confidence_min=0.55),
    FilterCfg("q002_conf060", quality_min=0.002, confidence_min=0.60),
    FilterCfg("q004_conf050", quality_min=0.004, confidence_min=0.50),
    FilterCfg("q004_conf055", quality_min=0.004, confidence_min=0.55),
    FilterCfg("short_only", side_mode="short"),
    FilterCfg("long_only", side_mode="long"),
    FilterCfg("cap15", max_active_notional=1.5),
    FilterCfg("cap20", max_active_notional=2.0),
    FilterCfg("hold288", max_hold_bars=288),
    FilterCfg("hold144", max_hold_bars=144),
    FilterCfg("q002_conf055_cap15", quality_min=0.002, confidence_min=0.55, max_active_notional=1.5),
    FilterCfg("q002_conf055_hold288", quality_min=0.002, confidence_min=0.55, max_hold_bars=288),
]

IQN_VARIANTS = [
    "baseline_modded_01965",
    "iqn_valrank_combo_cap3",
    "iqn_valrank_inverse_combo_cap2",
    "iqn_rollrank_noseed_combo_cap3",
    "iqn_rollrank_seeded_combo_cap3",
    "iqn_conflict_throttle_floor050",
    "iqn_downside_throttle_floor075",
    "iqn_direct_floor025_cap3",
]


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(dec["action"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    side = pd.to_numeric(dec["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return (action != ACTION_CASH) & (side != 0)


def _apply_filter(dec: pd.DataFrame, cfg: FilterCfg) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    keep = active.copy()
    if cfg.quality_min is not None:
        q = pd.to_numeric(out["quality_score"], errors="coerce").fillna(-999.0).to_numpy(dtype=np.float64)
        keep &= q >= float(cfg.quality_min)
    if cfg.confidence_min is not None:
        c = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        keep &= c >= float(cfg.confidence_min)
    if cfg.side_mode != "all":
        side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        if cfg.side_mode == "long":
            keep &= side > 0
        elif cfg.side_mode == "short":
            keep &= side < 0
        else:
            raise ValueError(f"unknown side_mode: {cfg.side_mode}")
    out.loc[active & ~keep, ["action", "side"]] = 0
    if cfg.max_active_notional is not None:
        active2 = _active(out)
        n = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        out["notional_exposure"] = np.where(active2, np.minimum(n, float(cfg.max_active_notional)), 0.0)
        if "position_fraction" in out.columns:
            lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
            out["position_fraction"] = np.clip(out["notional_exposure"].to_numpy(dtype=np.float64) / np.maximum(lev, 1e-12), 0.0, 1.0)
    if cfg.max_hold_bars is not None:
        active2 = _active(out)
        h = pd.to_numeric(out["max_hold_bars"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
        out["max_hold_bars"] = np.where(active2, np.minimum(h, int(cfg.max_hold_bars)), h).round().astype(int)
    return out


def _eval_dec(
    *,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    split: str,
    filter_name: str,
    iqn_variant: str,
    cost_mult: int,
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
    )
    return {
        "split": split,
        "filter": filter_name,
        "iqn_variant": iqn_variant,
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


def _selection_score(row: dict[str, Any]) -> float:
    pnl = float(row["pnl"])
    mdd = abs(float(row["mdd"]))
    wr = float(row["wr"])
    trades = int(row["trades"])
    if trades < 50:
        return -1e9 + pnl
    # PnL first, but require the model to keep the 01965 high-WR character.
    return pnl + 85.0 * wr - 0.55 * mdd - 0.015 * max(0, trades - 180)


def _safe_champion_score(row: dict[str, Any]) -> float:
    pnl = float(row["pnl"])
    mdd = abs(float(row["mdd"]))
    wr = float(row["wr"])
    trades = int(row["trades"])
    if trades < 50:
        return -1e9 + pnl
    return pnl + 120.0 * wr - 0.85 * mdd


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    stack = _load_stack()
    val_df, eval_df = _load_frames()
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    base_val = _apply_decision_mods(sources[str(cfg["source"])][0], cfg)
    base_eval = _apply_decision_mods(sources[str(cfg["source"])][1], cfg)

    model, payload, cat, _device = _load_iqn()
    val_ctx = _iqn_context(val_df, model=model, payload=payload, cat=cat)
    eval_ctx = _iqn_context(eval_df, model=model, payload=payload, cat=cat)

    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    for filter_cfg in FILTERS:
        val_filtered = _apply_filter(base_val, filter_cfg)
        eval_filtered = _apply_filter(base_eval, filter_cfg)
        rank_calibration = _fit_val_rank_calibration(val_filtered, val_ctx)
        val_seed = _active_seed(val_filtered, val_ctx)
        for variant in IQN_VARIANTS:
            rolling_seed = val_seed if variant in {
                "iqn_rollrank_seeded_combo_cap3",
                "iqn_conflict_throttle_floor050",
                "iqn_downside_throttle_floor075",
            } else None
            val_dec, val_audit = _apply_iqn_sizing(
                val_filtered,
                val_ctx,
                variant=variant,
                rank_calibration=rank_calibration,
                rolling_seed=None,
            )
            eval_dec, eval_audit = _apply_iqn_sizing(
                eval_filtered,
                eval_ctx,
                variant=variant,
                rank_calibration=rank_calibration,
                rolling_seed=rolling_seed,
            )
            cache[("val", filter_cfg.name, variant)] = val_dec
            cache[("oos", filter_cfg.name, variant)] = eval_dec
            val_row = _eval_dec(
                df=val_df,
                q=val_q,
                dec=val_dec,
                stack=stack,
                cfg=cfg,
                split="val",
                filter_name=filter_cfg.name,
                iqn_variant=variant,
                cost_mult=3,
            )
            val_row["selection_score"] = _selection_score(val_row)
            val_row["safe_champion_score"] = _safe_champion_score(val_row)
            val_row["active_before_iqn"] = val_audit["active_before"]
            val_row["active_after_iqn"] = val_audit["active_after"]
            rows.append(val_row)
            oos_row = _eval_dec(
                df=eval_df,
                q=eval_q,
                dec=eval_dec,
                stack=stack,
                cfg=cfg,
                split="oos",
                filter_name=filter_cfg.name,
                iqn_variant=variant,
                cost_mult=3,
            )
            oos_row["selection_score"] = np.nan
            oos_row["safe_champion_score"] = np.nan
            oos_row["active_before_iqn"] = eval_audit["active_before"]
            oos_row["active_after_iqn"] = eval_audit["active_after"]
            rows.append(oos_row)

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    val_grid = grid[grid["split"].eq("val")].copy()
    # Validation-only selectors. OOS is only read after these are fixed.
    best_pnl_wr = val_grid.sort_values(["selection_score", "pnl", "wr"], ascending=False).iloc[0].to_dict()
    best_safe = val_grid.sort_values(["safe_champion_score", "pnl", "wr"], ascending=False).iloc[0].to_dict()
    best_wr = val_grid[val_grid["trades"].ge(50)].sort_values(["wr", "pnl"], ascending=False).iloc[0].to_dict()

    selected_keys = {
        "best_pnl_wr": (str(best_pnl_wr["filter"]), str(best_pnl_wr["iqn_variant"])),
        "best_safe": (str(best_safe["filter"]), str(best_safe["iqn_variant"])),
        "best_wr": (str(best_wr["filter"]), str(best_wr["iqn_variant"])),
    }
    full_rows: list[dict[str, Any]] = []
    for label, (filter_name, variant) in selected_keys.items():
        for split, df, q in (("val", val_df, val_q), ("oos", eval_df, eval_q)):
            dec = cache[(split, filter_name, variant)]
            for cost in (1, 2, 3):
                row = _eval_dec(
                    df=df,
                    q=q,
                    dec=dec,
                    stack=stack,
                    cfg=cfg,
                    split=split,
                    filter_name=filter_name,
                    iqn_variant=variant,
                    cost_mult=cost,
                )
                row["selector"] = label
                full_rows.append(row)
    pd.DataFrame(full_rows).to_csv(TOP_GRID_OUT, index=False)

    def _paired(selector: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for split in ("val", "oos"):
            out[split] = {}
            for cost in (1, 2, 3):
                m = next(r for r in full_rows if r["selector"] == selector and r["split"] == split and r["cost"] == cost)
                out[split][f"cost{cost}"] = m
        return out

    summary = {
        "model_id": MODEL_ID,
        "candidate_base": CANDIDATE,
        "design": (
            "Validation-only clean champion search on 01965 alpha7_combo_primary_fallback. "
            "It avoids legacy regime features, DSAC fallback, high-turnover retraining, and OOS-selected 00596/01543 fallbacks. "
            "The search only applies explicit filters and IQN/Mamba/CatBoost sizing/veto variants to the 01965 decision stream."
        ),
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS read after validation selection",
            "feature_contract": "alpha7 live 93 features with clean_regime4_state24_sticky090_v2 only",
            "buggy_legacy_regime_allowed": False,
            "compat_alias_added": False,
            "retrained_model": False,
            "oos_diagnostic_candidate_00596_allowed": False,
            "cash_fallback_retrained": False,
        },
        "base_config": cfg,
        "filters": [asdict(f) for f in FILTERS],
        "iqn_variants": IQN_VARIANTS,
        "selected_by_validation": {
            "best_pnl_wr": best_pnl_wr,
            "best_safe": best_safe,
            "best_wr": best_wr,
        },
        "selected_metrics": {
            "best_pnl_wr": _paired("best_pnl_wr"),
            "best_safe": _paired("best_safe"),
            "best_wr": _paired("best_wr"),
        },
        "artifacts": {
            "grid": str(GRID_OUT),
            "top_costs": str(TOP_GRID_OUT),
            "source_iqn": str(ROOT / "tmp/causal_regen_20260516/alpha7_mamba_iqn_catboost_veto_20260527_promoted_v6_sideaware_notional300"),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": str(SUMMARY_OUT),
        "best_pnl_wr": summary["selected_metrics"]["best_pnl_wr"]["oos"]["cost3"],
        "best_safe": summary["selected_metrics"]["best_safe"]["oos"]["cost3"],
        "best_wr": summary["selected_metrics"]["best_wr"]["oos"]["cost3"],
    }, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
