#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_deep_stop_cd18_regime_direction_veto_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MONTHLY_OUT = OUT_DIR / "monthly_cost3.csv"


def _periods(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    out: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            out.append((f"{prefix}_{month}", mask))
    return out


def _parent_direction_veto(df: pd.DataFrame, dec: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = dec.copy().reset_index(drop=True)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(int)
    action = pd.to_numeric(out["action"], errors="coerce").fillna(0).astype(int)
    active = (action != ACTION_CASH) & (side != 0)
    regimes = df.apply(sweep._state24_dominant_regime, axis=1)
    bear_long = active & (side > 0) & regimes.eq("bear")
    bull_short = active & (side < 0) & regimes.eq("bull")
    blocked = bear_long | bull_short
    out.loc[
        blocked,
        [
            "action",
            "side",
            "notional_exposure",
            "position_fraction",
            "take_profit",
            "stop_loss",
            "max_hold_bars",
            "cooldown_bars",
        ],
    ] = 0
    out.loc[blocked, "leverage"] = 1.0
    return out, {
        "parent_bear_long_veto_rows": int(bear_long.sum()),
        "parent_bull_short_veto_rows": int(bull_short.sum()),
        "parent_total_veto_rows": int(blocked.sum()),
    }


def _row(variant: str, split: str, period: str, res: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "variant": variant,
        "split": split,
        "period": period,
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res.get("trades_per_day", 0.0)),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
        "runner_actions": json.dumps(res.get("runner_actions", {}), ensure_ascii=False, sort_keys=True),
    }
    if extra:
        row.update(extra)
    return row


def _eval_periods(
    *,
    variant_name: str,
    split: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period_name, mask in _periods(df, split):
        sub_df = df.loc[mask].reset_index(drop=True)
        sub_q = q[mask]
        sub_dec = dec.loc[mask].reset_index(drop=True)
        res = sweep._backtest_variant(
            df=sub_df,
            q=sub_q,
            dec=sub_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
            cost_mult=3,
            record=False,
            deep_gate=None,
        )
        rows.append(_row(variant_name, split, period_name, res, extra=extra))
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]

    val_parent_veto_dec, val_parent_counts = _parent_direction_veto(val_df, val_dec)
    eval_parent_veto_dec, eval_parent_counts = _parent_direction_veto(eval_df, eval_dec)

    baseline_variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    deep_veto_variant = sweep.Variant(
        "deep_stop_cd18_deep_regime_direction_veto",
        deep_stop_cooldown_extra=18,
        deep_block_long_in_bear_regime=True,
        deep_block_short_in_bull_regime=True,
    )

    all_rows: list[dict[str, Any]] = []
    tests = [
        ("deep_stop_cd18_baseline", baseline_variant, val_dec, eval_dec, {}, {}),
        ("deep_stop_cd18_deep_only_regime_veto", deep_veto_variant, val_dec, eval_dec, {}, {}),
        ("deep_stop_cd18_parent_only_regime_veto", baseline_variant, val_parent_veto_dec, eval_parent_veto_dec, val_parent_counts, eval_parent_counts),
        ("deep_stop_cd18_global_regime_veto", deep_veto_variant, val_parent_veto_dec, eval_parent_veto_dec, val_parent_counts, eval_parent_counts),
    ]
    for name, variant, vdec, odec, val_extra, oos_extra in tests:
        all_rows.extend(
            _eval_periods(
                variant_name=name,
                split="val",
                df=val_df,
                q=val_q,
                dec=vdec,
                stack=stack,
                cfg=cfg,
                variant=variant,
                extra=val_extra,
            )
        )
        all_rows.extend(
            _eval_periods(
                variant_name=name,
                split="oos",
                df=eval_df,
                q=eval_q,
                dec=odec,
                stack=stack,
                cfg=cfg,
                variant=variant,
                extra=oos_extra,
            )
        )

    grid = pd.DataFrame([r for r in all_rows if r["period"].endswith("_full")])
    monthly = pd.DataFrame([r for r in all_rows if not r["period"].endswith("_full")])
    grid.to_csv(GRID_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Regime-direction veto on deep_stop_cd18. Tests deep-only, parent-only, and global parent+deep veto.",
        "rule": "block LONG when clean_regime4_state24_sticky090_v2 dominant regime is bear; block SHORT when dominant regime is bull.",
        "grid": str(GRID_OUT),
        "monthly_grid": str(MONTHLY_OUT),
        "full_rows": grid.to_dict(orient="records"),
        "monthly_rows": monthly.to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "monthly": str(MONTHLY_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
