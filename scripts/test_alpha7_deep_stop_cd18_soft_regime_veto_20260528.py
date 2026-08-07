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
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_deep_stop_cd18_soft_regime_veto_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MONTHLY_OUT = OUT_DIR / "monthly_cost3.csv"


@dataclass(frozen=True)
class SoftVetoConfig:
    name: str
    deep_conf: float = 0.0
    deep_edge_mult: float = 1.0
    deep_margin_mult: float = 1.0
    parent_conf: float = 0.0
    parent_min_model_conf: float = 0.0
    parent_min_quality: float = 0.0
    parent_weak_mode: str = "any"
    deep_any_loss_cd: int = 0


def _regime_prob(row: pd.Series, regime: str) -> float:
    return float(row.get(f"clean_regime4_state24_sticky090_v2_{regime}_prob", 0.0) or 0.0)


def _counter_regime_prob(row: pd.Series, side: int) -> float:
    if int(side) > 0:
        return _regime_prob(row, "bear")
    if int(side) < 0:
        return _regime_prob(row, "bull")
    return 0.0


def _periods(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    out: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            out.append((f"{prefix}_{month}", mask))
    return out


def _deep_gate(config: SoftVetoConfig, edge_th: float, margin_th: float):
    def _gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        counter_prob = _counter_regime_prob(row, side)
        if counter_prob < float(config.deep_conf):
            return True, "soft_regime_pass"
        edge = max(float(ql), float(qs))
        margin = abs(float(ql) - float(qs))
        strong = edge >= float(edge_th * config.deep_edge_mult) and margin >= float(margin_th * config.deep_margin_mult)
        if strong:
            return True, "soft_regime_strong_pass"
        return False, "soft_regime_veto"

    return _gate


def _parent_soft_veto(df: pd.DataFrame, dec: pd.DataFrame, config: SoftVetoConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    out = dec.copy().reset_index(drop=True)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(int)
    action = pd.to_numeric(out["action"], errors="coerce").fillna(0).astype(int)
    active = (action != ACTION_CASH) & (side != 0)
    confidence = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    quality = pd.to_numeric(out["quality_score"], errors="coerce").fillna(-999.0)
    counter_prob = pd.Series([_counter_regime_prob(row, int(s)) for (_, row), s in zip(df.iterrows(), side)], index=out.index)
    in_counter_regime = active & (counter_prob >= float(config.parent_conf))
    weak_conf = confidence < float(config.parent_min_model_conf)
    weak_quality = quality < float(config.parent_min_quality)
    if config.parent_weak_mode == "both":
        weak_signal = weak_conf & weak_quality
    elif config.parent_weak_mode == "quality":
        weak_signal = weak_quality
    elif config.parent_weak_mode == "confidence":
        weak_signal = weak_conf
    else:
        weak_signal = weak_conf | weak_quality
    blocked = in_counter_regime & weak_signal
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
    bear_long = blocked & (side > 0)
    bull_short = blocked & (side < 0)
    return out, {
        "parent_soft_bear_long_veto_rows": int(bear_long.sum()),
        "parent_soft_bull_short_veto_rows": int(bull_short.sum()),
        "parent_soft_total_veto_rows": int(blocked.sum()),
    }


def _row(variant: str, split: str, period: str, res: dict[str, Any], config: SoftVetoConfig, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "variant": variant,
        "split": split,
        "period": period,
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
        "runner_actions": json.dumps(res.get("runner_actions", {}), ensure_ascii=False, sort_keys=True),
        **asdict(config),
    }
    if extra:
        row.update(extra)
    return row


def _eval_periods(
    *,
    split: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
    config: SoftVetoConfig,
    deep_gate: Any | None,
    extra: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period_name, mask in _periods(df, split):
        res = sweep._backtest_variant(
            df=df.loc[mask].reset_index(drop=True),
            q=q[mask],
            dec=dec.loc[mask].reset_index(drop=True),
            stack=stack,
            cfg=cfg,
            variant=variant,
            cost_mult=3,
            record=False,
            deep_gate=deep_gate,
        )
        rows.append(_row(config.name, split, period_name, res, config, extra))
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
    overlay = precision._overlay(stack["overlay"], cfg)

    configs = [
        SoftVetoConfig("baseline"),
        SoftVetoConfig("deep_soft_c55_e110_m110", deep_conf=0.55, deep_edge_mult=1.10, deep_margin_mult=1.10),
        SoftVetoConfig("deep_soft_c55_e125_m125", deep_conf=0.55, deep_edge_mult=1.25, deep_margin_mult=1.25),
        SoftVetoConfig("deep_soft_c65_e110_m110", deep_conf=0.65, deep_edge_mult=1.10, deep_margin_mult=1.10),
        SoftVetoConfig("deep_soft_c65_e125_m125", deep_conf=0.65, deep_edge_mult=1.25, deep_margin_mult=1.25),
        SoftVetoConfig("deep_soft_c70_e125_m125", deep_conf=0.70, deep_edge_mult=1.25, deep_margin_mult=1.25),
        SoftVetoConfig("deep_soft_c80_e125_m125", deep_conf=0.80, deep_edge_mult=1.25, deep_margin_mult=1.25),
        SoftVetoConfig("deep_soft_c80_e150_m150", deep_conf=0.80, deep_edge_mult=1.50, deep_margin_mult=1.50),
        SoftVetoConfig("deep_soft_c90_e125_m125", deep_conf=0.90, deep_edge_mult=1.25, deep_margin_mult=1.25),
        SoftVetoConfig("deep_soft_c80_e125_m125_losscd36", deep_conf=0.80, deep_edge_mult=1.25, deep_margin_mult=1.25, deep_any_loss_cd=36),
        SoftVetoConfig("parent_soft_c55_conf65_q035_any", parent_conf=0.55, parent_min_model_conf=0.65, parent_min_quality=0.035, parent_weak_mode="any"),
        SoftVetoConfig("parent_soft_c55_conf70_q040_any", parent_conf=0.55, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("parent_soft_c55_conf70_q040_both", parent_conf=0.55, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="both"),
        SoftVetoConfig("parent_soft_c65_conf65_q035_any", parent_conf=0.65, parent_min_model_conf=0.65, parent_min_quality=0.035, parent_weak_mode="any"),
        SoftVetoConfig("parent_soft_c65_conf70_q040_any", parent_conf=0.65, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("parent_soft_c65_conf70_q040_both", parent_conf=0.65, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="both"),
        SoftVetoConfig("parent_soft_c80_conf65_q035_any", parent_conf=0.80, parent_min_model_conf=0.65, parent_min_quality=0.035, parent_weak_mode="any"),
        SoftVetoConfig("parent_soft_c80_conf70_q040_any", parent_conf=0.80, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("parent_soft_c80_conf70_q040_both", parent_conf=0.80, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="both"),
        SoftVetoConfig("parent_soft_c90_conf70_q040_any", parent_conf=0.90, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("global_soft_c55", deep_conf=0.55, deep_edge_mult=1.10, deep_margin_mult=1.10, parent_conf=0.55, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("global_soft_c65", deep_conf=0.65, deep_edge_mult=1.10, deep_margin_mult=1.10, parent_conf=0.65, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("global_soft_c80", deep_conf=0.80, deep_edge_mult=1.25, deep_margin_mult=1.25, parent_conf=0.80, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
        SoftVetoConfig("global_soft_c90", deep_conf=0.90, deep_edge_mult=1.25, deep_margin_mult=1.25, parent_conf=0.90, parent_min_model_conf=0.70, parent_min_quality=0.040, parent_weak_mode="any"),
    ]

    all_rows: list[dict[str, Any]] = []
    for config in configs:
        variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18, deep_any_loss_cooldown_extra=int(config.deep_any_loss_cd))
        val_work = val_dec
        eval_work = eval_dec
        val_extra: dict[str, int] = {}
        eval_extra: dict[str, int] = {}
        if config.parent_conf > 0.0:
            val_work, val_extra = _parent_soft_veto(val_df, val_dec, config)
            eval_work, eval_extra = _parent_soft_veto(eval_df, eval_dec, config)
        gate = None
        if config.deep_conf > 0.0:
            gate = _deep_gate(config, edge_th=float(overlay.edge_th), margin_th=float(overlay.margin_th))
        all_rows.extend(
            _eval_periods(
                split="val",
                df=val_df,
                q=val_q,
                dec=val_work,
                stack=stack,
                cfg=cfg,
                variant=variant,
                config=config,
                deep_gate=gate,
                extra=val_extra,
            )
        )
        all_rows.extend(
            _eval_periods(
                split="oos",
                df=eval_df,
                q=eval_q,
                dec=eval_work,
                stack=stack,
                cfg=cfg,
                variant=variant,
                config=config,
                deep_gate=gate,
                extra=eval_extra,
            )
        )

    grid = pd.DataFrame([r for r in all_rows if r["period"].endswith("_full")])
    monthly = pd.DataFrame([r for r in all_rows if not r["period"].endswith("_full")])
    grid.to_csv(GRID_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Soft regime-direction veto on deep_stop_cd18 using regime confidence, deep edge/margin, and optional post-loss cooldown.",
        "deep_rule": "If counter-regime probability is above deep_conf, block only when deep edge/margin are not strong enough.",
        "parent_rule": "If counter-regime probability is above parent_conf, block only weak parent rows by confidence/quality.",
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
