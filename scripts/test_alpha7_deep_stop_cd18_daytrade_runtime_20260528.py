#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
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


MODEL_ID = "alpha7_deep_stop_cd18_daytrade_runtime_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
GRID_OUT = OUT_DIR / "grid.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"


@dataclass(frozen=True)
class DaytradeVariant:
    name: str
    cfg_overrides: dict[str, Any] = field(default_factory=dict)
    parent_daily_top_k: int = 0
    deep_daily_top_k: int = 0
    deep_side: str = "both"
    deep_stop_cooldown_extra: int = 18
    deep_block_long_in_bear_regime: bool = False
    deep_block_short_in_bull_regime: bool = False


def _active(dec: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH) & (
        pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int) != 0
    )


def _with_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out


def _filter_parent_daily_top_k(df: pd.DataFrame, dec: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    if top_k <= 0:
        return dec
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    if not bool(active.any()):
        return out
    ts = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    quality = pd.to_numeric(out.get("quality_score", 0.0), errors="coerce").fillna(-999.0)
    conf = pd.to_numeric(out.get("confidence", 0.0), errors="coerce").fillna(0.0)
    score = quality + 0.001 * conf
    allowed = pd.Series(False, index=out.index)
    for _, idx in out.index[active].to_series().groupby(ts[active].dt.date).groups.items():
        chosen = score.loc[list(idx)].sort_values(ascending=False).head(int(top_k)).index
        allowed.loc[chosen] = True
    out.loc[active & ~allowed, ["action", "side"]] = 0
    return out


def _deep_allowed_set(df: pd.DataFrame, q: np.ndarray, *, top_k: int) -> set[int]:
    if top_k <= 0:
        return set()
    ts = pd.to_datetime(df["timestamp"], errors="coerce").reset_index(drop=True)
    ql = pd.Series(q[:, 0], index=df.index, dtype="float64")
    qs = pd.Series(q[:, 1], index=df.index, dtype="float64")
    score = pd.Series(np.maximum(q[:, 0], q[:, 1]) + 0.25 * np.abs(q[:, 0] - q[:, 1]), index=df.index)
    eligible = pd.Series(np.arange(len(df)) >= 60, index=df.index)
    allowed: set[int] = set()
    for _, idx in df.index[eligible].to_series().groupby(ts[eligible].dt.date).groups.items():
        chosen = score.loc[list(idx)].sort_values(ascending=False).head(int(top_k)).index
        allowed.update(int(i) for i in chosen)
    return allowed


def _daytrade_score(res: dict[str, Any], *, target_tpd: float = 2.5) -> float:
    trades = int(res.get("trades", 0))
    if trades < 10:
        return -1e9 + float(res.get("pnl", 0.0))
    pnl = float(res.get("pnl", 0.0))
    mdd = abs(float(res.get("mdd", 0.0)))
    wr = float(res.get("wr", 0.0))
    tpd = float(res.get("trades_per_day", 0.0))
    sl = float(sweep._sl_ratio(res))
    return pnl - 1.25 * mdd + 30.0 * wr - 20.0 * abs(tpd - target_tpd) - 80.0 * sl


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    df = pd.DataFrame(records)
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    hold = pd.to_numeric(df["hold_bars"], errors="coerce").fillna(0.0)
    ts = pd.to_datetime(df["entry_time"], errors="coerce")
    return {
        "rows": int(len(df)),
        "trades_per_calendar_day": float(len(df) / max((ts.max().date() - ts.min().date()).days + 1, 1)) if ts.notna().any() else 0.0,
        "raw_sum": float(ret.sum()),
        "raw_mean": float(ret.mean()),
        "raw_wr": float((ret > 0).mean()),
        "median_hold_bars": float(hold.median()),
        "mean_hold_bars": float(hold.mean()),
        "max_hold_bars": int(hold.max()),
        "by_owner": {
            str(k): {
                "trades": int(len(g)),
                "raw_sum": float(pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0).sum()),
            }
            for k, g in df.groupby("owner")
        },
        "by_side": {
            str(k): {
                "trades": int(len(g)),
                "raw_sum": float(pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0).sum()),
            }
            for k, g in df.groupby("side")
        },
    }


def _eval_variant(
    *,
    variant: DaytradeVariant,
    split_name: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    base_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = _with_overrides(base_cfg, variant.cfg_overrides)
    cfg["name"] = variant.name
    dec2 = _filter_parent_daily_top_k(df, dec, top_k=variant.parent_daily_top_k)
    deep_allowed = _deep_allowed_set(df, q, top_k=variant.deep_daily_top_k)

    def deep_gate(i: int, side: int, ql: float, qs: float, row: pd.Series) -> tuple[bool, str]:
        if variant.deep_daily_top_k > 0 and int(i) not in deep_allowed:
            return False, "daily_topk"
        return True, ""

    sweep_variant = sweep.Variant(
        name=variant.name,
        deep_side=variant.deep_side,
        deep_stop_cooldown_extra=int(variant.deep_stop_cooldown_extra),
        deep_block_long_in_bear_regime=bool(variant.deep_block_long_in_bear_regime),
        deep_block_short_in_bull_regime=bool(variant.deep_block_short_in_bull_regime),
    )
    res = sweep._backtest_variant(
        df=df,
        q=q,
        dec=dec2,
        stack=stack,
        cfg=cfg,
        variant=sweep_variant,
        cost_mult=3,
        record=True,
        deep_gate=deep_gate if variant.deep_daily_top_k > 0 else None,
    )
    records = list(res.pop("trade_records", []))
    row = {
        **asdict(variant),
        "split": split_name,
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "legacy_score": float(sweep._score(res)),
        "daytrade_score": float(_daytrade_score(res)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
        "ledger_stats": json.dumps(_ledger_stats(records), ensure_ascii=False, sort_keys=True),
        "_records": records,
    }
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    base_cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    val_dec = sources[str(base_cfg["source"])][0]
    eval_dec = sources[str(base_cfg["source"])][1]

    variants = [
        DaytradeVariant("current_deep_stop_cd18"),
        DaytradeVariant(
            "dt_no_fallthrough_parent_strict",
            cfg_overrides={
                "entry_quality_min": 0.002,
                "entry_conf_min": 0.60,
                "same_side_entry_gap": 48,
                "soft_min_hold": 24,
                "giveback_min_hold": 24,
                "early_bars": 72,
            },
            deep_side="none",
        ),
        DaytradeVariant(
            "dt_parent_top3_deep_off",
            cfg_overrides={
                "entry_quality_min": 0.0015,
                "entry_conf_min": 0.55,
                "same_side_entry_gap": 48,
                "soft_min_hold": 24,
                "giveback_min_hold": 24,
                "early_bars": 72,
            },
            parent_daily_top_k=3,
            deep_side="none",
        ),
        DaytradeVariant(
            "dt_parent_top3_deep_top3",
            cfg_overrides={
                "entry_quality_min": 0.0015,
                "entry_conf_min": 0.55,
                "same_side_entry_gap": 48,
                "soft_min_hold": 24,
                "giveback_min_hold": 24,
                "early_bars": 72,
            },
            parent_daily_top_k=3,
            deep_daily_top_k=3,
        ),
        DaytradeVariant(
            "dt_parent_top2_deep_top2",
            cfg_overrides={
                "entry_quality_min": 0.002,
                "entry_conf_min": 0.60,
                "same_side_entry_gap": 72,
                "soft_min_hold": 30,
                "giveback_min_hold": 30,
                "early_bars": 96,
            },
            parent_daily_top_k=2,
            deep_daily_top_k=2,
        ),
        DaytradeVariant(
            "dt_parent_top2_deep_top2_regime_veto",
            cfg_overrides={
                "entry_quality_min": 0.002,
                "entry_conf_min": 0.60,
                "same_side_entry_gap": 72,
                "soft_min_hold": 30,
                "giveback_min_hold": 30,
                "early_bars": 96,
            },
            parent_daily_top_k=2,
            deep_daily_top_k=2,
            deep_block_long_in_bear_regime=True,
            deep_block_short_in_bull_regime=True,
        ),
        DaytradeVariant(
            "dt_parent_top1_deep_top2",
            cfg_overrides={
                "entry_quality_min": 0.0025,
                "entry_conf_min": 0.65,
                "same_side_entry_gap": 96,
                "soft_min_hold": 36,
                "giveback_min_hold": 36,
                "early_bars": 120,
            },
            parent_daily_top_k=1,
            deep_daily_top_k=2,
        ),
    ]

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        for split_name, df, q, dec in (("val", val_df, val_q, val_dec), ("oos", eval_df, eval_q, eval_dec)):
            row = _eval_variant(variant=variant, split_name=split_name, df=df, q=q, dec=dec, stack=stack, base_cfg=base_cfg)
            records = list(row.pop("_records", []))
            rows.append(row)
            if split_name == "oos":
                ledgers[variant.name] = records

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    oos = grid[grid["split"].eq("oos")].copy().sort_values(["daytrade_score", "pnl"], ascending=[False, False])
    best = str(oos.iloc[0]["name"]) if not oos.empty else ""
    best_ledger = OUT_DIR / f"{best}_oos_cost3_ledger.csv"
    pd.DataFrame(ledgers.get(best, [])).to_csv(best_ledger, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Runtime-only daytrade direction test. Model artifacts, feature contract, costs, and limit execution are frozen.",
        "base_model": "alpha7_submodel_01965_decontam_deep_stop_cd18_20260528",
        "target": "2-3 trades/day, longer thesis, reduced scalp fallback.",
        "grid": str(GRID_OUT),
        "best_by_daytrade_score": best,
        "best_oos_ledger": str(best_ledger),
        "oos_rows": oos.to_dict(orient="records"),
        "all_rows": grid.to_dict(orient="records"),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "best": best}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
