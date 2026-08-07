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


MODEL_ID = "alpha7_parent_soft_regime_veto_precision_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MONTHLY_OUT = OUT_DIR / "monthly_cost3.csv"
VAL_BASELINE_LEDGER = OUT_DIR / "val_baseline_cost3_ledger.csv"
VAL_CANDIDATE_LEDGER = OUT_DIR / "val_parent_soft_cost3_ledger.csv"
OOS_BASELINE_LEDGER = OUT_DIR / "oos_baseline_cost3_ledger.csv"
OOS_CANDIDATE_LEDGER = OUT_DIR / "oos_parent_soft_cost3_ledger.csv"
BLOCKED_ROWS_OUT = OUT_DIR / "blocked_parent_rows.csv"
LEDGER_DIFF_OUT = OUT_DIR / "ledger_diff_summary.csv"


PARENT_CONF = 0.65
PARENT_MIN_MODEL_CONF = 0.70
PARENT_MIN_QUALITY = 0.040


def _counter_regime_prob(row: pd.Series, side: int) -> float:
    if int(side) > 0:
        return float(row.get("clean_regime4_state24_sticky090_v2_bear_prob", 0.0) or 0.0)
    if int(side) < 0:
        return float(row.get("clean_regime4_state24_sticky090_v2_bull_prob", 0.0) or 0.0)
    return 0.0


def _periods(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    out: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            out.append((f"{prefix}_{month}", mask))
    return out


def _parent_soft_veto(df: pd.DataFrame, dec: pd.DataFrame, *, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out = dec.copy().reset_index(drop=True)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(int)
    action = pd.to_numeric(out["action"], errors="coerce").fillna(0).astype(int)
    active = (action != ACTION_CASH) & (side != 0)
    confidence = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    quality = pd.to_numeric(out["quality_score"], errors="coerce").fillna(-999.0)
    counter_prob = pd.Series([_counter_regime_prob(row, int(s)) for (_, row), s in zip(df.iterrows(), side)], index=out.index)
    in_counter_regime = active & (counter_prob >= PARENT_CONF)
    weak_signal = (confidence < PARENT_MIN_MODEL_CONF) | (quality < PARENT_MIN_QUALITY)
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

    rows = pd.DataFrame(
        {
            "split": split,
            "row_idx": np.flatnonzero(blocked.to_numpy()),
            "timestamp": df.loc[blocked.to_numpy(), "timestamp"].to_numpy(),
            "side": side.loc[blocked].to_numpy(),
            "counter_regime_prob": counter_prob.loc[blocked].to_numpy(dtype=np.float64),
            "confidence": confidence.loc[blocked].to_numpy(dtype=np.float64),
            "quality_score": quality.loc[blocked].to_numpy(dtype=np.float64),
            "notional_exposure": pd.to_numeric(dec.loc[blocked, "notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            "take_profit": pd.to_numeric(dec.loc[blocked, "take_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
            "stop_loss": pd.to_numeric(dec.loc[blocked, "stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        }
    )
    summary = {
        "blocked_rows": int(blocked.sum()),
        "blocked_bear_long_rows": int((blocked & (side > 0)).sum()),
        "blocked_bull_short_rows": int((blocked & (side < 0)).sum()),
        "active_rows": int(active.sum()),
        "blocked_active_ratio": float(blocked.sum() / max(active.sum(), 1)),
    }
    return out, rows, summary


def _row(variant: str, split: str, period: str, res: dict[str, Any]) -> dict[str, Any]:
    return {
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
    record_full: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for period_name, mask in _periods(df, split):
        record = bool(record_full and period_name == f"{split}_full")
        res = sweep._backtest_variant(
            df=df.loc[mask].reset_index(drop=True),
            q=q[mask],
            dec=dec.loc[mask].reset_index(drop=True),
            stack=stack,
            cfg=cfg,
            variant=variant,
            cost_mult=3,
            record=record,
            deep_gate=None,
        )
        if record:
            records = list(res.pop("trade_records", []))
        rows.append(_row(variant_name, split, period_name, res))
    return rows, records


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    df = pd.DataFrame(records)
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "return_sum": float(ret.sum()),
        "return_mean": float(ret.mean()),
        "return_median": float(ret.median()),
        "win_count": int((ret > 0).sum()),
        "loss_count": int((ret <= 0).sum()),
        "top5_return_sum": float(ret[ret > 0].sort_values(ascending=False).head(5).sum()),
        "bottom5_return_sum": float(ret[ret <= 0].sort_values().head(5).sum()),
        "final_cash_after": float(pd.to_numeric(df["cash_after"], errors="coerce").iloc[-1]) if "cash_after" in df.columns else 0.0,
    }
    for owner, g in df.groupby("owner"):
        gr = pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0)
        out[f"{owner}_trades"] = int(len(g))
        out[f"{owner}_return_sum"] = float(gr.sum())
        out[f"{owner}_wr"] = float((gr > 0).mean())
    return out


def _trade_key_frame(records: list[dict[str, Any]], prefix: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["trade_key"])
    df = pd.DataFrame(records).copy()
    for col in ["entry_signal_idx", "side", "owner"]:
        if col not in df.columns:
            raise RuntimeError(f"ledger missing required column: {col}")
    df["trade_key"] = df["entry_signal_idx"].astype(str) + "|" + df["side"].astype(str) + "|" + df["owner"].astype(str)
    keep = [
        "trade_key",
        "entry_signal_idx",
        "entry_time",
        "side",
        "owner",
        "exit_reason",
        "hold_bars",
        "trade_return",
        "cash_after",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df.add_prefix(f"{prefix}_").rename(columns={f"{prefix}_trade_key": "trade_key"})


def _ledger_diff(split: str, base_records: list[dict[str, Any]], cand_records: list[dict[str, Any]], blocked_rows: pd.DataFrame) -> dict[str, Any]:
    base = _trade_key_frame(base_records, "base")
    cand = _trade_key_frame(cand_records, "cand")
    merged = base.merge(cand, on="trade_key", how="outer", indicator=True)
    base_ret = pd.to_numeric(merged.get("base_trade_return"), errors="coerce").fillna(0.0)
    cand_ret = pd.to_numeric(merged.get("cand_trade_return"), errors="coerce").fillna(0.0)
    common = merged["_merge"].eq("both")
    base_only = merged["_merge"].eq("left_only")
    cand_only = merged["_merge"].eq("right_only")
    blocked_set = set(pd.to_numeric(blocked_rows["row_idx"], errors="coerce").dropna().astype(int).tolist())
    base_entry = pd.to_numeric(merged.get("base_entry_signal_idx"), errors="coerce")
    removed_blocked = base_only & base_entry.isin(blocked_set)
    out = {
        "split": split,
        "common_trades": int(common.sum()),
        "baseline_only_trades": int(base_only.sum()),
        "candidate_only_trades": int(cand_only.sum()),
        "common_return_delta_sum": float((cand_ret[common] - base_ret[common]).sum()),
        "baseline_only_return_sum": float(base_ret[base_only].sum()),
        "candidate_only_return_sum": float(cand_ret[cand_only].sum()),
        "baseline_only_blocked_parent_trades": int(removed_blocked.sum()),
        "baseline_only_blocked_parent_return_sum": float(base_ret[removed_blocked].sum()),
        "gross_return_delta_sum": float(cand_ret.sum() - base_ret.sum()),
    }
    detail_path = OUT_DIR / f"{split}_ledger_trade_key_diff.csv"
    merged.to_csv(detail_path, index=False)
    out["detail_path"] = str(detail_path)
    return out


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

    val_soft_dec, val_blocked, val_block_summary = _parent_soft_veto(val_df, val_dec, split="val")
    eval_soft_dec, eval_blocked, eval_block_summary = _parent_soft_veto(eval_df, eval_dec, split="oos")
    blocked_rows = pd.concat([val_blocked, eval_blocked], ignore_index=True)
    blocked_rows.to_csv(BLOCKED_ROWS_OUT, index=False)

    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    rows: list[dict[str, Any]] = []
    val_base_rows, val_base_records = _eval_periods(
        variant_name="baseline",
        split="val",
        df=val_df,
        q=val_q,
        dec=val_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        record_full=True,
    )
    val_cand_rows, val_cand_records = _eval_periods(
        variant_name="parent_soft_c65_conf70_q040_any",
        split="val",
        df=val_df,
        q=val_q,
        dec=val_soft_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        record_full=True,
    )
    oos_base_rows, oos_base_records = _eval_periods(
        variant_name="baseline",
        split="oos",
        df=eval_df,
        q=eval_q,
        dec=eval_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        record_full=True,
    )
    oos_cand_rows, oos_cand_records = _eval_periods(
        variant_name="parent_soft_c65_conf70_q040_any",
        split="oos",
        df=eval_df,
        q=eval_q,
        dec=eval_soft_dec,
        stack=stack,
        cfg=cfg,
        variant=variant,
        record_full=True,
    )
    rows.extend([r for r in val_base_rows + val_cand_rows + oos_base_rows + oos_cand_rows if r["period"].endswith("_full")])
    monthly = [r for r in val_base_rows + val_cand_rows + oos_base_rows + oos_cand_rows if not r["period"].endswith("_full")]
    pd.DataFrame(rows).to_csv(GRID_OUT, index=False)
    pd.DataFrame(monthly).to_csv(MONTHLY_OUT, index=False)
    pd.DataFrame(val_base_records).to_csv(VAL_BASELINE_LEDGER, index=False)
    pd.DataFrame(val_cand_records).to_csv(VAL_CANDIDATE_LEDGER, index=False)
    pd.DataFrame(oos_base_records).to_csv(OOS_BASELINE_LEDGER, index=False)
    pd.DataFrame(oos_cand_records).to_csv(OOS_CANDIDATE_LEDGER, index=False)

    ledger_diffs = [
        _ledger_diff("val", val_base_records, val_cand_records, val_blocked),
        _ledger_diff("oos", oos_base_records, oos_cand_records, eval_blocked),
    ]
    pd.DataFrame(ledger_diffs).to_csv(LEDGER_DIFF_OUT, index=False)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Ledger-level precision retest for parent_soft_c65_conf70_q040_any on deep_stop_cd18.",
        "rule": {
            "parent_conf": PARENT_CONF,
            "parent_min_model_conf": PARENT_MIN_MODEL_CONF,
            "parent_min_quality": PARENT_MIN_QUALITY,
            "weak_mode": "confidence < threshold OR quality < threshold",
        },
        "grid": str(GRID_OUT),
        "monthly_grid": str(MONTHLY_OUT),
        "blocked_rows": str(BLOCKED_ROWS_OUT),
        "ledger_diff_summary": str(LEDGER_DIFF_OUT),
        "ledgers": {
            "val_baseline": str(VAL_BASELINE_LEDGER),
            "val_candidate": str(VAL_CANDIDATE_LEDGER),
            "oos_baseline": str(OOS_BASELINE_LEDGER),
            "oos_candidate": str(OOS_CANDIDATE_LEDGER),
        },
        "block_summary": {"val": val_block_summary, "oos": eval_block_summary},
        "full_rows": rows,
        "monthly_rows": monthly,
        "ledger_stats": {
            "val_baseline": _ledger_stats(val_base_records),
            "val_candidate": _ledger_stats(val_cand_records),
            "oos_baseline": _ledger_stats(oos_base_records),
            "oos_candidate": _ledger_stats(oos_cand_records),
        },
        "ledger_diffs": ledger_diffs,
        "warning": "This is a precision retest on the existing validation/OOS frames, not an untouched deployment validation.",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "ledger_diff": str(LEDGER_DIFF_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
