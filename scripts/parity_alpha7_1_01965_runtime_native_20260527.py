#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import fields
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
    v31,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import (  # noqa: E402
    CANDIDATE,
    OOS_LEDGER_OUT as REF_OOS_LEDGER,
    _cfg_from_results,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402
from trading_bot import FinalGovernorRuntime  # noqa: E402


MODEL_ID = "alpha7_1_01965_runtime_native_parity_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
BOT_OOS_LEDGER_OUT = OUT_DIR / "bot_oos_cost3_ledger.csv"
BOT_DECISION_DIFF_OUT = OUT_DIR / "bot_decision_diff.csv"
LEDGER_DIFF_OUT = OUT_DIR / "ledger_diff.csv"


def _active(dec: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH) & (
        pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int) != 0
    )


def _apply_row(fn, dec: pd.DataFrame) -> pd.DataFrame:
    rows = [fn(row) for _, row in dec.iterrows()]
    return pd.DataFrame(rows).reset_index(drop=True)


def _runtime_parent_decisions(governor: FinalGovernorRuntime, frame: pd.DataFrame) -> pd.DataFrame:
    primary_result = governor._fully_learned_decision_frame(frame)
    if primary_result is None:
        raise RuntimeError("fully learned primary decision frame unavailable")
    primary_raw = primary_result[0].reset_index(drop=True)
    primary = _apply_row(governor._apply_fully_learned_primary_overlays, primary_raw)
    primary = _apply_row(governor._apply_fully_learned_runtime_config, primary)

    fallback_bundle = governor.fully_learned_fallback_policy_bundle
    if fallback_bundle is None:
        return primary
    fallback_result = governor._fully_learned_decision_frame(frame, bundle=fallback_bundle)
    if fallback_result is None:
        raise RuntimeError("fully learned fallback decision frame unavailable")
    fallback_raw = fallback_result[0].reset_index(drop=True)
    fallback_raw = governor._scale_fully_learned_decisions_with_runtime(
        fallback_raw,
        governor.fully_learned_fallback_scale_runtime,
    )
    fallback = _apply_row(governor._apply_fully_learned_fallback_overlays, fallback_raw)
    fallback = _apply_row(governor._apply_fully_learned_runtime_config, fallback)

    out = primary.copy().reset_index(drop=True)
    mask = (~_active(out)) & _active(fallback)
    for col in fallback.columns:
        if col in out.columns:
            out.loc[mask, col] = fallback.loc[mask, col].to_numpy()
    return out


def _runtime_overlay(governor: FinalGovernorRuntime) -> v31.OverlayConfig:
    cfg = dict(governor.v31_cfg or {})
    overlay_keys = [f.name for f in fields(v31.OverlayConfig)]
    missing = [k for k in overlay_keys if k not in cfg]
    if missing:
        raise RuntimeError(f"runtime v31 config missing overlay keys: {missing}")
    return v31.OverlayConfig(**{k: cfg[k] for k in overlay_keys})


def _decision_diff(expected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    always_cols = ["action", "side"]
    active_cols = [
        "notional_exposure",
        "leverage",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
        "cooldown_bars",
        "quality_score",
        "confidence",
    ]
    rows: list[dict[str, Any]] = []
    active_mask = (_active(expected) | _active(actual)).to_numpy(dtype=bool)
    for col in [c for c in [*always_cols, *active_cols] if c in expected.columns and c in actual.columns]:
        a = pd.to_numeric(expected[col], errors="coerce")
        b = pd.to_numeric(actual[col], errors="coerce")
        if col in {"action", "side", "max_hold_bars", "cooldown_bars"}:
            bad = a.fillna(-999999).astype(int).to_numpy() != b.fillna(-999999).astype(int).to_numpy()
        else:
            bad = ~np.isclose(a.fillna(np.nan).to_numpy(dtype=np.float64), b.fillna(np.nan).to_numpy(dtype=np.float64), atol=1e-10, rtol=1e-10, equal_nan=True)
        if col in active_cols:
            bad = bad & active_mask
        idx = np.where(bad)[0]
        for i in idx[:200]:
            rows.append({"row": int(i), "column": col, "expected": expected.iloc[i].get(col), "actual": actual.iloc[i].get(col)})
    return pd.DataFrame(rows)


def _ledger_diff(expected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    cols = ["entry_time", "side", "owner", "notional", "exit_time", "exit_reason", "trade_return"]
    rows: list[dict[str, Any]] = []
    if len(expected) != len(actual):
        rows.append({"row": -1, "column": "__len__", "expected": int(len(expected)), "actual": int(len(actual))})
    n = min(len(expected), len(actual))
    for col in cols:
        if col not in expected.columns or col not in actual.columns:
            rows.append({"row": -1, "column": col, "expected": "missing" if col not in expected.columns else "present", "actual": "missing" if col not in actual.columns else "present"})
            continue
        if col in {"notional", "trade_return"}:
            bad = ~np.isclose(
                pd.to_numeric(expected[col].iloc[:n], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float64),
                pd.to_numeric(actual[col].iloc[:n], errors="coerce").fillna(np.nan).to_numpy(dtype=np.float64),
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
            )
        else:
            bad = expected[col].iloc[:n].astype(str).to_numpy() != actual[col].iloc[:n].astype(str).to_numpy()
        idx = np.where(bad)[0]
        for i in idx[:200]:
            rows.append({"row": int(i), "column": col, "expected": expected.iloc[i].get(col), "actual": actual.iloc[i].get(col)})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    if str(cfg.get("name")) != CANDIDATE:
        raise RuntimeError(f"unexpected candidate: {cfg.get('name')}")
    stack = _load_stack()
    _val_df, eval_df = _load_frames()
    sources = _decision_sources(_val_df, eval_df, stack["parent"])
    expected_dec = _apply_decision_mods(sources[str(cfg["source"])][1], cfg).reset_index(drop=True)

    governor = FinalGovernorRuntime()
    bot_dec = _runtime_parent_decisions(governor, eval_df).reset_index(drop=True)
    dec_diff = _decision_diff(expected_dec, bot_dec)
    dec_diff.to_csv(BOT_DECISION_DIFF_OUT, index=False)

    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    res = backtest_signal_limit_exit_guard(
        eval_df.reset_index(drop=True),
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        eval_q,
        bot_dec.reset_index(drop=True),
        _runtime_overlay(governor),
        _default_limit_cfg(),
        _guard(cfg),
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=3.0,
        record=True,
    )
    bot_ledger = pd.DataFrame(res.get("trade_records", []))
    bot_ledger.to_csv(BOT_OOS_LEDGER_OUT, index=False)

    ref_ledger = pd.read_csv(REF_OOS_LEDGER)
    ledger_diff = _ledger_diff(ref_ledger, bot_ledger)
    ledger_diff.to_csv(LEDGER_DIFF_OUT, index=False)
    summary = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "runtime_model_id": governor.fully_learned_runtime_config.get("model_id"),
        "runtime_v31_cfg": dict(governor.v31_cfg or {}),
        "decision_mismatches": int(len(dec_diff)),
        "ledger_mismatches": int(len(ledger_diff)),
        "reference_ledger": str(REF_OOS_LEDGER),
        "bot_ledger": str(BOT_OOS_LEDGER_OUT),
        "decision_diff": str(BOT_DECISION_DIFF_OUT),
        "ledger_diff": str(LEDGER_DIFF_OUT),
        "oos_cost3": {
            "pnl": float(res["pnl"]),
            "mdd": float(res["mdd"]),
            "wr": float(res["wr"]),
            "trades": int(res["trades"]),
            "deep_entries": int(res.get("deep_entries", 0)),
            "long_entries": int(res.get("long_entries", 0)),
            "short_entries": int(res.get("short_entries", 0)),
            "exits": dict(res.get("exits", {})),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default))
    return 1 if summary["decision_mismatches"] or summary["ledger_mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
