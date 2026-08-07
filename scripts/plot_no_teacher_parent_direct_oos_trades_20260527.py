#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame, prepare_features  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.ablate_alpha3_teacher_layer_20260527 import _load_stack, _merge_state24  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import (  # noqa: E402
    ExitGuardConfig,
    _default_limit_cfg,
    backtest_signal_limit_exit_guard,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "alpha3_no_teacher_parent_direct_oos_trades_20260527"
REPORT_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_summary.json"
LEDGER_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_cost3_ledger.csv"
CHART_FULL_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_candles_cost3.png"
CHART_ZOOM_OUT = ROOT / f"data/ensemble/reports/{MODEL_ID}_trade_windows_cost3.png"


def _guard() -> ExitGuardConfig:
    return ExitGuardConfig(
        name="guard_soft3_hard1p45",
        hard_sl_mult=1.45,
        soft_sl_mult=1.0,
        early_bars=18,
        early_sl_mult=1.35,
        soft_min_hold=3,
        soft_persist_bars=3,
        regime_bad_th=0.50,
        flow_bad_th=0.02,
        giveback_trigger=0.72,
        giveback_min_mfe=0.014,
        giveback_min_hold=3,
        entry_quality_min=-999.0,
        entry_conf_min=0.0,
        same_side_entry_gap=0,
        cooldown_after_hard_stop=0,
        cooldown_after_soft_stop=0,
        cooldown_after_giveback=0,
    )


def _plot_candles(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    out: Path,
    *,
    title: str,
    start: int = 0,
    end: int | None = None,
) -> None:
    end = len(df) if end is None else int(min(end, len(df)))
    start = int(max(0, start))
    sub = df.iloc[start:end].reset_index(drop=False).rename(columns={"index": "bar_idx"})
    x = np.arange(len(sub), dtype=np.int64)
    o = pd.to_numeric(sub["open"], errors="coerce").ffill().to_numpy(dtype=float)
    h = pd.to_numeric(sub["high"], errors="coerce").ffill().to_numpy(dtype=float)
    l = pd.to_numeric(sub["low"], errors="coerce").ffill().to_numpy(dtype=float)
    c = pd.to_numeric(sub["close"], errors="coerce").ffill().to_numpy(dtype=float)
    up = c >= o

    fig, ax = plt.subplots(figsize=(22, 8), dpi=140)
    ax.vlines(x, l, h, color=np.where(up, "#1f9d55", "#c2410c"), linewidth=0.25, alpha=0.45)
    ax.vlines(x, o, c, color=np.where(up, "#15803d", "#b91c1c"), linewidth=0.75, alpha=0.78)

    visible = ledger[(ledger["entry_fill_idx"] >= start) & (ledger["entry_fill_idx"] < end)].copy()
    if not visible.empty:
        ex = visible["entry_fill_idx"].astype(int).to_numpy() - start
        ep = visible["entry_price"].astype(float).to_numpy()
        long_mask = visible["side"].astype(str).str.upper().to_numpy() == "LONG"
        short_mask = visible["side"].astype(str).str.upper().to_numpy() == "SHORT"
        ax.scatter(ex[long_mask], ep[long_mask], marker="^", s=20, color="#2563eb", zorder=4)
        ax.scatter(ex[short_mask], ep[short_mask], marker="v", s=20, color="#7c3aed", zorder=4)

    exits = ledger[(ledger["exit_fill_idx"] >= start) & (ledger["exit_fill_idx"] < end)].copy()
    if not exits.empty:
        xx = exits["exit_fill_idx"].astype(int).to_numpy() - start
        xp = exits["exit_price"].astype(float).to_numpy()
        ax.scatter(xx, xp, marker="x", s=18, color="#111827", linewidths=0.8, zorder=5)

    ax.set_title(title)
    ax.set_xlabel(f"5m bar index ({start}..{end})")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.16)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#2563eb", label="Long Entry", markersize=7),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#7c3aed", label="Short Entry", markersize=7),
            Line2D([0], [0], marker="x", color="#111827", label="Exit", markersize=7, linestyle="None"),
        ],
        loc="upper left",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parent, runner, add_cfg, overlay, _, teacher_model, teacher_cols, teacher_norm, teacher_buckets, deep_model, deep_payload, _ = _load_stack()
    fee = float(parent["config"]["fee"])
    slip = float(parent["config"]["slip"])
    eval_df = _read(v31.DEFAULT_EVAL)
    eval_df = _merge_state24(eval_df, alpha3_full.SIDE_CLEAN4_2026)

    eval_parent = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    _ = teacher._predict_deep(teacher_model, eval_features, teacher_cols, teacher_norm)
    decisions = eval_parent.copy()
    deep_q = v27._predict_all(deep_model, eval_df, deep_payload["seq_cols"], deep_payload["norm"])

    result = backtest_signal_limit_exit_guard(
        eval_df,
        parent,
        runner,
        add_cfg,
        deep_q,
        decisions,
        overlay,
        _default_limit_cfg(),
        _guard(),
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        record=True,
    )
    ledger = pd.DataFrame(result.pop("trade_records", []))
    ledger.to_csv(LEDGER_OUT, index=False)

    _plot_candles(
        eval_df,
        ledger,
        CHART_FULL_OUT,
        title="Alpha3 no_teacher_parent_direct - OOS Candles + Trades (Cost3)",
    )
    if len(ledger):
        start = max(0, int(ledger["entry_fill_idx"].min()) - 96)
        end = min(len(eval_df), int(ledger["exit_fill_idx"].max()) + 96)
    else:
        start, end = 0, min(len(eval_df), 2500)
    _plot_candles(
        eval_df,
        ledger,
        CHART_ZOOM_OUT,
        title="Alpha3 no_teacher_parent_direct - OOS Trade Window Candles (Cost3)",
        start=start,
        end=end,
    )

    report = {
        "model_id": MODEL_ID,
        "variant": "no_teacher_parent_direct",
        "metrics_cost3": result,
        "ledger_rows": int(len(ledger)),
        "ledger": str(LEDGER_OUT),
        "chart_full": str(CHART_FULL_OUT),
        "chart_zoom": str(CHART_ZOOM_OUT),
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
