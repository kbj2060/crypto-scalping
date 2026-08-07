#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


MODEL_ID = "omega4_4_cash_sleeve_filters_on_omega3_aggressive_20260623"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PRIMARY_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_current_baseline_growth_20260606"
OMEGA44_RISK_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623"
)
OMEGA44_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623"
    / "true_3head_tabm_bundle.pt"
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _load_primary(split: str) -> pd.DataFrame:
    path = PRIMARY_DIR / f"omega1_2_1_aggressive_compensated_scale200_cap090_{split}_trade_ledger_20260606.csv"
    df = pd.read_csv(path)
    df["model"] = "omega3_main"
    df["entry_ts"] = pd.to_datetime(df["entry_time"], errors="raise")
    df["exit_ts"] = pd.to_datetime(df["exit_time"], errors="raise")
    df["ret_dec"] = pd.to_numeric(df["net_trade_return_pct"], errors="raise") / 100.0
    df["mae_dec"] = pd.to_numeric(df["mae_pct"], errors="raise") / 100.0
    df["reason"] = df["exit_reason"].astype(str)
    df["side_num"] = df["side"].map({"LONG": 1, "SHORT": -1}).astype(int)
    return df


def _load_sleeve(split: str) -> pd.DataFrame:
    name = "validation_selected_risk_trade_ledger.csv" if split == "validation" else "oos_selected_risk_trade_ledger.csv"
    df = pd.read_csv(OMEGA44_RISK_DIR / name)
    df["model"] = "omega4_4_sleeve"
    df["entry_ts"] = pd.to_datetime(df["entry_timestamp"], errors="raise")
    df["exit_ts"] = pd.to_datetime(df["exit_timestamp"], errors="raise")
    df["ret_dec"] = pd.to_numeric(df["risk_trade_return"], errors="raise")
    df["mae_dec"] = pd.to_numeric(df["mae_price_move"], errors="raise") * pd.to_numeric(df["risk_notional"], errors="raise")
    df["reason"] = df["reason"].astype(str)
    df["side_num"] = pd.to_numeric(df["side"], errors="raise").astype(int)
    df["risk_notional"] = pd.to_numeric(df["risk_notional"], errors="raise")
    df["risk_margin_fraction"] = pd.to_numeric(df["risk_margin_fraction"], errors="raise")
    df["risk_leverage"] = pd.to_numeric(df["risk_leverage"], errors="raise")
    return df


def _overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    return bool(a_start < b_end and b_start < a_end)


def _primary_gap_sleeve(primary: pd.DataFrame, sleeve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    intervals = list(zip(primary["entry_ts"], primary["exit_ts"]))
    kept: list[pd.Series] = []
    skipped: list[pd.Series] = []
    for _, row in sleeve.sort_values("entry_ts").iterrows():
        has_overlap = any(_overlap(row["entry_ts"], row["exit_ts"], start, end) for start, end in intervals)
        if has_overlap:
            skipped.append(row)
        else:
            kept.append(row)
    kept_df = pd.DataFrame(kept) if kept else sleeve.iloc[0:0].copy()
    skipped_df = pd.DataFrame(skipped) if skipped else sleeve.iloc[0:0].copy()
    combined = pd.concat([primary, kept_df], ignore_index=True).sort_values(["entry_ts", "model"]).reset_index(drop=True)
    return combined, kept_df.reset_index(drop=True), skipped_df.reset_index(drop=True)


def _metrics(trades: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    trades = trades.sort_values(["entry_ts", "model"]).reset_index(drop=True)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    rows: list[dict[str, Any]] = []
    for _, row in trades.iterrows():
        peak = max(peak, cash)
        adverse = cash * (1.0 + min(float(row["mae_dec"]), 0.0))
        mdd = min(mdd, adverse / max(peak, 1.0e-12) - 1.0)
        before = cash
        cash *= 1.0 + float(row["ret_dec"])
        wins += int(cash > before)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        out = row.to_dict()
        out["cash_after_combo"] = cash
        rows.append(out)
    n = len(trades)
    days = 1.0
    if n:
        days = max((trades["exit_ts"].max() - trades["entry_ts"].min()).total_seconds() / 86400.0, 1.0e-9)
    side = trades["side_num"] if n else pd.Series(dtype=int)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(n),
            "wr": float(wins / max(n, 1)),
            "trades_per_day": float(n / days),
            "long_entries": int((side > 0).sum()),
            "short_entries": int((side < 0).sum()),
            "main_trades": int(trades["model"].eq("omega3_main").sum()) if n else 0,
            "sleeve_trades": int(trades["model"].eq("omega4_4_sleeve").sum()) if n else 0,
            "exit_reasons": {str(k): int(v) for k, v in trades["reason"].value_counts().to_dict().items()} if n else {},
        },
        pd.DataFrame(rows),
    )


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _entry_parent_features() -> dict[str, pd.DataFrame]:
    device = _device()
    bundle = torch.load(OMEGA44_BUNDLE, map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    out: dict[str, pd.DataFrame] = {}
    for split, frame, oof in (("validation", frames["val_raw"], True), ("oos", frames["oos_raw"], False)):
        _, src, dec = risk_sidecar._predict_decisions(
            frame,
            oof=oof,
            models=models,
            base_cols=base_cols,
            quality_threshold=0.70,
            device=device,
        )
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        feat = pd.DataFrame(
            {
                "entry_signal_i": np.arange(len(frame), dtype=np.int64),
                "timestamp": pd.to_datetime(frame["timestamp"], errors="raise"),
                "parent_quality_score": pd.to_numeric(dec["quality_score"], errors="raise").to_numpy(dtype=np.float64),
                "parent_confidence": pd.to_numeric(dec["confidence"], errors="raise").to_numpy(dtype=np.float64),
                "parent_action": pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64),
                "parent_side": pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64),
                "router_expert": src[f"{prefix}_router_expert"].astype(str).to_numpy(),
                "router_confidence": pd.to_numeric(src[f"{prefix}_router_confidence"], errors="raise").to_numpy(dtype=np.float64),
                "router_margin": pd.to_numeric(src[f"{prefix}_router_margin"], errors="raise").to_numpy(dtype=np.float64),
                "dir_confidence": pd.to_numeric(src[f"{prefix}_dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
                "dir_trade_prob": pd.to_numeric(src[f"{prefix}_dir_trade_prob"], errors="raise").to_numpy(dtype=np.float64),
                "quality_for_action": pd.to_numeric(src[f"{prefix}_quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
                "entry_hour": pd.to_datetime(frame["timestamp"], errors="raise").dt.hour.to_numpy(dtype=np.int64),
            }
        )
        out[split] = feat
    return out


def _attach_parent_features(sleeve: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "entry_signal_i",
        "parent_quality_score",
        "parent_confidence",
        "router_expert",
        "router_confidence",
        "router_margin",
        "dir_confidence",
        "dir_trade_prob",
        "quality_for_action",
        "entry_hour",
    ]
    out = sleeve.merge(feat[cols], on="entry_signal_i", how="left", validate="many_to_one")
    if out[cols[1:]].isna().any(axis=None):
        raise RuntimeError("missing parent features after sleeve merge")
    return out


def _filter_sleeve(sleeve: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    mask = np.ones(len(sleeve), dtype=bool)
    mask &= pd.to_numeric(sleeve["risk_notional"], errors="raise").to_numpy(dtype=np.float64) <= float(cfg["risk_notional_cap"])
    mask &= pd.to_numeric(sleeve["parent_quality_score"], errors="raise").to_numpy(dtype=np.float64) >= float(cfg["quality_min"])
    mask &= pd.to_numeric(sleeve["parent_confidence"], errors="raise").to_numpy(dtype=np.float64) >= float(cfg["confidence_min"])
    mask &= pd.to_numeric(sleeve["dir_confidence"], errors="raise").to_numpy(dtype=np.float64) >= float(cfg["dir_confidence_min"])
    mask &= pd.to_numeric(sleeve["router_confidence"], errors="raise").to_numpy(dtype=np.float64) >= float(cfg["router_confidence_min"])
    side = str(cfg["side"])
    if side == "long":
        mask &= pd.to_numeric(sleeve["side_num"], errors="raise").to_numpy(dtype=np.int64) > 0
    elif side == "short":
        mask &= pd.to_numeric(sleeve["side_num"], errors="raise").to_numpy(dtype=np.int64) < 0
    elif side != "all":
        raise RuntimeError(f"unknown side filter: {side}")
    hour_group = str(cfg["hour_group"])
    hour = pd.to_numeric(sleeve["entry_hour"], errors="raise").to_numpy(dtype=np.int64)
    if hour_group == "asia":
        mask &= (hour >= 0) & (hour < 8)
    elif hour_group == "europe":
        mask &= (hour >= 8) & (hour < 16)
    elif hour_group == "us":
        mask &= (hour >= 16) & (hour < 24)
    elif hour_group != "all":
        raise RuntimeError(f"unknown hour group: {hour_group}")
    return sleeve.loc[mask].reset_index(drop=True)


def _candidate_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for risk_notional_cap, quality_min, confidence_min, dir_confidence_min, router_confidence_min, side, hour_group in itertools.product(
        [0.28, 0.30, 0.32, 0.34, 0.36, 0.40, 0.45, 0.50],
        [0.70, 0.72, 0.75, 0.78, 0.80],
        [0.00, 0.45, 0.50, 0.55],
        [0.00, 0.45, 0.50, 0.55],
        [0.00, 0.40, 0.50, 0.60],
        ["all", "long", "short"],
        ["all", "asia", "europe", "us"],
    ):
        rows.append(
            {
                "risk_notional_cap": risk_notional_cap,
                "quality_min": quality_min,
                "confidence_min": confidence_min,
                "dir_confidence_min": dir_confidence_min,
                "router_confidence_min": router_confidence_min,
                "side": side,
                "hour_group": hour_group,
            }
        )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parent_features = _entry_parent_features()
    primary = {split: _load_primary(split) for split in ("validation", "oos")}
    sleeve = {split: _attach_parent_features(_load_sleeve(split), parent_features[split]) for split in ("validation", "oos")}
    primary_metrics: dict[str, dict[str, Any]] = {}
    for split in ("validation", "oos"):
        primary_metrics[split], _ = _metrics(primary[split])

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    added_ledgers: dict[str, pd.DataFrame] = {}
    for idx, cfg in enumerate(_candidate_grid()):
        rec: dict[str, Any] = {"candidate_id": f"sleeve_{idx:05d}", **cfg}
        for split in ("validation", "oos"):
            filtered = _filter_sleeve(sleeve[split], cfg)
            combo, added, skipped = _primary_gap_sleeve(primary[split], filtered)
            metric, ledger = _metrics(combo)
            rec.update(
                {
                    f"{split}_pnl": metric["pnl"],
                    f"{split}_mdd": metric["mdd"],
                    f"{split}_trades": metric["trades"],
                    f"{split}_wr": metric["wr"],
                    f"{split}_sleeve_trades": metric["sleeve_trades"],
                    f"{split}_skipped_overlap": int(len(skipped)),
                    f"{split}_delta_pnl": metric["pnl"] - primary_metrics[split]["pnl"],
                    f"{split}_delta_mdd": metric["mdd"] - primary_metrics[split]["mdd"],
                    f"{split}_delta_wr_pp": (metric["wr"] - primary_metrics[split]["wr"]) * 100.0,
                }
            )
            if split == "validation":
                ledgers[rec["candidate_id"]] = ledger
                added_ledgers[rec["candidate_id"]] = added
        rows.append(rec)

    ranking = pd.DataFrame(rows)
    valid = ranking[
        (ranking["validation_sleeve_trades"] >= 1)
        & (ranking["validation_delta_pnl"] > 0.0)
        & (ranking["validation_delta_mdd"] >= -0.25)
    ].copy()
    if valid.empty:
        selected = ranking.sort_values(["validation_delta_pnl", "validation_delta_mdd"], ascending=[False, False]).iloc[0].to_dict()
        selection_status = "no_validation_positive_candidate"
    else:
        # OOS is not used for selection. Conservative tie-break prefers lower sleeve exposure.
        valid = valid.sort_values(
            [
                "validation_delta_pnl",
                "risk_notional_cap",
                "validation_sleeve_trades",
                "validation_delta_mdd",
            ],
            ascending=[False, True, True, False],
        )
        selected = valid.iloc[0].to_dict()
        selection_status = "validation_only_selected"

    selected_id = str(selected["candidate_id"])
    selected_cfg = {k: selected[k] for k in ["risk_notional_cap", "quality_min", "confidence_min", "dir_confidence_min", "router_confidence_min", "side", "hour_group"]}
    selected_outputs: dict[str, Any] = {}
    for split in ("validation", "oos"):
        filtered = _filter_sleeve(sleeve[split], selected_cfg)
        combo, added, skipped = _primary_gap_sleeve(primary[split], filtered)
        metric, ledger = _metrics(combo)
        ledger.to_csv(OUT_DIR / f"{split}_{selected_id}_combined_ledger.csv", index=False)
        added.to_csv(OUT_DIR / f"{split}_{selected_id}_added_sleeve_trades.csv", index=False)
        skipped.to_csv(OUT_DIR / f"{split}_{selected_id}_skipped_sleeve_trades.csv", index=False)
        selected_outputs[split] = {
            "combo": metric,
            "primary": primary_metrics[split],
            "delta_vs_primary": {
                "pnl": metric["pnl"] - primary_metrics[split]["pnl"],
                "mdd": metric["mdd"] - primary_metrics[split]["mdd"],
                "trades": metric["trades"] - primary_metrics[split]["trades"],
                "wr_pp": (metric["wr"] - primary_metrics[split]["wr"]) * 100.0,
            },
            "added_sleeve_trades": int(len(added)),
            "skipped_overlap_trades": int(len(skipped)),
        }

    ranking.sort_values(["validation_delta_pnl", "risk_notional_cap"], ascending=[False, True]).to_csv(OUT_DIR / "validation_only_sleeve_filter_ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "selection_status": selection_status,
        "selection_policy": "validation-only: require at least one gap sleeve trade, positive validation delta_pnl, and no validation MDD worsening beyond 0.25pp; OOS excluded from filter/sort/tie-break; conservative tie-break prefers lower risk_notional_cap.",
        "selected_candidate_id": selected_id,
        "selected_filter": selected_cfg,
        "selected_validation_row": selected,
        "primary_model_id": "omega3_aggressive_compensated_scale200_cap090_20260618",
        "sleeve_model_id": "omega4_4_topdown_reproducible_architecture_baseline_20260623",
        "results": selected_outputs,
        "artifacts": {
            "ranking": str(OUT_DIR / "validation_only_sleeve_filter_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
        "caveat": "Ledger-level cash sleeve replay. It preserves primary omega3 positions and only adds Omega4.4 trades whose full interval does not overlap primary positions. Runtime integration still needs live parity and same-bar arbitration tests.",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
