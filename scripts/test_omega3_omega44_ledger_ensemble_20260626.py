#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega3_omega44_ledger_ensemble_20260626"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

OMEGA3_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_current_baseline_growth_20260606"
OMEGA44_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v18_topdown_best_parent_exit075_live_exposure_dynamic_leverage_valonly_logrisk_tail050_minavg075_20260624"
)
OMEGA44_PARTIAL_DIR = ROOT / "tmp/causal_regen_20260516/omega4_4_v18_short_aged_profit_overlay_full_replay_20260625"


@dataclass(frozen=True)
class ModelLedger:
    name: str
    validation_path: Path
    oos_path: Path
    return_col: str
    return_scale: float
    entry_time_col: str
    exit_time_col: str
    reason_col: str


LEDGERS = {
    "omega3": ModelLedger(
        name="omega3",
        validation_path=OMEGA3_DIR / "omega1_2_1_aggressive_compensated_scale200_cap090_validation_trade_ledger_20260606.csv",
        oos_path=OMEGA3_DIR / "omega1_2_1_aggressive_compensated_scale200_cap090_oos_trade_ledger_20260606.csv",
        return_col="net_trade_return_pct",
        return_scale=0.01,
        entry_time_col="entry_time",
        exit_time_col="exit_time",
        reason_col="exit_reason",
    ),
    "omega44": ModelLedger(
        name="omega44",
        validation_path=OMEGA44_DIR / "validation_selected_risk_replayed_trade_ledger.csv",
        oos_path=OMEGA44_DIR / "oos_selected_risk_replayed_trade_ledger.csv",
        return_col="trade_return",
        return_scale=1.0,
        entry_time_col="entry_timestamp",
        exit_time_col="exit_timestamp",
        reason_col="reason",
    ),
    "omega44_partial": ModelLedger(
        name="omega44_partial",
        validation_path=OMEGA44_PARTIAL_DIR / "validation_short_partial_cap1152_u0.035_p0.50_ledger.csv",
        oos_path=OMEGA44_PARTIAL_DIR / "oos_short_partial_cap1152_u0.035_p0.50_ledger.csv",
        return_col="risk_trade_return",
        return_scale=1.0,
        entry_time_col="entry_timestamp",
        exit_time_col="exit_timestamp",
        reason_col="reason",
    ),
}


def _read_ledger(spec: ModelLedger, split: str) -> pd.DataFrame:
    path = spec.validation_path if split == "validation" else spec.oos_path
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "model": spec.name,
            "entry_i": pd.to_numeric(df["entry_i"], errors="raise").astype(int),
            "exit_i": pd.to_numeric(df["exit_i"], errors="raise").astype(int),
            "entry_time": pd.to_datetime(df[spec.entry_time_col], errors="raise"),
            "exit_time": pd.to_datetime(df[spec.exit_time_col], errors="raise"),
            "side": df["side"].map(lambda x: 1 if str(x).upper() == "LONG" else (-1 if str(x).upper() == "SHORT" else int(x))).astype(int),
            "ret": pd.to_numeric(df[spec.return_col], errors="raise").astype(float) * float(spec.return_scale),
            "reason": df[spec.reason_col].astype(str),
            "notional": pd.to_numeric(df.get("notional", 0.0), errors="coerce").fillna(0.0).astype(float),
            "leverage": pd.to_numeric(df.get("leverage", 0.0), errors="coerce").fillna(0.0).astype(float),
        }
    )
    out["win"] = out["ret"] > 0.0
    return out.sort_values(["entry_i", "exit_i", "model"]).reset_index(drop=True)


def _model_active_at(trades: pd.DataFrame, entry_i: int) -> int:
    active = trades[(trades["entry_i"] <= entry_i) & (entry_i < trades["exit_i"])]
    if active.empty:
        return 0
    # If multiple historical ledger rows overlap due reproduction artifacts, use the most recent entry.
    row = active.sort_values(["entry_i", "exit_i"]).iloc[-1]
    return int(row["side"])


def _metrics(trades: pd.DataFrame, *, variant: str, split: str) -> dict:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    rows = []
    for n, row in enumerate(trades.sort_values(["entry_i", "exit_i", "model"]).itertuples(index=False), start=1):
        cash *= 1.0 + float(row.ret)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        rec = row._asdict()
        rec["ensemble_trade_id"] = n
        rec["cash_after"] = cash
        rows.append(rec)
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        ledger = ledger[
            [
                "ensemble_trade_id",
                "model",
                "entry_i",
                "exit_i",
                "entry_time",
                "exit_time",
                "side",
                "ret",
                "reason",
                "notional",
                "leverage",
                "win",
                "cash_after",
            ]
        ]
    counts = trades["model"].value_counts().to_dict() if not trades.empty else {}
    return {
        "variant": variant,
        "split": split,
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(trades)),
        "wr": float((trades["ret"] > 0.0).mean()) if len(trades) else 0.0,
        "long_entries": int((trades["side"] > 0).sum()) if len(trades) else 0,
        "short_entries": int((trades["side"] < 0).sum()) if len(trades) else 0,
        "omega3_trades": int(counts.get("omega3", 0)),
        "omega44_trades": int(counts.get("omega44", 0)),
        "omega44_partial_trades": int(counts.get("omega44_partial", 0)),
        "ledger": ledger,
    }


def _single(name: str, ledgers: dict[str, pd.DataFrame], *, side: int = 0) -> pd.DataFrame:
    df = ledgers[name].copy()
    if side:
        df = df[df["side"] == side].copy()
    return df


def _priority_merge(primary: pd.DataFrame, secondary: pd.DataFrame, *, secondary_when_idle_only: bool = True) -> pd.DataFrame:
    all_rows = pd.concat([primary.assign(priority=0), secondary.assign(priority=1)], ignore_index=True)
    all_rows = all_rows.sort_values(["entry_i", "priority", "exit_i", "model"]).reset_index(drop=True)
    accepted = []
    busy_until = -1
    for row in all_rows.itertuples(index=False):
        if int(row.entry_i) < int(busy_until):
            continue
        if secondary_when_idle_only and int(row.priority) == 1 and int(row.entry_i) < int(busy_until):
            continue
        accepted.append(row._asdict())
        busy_until = int(row.exit_i)
    if not accepted:
        return all_rows.iloc[0:0].drop(columns=["priority"])
    return pd.DataFrame(accepted).drop(columns=["priority"]).sort_values(["entry_i", "exit_i", "model"]).reset_index(drop=True)


def _first_available(candidates: Iterable[pd.DataFrame]) -> pd.DataFrame:
    all_rows = pd.concat([df for df in candidates if not df.empty], ignore_index=True)
    if all_rows.empty:
        return all_rows
    all_rows = all_rows.sort_values(["entry_i", "exit_i", "model"]).reset_index(drop=True)
    accepted = []
    busy_until = -1
    for row in all_rows.itertuples(index=False):
        if int(row.entry_i) < int(busy_until):
            continue
        accepted.append(row._asdict())
        busy_until = int(row.exit_i)
    return pd.DataFrame(accepted).sort_values(["entry_i", "exit_i", "model"]).reset_index(drop=True)


def _omega3_v18_state_gate(omega3: pd.DataFrame, v18: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    keep = []
    for row in omega3.itertuples(index=False):
        active_side = _model_active_at(v18, int(row.entry_i))
        side = int(row.side)
        if mode == "allow_if_no_opposite":
            ok = active_side in (0, side)
        elif mode == "require_same_active":
            ok = active_side == side
        elif mode == "require_v18_cash":
            ok = active_side == 0
        elif mode == "veto_if_same_active":
            ok = active_side != side
        else:
            raise ValueError(mode)
        if ok:
            keep.append(row._asdict())
    return pd.DataFrame(keep, columns=omega3.columns).reset_index(drop=True) if keep else omega3.iloc[0:0].copy()


def _build_variants(ledgers: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    o3 = ledgers["omega3"]
    o44 = ledgers["omega44"]
    o44p = ledgers["omega44_partial"]
    variants: dict[str, pd.DataFrame] = {}
    for name in ("omega3", "omega44", "omega44_partial"):
        variants[name] = _single(name, ledgers)
        variants[f"{name}_long_only"] = _single(name, ledgers, side=1)
        variants[f"{name}_short_only"] = _single(name, ledgers, side=-1)
    for sec_name, sec in (("omega44", o44), ("omega44p", o44p)):
        variants[f"omega3_primary_{sec_name}_cash_sleeve"] = _priority_merge(o3, sec)
        variants[f"{sec_name}_primary_omega3_cash_sleeve"] = _priority_merge(sec, o3)
        variants[f"first_available_omega3_{sec_name}"] = _first_available([o3, sec])
        variants[f"omega3_short_{sec_name}_long"] = _first_available([o3[o3["side"] < 0], sec[sec["side"] > 0]])
        variants[f"omega3_long_{sec_name}_short"] = _first_available([o3[o3["side"] > 0], sec[sec["side"] < 0]])
        variants[f"omega3_plus_{sec_name}_short_cash"] = _priority_merge(o3, sec[sec["side"] < 0])
        variants[f"omega3_plus_{sec_name}_long_cash"] = _priority_merge(o3, sec[sec["side"] > 0])
        for mode in ("allow_if_no_opposite", "require_same_active", "require_v18_cash", "veto_if_same_active"):
            gated = _omega3_v18_state_gate(o3, sec, mode=mode)
            variants[f"omega3_gate_{sec_name}_{mode}"] = gated
            variants[f"omega3_gate_{sec_name}_{mode}_plus_{sec_name}_cash"] = _priority_merge(gated, sec)
    return variants


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    ledgers_by_split: dict[str, dict[str, pd.DataFrame]] = {}
    for split in ("validation", "oos"):
        ledgers_by_split[split] = {name: _read_ledger(spec, split) for name, spec in LEDGERS.items()}
    variant_ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for split, ledgers in ledgers_by_split.items():
        for variant, trades in _build_variants(ledgers).items():
            m = _metrics(trades, variant=variant, split=split)
            ledger = m.pop("ledger")
            variant_ledgers[(split, variant)] = ledger
            rows.append(m)
    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(index="variant", columns="split")
    flat = pd.DataFrame(index=wide.index)
    for col in ("pnl", "mdd", "trades", "wr", "long_entries", "short_entries", "omega3_trades", "omega44_trades", "omega44_partial_trades"):
        for split in ("validation", "oos"):
            flat[f"{split}_{col}"] = wide[col][split]
    base_o3 = flat.loc["omega3"]
    base_o44 = flat.loc["omega44"]
    for split in ("validation", "oos"):
        flat[f"{split}_delta_vs_omega3_pnl"] = flat[f"{split}_pnl"] - float(base_o3[f"{split}_pnl"])
        flat[f"{split}_delta_vs_omega3_mdd"] = flat[f"{split}_mdd"] - float(base_o3[f"{split}_mdd"])
        flat[f"{split}_delta_vs_omega44_pnl"] = flat[f"{split}_pnl"] - float(base_o44[f"{split}_pnl"])
        flat[f"{split}_delta_vs_omega44_mdd"] = flat[f"{split}_mdd"] - float(base_o44[f"{split}_mdd"])
    flat["validation_score"] = (
        flat["validation_pnl"]
        + 3.0 * np.minimum(0.0, flat["validation_delta_vs_omega3_mdd"])
        - 0.10 * np.maximum(0.0, flat["validation_trades"] - base_o3["validation_trades"])
    )
    flat = flat.reset_index().sort_values(["validation_score", "validation_pnl"], ascending=False)
    flat.to_csv(OUT_DIR / "ledger_ensemble_grid.csv", index=False)

    selected = flat.iloc[0].to_dict()
    selected_variant = str(selected["variant"])
    for split in ("validation", "oos"):
        variant_ledgers[(split, selected_variant)].to_csv(OUT_DIR / f"{split}_{selected_variant}_ledger.csv", index=False)

    strict = flat[
        (flat["validation_pnl"] > base_o3["validation_pnl"])
        & (flat["validation_mdd"] >= base_o3["validation_mdd"])
        & (flat["oos_pnl"] > base_o3["oos_pnl"])
        & (flat["oos_mdd"] >= base_o3["oos_mdd"])
    ].copy()
    strict.to_csv(OUT_DIR / "strict_beats_omega3_grid.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "method": "ledger_level_non_overlapping_ensemble_scan",
        "caveat": "Fast triage only. MDD is close-to-close ledger MDD, not intrabar path MDD. Winning candidates require full runtime replay.",
        "baselines": {
            "omega3": flat[flat["variant"].eq("omega3")].iloc[0].to_dict(),
            "omega44": flat[flat["variant"].eq("omega44")].iloc[0].to_dict(),
            "omega44_partial": flat[flat["variant"].eq("omega44_partial")].iloc[0].to_dict(),
        },
        "selected_by_validation_score": selected,
        "strict_beats_omega3_count": int(len(strict)),
        "top10": flat.head(10).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "ledger_ensemble_grid.csv"),
            "strict": str(OUT_DIR / "strict_beats_omega3_grid.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected_variant, "strict_beats_omega3_count": int(len(strict))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
