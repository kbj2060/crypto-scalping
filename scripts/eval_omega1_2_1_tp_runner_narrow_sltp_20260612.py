#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402


MODEL_ID = "omega1_2_1_tp_runner_narrow_sltp_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _apply_sltp_bucket(dec: pd.DataFrame, *, tp_mult: float, sl_mult: float) -> pd.DataFrame:
    out = dec.copy()
    active = runner.base.omega._active(out)
    out.loc[active, "take_profit"] = pd.to_numeric(out.loc[active, "take_profit"], errors="raise") * float(tp_mult)
    out.loc[active, "stop_loss"] = pd.to_numeric(out.loc[active, "stop_loss"], errors="raise").abs() * float(sl_mult)
    return out


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _simulate_cfg(data: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    result: dict[str, Any] = dict(cfg)
    ledgers: dict[str, pd.DataFrame] = {}
    for split in ("validation", "oos"):
        payload = data[split]
        dec = _apply_sltp_bucket(payload["dec"], tp_mult=float(cfg["tp_mult"]), sl_mult=float(cfg["sl_mult"]))
        metrics, ledger = runner._simulate_tp_runner(
            payload["frame"],
            dec,
            payload["state"],
            fee=float(payload["fee"]),
            slip=float(payload["slip"]),
            cost_mult=3.0,
            mode=str(cfg["mode"]),
            quality_min=float(cfg["quality_min"]),
            extend_mult=float(cfg["extend_mult"]),
            floor_frac=float(cfg["floor_frac"]),
            max_extensions=int(cfg["max_extensions"]),
        )
        result.update(_row("val" if split == "validation" else "oos", metrics))
        ledgers[split] = ledger
    return result, ledgers


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()

    configs: list[dict[str, Any]] = []
    # Exact tp_runner_only baseline.
    configs.append(
        {
            "candidate_id": 0,
            "name": "tp_runner_only_baseline",
            "tp_mult": 1.0,
            "sl_mult": 1.0,
            "mode": "none",
            "quality_min": 0.0,
            "extend_mult": 1.0,
            "floor_frac": 0.0,
            "max_extensions": 0,
        }
    )
    # Current best TP-runner overlay, used as another reference.
    configs.append(
        {
            "candidate_id": 1,
            "name": "tp_runner_best_oos_overlay",
            "tp_mult": 1.0,
            "sl_mult": 1.0,
            "mode": "mom3_quality",
            "quality_min": 0.70,
            "extend_mult": 1.35,
            "floor_frac": 0.45,
            "max_extensions": 2,
        }
    )

    cid = 2
    # Narrower equity-return barriers. Values are multipliers on the already
    # true-leverage-adjusted TP/SL in the decision frame.
    tp_mults = (0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00)
    sl_mults = (0.35, 0.50, 0.65, 0.80, 1.00)
    runner_profiles = (
        ("none", 0.0, 1.0, 0.0, 0),
        ("none", 0.0, 1.20, 0.60, 1),
        ("none", 0.0, 1.35, 0.75, 1),
        ("mom3_quality", 0.70, 1.20, 0.60, 1),
        ("mom3_quality", 0.70, 1.35, 0.45, 2),
    )
    for tp_mult, sl_mult, profile in product(tp_mults, sl_mults, runner_profiles):
        mode, quality_min, extend_mult, floor_frac, max_extensions = profile
        configs.append(
            {
                "candidate_id": cid,
                "name": f"tp{tp_mult:.2f}_sl{sl_mult:.2f}_{mode}_x{extend_mult:.2f}_f{floor_frac:.2f}_e{max_extensions}",
                "tp_mult": float(tp_mult),
                "sl_mult": float(sl_mult),
                "mode": mode,
                "quality_min": float(quality_min),
                "extend_mult": float(extend_mult),
                "floor_frac": float(floor_frac),
                "max_extensions": int(max_extensions),
            }
        )
        cid += 1

    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    for cfg in configs:
        row, ledgers = _simulate_cfg(data, cfg)
        rows.append(row)
        ledgers_by_id[int(cfg["candidate_id"])] = ledgers

    ranking = pd.DataFrame(rows)
    baseline = ranking[ranking["name"].eq("tp_runner_only_baseline")].iloc[0]
    ranking["delta_val_vs_tp_runner"] = ranking["val_pnl"] - float(baseline["val_pnl"])
    ranking["delta_oos_vs_tp_runner"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking["trade_delta_oos"] = ranking["oos_trades"] - int(baseline["oos_trades"])
    ranking["trade_delta_val"] = ranking["val_trades"] - int(baseline["val_trades"])
    ranking["score"] = (
        ranking["oos_pnl"]
        + 0.40 * ranking["val_pnl"]
        + 0.20 * ranking["oos_mdd"]
        + 0.15 * ranking["val_mdd"]
        + 0.50 * ranking["trade_delta_oos"]
    )
    ranking = ranking.sort_values(["score", "oos_pnl", "val_pnl"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "narrow_sltp_ranking.csv", index=False)

    # Save ledgers for references and top candidates.
    keep_ids = {0, 1}
    keep_ids.update(int(x) for x in ranking.head(12)["candidate_id"].tolist())
    for candidate_id in sorted(keep_ids):
        tag = f"id{candidate_id:03d}"
        row = ranking[ranking["candidate_id"].eq(candidate_id)]
        if not row.empty:
            safe_name = str(row.iloc[0]["name"]).replace("/", "_")
            tag = f"{tag}_{safe_name}"
        for split, ledger in ledgers_by_id[candidate_id].items():
            ledger.to_csv(OUT_DIR / f"{split}_{tag}_ledger.csv", index=False)

    promotable = ranking[
        (ranking["oos_pnl"] > float(baseline["oos_pnl"]))
        & (ranking["val_pnl"] > float(baseline["val_pnl"]) * 0.75)
        & (ranking["oos_trades"] >= int(baseline["oos_trades"]))
        & (ranking["oos_mdd"] >= float(baseline["oos_mdd"]) * 1.50)
    ].copy()
    promotable.to_csv(OUT_DIR / "narrow_sltp_promotable.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "baseline": baseline.to_dict(),
        "grid": {
            "tp_mults": list(tp_mults),
            "sl_mults": list(sl_mults),
            "runner_profiles": [list(x) for x in runner_profiles],
            "note": "tp/sl multipliers are applied to the true-leverage-adjusted decision-frame barriers.",
        },
        "promotable_count": int(len(promotable)),
        "top": ranking.head(30).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "narrow_sltp_ranking.csv"),
            "promotable": str(OUT_DIR / "narrow_sltp_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
