#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _load_bundle(path: Path, *, device: torch.device) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    models = payload["models"]
    base_cols = list(payload["base_cols"])
    return models, base_cols


def _predict_decisions(
    frame: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    base_cols: list[str],
    *,
    q: float,
    oof: bool,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = parent._base_input(frame, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    src = parent._prediction_output(frame, direction, quality, threshold=float(q), prefix=prefix)
    dec = parent._to_decisions(src, oof=oof)
    return x, dec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--direction-label-dir", type=Path, default=omega4.LABEL_DIR)
    ap.add_argument("--quality-mode", choices=["same_as_direction", "hard_rule", "quality_label_action", "quality_label_hard_rule"], default="same_as_direction")
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-thresholds", default="0.45,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    device = parent._device(str(args.device))
    models, base_cols = _load_bundle(Path(args.bundle), device=device)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode=str(args.quality_mode),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    x_val, val_dec = _predict_decisions(val_raw, models, base_cols, q=float(args.quality_threshold), oof=True, device=device)
    x_oos, oos_dec = _predict_decisions(oos_raw, models, base_cols, q=float(args.quality_threshold), oof=False, device=device)
    no_exit = {
        "validation": omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        "oos": omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    }
    loaded = parent._load_payloads(models, device=device)
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {"no_exit_head": no_exit}
    for thr in [float(x.strip()) for x in str(args.exit_thresholds).split(",") if x.strip()]:
        val = parent._metrics_with_shared_exit(val_raw, x_val, val_dec, loaded, threshold=thr, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
        oos = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec, loaded, threshold=thr, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
        key = f"exit_thr_{thr:.2f}".replace(".", "p")
        results[key] = {"validation": val, "oos": oos}
        rows.append(
            {
                "variant": key,
                "exit_threshold": thr,
                "validation_pnl": float(val["pnl"]),
                "validation_mdd": float(val["mdd"]),
                "validation_wr": float(val["wr"]),
                "validation_trades": int(val["trades"]),
                "oos_pnl": float(oos["pnl"]),
                "oos_mdd": float(oos["mdd"]),
                "oos_wr": float(oos["wr"]),
                "oos_trades": int(oos["trades"]),
            }
        )
    rows.sort(key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True)
    report = {
        "bundle": str(args.bundle),
        "quality_threshold": float(args.quality_threshold),
        "results": results,
        "ranking_by_oos_pnl": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out_json), "top": rows[:5], "no_exit_head": no_exit}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
