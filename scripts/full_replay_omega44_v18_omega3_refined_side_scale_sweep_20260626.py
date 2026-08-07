#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import full_replay_omega4_4_v18_short_aged_profit_overlays_20260625 as v18  # noqa: E402
import full_replay_omega44_v18_omega3_exposure_fine_sweep_20260626 as fine  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402


MODEL_ID = "omega44_v18_omega3_refined_side_scale_sweep_20260626"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class RefinedSpec:
    variant: str
    long_scale: float
    short_scale: float
    cap: float
    leverage: float = 2.0
    partial_fraction: float = 0.50
    cap_bars: int = 1152
    min_unreal: float = 0.035


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


def _tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _specs() -> list[RefinedSpec]:
    specs: list[RefinedSpec] = []
    for long_scale in (0.80, 0.85, 0.90, 0.95, 1.00):
        for short_scale in (1.25, 1.35, 1.45, 1.55, 1.65):
            for cap in (0.95, 1.00, 1.05, 1.10, 1.15, 1.20):
                specs.append(
                    RefinedSpec(
                        variant=f"side_l{_tag(long_scale)}_s{_tag(short_scale)}_cap{_tag(cap)}_shortpartial",
                        long_scale=long_scale,
                        short_scale=short_scale,
                        cap=cap,
                    )
                )
    return specs


def _risk_arrays(spec: RefinedSpec, dec: pd.DataFrame, base_margin: np.ndarray, base_leverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_notional = np.asarray(base_margin, dtype=np.float64) * np.asarray(base_leverage, dtype=np.float64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    scale = np.where(side < 0, float(spec.short_scale), np.where(side > 0, float(spec.long_scale), 1.0))
    notional = np.minimum(base_notional * scale, float(spec.cap))
    leverage = np.full_like(base_notional, float(spec.leverage), dtype=np.float64)
    margin = notional / max(float(spec.leverage), 1.0e-12)
    return margin, leverage


def _overlay_spec(spec: RefinedSpec) -> v18.OverlaySpec:
    return v18.OverlaySpec(
        f"{spec.variant}_shortpartial",
        "partial_deleverage",
        -1,
        int(spec.cap_bars),
        float(spec.min_unreal),
        partial_fraction=float(spec.partial_fraction),
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(v18.REPORT_PATH.read_text(encoding="utf-8"))
    device = parent._device("cuda")
    payload, extra = v18._prepare_payload(report, device)
    fee, slip = v18.omega._load_fee_slip()

    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] = {"variant": "baseline_v18", "long_scale": 1.0, "short_scale": 1.0, "cap": 0.0}
    baseline_spec = fine.ExposureSpec("baseline_v18", "sidecar")
    for split, (frame, base_x, dec, base_margin, base_leverage) in payload.items():
        metrics, _ledger = v18._replay_overlay(
            frame,
            base_x,
            dec,
            extra["loaded"],
            np.asarray(base_margin, dtype=np.float64),
            np.asarray(base_leverage, dtype=np.float64),
            fine._overlay_spec(baseline_spec),
            report=report,
            fee=fee,
            slip=slip,
            device=device,
        )
        for key, value in metrics.items():
            baseline_row[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
    rows.append(baseline_row)

    specs = _specs()
    for idx, spec in enumerate(specs, start=1):
        row: dict[str, Any] = {
            "variant": spec.variant,
            "long_scale": spec.long_scale,
            "short_scale": spec.short_scale,
            "cap": spec.cap,
            "leverage": spec.leverage,
            "partial_fraction": spec.partial_fraction,
            "cap_bars": spec.cap_bars,
            "min_unreal": spec.min_unreal,
        }
        for split, (frame, base_x, dec, base_margin, base_leverage) in payload.items():
            margin, leverage = _risk_arrays(spec, dec, base_margin, base_leverage)
            metrics, _ledger = v18._replay_overlay(
                frame,
                base_x,
                dec,
                extra["loaded"],
                margin,
                leverage,
                _overlay_spec(spec),
                report=report,
                fee=fee,
                slip=slip,
                device=device,
            )
            for key, value in metrics.items():
                row[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
        rows.append(row)
        print(
            json.dumps(
                {
                    "idx": idx,
                    "total": len(specs),
                    "variant": spec.variant,
                    "validation_pnl": row["validation_pnl"],
                    "validation_mdd": row["validation_mdd"],
                    "oos_pnl": row["oos_pnl"],
                    "oos_mdd": row["oos_mdd"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    df = pd.DataFrame(rows)
    base = df.loc[df["variant"].eq("baseline_v18")].iloc[0]
    for split in ("validation", "oos"):
        df[f"{split}_delta_vs_v18_pnl"] = df[f"{split}_pnl"] - float(base[f"{split}_pnl"])
        df[f"{split}_delta_vs_v18_mdd"] = df[f"{split}_mdd"] - float(base[f"{split}_mdd"])
        df[f"{split}_delta_vs_v18_log_risk"] = df[f"{split}_log_risk_utility"] - float(base[f"{split}_log_risk_utility"])
    df["score_strict_like"] = df["validation_pnl"] + 2.0 * df["validation_delta_vs_v18_mdd"] + 0.5 * df["oos_delta_vs_v18_pnl"] + 2.0 * df["oos_delta_vs_v18_mdd"]
    df["score_mdd15"] = df["validation_pnl"] - 5.0 * np.maximum(0.0, -15.0 - df["validation_mdd"])
    df["score_mdd16"] = df["validation_pnl"] - 4.0 * np.maximum(0.0, -16.0 - df["validation_mdd"])
    df["score_mdd18"] = df["validation_pnl"] - 3.0 * np.maximum(0.0, -18.0 - df["validation_mdd"])
    df = df.sort_values(["score_mdd16", "validation_pnl"], ascending=False).reset_index(drop=True)
    grid_path = OUT_DIR / "refined_side_scale_grid.csv"
    df.to_csv(grid_path, index=False)

    views = {
        "strict_beats_v18": df[
            (df["validation_pnl"] > base["validation_pnl"])
            & (df["validation_mdd"] >= base["validation_mdd"])
            & (df["oos_pnl"] > base["oos_pnl"])
            & (df["oos_mdd"] >= base["oos_mdd"])
        ].copy(),
        "validation_mdd15": df[df["validation_mdd"] >= -15.0].sort_values(["validation_pnl"], ascending=False).copy(),
        "validation_mdd16": df[df["validation_mdd"] >= -16.0].sort_values(["validation_pnl"], ascending=False).copy(),
        "validation_mdd18": df[df["validation_mdd"] >= -18.0].sort_values(["validation_pnl"], ascending=False).copy(),
        "oos_pnl_champions": df.sort_values(["oos_pnl"], ascending=False).copy(),
    }
    for name, view in views.items():
        view.to_csv(OUT_DIR / f"{name}.csv", index=False)

    selected = {
        name: (view.iloc[0].to_dict() if len(view) else None)
        for name, view in views.items()
    }
    report_out = {
        "model_id": MODEL_ID,
        "source_model": "omega4_4_v18_baseline_20260624",
        "borrowed_from": "omega3_aggressive_compensated_scale200_cap090_20260618",
        "fixed_overlay": {
            "mode": "short_aged_profit_partial_deleverage",
            "cap_bars": 1152,
            "min_unrealized_price_move": 0.035,
            "partial_fraction": 0.50,
        },
        "baseline": base.to_dict(),
        "selected": selected,
        "counts": {name: int(len(view)) for name, view in views.items()},
        "top20": df.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(grid_path),
        },
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report_out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "selected": {k: (v or {}).get("variant") for k, v in selected.items()},
                "counts": report_out["counts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
