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
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402


MODEL_ID = "omega44_v18_omega3_exposure_fine_sweep_20260626"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"


@dataclass(frozen=True)
class ExposureSpec:
    variant: str
    mode: str
    scale: float = 1.0
    cap: float = 0.0
    fixed_notional: float = 0.0
    long_scale: float = 1.0
    short_scale: float = 1.0
    leverage: float = 2.0
    short_partial: bool = False


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


def _specs() -> list[ExposureSpec]:
    specs = [ExposureSpec("baseline_v18", "sidecar")]
    for scale in (1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40):
        for cap in (0.90, 0.95, 1.00, 1.05):
            for partial in (False, True):
                specs.append(
                    ExposureSpec(
                        f"scaled_s{_tag(scale)}_cap{_tag(cap)}{'_shortpartial' if partial else ''}",
                        "scaled",
                        scale=scale,
                        cap=cap,
                        short_partial=partial,
                    )
                )
    for fixed in (0.82, 0.86, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20):
        for partial in (False, True):
            specs.append(
                ExposureSpec(
                    f"fixed{_tag(fixed)}{'_shortpartial' if partial else ''}",
                    "fixed",
                    fixed_notional=fixed,
                    short_partial=partial,
                )
            )
    for long_scale in (0.90, 1.00, 1.10):
        for short_scale in (1.15, 1.35, 1.55, 1.80):
            for cap in (1.00, 1.10, 1.20, 1.35):
                for partial in (False, True):
                    specs.append(
                        ExposureSpec(
                            f"side_l{_tag(long_scale)}_s{_tag(short_scale)}_cap{_tag(cap)}{'_shortpartial' if partial else ''}",
                            "side_scaled",
                            long_scale=long_scale,
                            short_scale=short_scale,
                            cap=cap,
                            short_partial=partial,
                        )
                    )
    return specs


def _risk_arrays(spec: ExposureSpec, dec: pd.DataFrame, base_margin: np.ndarray, base_leverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_notional = np.asarray(base_margin, dtype=np.float64) * np.asarray(base_leverage, dtype=np.float64)
    if spec.mode == "sidecar":
        return np.asarray(base_margin, dtype=np.float64).copy(), np.asarray(base_leverage, dtype=np.float64).copy()
    if spec.mode == "scaled":
        notional = base_notional * float(spec.scale)
        if spec.cap > 0.0:
            notional = np.minimum(notional, float(spec.cap))
    elif spec.mode == "fixed":
        notional = np.full_like(base_notional, float(spec.fixed_notional), dtype=np.float64)
    elif spec.mode == "side_scaled":
        side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
        scale = np.where(side < 0, float(spec.short_scale), np.where(side > 0, float(spec.long_scale), 1.0))
        notional = base_notional * scale
        if spec.cap > 0.0:
            notional = np.minimum(notional, float(spec.cap))
    else:
        raise ValueError(spec.mode)
    leverage = np.full_like(base_notional, float(spec.leverage), dtype=np.float64)
    margin = notional / max(float(spec.leverage), 1.0e-12)
    return margin, leverage


def _overlay_spec(spec: ExposureSpec) -> v18.OverlaySpec:
    if spec.short_partial:
        return v18.OverlaySpec(
            f"{spec.variant}_shortpartial",
            "partial_deleverage",
            -1,
            1152,
            0.035,
            partial_fraction=0.50,
        )
    return v18.OverlaySpec(spec.variant, "none", -1, 0, 0.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(v18.REPORT_PATH.read_text(encoding="utf-8"))
    device = parent._device("cuda")
    payload, extra = v18._prepare_payload(report, device)
    fee, slip = v18.omega._load_fee_slip()
    rows: list[dict[str, Any]] = []
    specs = _specs()
    for idx, spec in enumerate(specs, start=1):
        row: dict[str, Any] = {
            "variant": spec.variant,
            "mode": spec.mode,
            "scale": spec.scale,
            "cap": spec.cap,
            "fixed_notional": spec.fixed_notional,
            "long_scale": spec.long_scale,
            "short_scale": spec.short_scale,
            "leverage": spec.leverage,
            "short_partial": spec.short_partial,
        }
        for split, (frame, base_x, dec, base_margin, base_leverage) in payload.items():
            margin, leverage = _risk_arrays(spec, dec, base_margin, base_leverage)
            metrics, ledger = v18._replay_overlay(
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
                if key == "exit_reasons":
                    row[f"{split}_exit_reasons"] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    row[f"{split}_{key}"] = value
            if spec.variant == "baseline_v18" or idx % 15 == 0:
                ledger.to_csv(LEDGER_DIR / f"{split}_{spec.variant}_ledger.csv", index=False)
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
    df["validation_score_mdd16"] = df["validation_pnl"] - 4.0 * np.maximum(0.0, -16.0 - df["validation_mdd"])
    df["validation_score_mdd18"] = df["validation_pnl"] - 3.0 * np.maximum(0.0, -18.0 - df["validation_mdd"])
    df["validation_score_growth"] = df["validation_pnl"] - 1.5 * np.maximum(0.0, -22.0 - df["validation_mdd"])
    df = df.sort_values(["validation_score_mdd16", "validation_pnl"], ascending=False).reset_index(drop=True)
    df.to_csv(OUT_DIR / "fine_exposure_grid.csv", index=False)
    views = {
        "strict_beats_v18": df[
            (df["validation_pnl"] > base["validation_pnl"])
            & (df["validation_mdd"] >= base["validation_mdd"])
            & (df["oos_pnl"] > base["oos_pnl"])
            & (df["oos_mdd"] >= base["oos_mdd"])
        ].copy(),
        "validation_mdd16": df[df["validation_mdd"] >= -16.0].sort_values(["validation_pnl"], ascending=False).copy(),
        "validation_mdd18": df[df["validation_mdd"] >= -18.0].sort_values(["validation_pnl"], ascending=False).copy(),
        "growth_mdd22": df[df["validation_mdd"] >= -22.0].sort_values(["validation_pnl"], ascending=False).copy(),
        "pnl_champions": df.sort_values(["validation_pnl"], ascending=False).copy(),
    }
    for name, view in views.items():
        view.to_csv(OUT_DIR / f"{name}.csv", index=False)
    selected = {
        "mdd16": views["validation_mdd16"].iloc[0].to_dict() if len(views["validation_mdd16"]) else None,
        "mdd18": views["validation_mdd18"].iloc[0].to_dict() if len(views["validation_mdd18"]) else None,
        "growth_mdd22": views["growth_mdd22"].iloc[0].to_dict() if len(views["growth_mdd22"]) else None,
        "pnl_champion": views["pnl_champions"].iloc[0].to_dict() if len(views["pnl_champions"]) else None,
        "strict": views["strict_beats_v18"].iloc[0].to_dict() if len(views["strict_beats_v18"]) else None,
    }
    report_out = {
        "model_id": MODEL_ID,
        "source_model": "omega4_4_v18_baseline_20260624",
        "borrowed_from": "omega3_aggressive_compensated_scale200_cap090_20260618",
        "borrowed_pipeline_parts": [
            "aggressive exposure scaling",
            "fixed 2x leverage accounting",
            "short-heavy side-specific exposure probes",
            "short aged-profit partial de-risk overlay probes",
        ],
        "kept_from_omega44": [
            "parent direction/quality/exit heads",
            "risk sidecar score ordering before exposure remap",
            "ATR safety TP/SL price barriers",
            "exit head actual in-position exit replay",
        ],
        "baseline": base.to_dict(),
        "selected": selected,
        "counts": {name: int(len(view)) for name, view in views.items()},
        "top20": df.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "fine_exposure_grid.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report_out, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": {k: (v or {}).get("variant") for k, v in selected.items()}, "counts": report_out["counts"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
