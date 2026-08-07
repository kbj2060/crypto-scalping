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

from features.engineering import FeatureEngineer  # noqa: E402
from features.schema import STATE_DIRECTION_ALPHA  # noqa: E402
from scripts.analyze_all_feature_usage_20260528 import (  # noqa: E402
    _add_targets,
    _load_frames,
    _psi,
    _spearman,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "directional_alpha_feature_audit_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
FEATURE_OUT = OUT_DIR / "directional_alpha_feature_scores.csv"
REDUNDANCY_OUT = OUT_DIR / "directional_alpha_redundancy_abs090.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"
REPORT_OUT = ROOT / "docs/audits/directional_alpha_feature_audit_20260528.md"


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    return FeatureEngineer(keep_only_active=False)._create_directional_alpha_features(out)


def _redundancy(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sample = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = sample.corr(method="spearman").abs().fillna(0.0)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = float(corr.loc[a, b])
            if v >= 0.90:
                rows.append({"feature_a": a, "feature_b": b, "abs_spearman": v})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)


def _verdict(row: pd.Series) -> tuple[str, str]:
    f = str(row["feature"])
    psi = float(row["psi_oos"]) if pd.notna(row["psi_oos"]) else np.inf
    ret = float(row["max_abs_ret_ic_oos"]) if pd.notna(row["max_abs_ret_ic_oos"]) else 0.0
    vol = float(row["max_abs_vol_ic_oos"]) if pd.notna(row["max_abs_vol_ic_oos"]) else 0.0
    if psi >= 0.50:
        return "MONITOR_OR_NORMALIZE", f"high OOS drift PSI={psi:.3f}; normalize/re-audit before active direct input"
    if ret >= 0.07 and ret >= 0.75 * max(vol, 1e-12):
        return "KEEP_ENTRY_CONTEXT", "direction-context candidate; test in entry layer only"
    if vol >= 0.12 and vol > ret:
        return "KEEP_RISK_CONTEXT", "stronger risk/volatility utility than direction"
    if max(ret, vol) < 0.03:
        return "LOW_SIGNAL_SECONDARY", "low standalone OOS tendency"
    return "SECONDARY_CONTEXT", "moderate context; require ablation before promotion"


def _fmt(x: Any, n: int = 3) -> str:
    try:
        if pd.isna(x):
            return "nan"
        return f"{float(x):.{n}f}"
    except Exception:
        return "nan"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, oos = _load_frames()
    train = _add_targets(_augment(train))
    val = _add_targets(_augment(val))
    oos = _add_targets(_augment(oos))

    cols = [c for c in STATE_DIRECTION_ALPHA if c in train.columns and c in val.columns and c in oos.columns]
    if len(cols) != len(STATE_DIRECTION_ALPHA):
        missing = sorted(set(STATE_DIRECTION_ALPHA) - set(cols))
        raise RuntimeError(f"missing directional alpha features: {missing}")

    rows: list[dict[str, Any]] = []
    for c in cols:
        row: dict[str, Any] = {
            "feature": c,
            "train_missing": float(train[c].isna().mean()),
            "val_missing": float(val[c].isna().mean()),
            "oos_missing": float(oos[c].isna().mean()),
            "train_std": float(pd.to_numeric(train[c], errors="coerce").std()),
            "train_nunique": int(pd.to_numeric(train[c], errors="coerce").nunique(dropna=True)),
            "psi_val": _psi(train[c], val[c]),
            "psi_oos": _psi(train[c], oos[c]),
        }
        for h in (6, 12, 24, 48):
            row[f"ic_ret_val_{h}"] = _spearman(val[c], val[f"fwd_ret_{h}"])
            row[f"ic_abs_val_{h}"] = _spearman(val[c], val[f"fwd_abs_{h}"])
            row[f"ic_ret_oos_{h}"] = _spearman(oos[c], oos[f"fwd_ret_{h}"])
            row[f"ic_abs_oos_{h}"] = _spearman(oos[c], oos[f"fwd_abs_{h}"])
        row["max_abs_ret_ic_oos"] = float(np.nanmax([abs(row[f"ic_ret_oos_{h}"]) for h in (6, 12, 24, 48)]))
        row["max_abs_vol_ic_oos"] = float(np.nanmax([abs(row[f"ic_abs_oos_{h}"]) for h in (6, 12, 24, 48)]))
        row["max_abs_ret_ic_val"] = float(np.nanmax([abs(row[f"ic_ret_val_{h}"]) for h in (6, 12, 24, 48)]))
        row["max_abs_vol_ic_val"] = float(np.nanmax([abs(row[f"ic_abs_val_{h}"]) for h in (6, 12, 24, 48)]))
        verdict, reason = _verdict(pd.Series(row))
        row["verdict"] = verdict
        row["reason"] = reason
        rows.append(row)

    feature_df = pd.DataFrame(rows).sort_values(
        ["verdict", "max_abs_ret_ic_oos", "max_abs_vol_ic_oos"],
        ascending=[True, False, False],
    )
    feature_df.to_csv(FEATURE_OUT, index=False)

    red = _redundancy(train, cols)
    red.to_csv(REDUNDANCY_OUT, index=False)

    summary = {
        "model_id": MODEL_ID,
        "n_features": int(len(cols)),
        "verdict_counts": feature_df["verdict"].value_counts().to_dict(),
        "top_ret_ic": feature_df.sort_values("max_abs_ret_ic_oos", ascending=False).head(12).to_dict(orient="records"),
        "top_vol_ic": feature_df.sort_values("max_abs_vol_ic_oos", ascending=False).head(12).to_dict(orient="records"),
        "high_drift": feature_df.sort_values("psi_oos", ascending=False).head(12).to_dict(orient="records"),
        "redundancy_count_abs090": int(len(red)),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Directional Alpha Feature Audit - 2026-05-28")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report audits the 48 newly added direction-oriented features in `features/engineering.py`, including BTC lead-lag features.")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("")
    lines.append(f"- Feature scores: `{FEATURE_OUT.relative_to(ROOT)}`")
    lines.append(f"- Redundancy pairs: `{REDUNDANCY_OUT.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{SUMMARY_OUT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Verdict Counts")
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---|---:|")
    for k, v in feature_df["verdict"].value_counts().items():
        lines.append(f"| `{k}` | {int(v)} |")
    lines.append("")
    lines.append("## Top Return-Tendency Features")
    lines.append("")
    lines.append("| Feature | Verdict | Ret IC | Vol IC | PSI | Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in feature_df.sort_values("max_abs_ret_ic_oos", ascending=False).head(15).iterrows():
        lines.append(
            f"| `{r['feature']}` | `{r['verdict']}` | {_fmt(r['max_abs_ret_ic_oos'])} | {_fmt(r['max_abs_vol_ic_oos'])} | {_fmt(r['psi_oos'])} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Top Risk/Volatility-Tendency Features")
    lines.append("")
    lines.append("| Feature | Verdict | Ret IC | Vol IC | PSI | Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in feature_df.sort_values("max_abs_vol_ic_oos", ascending=False).head(15).iterrows():
        lines.append(
            f"| `{r['feature']}` | `{r['verdict']}` | {_fmt(r['max_abs_ret_ic_oos'])} | {_fmt(r['max_abs_vol_ic_oos'])} | {_fmt(r['psi_oos'])} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Per-Feature Table")
    lines.append("")
    lines.append("| Feature | Verdict | Ret IC | Vol IC | PSI | Reason |")
    lines.append("|---|---|---:|---:|---:|---|")
    for _, r in feature_df.sort_values("feature").iterrows():
        lines.append(
            f"| `{r['feature']}` | `{r['verdict']}` | {_fmt(r['max_abs_ret_ic_oos'])} | {_fmt(r['max_abs_vol_ic_oos'])} | {_fmt(r['psi_oos'])} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Secondary Feature Retraining Implication")
    lines.append("")
    lines.append("M7, AI, and regime-derived outputs do not automatically use these new features. If the goal is to let those second-order artifacts learn from the new direction block, their 2024-only training artifacts must be regenerated, then 2025/2026 sidecars rescored under the same causal split policy. Existing live artifacts remain valid but are blind to this new feature block.")
    lines.append("")
    lines.append("## Source-Required Direction Features")
    lines.append("")
    lines.append("The current offline active frame supports BTC lead-lag through `close_btc`, `volume_btc`, and `quote_volume_btc`. True orderbook imbalance, liquidation map/cluster distance, cross-exchange premium/basis, side-specific OI, and on-chain exchange-flow features are not present in the offline training frame. They must not be added as zero-filled active inputs; add them only after the historical source is persisted and the feature contract can fail fast on missing columns.")
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_OUT)
    print(FEATURE_OUT)
    print(SUMMARY_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
