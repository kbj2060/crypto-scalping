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
    _family,
    _feature_cols,
    _load_frames,
    _probe_auc,
    _psi,
    _spearman,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "directional_feature_universe_audit_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
FEATURE_OUT = OUT_DIR / "directional_feature_universe_scores.csv"
FAMILY_OUT = OUT_DIR / "directional_feature_family_summary.csv"
PROBE_OUT = OUT_DIR / "directional_feature_family_probe_auc.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"
REPORT_OUT = ROOT / "docs/audits/directional_feature_universe_audit_20260528.md"

PER_FEATURE_AUDIT = ROOT / "tmp/causal_regen_20260516/features_folder_per_feature_audit_20260528/per_feature_verdict.csv"

DIRECTION_TOKENS = (
    "action",
    "ai_",
    "bear",
    "btc",
    "bull",
    "cash",
    "chop",
    "confidence",
    "conf_",
    "count_long_short",
    "crowd",
    "cvd",
    "direction",
    "dlinear",
    "down",
    "edge",
    "flow",
    "funding",
    "long",
    "m7_",
    "margin",
    "market_state",
    "momentum",
    "net_taker",
    "oi_",
    "open_interest",
    "patchtst",
    "pred_",
    "prob",
    "regime",
    "ret",
    "short",
    "side",
    "squeeze",
    "teacher",
    "tide",
    "timesnet",
    "trend",
    "up",
    "vwap",
    "whale",
    "whipsaw",
)


def _augment(frame: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineer(keep_only_active=False)._create_directional_alpha_features(frame.copy())


def _load_prior_verdicts() -> dict[str, dict[str, str]]:
    if not PER_FEATURE_AUDIT.exists():
        return {}
    df = pd.read_csv(PER_FEATURE_AUDIT)
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        out[str(r["feature"])] = {
            "prior_verdict": str(r.get("verdict", "")),
            "prior_layer": str(r.get("recommended_layer", "")),
        }
    return out


def _is_direction_candidate(feature: str, family: str) -> bool:
    if feature in STATE_DIRECTION_ALPHA:
        return True
    if family in {
        "ai",
        "funding",
        "m7",
        "market_state",
        "microstructure",
        "open_interest",
        "patchtst",
        "regime_pred",
        "regime_sticky_v2",
        "teacher",
        "technical",
        "ts_model",
    }:
        return True
    lf = feature.lower()
    return any(tok in lf for tok in DIRECTION_TOKENS)


def _verdict(row: pd.Series) -> tuple[str, str]:
    psi = float(row["psi_oos"]) if pd.notna(row["psi_oos"]) else np.inf
    ret = float(row["max_abs_ret_ic_oos"]) if pd.notna(row["max_abs_ret_ic_oos"]) else 0.0
    ret_val = float(row["max_abs_ret_ic_val"]) if pd.notna(row["max_abs_ret_ic_val"]) else 0.0
    vol = float(row["max_abs_vol_ic_oos"]) if pd.notna(row["max_abs_vol_ic_oos"]) else 0.0
    prior = str(row.get("prior_verdict", ""))
    if "BUG_RISK" in prior:
        return "BUG_RISK_CHECK", "prior audit marked bug-risk; regenerate/prove before direct use"
    if prior == "DROP_RAW_LEVEL":
        return "DROP_RAW_LEVEL", "prior audit excludes raw level/price-like features from direct active input"
    if prior == "DEDUP_DROP":
        return "DEDUP_DROP_USE_REPRESENTATIVE", "prior audit marked as duplicate; use representative feature instead"
    if prior == "MONITOR_OR_VETO_ONLY" and psi >= 0.25:
        return "MONITOR_OR_VETO_ONLY", f"prior monitor/veto feature with OOS drift PSI={psi:.3f}"
    if psi >= 0.50:
        return "MONITOR_OR_VETO_ONLY", f"high OOS drift PSI={psi:.3f}; avoid direct owner input"
    if ret >= 0.055 and ret_val >= 0.015:
        return "KEEP_ENTRY_CONTEXT", "best current direction-context candidate; ablate in entry/meta layer"
    if vol >= 0.12 and vol > ret:
        return "KEEP_RISK_CONTEXT", "stronger risk/volatility utility than pure direction"
    if ret >= 0.035:
        return "SECONDARY_ENTRY_CONTEXT", "moderate direction tendency; use only inside compact ablations"
    if max(ret, vol) < 0.025:
        return "LOW_SIGNAL_SECONDARY", "low standalone OOS tendency"
    return "SECONDARY_CONTEXT", "mixed context; not a hard direction owner"


def _fmt(x: Any, n: int = 3) -> str:
    try:
        if pd.isna(x):
            return "nan"
        return f"{float(x):.{n}f}"
    except Exception:
        return "nan"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = _load_prior_verdicts()
    train, val, oos = _load_frames()
    train = _add_targets(_augment(train))
    val = _add_targets(_augment(val))
    oos = _add_targets(_augment(oos))

    all_cols = _feature_cols(train, val, oos)
    cols = [c for c in all_cols if _is_direction_candidate(c, _family(c))]
    if not cols:
        raise RuntimeError("no direction candidates found")

    rows: list[dict[str, Any]] = []
    for c in cols:
        row: dict[str, Any] = {
            "feature": c,
            "family": _family(c),
            "in_state_direction_alpha": bool(c in STATE_DIRECTION_ALPHA),
            "train_missing": float(train[c].isna().mean()),
            "val_missing": float(val[c].isna().mean()),
            "oos_missing": float(oos[c].isna().mean()),
            "train_std": float(pd.to_numeric(train[c], errors="coerce").std()),
            "train_nunique": int(pd.to_numeric(train[c], errors="coerce").nunique(dropna=True)),
            "psi_val": _psi(train[c], val[c]),
            "psi_oos": _psi(train[c], oos[c]),
        }
        row.update(prior.get(c, {"prior_verdict": "", "prior_layer": ""}))
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

    family_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    vol_cut = float(train["fwd_abs_24"].quantile(0.75))
    for df in (train, val, oos):
        df["high_abs24"] = (df["fwd_abs_24"] >= vol_cut).astype(int)
    for fam in sorted(feature_df["family"].unique()):
        sub = feature_df[feature_df["family"].eq(fam)]
        fam_cols = sub["feature"].tolist()
        family_rows.append(
            {
                "family": fam,
                "n_features": int(len(fam_cols)),
                "mean_ret_ic_oos": float(sub["max_abs_ret_ic_oos"].mean()),
                "mean_vol_ic_oos": float(sub["max_abs_vol_ic_oos"].mean()),
                "median_psi_oos": float(sub["psi_oos"].median()),
                "max_psi_oos": float(sub["psi_oos"].max()),
                "best_ret_feature": str(sub.sort_values("max_abs_ret_ic_oos", ascending=False).iloc[0]["feature"]),
                "best_ret_ic": float(sub["max_abs_ret_ic_oos"].max()),
                "best_vol_feature": str(sub.sort_values("max_abs_vol_ic_oos", ascending=False).iloc[0]["feature"]),
                "best_vol_ic": float(sub["max_abs_vol_ic_oos"].max()),
            }
        )
        probe_rows.append({"family": fam, "target": "dir24_up", **_probe_auc(train, val, oos, fam_cols, "dir24_up")})
        probe_rows.append({"family": fam, "target": "high_abs24", **_probe_auc(train, val, oos, fam_cols, "high_abs24")})

    family_df = pd.DataFrame(family_rows).sort_values(["best_ret_ic", "mean_ret_ic_oos"], ascending=False)
    probe_df = pd.DataFrame(probe_rows)
    family_df.to_csv(FAMILY_OUT, index=False)
    probe_df.to_csv(PROBE_OUT, index=False)

    summary = {
        "model_id": MODEL_ID,
        "n_candidates": int(len(feature_df)),
        "n_state_direction_alpha": int(feature_df["in_state_direction_alpha"].sum()),
        "verdict_counts": feature_df["verdict"].value_counts().to_dict(),
        "family_summary": family_df.to_dict(orient="records"),
        "probe_auc": probe_df.to_dict(orient="records"),
        "top_direction": feature_df.sort_values("max_abs_ret_ic_oos", ascending=False).head(40).to_dict(orient="records"),
        "top_risk_vol": feature_df.sort_values("max_abs_vol_ic_oos", ascending=False).head(30).to_dict(orient="records"),
        "artifacts": {
            "feature_scores": str(FEATURE_OUT),
            "family_summary": str(FAMILY_OUT),
            "probe_auc": str(PROBE_OUT),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Directional Feature Universe Audit - 2026-05-28")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report re-audits all direction-like candidates available in the active/offline frame plus the 48 engineered `STATE_DIRECTION_ALPHA` features.")
    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"- Feature scores: `{FEATURE_OUT.relative_to(ROOT)}`")
    lines.append(f"- Family summary: `{FAMILY_OUT.relative_to(ROOT)}`")
    lines.append(f"- Family probe AUC: `{PROBE_OUT.relative_to(ROOT)}`")
    lines.append(f"- Summary: `{SUMMARY_OUT.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Verdict Counts")
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---|---:|")
    for k, v in feature_df["verdict"].value_counts().items():
        lines.append(f"| `{k}` | {int(v)} |")
    lines.append("")
    lines.append("## Best Direction-Tendency Candidates")
    lines.append("")
    lines.append("| Feature | Family | Verdict | Ret IC | Vol IC | PSI | Prior | Reason |")
    lines.append("|---|---|---|---:|---:|---:|---|---|")
    for _, r in feature_df.sort_values("max_abs_ret_ic_oos", ascending=False).head(30).iterrows():
        lines.append(
            f"| `{r['feature']}` | `{r['family']}` | `{r['verdict']}` | {_fmt(r['max_abs_ret_ic_oos'])} | {_fmt(r['max_abs_vol_ic_oos'])} | {_fmt(r['psi_oos'])} | `{r.get('prior_verdict','')}` | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Best Risk/Volatility Context")
    lines.append("")
    lines.append("| Feature | Family | Verdict | Ret IC | Vol IC | PSI | Reason |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for _, r in feature_df.sort_values("max_abs_vol_ic_oos", ascending=False).head(20).iterrows():
        lines.append(
            f"| `{r['feature']}` | `{r['family']}` | `{r['verdict']}` | {_fmt(r['max_abs_ret_ic_oos'])} | {_fmt(r['max_abs_vol_ic_oos'])} | {_fmt(r['psi_oos'])} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## Family Summary")
    lines.append("")
    lines.append("| Family | N | Best Ret Feature | Best Ret IC | Best Vol Feature | Best Vol IC | Median PSI |")
    lines.append("|---|---:|---|---:|---|---:|---:|")
    for _, r in family_df.iterrows():
        lines.append(
            f"| `{r['family']}` | {int(r['n_features'])} | `{r['best_ret_feature']}` | {_fmt(r['best_ret_ic'])} | `{r['best_vol_feature']}` | {_fmt(r['best_vol_ic'])} | {_fmt(r['median_psi_oos'])} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Available offline whale/OI features are proxy-level: `whale_conviction`, `whale_retail_ratio`, aggregate OI, top-trader ratios, taker ratios, and engineered funding/OI interactions.")
    lines.append("- True liquidation flow, orderbook imbalance, side-specific OI, and on-chain exchange-flow features are still source-required and are not scored here because they are absent from the active/offline training frame.")
    lines.append("- Direction IC is a cheap screen, not a trade-level proof. Candidates marked `KEEP_ENTRY_CONTEXT` or `SECONDARY_ENTRY_CONTEXT` still need layer-specific ablation.")
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_OUT)
    print(FEATURE_OUT)
    print(SUMMARY_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
