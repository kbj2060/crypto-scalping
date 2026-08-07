#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_all_feature_usage_20260528 import (  # noqa: E402
    _feature_cols,
    _load_frames,
    _spearman,
)
from scripts.analyze_directional_feature_universe_20260528 import (  # noqa: E402
    _augment,
    _is_direction_candidate,
    _load_prior_verdicts,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default  # noqa: E402


MODEL_ID = "entry_horizon_timing_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
FEATURE_OUT = OUT_DIR / "entry_horizon_feature_scores.csv"
FAMILY_OUT = OUT_DIR / "entry_horizon_family_summary.csv"
TOP_OUT = OUT_DIR / "entry_horizon_top_by_horizon.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"
REPORT_OUT = ROOT / "docs/audits/entry_horizon_timing_20260528.md"

HORIZONS = (12, 24, 36, 48, 64, 96)


def _family(c: str) -> str:
    from scripts.analyze_all_feature_usage_20260528 import _family as fam

    return fam(c)


def _add_horizon_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = pd.Series(_close(out), index=out.index).astype(float).clip(lower=1e-12)
    for h in HORIZONS:
        ret = close.shift(-h) / close - 1.0
        out[f"fwd_ret_{h}"] = ret.replace([np.inf, -np.inf], np.nan)
        out[f"dir_up_{h}"] = (out[f"fwd_ret_{h}"] > 0.0).astype(int)
    return out


def _auc(feature: pd.Series, target: pd.Series) -> float:
    x = pd.to_numeric(feature, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(target, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 500 or int(y[mask].nunique()) < 2 or float(x[mask].std()) <= 1e-12:
        return float("nan")
    return float(roc_auc_score(y[mask].astype(int), x[mask].astype(float)))


def _verdict_for_feature(feature: str, prior: dict[str, dict[str, str]]) -> tuple[str, str]:
    p = prior.get(feature, {})
    return str(p.get("prior_verdict", "")), str(p.get("prior_layer", ""))


def _clean_for_owner(row: pd.Series) -> bool:
    verdict = str(row.get("prior_verdict", ""))
    psi = float(row.get("psi_oos", np.nan))
    if verdict in {"DROP_RAW_LEVEL", "DEDUP_DROP", "BUG_RISK_REGENERATE"}:
        return False
    if "BUG_RISK" in verdict:
        return False
    if verdict == "MONITOR_OR_VETO_ONLY":
        return False
    if pd.notna(psi) and psi >= 0.50:
        return False
    return True


def _load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, val, oos = _load_frames()
    return _add_horizon_targets(_augment(train)), _add_horizon_targets(_augment(val)), _add_horizon_targets(_augment(oos))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prior = _load_prior_verdicts()
    train, val, oos = _load_base()
    cols = [
        c
        for c in _feature_cols(train, val, oos)
        if _is_direction_candidate(c, _family(c)) and not c.startswith(("dir_up_", "fwd_ret_"))
    ]
    if not cols:
        raise RuntimeError("no direction candidate features found")

    base_universe = ROOT / "tmp/causal_regen_20260516/directional_feature_universe_audit_20260528/directional_feature_universe_scores.csv"
    psi_map: dict[str, float] = {}
    if base_universe.exists():
        base = pd.read_csv(base_universe)
        psi_map = dict(zip(base["feature"].astype(str), pd.to_numeric(base["psi_oos"], errors="coerce")))

    rows: list[dict[str, Any]] = []
    for c in cols:
        prior_verdict, prior_layer = _verdict_for_feature(c, prior)
        row: dict[str, Any] = {
            "feature": c,
            "family": _family(c),
            "prior_verdict": prior_verdict,
            "prior_layer": prior_layer,
            "psi_oos": float(psi_map.get(c, np.nan)),
        }
        for h in HORIZONS:
            val_ic = _spearman(val[c], val[f"fwd_ret_{h}"])
            oos_ic = _spearman(oos[c], oos[f"fwd_ret_{h}"])
            val_auc = _auc(val[c], val[f"dir_up_{h}"])
            oos_auc = _auc(oos[c], oos[f"dir_up_{h}"])
            row[f"ic_val_{h}"] = val_ic
            row[f"ic_oos_{h}"] = oos_ic
            row[f"auc_val_{h}"] = val_auc
            row[f"auc_oos_{h}"] = oos_auc
            row[f"dir_auc_oos_{h}"] = float(max(oos_auc, 1.0 - oos_auc)) if pd.notna(oos_auc) else float("nan")
            row[f"sign_stable_{h}"] = bool(pd.notna(val_ic) and pd.notna(oos_ic) and np.sign(val_ic) == np.sign(oos_ic))
        abs_ics = {h: abs(float(row[f"ic_oos_{h}"])) if pd.notna(row[f"ic_oos_{h}"]) else -1.0 for h in HORIZONS}
        best_h = max(abs_ics, key=abs_ics.get)
        row["best_horizon"] = int(best_h)
        row["best_abs_ic_oos"] = float(abs_ics[best_h])
        row["best_ic_oos"] = float(row[f"ic_oos_{best_h}"])
        row["best_dir_auc_oos"] = float(row[f"dir_auc_oos_{best_h}"])
        row["best_sign_stable"] = bool(row[f"sign_stable_{best_h}"])
        rows.append(row)

    feat = pd.DataFrame(rows)
    feat["clean_for_owner"] = feat.apply(_clean_for_owner, axis=1)
    feat = feat.sort_values(["clean_for_owner", "best_abs_ic_oos", "best_dir_auc_oos"], ascending=[False, False, False])
    feat.to_csv(FEATURE_OUT, index=False)

    top_rows: list[pd.DataFrame] = []
    for h in HORIZONS:
        cols_out = [
            "feature",
            "family",
            "prior_verdict",
            "clean_for_owner",
            f"ic_val_{h}",
            f"ic_oos_{h}",
            f"auc_oos_{h}",
            f"dir_auc_oos_{h}",
            f"sign_stable_{h}",
            "psi_oos",
        ]
        sub = feat.assign(abs_ic_h=feat[f"ic_oos_{h}"].abs()).sort_values(
            ["clean_for_owner", "abs_ic_h", f"dir_auc_oos_{h}"], ascending=[False, False, False]
        )[cols_out + ["abs_ic_h"]].head(30)
        sub.insert(0, "horizon", h)
        top_rows.append(sub)
    top = pd.concat(top_rows, ignore_index=True)
    top.to_csv(TOP_OUT, index=False)

    fam_rows: list[dict[str, Any]] = []
    for fam, sub in feat.groupby("family"):
        row: dict[str, Any] = {"family": fam, "n_features": int(len(sub))}
        clean = sub[sub["clean_for_owner"]]
        row["n_clean"] = int(len(clean))
        basis = clean if len(clean) else sub
        for h in HORIZONS:
            idx = basis[f"ic_oos_{h}"].abs().idxmax()
            row[f"best_feature_{h}"] = str(basis.loc[idx, "feature"])
            row[f"best_ic_oos_{h}"] = float(basis.loc[idx, f"ic_oos_{h}"])
            row[f"best_dir_auc_oos_{h}"] = float(basis.loc[idx, f"dir_auc_oos_{h}"])
        best_idx = basis["best_abs_ic_oos"].idxmax()
        row["best_overall_feature"] = str(basis.loc[best_idx, "feature"])
        row["best_overall_horizon"] = int(basis.loc[best_idx, "best_horizon"])
        row["best_overall_abs_ic"] = float(basis.loc[best_idx, "best_abs_ic_oos"])
        fam_rows.append(row)
    fam = pd.DataFrame(fam_rows).sort_values("best_overall_abs_ic", ascending=False)
    fam.to_csv(FAMILY_OUT, index=False)

    clean_top = feat[feat["clean_for_owner"]].head(40)
    summary = {
        "model_id": MODEL_ID,
        "horizons": list(HORIZONS),
        "n_candidates": int(len(feat)),
        "n_clean_for_owner": int(feat["clean_for_owner"].sum()),
        "best_clean_features": clean_top.to_dict(orient="records"),
        "family_summary": fam.to_dict(orient="records"),
        "artifacts": {
            "feature_scores": str(FEATURE_OUT),
            "family_summary": str(FAMILY_OUT),
            "top_by_horizon": str(TOP_OUT),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# Entry Horizon Timing Audit - 2026-05-28")
    lines.append("")
    lines.append("Horizons: `12`, `24`, `36`, `48`, `64`, `96` bars.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Feature scores: `{FEATURE_OUT}`")
    lines.append(f"- Family summary: `{FAMILY_OUT}`")
    lines.append(f"- Top by horizon: `{TOP_OUT}`")
    lines.append("")
    lines.append("## Top Clean Features By Horizon")
    lines.append("")
    lines.append("| Horizon | Feature | Family | IC OOS | Direction AUC OOS | Stable Sign | PSI |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for h in HORIZONS:
        sub = top[(top["horizon"] == h) & (top["clean_for_owner"])].head(10)
        for _, r in sub.iterrows():
            lines.append(
                f"| {h} | `{r['feature']}` | `{r['family']}` | {float(r[f'ic_oos_{h}']):.3f} | "
                f"{float(r[f'dir_auc_oos_{h}']):.3f} | {bool(r[f'sign_stable_{h}'])} | {float(r['psi_oos']) if pd.notna(r['psi_oos']) else float('nan'):.3f} |"
            )
    lines.append("")
    lines.append("## Family Winners")
    lines.append("")
    lines.append("| Family | Best Feature | Best Horizon | Best Abs IC |")
    lines.append("|---|---|---:|---:|")
    for _, r in fam.head(20).iterrows():
        lines.append(
            f"| `{r['family']}` | `{r['best_overall_feature']}` | {int(r['best_overall_horizon'])} | {float(r['best_overall_abs_ic']):.3f} |"
        )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_DIR}")
    print(f"report: {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
