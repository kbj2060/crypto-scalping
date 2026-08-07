#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import PRIMARY_EVAL_CSV, PRIMARY_TRAIN_CSV  # noqa: E402
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402


MODEL_ID = "regime_feature_validity_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
FEATURE_TABLE_OUT = OUT_DIR / "feature_validity.csv"
FAMILY_TABLE_OUT = OUT_DIR / "family_validity.csv"
PROBE_TABLE_OUT = OUT_DIR / "probe_auc.csv"
CORR_TABLE_OUT = OUT_DIR / "redundancy_top_pairs.csv"

STICKY_PREFIX = "clean_regime4_state24_sticky090_v2_"
LEGACY_V4_PREFIX = "clean_regime_2024_unsup_v4_"
PRED_PREFIX = "regime4_pred_"


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = combo._merge_state24(_read(combo.v31.DEFAULT_TRAIN), combo.alpha3_full.SIDE_CLEAN4_2025)
    eval_df = combo._merge_state24(_read(combo.v31.DEFAULT_EVAL), combo.alpha3_full.SIDE_CLEAN4_2026)
    a7_train = _rename_clean4_v2(_read(PRIMARY_TRAIN_CSV))
    a7_eval = _rename_clean4_v2(_read(PRIMARY_EVAL_CSV))
    train_all = combo._augment_with_alpha7_features(train_all, a7_train)
    eval_df = combo._augment_with_alpha7_features(eval_df, a7_eval)
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"], errors="raise")
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], errors="raise")
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    return train, val, eval_df.reset_index(drop=True)


def _family(col: str) -> str:
    if col.startswith(STICKY_PREFIX):
        return "sticky_v2_current"
    if col.startswith(LEGACY_V4_PREFIX):
        return "legacy_v4_current"
    if col.startswith(PRED_PREFIX):
        return "regime4_pred_future"
    if col.startswith("market_state_"):
        return "market_state"
    if col in {"ai_vol_regime_pct", "patchtst_regime_sim", "cvp_regime", "regime_trending"}:
        return "aux_regime"
    return "other_regime"


def _regime_cols(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in frame.columns:
        lc = str(c).lower()
        if not pd.api.types.is_numeric_dtype(frame[c]):
            continue
        if "regime" not in lc and "market_state" not in lc:
            continue
        if any(tok in lc for tok in ("target", "label", "future_ret", "cash_after", "pnl_after")):
            continue
        out.append(str(c))
    return sorted(out)


def _psi(train: pd.Series, other: pd.Series, bins: int = 10) -> float:
    a = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    b = pd.to_numeric(other, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(a) < 100 or len(b) < 100 or float(np.nanstd(a)) <= 1e-12:
        return float("nan")
    qs = np.unique(np.quantile(a, np.linspace(0.0, 1.0, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0] = -np.inf
    qs[-1] = np.inf
    ca = np.histogram(a, bins=qs)[0].astype(float)
    cb = np.histogram(b, bins=qs)[0].astype(float)
    pa = np.clip(ca / max(ca.sum(), 1.0), 1e-6, 1.0)
    pb = np.clip(cb / max(cb.sum(), 1.0), 1e-6, 1.0)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = _close(out)
    for h in (12, 24, 48):
        out[f"fwd_ret_{h}"] = pd.Series(close).shift(-h).to_numpy() / np.maximum(close, 1e-12) - 1.0
        out[f"fwd_abs_{h}"] = np.abs(out[f"fwd_ret_{h}"].to_numpy(dtype=float))
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 500 or float(x[mask].std()) <= 1e-12 or float(y[mask].std()) <= 1e-12:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def _probe_auc(train: pd.DataFrame, val: pd.DataFrame, oos: pd.DataFrame, cols: list[str], target: str) -> dict[str, float]:
    cols = [c for c in cols if c in train.columns and c in val.columns and c in oos.columns]
    if not cols:
        return {"train_auc": float("nan"), "val_auc": float("nan"), "oos_auc": float("nan"), "n_features": 0}
    tr = train.dropna(subset=[target]).reset_index(drop=True)
    va = val.dropna(subset=[target]).reset_index(drop=True)
    oo = oos.dropna(subset=[target]).reset_index(drop=True)
    if tr[target].nunique() < 2 or va[target].nunique() < 2 or oo[target].nunique() < 2:
        return {"train_auc": float("nan"), "val_auc": float("nan"), "oos_auc": float("nan"), "n_features": len(cols)}
    model = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.10, random_state=7)
    xtr = tr[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model.fit(xtr, tr[target].astype(int))
    out: dict[str, float] = {"n_features": float(len(cols))}
    for name, df in (("train", tr), ("val", va), ("oos", oo)):
        p = model.predict_proba(df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1]
        out[f"{name}_auc"] = float(roc_auc_score(df[target].astype(int), p))
    return out


def _redundancy(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = x.corr().abs().fillna(0.0)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = float(corr.loc[a, b])
            if v >= 0.95:
                rows.append({"feature_a": a, "feature_b": b, "abs_corr": v, "family_a": _family(a), "family_b": _family(b)})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, oos = _load_frames()
    train = _add_targets(train)
    val = _add_targets(val)
    oos = _add_targets(oos)
    cols = _regime_cols(train)
    common = [c for c in cols if c in val.columns and c in oos.columns]
    if not common:
        raise RuntimeError("no common regime columns")

    train["dir24_up"] = (train["fwd_ret_24"] > 0.0).astype(int)
    val["dir24_up"] = (val["fwd_ret_24"] > 0.0).astype(int)
    oos["dir24_up"] = (oos["fwd_ret_24"] > 0.0).astype(int)
    vol_cut = float(train["fwd_abs_24"].quantile(0.75))
    train["high_abs24"] = (train["fwd_abs_24"] >= vol_cut).astype(int)
    val["high_abs24"] = (val["fwd_abs_24"] >= vol_cut).astype(int)
    oos["high_abs24"] = (oos["fwd_abs_24"] >= vol_cut).astype(int)

    feature_rows: list[dict[str, Any]] = []
    for c in common:
        row: dict[str, Any] = {
            "feature": c,
            "family": _family(c),
            "train_missing": float(train[c].isna().mean()),
            "val_missing": float(val[c].isna().mean()),
            "oos_missing": float(oos[c].isna().mean()),
            "train_std": float(pd.to_numeric(train[c], errors="coerce").std()),
            "train_nunique": int(pd.to_numeric(train[c], errors="coerce").nunique(dropna=True)),
            "psi_val": _psi(train[c], val[c]),
            "psi_oos": _psi(train[c], oos[c]),
        }
        for h in (12, 24, 48):
            row[f"ic_ret_val_{h}"] = _spearman(val[c], val[f"fwd_ret_{h}"])
            row[f"ic_abs_val_{h}"] = _spearman(val[c], val[f"fwd_abs_{h}"])
            row[f"ic_ret_oos_{h}"] = _spearman(oos[c], oos[f"fwd_ret_{h}"])
            row[f"ic_abs_oos_{h}"] = _spearman(oos[c], oos[f"fwd_abs_{h}"])
        row["max_abs_ic_val"] = float(np.nanmax([abs(row[k]) for k in row if k.startswith("ic_") and "_val_" in k]))
        row["max_abs_ic_oos"] = float(np.nanmax([abs(row[k]) for k in row if k.startswith("ic_") and "_oos_" in k]))
        row["ic_stability"] = float(row["max_abs_ic_oos"] / max(row["max_abs_ic_val"], 1e-9))
        feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows).sort_values(["family", "max_abs_ic_oos"], ascending=[True, False])
    feature_df.to_csv(FEATURE_TABLE_OUT, index=False)

    family_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for fam in sorted({_family(c) for c in common}):
        fam_cols = [c for c in common if _family(c) == fam]
        sub = feature_df[feature_df["family"].eq(fam)]
        family_rows.append(
            {
                "family": fam,
                "n_features": int(len(fam_cols)),
                "mean_psi_oos": float(sub["psi_oos"].mean()),
                "median_psi_oos": float(sub["psi_oos"].median()),
                "max_psi_oos": float(sub["psi_oos"].max()),
                "mean_max_abs_ic_val": float(sub["max_abs_ic_val"].mean()),
                "mean_max_abs_ic_oos": float(sub["max_abs_ic_oos"].mean()),
                "best_oos_feature": str(sub.sort_values("max_abs_ic_oos", ascending=False).iloc[0]["feature"]),
                "best_oos_ic": float(sub["max_abs_ic_oos"].max()),
            }
        )
        for target in ("dir24_up", "high_abs24"):
            auc = _probe_auc(train, val, oos, fam_cols, target)
            probe_rows.append({"family": fam, "target": target, **auc})

    all_regime_cols = common
    for target in ("dir24_up", "high_abs24"):
        auc = _probe_auc(train, val, oos, all_regime_cols, target)
        probe_rows.append({"family": "all_regime", "target": target, **auc})

    family_df = pd.DataFrame(family_rows).sort_values("mean_max_abs_ic_oos", ascending=False)
    probe_df = pd.DataFrame(probe_rows)
    corr_df = _redundancy(train, common)
    family_df.to_csv(FAMILY_TABLE_OUT, index=False)
    probe_df.to_csv(PROBE_TABLE_OUT, index=False)
    corr_df.to_csv(CORR_TABLE_OUT, index=False)

    summary = {
        "model_id": MODEL_ID,
        "split_policy": "train=2025 before 2025-10-01, val=2025-10-01..end, oos=2026 frame",
        "n_rows": {"train": int(len(train)), "val": int(len(val)), "oos": int(len(oos))},
        "n_regime_features_common": int(len(common)),
        "families": family_df.to_dict(orient="records"),
        "probe_auc": probe_df.to_dict(orient="records"),
        "top_features_by_oos_ic": feature_df.sort_values("max_abs_ic_oos", ascending=False).head(20).to_dict(orient="records"),
        "high_abs24_train_cut": vol_cut,
        "artifacts": {
            "feature_table": str(FEATURE_TABLE_OUT),
            "family_table": str(FAMILY_TABLE_OUT),
            "probe_table": str(PROBE_TABLE_OUT),
            "redundancy_top_pairs": str(CORR_TABLE_OUT),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "feature_table": str(FEATURE_TABLE_OUT), "family_table": str(FAMILY_TABLE_OUT), "probe_table": str(PROBE_TABLE_OUT), "redundancy": str(CORR_TABLE_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
