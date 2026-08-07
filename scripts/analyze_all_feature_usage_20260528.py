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


MODEL_ID = "all_feature_usage_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
FEATURE_TABLE_OUT = OUT_DIR / "feature_usage.csv"
FAMILY_TABLE_OUT = OUT_DIR / "family_usage.csv"
PROBE_TABLE_OUT = OUT_DIR / "family_probe_auc.csv"
REDUNDANCY_OUT = OUT_DIR / "redundancy_top_pairs.csv"

FORBIDDEN_NAME_TOKENS = (
    "label",
    "future",
    "fwd_",
    "cash_after",
    "pnl_after",
    "realized_net",
    "exit_reason",
    "dir24",
    "high_abs",
    "large_down",
    "target_",  # direct target columns. m7_target_* handled explicitly below.
)
DERIVABLE_SKIP = {"timestamp", "symbol"}


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


def _family(c: str) -> str:
    if c.startswith("clean_regime4_state24_sticky090_v2_"):
        return "regime_sticky_v2"
    if c.startswith("regime4_pred_"):
        return "regime_pred"
    if c.startswith("clean_regime_2024_unsup_v4_") or c.startswith("clean_regime4_2024_unsup_v1_"):
        return "regime_legacy"
    if c.startswith("market_state_"):
        return "market_state"
    if c.startswith("teacher_"):
        return "teacher"
    if c.startswith("m7_"):
        return "m7"
    if c.startswith("ai_"):
        return "ai"
    if c.startswith(("pred_patchtst", "conf_patchtst", "patchtst_")):
        return "patchtst"
    if c.startswith(("timesnet_", "dlinear_", "tide_")):
        return "ts_model"
    if "funding" in c:
        return "funding"
    if "oi_" in c or "open_interest" in c:
        return "open_interest"
    if any(tok in c for tok in ("taker", "flow", "volume", "trade_intensity", "amihud", "liquidity")):
        return "microstructure"
    if any(tok in c for tok in ("vol", "atr", "bb_width", "garch", "realized", "skew", "kurt")):
        return "volatility"
    if any(tok in c for tok in ("rsi", "macd", "trend", "mom", "return", "chop", "squeeze")):
        return "technical"
    if any(tok in c for tok in ("hour", "session", "weekday", "funding_window")):
        return "calendar"
    return "other"


def _is_allowed_feature(c: str, s: pd.Series) -> bool:
    if c in DERIVABLE_SKIP:
        return False
    if not pd.api.types.is_numeric_dtype(s):
        return False
    lc = c.lower()
    if c.startswith("m7_target_"):
        return True
    if any(tok in lc for tok in FORBIDDEN_NAME_TOKENS):
        return False
    return True


def _feature_cols(train: pd.DataFrame, val: pd.DataFrame, oos: pd.DataFrame) -> list[str]:
    cols = []
    for c in train.columns:
        if c not in val.columns or c not in oos.columns:
            continue
        if _is_allowed_feature(str(c), train[c]):
            cols.append(str(c))
    return sorted(cols)


def _add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = _close(out)
    for h in (6, 12, 24, 48):
        ret = pd.Series(close).shift(-h).to_numpy() / np.maximum(close, 1e-12) - 1.0
        out[f"fwd_ret_{h}"] = ret
        out[f"fwd_abs_{h}"] = np.abs(ret)
    out["dir24_up"] = (out["fwd_ret_24"] > 0.0).astype(int)
    return out


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


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 500 or float(x[mask].std()) <= 1e-12 or float(y[mask].std()) <= 1e-12:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def _layer_recommendation(row: dict[str, Any]) -> str:
    fam = row["family"]
    drift = float(row.get("psi_oos") or 0.0)
    ret_ic = max(abs(float(row.get(f"ic_ret_oos_{h}") or 0.0)) for h in (6, 12, 24, 48))
    abs_ic = max(abs(float(row.get(f"ic_abs_oos_{h}") or 0.0)) for h in (6, 12, 24, 48))
    if drift >= 0.50:
        return "monitor_or_veto_only"
    if fam in {"regime_sticky_v2", "regime_pred", "market_state"}:
        return "risk_meta_layer" if abs_ic >= ret_ic else "entry_meta_context"
    if fam in {"teacher", "m7", "ai", "patchtst", "ts_model"}:
        if ret_ic >= 0.08:
            return "entry_quality_or_direction"
        if abs_ic >= 0.08:
            return "risk_sizing_or_exit"
        return "secondary_context"
    if fam in {"microstructure", "funding", "open_interest"}:
        if abs_ic >= 0.06:
            return "execution_risk_sizing"
        return "entry_context"
    if fam in {"volatility"}:
        return "risk_sizing_or_exit"
    if fam in {"technical"}:
        return "entry_context"
    return "secondary_context"


def _probe_auc(train: pd.DataFrame, val: pd.DataFrame, oos: pd.DataFrame, cols: list[str], target: str) -> dict[str, float]:
    cols = [c for c in cols if c in train.columns and c in val.columns and c in oos.columns]
    if not cols:
        return {"n_features": 0.0, "train_auc": float("nan"), "val_auc": float("nan"), "oos_auc": float("nan")}
    tr = train.dropna(subset=[target]).reset_index(drop=True)
    va = val.dropna(subset=[target]).reset_index(drop=True)
    oo = oos.dropna(subset=[target]).reset_index(drop=True)
    if tr[target].nunique() < 2 or va[target].nunique() < 2 or oo[target].nunique() < 2:
        return {"n_features": float(len(cols)), "train_auc": float("nan"), "val_auc": float("nan"), "oos_auc": float("nan")}
    model = HistGradientBoostingClassifier(max_iter=180, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.10, random_state=17)
    model.fit(tr[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0), tr[target].astype(int))
    out = {"n_features": float(len(cols))}
    for name, df in (("train", tr), ("val", va), ("oos", oo)):
        p = model.predict_proba(df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1]
        out[f"{name}_auc"] = float(roc_auc_score(df[target].astype(int), p))
    return out


def _redundancy(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sample = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = sample.corr().abs().fillna(0.0)
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
    cols = _feature_cols(train, val, oos)
    if not cols:
        raise RuntimeError("no usable feature columns")
    vol_cut = float(train["fwd_abs_24"].quantile(0.75))
    for df in (train, val, oos):
        df["high_abs24"] = (df["fwd_abs_24"] >= vol_cut).astype(int)
        df["large_down24"] = (df["fwd_ret_24"] <= train["fwd_ret_24"].quantile(0.25)).astype(int)

    rows: list[dict[str, Any]] = []
    for c in cols:
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
        for h in (6, 12, 24, 48):
            row[f"ic_ret_val_{h}"] = _spearman(val[c], val[f"fwd_ret_{h}"])
            row[f"ic_abs_val_{h}"] = _spearman(val[c], val[f"fwd_abs_{h}"])
            row[f"ic_ret_oos_{h}"] = _spearman(oos[c], oos[f"fwd_ret_{h}"])
            row[f"ic_abs_oos_{h}"] = _spearman(oos[c], oos[f"fwd_abs_{h}"])
        row["max_abs_ret_ic_oos"] = float(np.nanmax([abs(row[f"ic_ret_oos_{h}"]) for h in (6, 12, 24, 48)]))
        row["max_abs_vol_ic_oos"] = float(np.nanmax([abs(row[f"ic_abs_oos_{h}"]) for h in (6, 12, 24, 48)]))
        row["max_abs_ret_ic_val"] = float(np.nanmax([abs(row[f"ic_ret_val_{h}"]) for h in (6, 12, 24, 48)]))
        row["max_abs_vol_ic_val"] = float(np.nanmax([abs(row[f"ic_abs_val_{h}"]) for h in (6, 12, 24, 48)]))
        row["recommended_layer"] = _layer_recommendation(row)
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(FEATURE_TABLE_OUT, index=False)

    family_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for fam in sorted(feature_df["family"].unique()):
        fam_cols = feature_df.loc[feature_df["family"].eq(fam), "feature"].tolist()
        sub = feature_df[feature_df["family"].eq(fam)]
        family_rows.append(
            {
                "family": fam,
                "n_features": int(len(fam_cols)),
                "mean_psi_oos": float(sub["psi_oos"].mean()),
                "median_psi_oos": float(sub["psi_oos"].median()),
                "max_psi_oos": float(sub["psi_oos"].max()),
                "mean_ret_ic_oos": float(sub["max_abs_ret_ic_oos"].mean()),
                "mean_vol_ic_oos": float(sub["max_abs_vol_ic_oos"].mean()),
                "best_ret_feature": str(sub.sort_values("max_abs_ret_ic_oos", ascending=False).iloc[0]["feature"]),
                "best_ret_ic": float(sub["max_abs_ret_ic_oos"].max()),
                "best_vol_feature": str(sub.sort_values("max_abs_vol_ic_oos", ascending=False).iloc[0]["feature"]),
                "best_vol_ic": float(sub["max_abs_vol_ic_oos"].max()),
            }
        )
        for target in ("dir24_up", "high_abs24", "large_down24"):
            probe_rows.append({"family": fam, "target": target, **_probe_auc(train, val, oos, fam_cols, target)})

    family_df = pd.DataFrame(family_rows).sort_values(["mean_ret_ic_oos", "mean_vol_ic_oos"], ascending=False)
    probe_df = pd.DataFrame(probe_rows)
    corr_df = _redundancy(train, cols)
    family_df.to_csv(FAMILY_TABLE_OUT, index=False)
    probe_df.to_csv(PROBE_TABLE_OUT, index=False)
    corr_df.to_csv(REDUNDANCY_OUT, index=False)

    summary = {
        "model_id": MODEL_ID,
        "split_policy": "train=2025 before 2025-10-01, val=2025-10-01..end, oos=2026 frame",
        "n_rows": {"train": int(len(train)), "val": int(len(val)), "oos": int(len(oos))},
        "n_features_common_numeric": int(len(cols)),
        "families": family_df.to_dict(orient="records"),
        "probe_auc": probe_df.to_dict(orient="records"),
        "top_entry_direction_candidates": feature_df.sort_values("max_abs_ret_ic_oos", ascending=False).head(30).to_dict(orient="records"),
        "top_risk_vol_candidates": feature_df.sort_values("max_abs_vol_ic_oos", ascending=False).head(30).to_dict(orient="records"),
        "recommended_layers_count": feature_df["recommended_layer"].value_counts().to_dict(),
        "artifacts": {
            "feature_table": str(FEATURE_TABLE_OUT),
            "family_table": str(FAMILY_TABLE_OUT),
            "probe_table": str(PROBE_TABLE_OUT),
            "redundancy_top_pairs": str(REDUNDANCY_OUT),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "feature_table": str(FEATURE_TABLE_OUT), "family_table": str(FAMILY_TABLE_OUT), "probe_table": str(PROBE_TABLE_OUT), "redundancy": str(REDUNDANCY_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
