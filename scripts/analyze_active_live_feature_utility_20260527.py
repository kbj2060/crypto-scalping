#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_regression


ROOT = Path(__file__).resolve().parents[1]
FRAME_2025 = ROOT / "tmp/causal_regen_20260516/certified/features_2025.csv"
FRAME_2026 = ROOT / "tmp/causal_regen_20260516/certified/features_2026.csv"
OUT_DIR = ROOT / "data/ensemble/reports/active_live_feature_analysis_20260527"

HORIZONS = (1, 3, 6, 12, 24)
ACTIVE_PREFIXES = ("teacher_", "m7_", "ai_", "patchtst_", "tide_", "timesnet_", "dlinear_")
ACTIVE_EXACT = {"pred_patchtst", "conf_patchtst"}
BASE_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "sum_open_interest_value",
    "last_funding_rate",
    "funding_abs",
    "funding_pressure",
    "volatility_z",
    "garch_vol_z",
    "net_taker_ratio",
    "clean_regime_2024_unsup_v4_trend_bias",
    "clean_regime_2024_unsup_v4_risk_off_prob",
    "clean_regime_2024_unsup_v4_transition_risk",
    "clean_regime_2024_unsup_v4_whipsaw_prob",
    "clean_regime_2024_unsup_v4_bull_prob",
    "clean_regime_2024_unsup_v4_bear_prob",
    "clean_regime_2024_unsup_v4_chop_prob",
]


def _header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _active_columns() -> list[str]:
    h25 = set(_header(FRAME_2025))
    h26 = set(_header(FRAME_2026))
    common = sorted(h25 & h26)
    cols = [
        c
        for c in common
        if c in ACTIVE_EXACT or c.startswith(ACTIVE_PREFIXES)
    ]
    if not cols:
        raise RuntimeError("no active/live feature columns found")
    return cols


def _load(path: Path, active_cols: list[str]) -> pd.DataFrame:
    available = set(_header(path))
    required = [c for c in BASE_COLS if c in available] + active_cols
    missing_active = [c for c in active_cols if c not in available]
    if missing_active:
        raise RuntimeError(f"{path} missing active columns: {missing_active[:20]}")
    df = pd.read_csv(path, usecols=required)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def _finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    return xv[mask], yv[mask]


def _spearman(x: pd.Series, y: pd.Series) -> float:
    xv, yv = _finite_pair(x, y)
    if len(xv) < 100 or np.nanstd(xv) <= 1e-12 or np.nanstd(yv) <= 1e-12:
        return 0.0
    val = spearmanr(xv, yv).correlation
    return float(0.0 if not np.isfinite(val) else val)


def _auc(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    xv, yv = _finite_pair(x, y)
    if len(xv) < 100 or np.nanstd(xv) <= 1e-12:
        return 0.5, 0.5
    label = (yv > 0.0).astype(int)
    if label.min() == label.max():
        return 0.5, 0.5
    raw = float(roc_auc_score(label, xv))
    return raw, max(raw, 1.0 - raw)


def _psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    a = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    b = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(a) < 100 or len(b) < 100 or np.nanstd(a) <= 1e-12:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    pa = np.histogram(a, bins=edges)[0] / max(len(a), 1)
    pb = np.histogram(b, bins=edges)[0] / max(len(b), 1)
    pa = np.clip(pa, 1e-6, None)
    pb = np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    out["current_ret_1"] = close.pct_change().replace([np.inf, -np.inf], np.nan)
    out["current_abs_ret_1"] = out["current_ret_1"].abs()
    out["current_intrabar_range"] = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if "sum_open_interest_value" in out.columns:
        out["current_oi_change"] = pd.to_numeric(out["sum_open_interest_value"], errors="coerce").pct_change()
    if "volume" in out.columns:
        out["current_log_volume"] = np.log1p(pd.to_numeric(out["volume"], errors="coerce").clip(lower=0.0))
    for h in HORIZONS:
        out[f"fwd_ret_{h}"] = (close.shift(-h) / close - 1.0).replace([np.inf, -np.inf], np.nan)
        out[f"fwd_abs_ret_{h}"] = out[f"fwd_ret_{h}"].abs()
    return out


def _feature_family(col: str) -> str:
    if col.startswith("teacher_"):
        return "teacher"
    if col.startswith("m7_"):
        return "m7"
    if col.startswith("ai_"):
        return "ai"
    return "nf_direct"


def _health(df25: pd.DataFrame, df26: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in cols:
        s25 = pd.to_numeric(df25[c], errors="coerce")
        s26 = pd.to_numeric(df26[c], errors="coerce")
        out[c] = {
            "family": _feature_family(c),
            "missing_2025": float(s25.isna().mean()),
            "missing_2026": float(s26.isna().mean()),
            "std_2025": float(s25.std(skipna=True) or 0.0),
            "std_2026": float(s26.std(skipna=True) or 0.0),
            "unique_2025": int(s25.nunique(dropna=True)),
            "unique_2026": int(s26.nunique(dropna=True)),
            "psi_2026_vs_2025": _psi(s25, s26),
        }
    return out


def _future_scores(df25: pd.DataFrame, df26: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(704)
    mi_sample_n = min(25000, len(df26))
    mi_idx = rng.choice(len(df26), size=mi_sample_n, replace=False) if len(df26) > mi_sample_n else np.arange(len(df26))
    for c in cols:
        per_h: list[dict[str, Any]] = []
        for h in HORIZONS:
            y = f"fwd_ret_{h}"
            ic25 = _spearman(df25[c], df25[y])
            ic26 = _spearman(df26[c], df26[y])
            auc_raw, auc_dir = _auc(df26[c], df26[y])
            per_h.append({
                "horizon": h,
                "ic_2025": ic25,
                "ic_2026": ic26,
                "auc_raw_2026": auc_raw,
                "auc_directional_2026": auc_dir,
                "sign_consistent": int(np.sign(ic25) == np.sign(ic26) and abs(ic25) > 1e-6 and abs(ic26) > 1e-6),
            })
        best = max(per_h, key=lambda r: (abs(r["ic_2026"]), r["auc_directional_2026"]))
        x_mi = pd.to_numeric(df26[c], errors="coerce").iloc[mi_idx].to_numpy(dtype=float).reshape(-1, 1)
        y_mi = pd.to_numeric(df26[f"fwd_ret_{best['horizon']}"], errors="coerce").iloc[mi_idx].to_numpy(dtype=float)
        mask = np.isfinite(x_mi[:, 0]) & np.isfinite(y_mi)
        mi = 0.0
        if mask.sum() >= 500 and np.nanstd(x_mi[mask, 0]) > 1e-12:
            mi = float(mutual_info_regression(x_mi[mask], y_mi[mask], random_state=704, n_neighbors=5)[0])
        rows.append({
            "feature": c,
            "family": _feature_family(c),
            "best_horizon": best["horizon"],
            "best_ic_2025": best["ic_2025"],
            "best_ic_2026": best["ic_2026"],
            "best_abs_ic_2026": abs(best["ic_2026"]),
            "best_auc_raw_2026": best["auc_raw_2026"],
            "best_auc_directional_2026": best["auc_directional_2026"],
            "best_sign_consistent": best["sign_consistent"],
            "mi_2026_best_horizon": mi,
        })
    return pd.DataFrame(rows)


def _current_scores(df25: pd.DataFrame, df26: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    targets = [
        c for c in [
            "current_ret_1",
            "current_abs_ret_1",
            "current_intrabar_range",
            "current_oi_change",
            "current_log_volume",
            "last_funding_rate",
            "funding_abs",
            "funding_pressure",
            "volatility_z",
            "garch_vol_z",
            "net_taker_ratio",
            "clean_regime_2024_unsup_v4_trend_bias",
            "clean_regime_2024_unsup_v4_risk_off_prob",
            "clean_regime_2024_unsup_v4_transition_risk",
            "clean_regime_2024_unsup_v4_whipsaw_prob",
        ]
        if c in df25.columns and c in df26.columns
    ]
    rows = []
    for c in cols:
        vals = []
        for t in targets:
            vals.append((t, _spearman(df25[c], df25[t]), _spearman(df26[c], df26[t])))
        best = max(vals, key=lambda r: abs(r[2])) if vals else ("", 0.0, 0.0)
        rows.append({
            "feature": c,
            "family": _feature_family(c),
            "best_current_target": best[0],
            "best_current_corr_2025": best[1],
            "best_current_corr_2026": best[2],
            "best_abs_current_corr_2026": abs(best[2]),
        })
    return pd.DataFrame(rows)


def _corr_outputs(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[list[str]]]:
    corr = df[cols].corr(method="spearman", min_periods=1000).fillna(0.0)
    edges = []
    graph: dict[str, set[str]] = {c: set() for c in cols}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = float(corr.loc[a, b])
            if abs(v) >= 0.90:
                edges.append({"feature_a": a, "feature_b": b, "spearman_corr_2025": v, "abs_corr": abs(v)})
            if abs(v) >= 0.95:
                graph[a].add(b)
                graph[b].add(a)
    seen: set[str] = set()
    clusters: list[list[str]] = []
    for c in cols:
        if c in seen:
            continue
        stack = [c]
        comp = []
        seen.add(c)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(comp) > 1:
            clusters.append(sorted(comp))
    return pd.DataFrame(edges).sort_values("abs_corr", ascending=False), clusters


def _classify(row: pd.Series) -> str:
    future = float(row["future_score_0_100"])
    current = float(row["current_score_0_100"])
    psi = float(row["psi_2026_vs_2025"])
    if future >= 65 and psi <= 1.0:
        return "core_future"
    if future >= 45 and psi <= 1.5:
        return "useful_future"
    if current >= 60 and future < 35:
        return "current_context"
    if psi > 2.0:
        return "drift_risk"
    return "weak_or_redundant"


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [cols, ["---"] * len(cols)]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        rows.append(vals)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_cols = _active_columns()
    df25 = _add_targets(_load(FRAME_2025, active_cols))
    df26 = _add_targets(_load(FRAME_2026, active_cols))

    health = _health(df25, df26, active_cols)
    fut = _future_scores(df25, df26, active_cols)
    cur = _current_scores(df25, df26, active_cols)
    scores = fut.merge(cur, on=["feature", "family"], how="left")
    for c, h in health.items():
        for k, v in h.items():
            if k != "family":
                scores.loc[scores["feature"] == c, k] = v
    drift_factor = 1.0 / (1.0 + scores["psi_2026_vs_2025"].fillna(0.0).clip(lower=0.0))
    ic_component = (scores["best_abs_ic_2026"].clip(0.0, 0.05) / 0.05)
    auc_component = ((scores["best_auc_directional_2026"] - 0.5).clip(0.0, 0.05) / 0.05)
    consistency_component = scores["best_sign_consistent"].fillna(0.0)
    mi_component = (scores["mi_2026_best_horizon"].fillna(0.0).clip(0.0, 0.003) / 0.003)
    scores["future_score_0_100"] = 100.0 * (0.45 * ic_component + 0.25 * auc_component + 0.15 * consistency_component + 0.15 * mi_component) * drift_factor
    scores["current_score_0_100"] = 100.0 * (scores["best_abs_current_corr_2026"].fillna(0.0).clip(0.0, 0.55) / 0.55) * drift_factor
    scores["utility_bucket"] = scores.apply(_classify, axis=1)
    scores = scores.sort_values(["future_score_0_100", "current_score_0_100"], ascending=False)

    corr_edges, clusters = _corr_outputs(df25, active_cols)
    family_summary = (
        scores.groupby("family")
        .agg(
            feature_count=("feature", "count"),
            mean_future_score=("future_score_0_100", "mean"),
            max_future_score=("future_score_0_100", "max"),
            mean_current_score=("current_score_0_100", "mean"),
            max_current_score=("current_score_0_100", "max"),
            mean_psi=("psi_2026_vs_2025", "mean"),
        )
        .reset_index()
        .sort_values("max_future_score", ascending=False)
    )

    scores.to_csv(OUT_DIR / "active_live_feature_scores.csv", index=False)
    corr_edges.to_csv(OUT_DIR / "active_live_feature_corr_edges_abs090.csv", index=False)
    family_summary.to_csv(OUT_DIR / "active_live_feature_family_summary.csv", index=False)
    (OUT_DIR / "active_live_feature_corr_clusters_abs095.json").write_text(
        json.dumps({"threshold": 0.95, "clusters": clusters}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "status": "pass",
        "frames": {
            "2025": str(FRAME_2025),
            "2026": str(FRAME_2026),
            "rows_2025": int(len(df25)),
            "rows_2026": int(len(df26)),
        },
        "feature_count": len(active_cols),
        "score_formula": {
            "future_score_0_100": "100*(0.45*absIC26/0.05 + 0.25*(AUC26-0.5)/0.05 + 0.15*sign_consistency + 0.15*MI26/0.003)/(1+PSI)",
            "current_score_0_100": "100*(abs best same-bar context Spearman corr in 2026 / 0.55)/(1+PSI)",
        },
        "top_future": scores.head(20).to_dict(orient="records"),
        "top_current": scores.sort_values("current_score_0_100", ascending=False).head(20).to_dict(orient="records"),
        "family_summary": family_summary.to_dict(orient="records"),
        "high_corr_edge_count_abs090": int(len(corr_edges)),
        "high_corr_cluster_count_abs095": int(len(clusters)),
    }
    (OUT_DIR / "active_live_feature_analysis_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Active/Live Feature Utility Analysis",
        "",
        f"- features: {len(active_cols)}",
        f"- rows 2025: {len(df25):,}",
        f"- rows 2026: {len(df26):,}",
        f"- high corr edges abs>=0.90: {len(corr_edges):,}",
        f"- high corr clusters abs>=0.95: {len(clusters):,}",
        "",
        "## Top Future Predictive Features",
        _markdown_table(scores.head(20)[["feature", "family", "future_score_0_100", "best_horizon", "best_ic_2026", "best_auc_directional_2026", "psi_2026_vs_2025", "utility_bucket"]]),
        "",
        "## Top Current Context Features",
        _markdown_table(scores.sort_values("current_score_0_100", ascending=False).head(20)[["feature", "family", "current_score_0_100", "best_current_target", "best_current_corr_2026", "psi_2026_vs_2025", "utility_bucket"]]),
    ]
    (OUT_DIR / "active_live_feature_analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "pass", "out_dir": str(OUT_DIR), "feature_count": len(active_cols)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
