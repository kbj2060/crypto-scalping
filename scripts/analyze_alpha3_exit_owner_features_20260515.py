#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha3_deep_exit_oracle_20260514 as deep_exit  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_alpha3_rl_exit_owner_fulltrain_20260514 as base_exit  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_exit_owner_feature_analysis_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_exit_owner_feature_analysis_20260515.json"
RANKING_OUT = ROOT / "data/ensemble/reports/alpha3_exit_owner_feature_analysis_20260515_ranking.csv"
REDUNDANT_OUT = ROOT / "data/ensemble/reports/alpha3_exit_owner_feature_analysis_20260515_redundant_pairs.csv"
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-10-01")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_std = float(np.std(y))
    if y_std <= 1e-12:
        return np.zeros(x.shape[1], dtype=np.float64)
    x_center = x - np.nanmean(x, axis=0, keepdims=True)
    y_center = y - float(np.nanmean(y))
    x_std = np.nanstd(x, axis=0)
    denom = np.maximum(x_std * y_std, 1e-12)
    return np.nanmean(x_center * y_center[:, None], axis=0) / denom


def _rank_features(x: np.ndarray, y: np.ndarray, feature_names: list[str], action_names: list[str]) -> pd.DataFrame:
    x_imp = SimpleImputer(strategy="median").fit_transform(x.astype(np.float64))
    best_i = np.argmax(y, axis=1)
    hold_i = action_names.index("hold")
    close_label = (best_i != hold_i).astype(np.int64)
    best_exit_q = np.max(y[:, 1:], axis=1)
    hold_q = y[:, hold_i]
    adv = np.clip(best_exit_q - hold_q, -0.25, 0.25)
    best_q = np.clip(np.max(y, axis=1), -0.25, 0.25)
    giveback_idx = feature_names.index("giveback") if "giveback" in feature_names else -1

    mi_close = mutual_info_classif(x_imp, close_label, discrete_features=False, random_state=20260515)
    mi_adv = mutual_info_regression(x_imp, adv, random_state=20260515)
    corr_close = np.abs(_safe_corr(x_imp, close_label.astype(np.float64)))
    corr_adv = np.abs(_safe_corr(x_imp, adv))
    corr_best = np.abs(_safe_corr(x_imp, best_q))
    missing = np.mean(~np.isfinite(x), axis=0)
    std = np.nanstd(x_imp, axis=0)
    zeroish = std <= 1e-8

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "block": ["base_market" if i < len(feature_names) - 30 else "position_state" for i in range(len(feature_names))],
            "missing_rate": missing,
            "std": std,
            "zeroish": zeroish,
            "mi_close": mi_close,
            "mi_exit_adv": mi_adv,
            "abs_corr_close": corr_close,
            "abs_corr_exit_adv": corr_adv,
            "abs_corr_best_q": corr_best,
        }
    )
    out["rank_score"] = (
        out["mi_close"].rank(pct=True)
        + out["mi_exit_adv"].rank(pct=True)
        + out["abs_corr_close"].rank(pct=True)
        + out["abs_corr_exit_adv"].rank(pct=True)
        + out["abs_corr_best_q"].rank(pct=True)
    )
    if giveback_idx >= 0:
        giveback = x_imp[:, giveback_idx]
        out["abs_corr_giveback"] = np.abs(_safe_corr(x_imp, giveback))
    out = out.sort_values(["zeroish", "rank_score"], ascending=[True, False]).reset_index(drop=True)
    return out


def _pca_summary(x: np.ndarray, feature_names: list[str], start: int, end: int, *, prefix: str) -> dict[str, Any]:
    cols = feature_names[start:end]
    if not cols:
        return {"prefix": prefix, "n_features": 0}
    x_imp = SimpleImputer(strategy="median").fit_transform(x[:, start:end].astype(np.float64))
    x_z = StandardScaler().fit_transform(x_imp)
    n_components = min(x_z.shape[0], x_z.shape[1])
    pca = PCA(n_components=n_components, random_state=20260515)
    pca.fit(x_z)
    csum = np.cumsum(pca.explained_variance_ratio_)
    def need(th: float) -> int:
        return int(np.searchsorted(csum, th) + 1)
    return {
        "prefix": prefix,
        "n_features": int(len(cols)),
        "components_for_80pct": need(0.80),
        "components_for_90pct": need(0.90),
        "components_for_95pct": need(0.95),
        "components_for_99pct": need(0.99),
        "first_12_explained": [float(v) for v in pca.explained_variance_ratio_[:12]],
    }


def _pls_summary(x: np.ndarray, y: np.ndarray, feature_names: list[str], start: int, end: int, *, prefix: str) -> dict[str, Any]:
    cols = feature_names[start:end]
    if not cols:
        return {"prefix": prefix, "n_features": 0}
    x_imp = SimpleImputer(strategy="median").fit_transform(x[:, start:end].astype(np.float64))
    best_i = np.argmax(y, axis=1)
    hold_i = 0
    close_label = (best_i != hold_i).astype(np.float64)
    best_exit_q = np.max(y[:, 1:], axis=1)
    adv = np.clip(best_exit_q - y[:, hold_i], -0.25, 0.25)
    best_q = np.clip(np.max(y, axis=1), -0.25, 0.25)
    target = np.column_stack([close_label, adv, best_q]).astype(np.float64)
    max_comp = min(12, x_imp.shape[1], x_imp.shape[0] - 1)
    rows: list[dict[str, Any]] = []
    for n_comp in range(1, max_comp + 1):
        pls = PLSRegression(n_components=n_comp, scale=True)
        scores = pls.fit_transform(x_imp, target)[0]
        pred = pls.predict(x_imp)
        mse = np.mean((target - pred) ** 2, axis=0)
        var = np.maximum(np.var(target, axis=0), 1e-12)
        rows.append(
            {
                "n_components": int(n_comp),
                "r2_close": float(1.0 - mse[0] / var[0]),
                "r2_exit_adv": float(1.0 - mse[1] / var[1]),
                "r2_best_q": float(1.0 - mse[2] / var[2]),
                "score_std_mean": float(np.mean(np.std(scores, axis=0))),
            }
        )
    return {"prefix": prefix, "n_features": int(len(cols)), "targets": ["close_label", "exit_advantage_clipped", "best_q_clipped"], "grid": rows}


def _redundant_pairs(x: np.ndarray, feature_names: list[str], ranking: pd.DataFrame, *, max_rows: int = 250) -> pd.DataFrame:
    x_imp = SimpleImputer(strategy="median").fit_transform(x.astype(np.float64))
    if len(x_imp) > 20000:
        rng = np.random.default_rng(20260515)
        idx = rng.choice(len(x_imp), size=20000, replace=False)
        x_imp = x_imp[idx]
    x_z = StandardScaler().fit_transform(x_imp)
    corr = np.corrcoef(x_z, rowvar=False)
    score = dict(zip(ranking["feature"], ranking["rank_score"]))
    rows: list[dict[str, Any]] = []
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr[i, j])
            if not np.isfinite(c) or abs(c) < 0.97:
                continue
            fi, fj = feature_names[i], feature_names[j]
            keep = fi if score.get(fi, 0.0) >= score.get(fj, 0.0) else fj
            drop = fj if keep == fi else fi
            rows.append({"feature_a": fi, "feature_b": fj, "abs_corr": abs(c), "keep_by_rank": keep, "drop_candidate": drop})
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).head(max_rows).reset_index(drop=True)


def main() -> int:
    print(f"[{MODEL_ID}] loading Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    arms = deep_exit._arm_configs()
    arm_by_name = {a.name: a for a in arms}
    entry_cfg = arm_by_name["baseline_exit2_pen05"]
    base_cols = list(stack["teacher_payload"]["feature_cols"])
    feature_names = deep_exit._feature_names(base_cols)
    action_names = base_exit._action_names(arms)

    train_all = _read(v31.DEFAULT_TRAIN)
    train_df = train_all[(train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] < TRAIN_END)].reset_index(drop=True)
    print(f"[{MODEL_ID}] rebuilding Alpha3 decisions/q for {len(train_df)} rows", flush=True)
    train_dec, train_q = front_run._decisions_and_q(train_df, stack)
    print(f"[{MODEL_ID}] collecting Alpha3 exit-owner DP replay", flush=True)
    x, y, dataset_meta = base_exit.collect_q_dataset(
        train_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        train_q,
        train_dec,
        stack["overlay"],
        entry_cfg,
        arms,
        base_cols,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    if x.shape[1] != len(feature_names):
        raise RuntimeError(f"feature mismatch: x={x.shape[1]} names={len(feature_names)}")
    print(f"[{MODEL_ID}] ranking {x.shape[1]} features on {x.shape[0]} states", flush=True)
    ranking = _rank_features(x, y, feature_names, action_names)
    ranking.to_csv(RANKING_OUT, index=False)
    redundant = _redundant_pairs(x, feature_names, ranking)
    redundant.to_csv(REDUNDANT_OUT, index=False)

    state_start = len(feature_names) - 30
    top_state = ranking[ranking["block"] == "position_state"].head(30)["feature"].tolist()
    top_base = ranking[ranking["block"] == "base_market"].head(40)["feature"].tolist()
    must_keep_state = [
        "pos",
        "owner_deep",
        "owner_parent",
        "hold_norm",
        "unreal",
        "mfe",
        "mae",
        "giveback",
        "notional",
        "parent_notional",
        "effective_tp",
        "effective_sl",
        "q_same",
        "q_opp",
        "q_margin",
        "row_vol_anchor",
    ]
    selected_state = [f for f in top_state if f in must_keep_state] + [f for f in must_keep_state if f not in top_state]
    selected_state = list(dict.fromkeys(selected_state))
    recommended = {
        "minimal_state_only": selected_state,
        "state_plus_top_base20": selected_state + top_base[:20],
        "state_plus_top_base30": selected_state + top_base[:30],
        "state_plus_top_base40": selected_state + top_base[:40],
        "compressed_base_pls8_plus_state": selected_state + [f"base_pls_{i}" for i in range(8)],
        "compressed_base_pls12_plus_state": selected_state + [f"base_pls_{i}" for i in range(12)],
    }
    report = {
        "model_id": MODEL_ID,
        "date": "2026-05-15",
        "scope": "Alpha3 DSAC exit-owner feature analysis. L2/orderbook/queue data intentionally excluded.",
        "dataset": {
            **dataset_meta,
            "train_start": str(train_df["timestamp"].iloc[0]) if len(train_df) else None,
            "train_end": str(train_df["timestamp"].iloc[-1]) if len(train_df) else None,
            "state_dim": int(x.shape[1]),
            "base_market_features": int(len(base_cols)),
            "position_state_features": 30,
            "actions": action_names,
            "target_argmax_counts": dict(zip(action_names, np.bincount(np.argmax(y, axis=1), minlength=len(action_names)).astype(int).tolist())),
        },
        "top20_features": ranking.head(20).to_dict(orient="records"),
        "top20_base_market": ranking[ranking["block"] == "base_market"].head(20).to_dict(orient="records"),
        "top20_position_state": ranking[ranking["block"] == "position_state"].head(20).to_dict(orient="records"),
        "zeroish_features": ranking[ranking["zeroish"]].loc[:, ["feature", "block", "std", "missing_rate"]].to_dict(orient="records"),
        "pca": {
            "base_market_93": _pca_summary(x, feature_names, 0, state_start, prefix="base_market"),
            "position_state_30": _pca_summary(x, feature_names, state_start, len(feature_names), prefix="position_state"),
            "all_123": _pca_summary(x, feature_names, 0, len(feature_names), prefix="all"),
        },
        "pls": {
            "base_market_93": _pls_summary(x, y, feature_names, 0, state_start, prefix="base_market"),
            "position_state_30": _pls_summary(x, y, feature_names, state_start, len(feature_names), prefix="position_state"),
            "all_123": _pls_summary(x, y, feature_names, 0, len(feature_names), prefix="all"),
        },
        "redundant_pair_count_abs_corr_ge_0_97_top250": int(len(redundant)),
        "recommended_feature_sets": {k: {"n_features": len(v), "features": v} for k, v in recommended.items()},
        "artifacts": {
            "report": str(REPORT_OUT.relative_to(ROOT)),
            "ranking": str(RANKING_OUT.relative_to(ROOT)),
            "redundant_pairs": str(REDUNDANT_OUT.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "ranking": str(RANKING_OUT), "top5": ranking.head(5)["feature"].tolist()}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
