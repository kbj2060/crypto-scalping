#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/home/llewyn/crypto-scalping")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASE_CSV = ROOT / "tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260521_stable48/08_alpha5_direction_router_rl_2024_to_2025/rl_training_2025_direction_router.csv"
DEFAULT_MARKET_STATE_CSV = ROOT / "data/ensemble/retrained_v22_market_state_v5_20260511/market_state_v5_2025.csv"
DEFAULT_CLEAN4_CSV = ROOT / "data/ensemble/supervised/clean_regime4_state24_sticky090_v2_20260517/training_features_2025_clean_regime4_state24_sticky090_v2.csv"
DEFAULT_PRED4_CSV = ROOT / "data/ensemble/supervised/regime4_pred_tft_h12_nomdjd_all74_20260517/training_features_2025_regime4_pred_tft_vsn_selected.csv"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521"
SOURCE_CLEAN4_PREFIX = "clean_regime4_2024_unsup_v1_"
STATE24_CLEAN4_PREFIX = "clean_regime4_state24_sticky090_v2_"


FORBIDDEN_PREFIXES = (
    "label_",
    "meta_",
    "entry_",
    "direction_",
    "path_",
)
FORBIDDEN_COLS = {
    "timestamp",
}


def _set_dsac_env() -> None:
    os.environ["DSAC_ALPHA5_V2_STATE_ENABLE"] = "1"
    os.environ["DSAC_V2_MULTI_ACTION_ENABLE"] = "1"
    os.environ["DSAC_ALL_FEATURES_ENABLE"] = "1"
    os.environ["DSAC_EXTRA_PCA_ENABLE"] = "1"
    os.environ["DSAC_EXTRA_PCA_COMPONENTS"] = "32"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"timestamp column missing: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DSAC feature inventory with explicit sidecar provenance.")
    p.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    p.add_argument("--market-state-csv", type=Path, default=DEFAULT_MARKET_STATE_CSV)
    p.add_argument("--clean4-csv", type=Path, default=DEFAULT_CLEAN4_CSV)
    p.add_argument("--pred4-csv", type=Path, default=DEFAULT_PRED4_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--clean4-rename-prefix",
        default=STATE24_CLEAN4_PREFIX,
        help="Downstream prefix used for the state24 clean-regime sidecar to avoid raw-state12 prefix ambiguity.",
    )
    p.add_argument(
        "--keep-legacy-clean4-prefix",
        action="store_true",
        help="Keep pre-existing clean_regime4_2024_unsup_v1_* columns from the base CSV. Default drops them and replaces them with the renamed state24 sidecar.",
    )
    return p.parse_args()


def _merge_exact(base: pd.DataFrame, side: pd.DataFrame, cols: list[str], tag: str) -> pd.DataFrame:
    use_cols = ["timestamp", *[c for c in cols if c in side.columns and c != "timestamp"]]
    merged = base.merge(side[use_cols], on="timestamp", how="left", suffixes=("", f"__{tag}"))
    for col in use_cols:
        if col == "timestamp":
            continue
        aux = f"{col}__{tag}"
        if aux not in merged.columns:
            if col not in merged.columns:
                merged[col] = np.nan
            continue
        aux_num = pd.to_numeric(merged[aux], errors="coerce")
        if col in merged.columns:
            base_num = pd.to_numeric(merged[col], errors="coerce")
            merged[col] = aux_num.where(aux_num.notna(), base_num)
        else:
            merged[col] = aux_num
        merged = merged.drop(columns=[aux])
    return merged


def _rename_clean4_sidecar(side: pd.DataFrame, target_prefix: str) -> pd.DataFrame:
    target_prefix = str(target_prefix).strip()
    if not target_prefix:
        raise ValueError("clean4 rename prefix cannot be empty")
    renamed = side.copy()
    mapping: dict[str, str] = {}
    for col in renamed.columns:
        if col.startswith(SOURCE_CLEAN4_PREFIX):
            mapping[col] = target_prefix + col[len(SOURCE_CLEAN4_PREFIX) :]
    return renamed.rename(columns=mapping)


def _drop_legacy_clean4_cols(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in frame.columns if c.startswith(SOURCE_CLEAN4_PREFIX)]
    if not cols:
        return frame, []
    return frame.drop(columns=cols), cols


def _feature_family(name: str) -> str:
    if name.startswith("market_state_"):
        return "market_state"
    if name.startswith(STATE24_CLEAN4_PREFIX):
        return "clean_regime4_state24"
    if name.startswith("clean_regime4_"):
        return "clean_regime4_raw_state12"
    if name.startswith("regime4_pred_"):
        return "regime4_pred"
    if name.startswith("a5dir_"):
        return "a5dir"
    if name.startswith("m7_"):
        return "m7"
    if name.startswith("ai_"):
        return "ai_family"
    if name.startswith("patchtst_") or name in {"pred_patchtst", "conf_patchtst"}:
        return "patchtst_family"
    if name.startswith("timesnet_"):
        return "timesnet_family"
    if name.startswith("tide_"):
        return "tide_family"
    if name.startswith("dlinear_"):
        return "dlinear_family"
    if name.startswith("teacher_"):
        return "teacher_family"
    if name.startswith("cvp_"):
        return "cvp"
    if name.startswith("funding_") or name in {"last_funding_rate", "mta_funding", "ou_funding_z"}:
        return "funding"
    if name.startswith("session_") or name.endswith("_sin") or name.endswith("_cos") or name == "is_hour_open":
        return "time"
    if name in {"open", "high", "low", "close", "volume", "quote_volume", "trades", "close_btc", "volume_btc", "quote_volume_btc"}:
        return "raw_level"
    return "other"


def _psi(train: pd.Series, val: pd.Series, bins: int = 10) -> float:
    x = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    y = pd.to_numeric(val, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 64 or len(y) < 64:
        return 0.0
    quantiles = np.unique(np.nanquantile(x.to_numpy(dtype=float), np.linspace(0.0, 1.0, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    q = quantiles.copy()
    q[0] = -np.inf
    q[-1] = np.inf
    px = np.histogram(x.to_numpy(dtype=float), bins=q)[0].astype(np.float64)
    py = np.histogram(y.to_numpy(dtype=float), bins=q)[0].astype(np.float64)
    px = np.clip(px / np.clip(px.sum(), 1e-12, None), 1e-6, None)
    py = np.clip(py / np.clip(py.sum(), 1e-12, None), 1e-6, None)
    return float(np.sum((px - py) * np.log(px / py)))


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    _set_dsac_env()
    from ensemble import train_rl_dsac_agent as dsac

    base = _read_csv(args.base_csv)
    market_state = _read_csv(args.market_state_csv)
    clean4 = _rename_clean4_sidecar(_read_csv(args.clean4_csv), str(args.clean4_rename_prefix))
    pred4 = _read_csv(args.pred4_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dropped_legacy_clean4_cols: list[str] = []
    if not bool(args.keep_legacy_clean4_prefix):
        base, dropped_legacy_clean4_cols = _drop_legacy_clean4_cols(base)

    merged = _merge_exact(base, market_state, [c for c in market_state.columns if c != "timestamp"], "market_state")
    merged = _merge_exact(merged, clean4, [c for c in clean4.columns if c.startswith(str(args.clean4_rename_prefix))], "clean4_state24")
    merged = _merge_exact(merged, pred4, [c for c in pred4.columns if c.startswith("regime4_pred_")], "pred4")
    merged_path = out_dir / "rl_training_2025_direction_router_feature_inventory_base.csv"
    merged.to_csv(merged_path, index=False)

    represented_tail = set(dsac.DSAC_ALL_FEATURE_COLS)
    represented_base = set(dsac.DSAC_DUPLICATE_SOURCE_COLS)
    represented_market = set(dsac.DSAC_MARKET_STATE_COLS)
    represented_router = set(dsac.ALPHA5_ROUTER_STATE_COLS)
    represented_major = set(dsac.ALPHA5_CATBOOST_MAJOR_COLS)

    ts = pd.to_datetime(merged["timestamp"], errors="coerce")
    train_mask = ts < pd.Timestamp("2025-10-01")
    val_mask = ts >= pd.Timestamp("2025-10-01")
    close = pd.to_numeric(merged.get("close", 0.0), errors="coerce")
    fwd12 = close.shift(-12) / close - 1.0
    fwd48 = close.shift(-48) / close - 1.0

    rows: list[dict[str, Any]] = []
    family_lists: dict[str, list[str]] = {}

    for col in merged.columns:
        if col in FORBIDDEN_COLS or col.startswith(FORBIDDEN_PREFIXES):
            status = "forbidden"
            family = _feature_family(col)
            rows.append({"feature": col, "family": family, "status": status})
            continue
        if col == "timestamp":
            continue
        series = pd.to_numeric(merged[col], errors="coerce")
        numeric_like = float(series.notna().mean()) >= 0.95
        if not numeric_like:
            status = "non_numeric_or_sparse"
        elif col in represented_tail:
            status = "current_tail"
        elif col in represented_base:
            status = "represented_in_base_state"
        elif col in represented_market:
            status = "market_state_explicit"
        elif col in represented_router:
            status = "router_explicit"
        elif col in represented_major:
            status = "catboost_major_explicit"
        else:
            status = "novel_candidate"
        family = _feature_family(col)
        valid = series.dropna()
        std = float(valid.std(ddof=0)) if len(valid) else 0.0
        nunique = int(valid.nunique(dropna=True)) if len(valid) else 0
        zero_rate = float(np.isclose(valid.to_numpy(dtype=float), 0.0).mean()) if len(valid) else 1.0
        missing_rate = float(1.0 - series.notna().mean())
        near_constant = bool(nunique <= 1 or std <= 1e-12)
        psi_val = _psi(series[train_mask], series[val_mask]) if numeric_like else 0.0
        rho12 = float(series.corr(fwd12, method="spearman")) if numeric_like else 0.0
        rho48 = float(series.corr(fwd48, method="spearman")) if numeric_like else 0.0
        row = {
            "feature": col,
            "family": family,
            "status": status,
            "numeric_like": bool(numeric_like),
            "missing_rate": missing_rate,
            "zero_rate": zero_rate,
            "nunique": nunique,
            "std": std,
            "near_constant": near_constant,
            "psi_2025_train_val": psi_val,
            "spearman_fwd12": rho12,
            "spearman_fwd48": rho48,
            "abs_spearman_fwd48": abs(rho48),
        }
        rows.append(row)
        if status == "novel_candidate" and numeric_like and not near_constant and missing_rate <= 0.05:
            family_lists.setdefault(family, []).append(col)

    inv = pd.DataFrame(rows).sort_values(["status", "family", "abs_spearman_fwd48", "psi_2025_train_val"], ascending=[True, True, False, True])
    inv.to_csv(out_dir / "candidate_inventory.csv", index=False)

    novel_all = sorted({c for cols in family_lists.values() for c in cols})
    _json_write(
        out_dir / "clean_candidate_list.json",
        {
            "name": "dsac_novel_clean_candidates_2025",
            "feature_count": len(novel_all),
            "features": novel_all,
        },
    )
    for family, cols in sorted(family_lists.items()):
        _json_write(
            out_dir / "families" / f"{family}.json",
            {
                "name": f"dsac_family_{family}",
                "feature_count": len(cols),
                "features": cols,
            },
        )

    summary = {
        "base_csv": str(args.base_csv),
        "merged_csv": str(merged_path),
        "market_state_csv": str(args.market_state_csv),
        "clean4_state24_csv": str(args.clean4_csv),
        "clean4_state24_rename_prefix": str(args.clean4_rename_prefix),
        "legacy_clean4_prefix": SOURCE_CLEAN4_PREFIX,
        "legacy_clean4_prefix_policy": "kept" if bool(args.keep_legacy_clean4_prefix) else "dropped_and_replaced_by_clean4_state24_rename_prefix",
        "dropped_legacy_clean4_cols": dropped_legacy_clean4_cols,
        "pred4_csv": str(args.pred4_csv),
        "provenance_policy": "state24 clean-regime sidecar is renamed before merge; ambiguous clean_regime4_2024_unsup_v1_* columns from the base CSV are dropped by default.",
        "rows": int(len(merged)),
        "train_rows_2025_jan_sep": int(train_mask.sum()),
        "val_rows_2025_oct_dec": int(val_mask.sum()),
        "current_tail_count": int(len(dsac.DSAC_ALL_FEATURE_COLS)),
        "novel_clean_candidate_count": int(len(novel_all)),
        "novel_candidates_by_family": {k: len(v) for k, v in sorted(family_lists.items())},
        "families": sorted(family_lists.keys()),
    }
    _json_write(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
