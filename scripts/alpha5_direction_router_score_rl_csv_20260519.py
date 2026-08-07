#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path("/home/llewyn/crypto-scalping")
DEFAULT_ROUTER_MODEL = ROOT / "tmp/causal_regen_20260516/alpha5_router_v2_contracts_20260519/router3_catboost_gpu.cbm"
DEFAULT_ROUTER_META = ROOT / "tmp/causal_regen_20260516/alpha5_router_v5_ensemble_contracts_20260520/router_ensemble_meta.joblib"
DEFAULT_ALPHA5_PARQUETS = [
    ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519/alpha5_29_hier_label_factory_train.parquet",
    ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519/alpha5_29_hier_label_factory_val.parquet",
    ROOT / "tmp/causal_regen_20260516/alpha5_29_hier_label_factory_20260519/alpha5_29_hier_label_factory_oos.parquet",
    ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519/alpha5_30_direction_learnable_train.parquet",
    ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519/alpha5_30_direction_learnable_val.parquet",
    ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519/alpha5_30_direction_learnable_oos.parquet",
]
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_direction_router_rl_20260519"


def _default_router_model() -> Path:
    return Path(os.getenv("A5DIR_ROUTER_MODEL_PATH", str(DEFAULT_ROUTER_MODEL)))


def _default_router_meta() -> Path:
    return Path(os.getenv("A5DIR_ROUTER_META_PATH", str(DEFAULT_ROUTER_META)))


def _default_aux_paths() -> list[Path]:
    raw = os.getenv("A5DIR_ROUTER_AUX_PARQUETS", "").strip()
    if not raw:
        return list(DEFAULT_ALPHA5_PARQUETS)
    parts = [x.strip() for x in raw.split(os.pathsep) if x.strip()]
    return [Path(x) for x in parts]


def _load_router_meta(meta_path: Path) -> dict:
    meta = joblib.load(meta_path)
    if not isinstance(meta, dict) or "feature_cols" not in meta:
        raise ValueError(f"invalid router meta: {meta_path}")
    return meta


def _score_router_proba(meta: dict, feature_frame: pd.DataFrame, router_model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    router_type = str(meta.get("type", "router3"))
    if router_type == "router_ensemble":
        components = list(meta.get("components", []))
        if not components:
            raise ValueError("router_ensemble meta missing components")
        p_none = np.zeros(len(feature_frame), dtype=np.float32)
        p_long = np.zeros(len(feature_frame), dtype=np.float32)
        p_short = np.zeros(len(feature_frame), dtype=np.float32)
        total_weight = 0.0
        for comp in components:
            weight = float(comp.get("weight", 1.0))
            comp_meta = _load_router_meta(Path(comp["meta_path"]))
            c_none, c_long, c_short = _score_router_proba(comp_meta, feature_frame, Path(comp["model_path"]))
            p_none += weight * c_none
            p_long += weight * c_long
            p_short += weight * c_short
            total_weight += weight
        total_weight = max(total_weight, 1e-9)
        p_none /= total_weight
        p_long /= total_weight
        p_short /= total_weight
        return p_none.astype(np.float32), p_long.astype(np.float32), p_short.astype(np.float32)

    if router_type == "router3":
        model = CatBoostClassifier()
        model.load_model(str(router_model_path))
        proba = np.asarray(model.predict_proba(feature_frame), dtype=np.float32)
        classes = list(getattr(model, "classes_", meta.get("classes", [0, 1, 2])))
        if 0 not in classes or 1 not in classes or 2 not in classes:
            raise ValueError(f"unexpected router classes: {classes}")
        p_none = proba[:, classes.index(0)].astype(np.float32)
        p_long = proba[:, classes.index(1)].astype(np.float32)
        p_short = proba[:, classes.index(2)].astype(np.float32)
        return p_none, p_long, p_short

    if router_type == "router_ovr_pair":
        long_model_path = Path(meta["long_model_path"])
        short_model_path = Path(meta["short_model_path"])
        long_model = CatBoostClassifier()
        short_model = CatBoostClassifier()
        long_model.load_model(str(long_model_path))
        short_model.load_model(str(short_model_path))
        p_long = np.asarray(long_model.predict_proba(feature_frame), dtype=np.float32)[:, 1]
        p_short = np.asarray(short_model.predict_proba(feature_frame), dtype=np.float32)[:, 1]
        p_none = np.clip(1.0 - np.maximum(p_long, p_short), 0.0, 1.0).astype(np.float32)
        return p_none, p_long.astype(np.float32), p_short.astype(np.float32)

    if router_type == "router4_collapse":
        model = CatBoostClassifier()
        model.load_model(str(router_model_path))
        proba = np.asarray(model.predict_proba(feature_frame), dtype=np.float32)
        classes = list(getattr(model, "classes_", meta.get("classes", [0, 1, 2, 3])))
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        for cls in (0, 1, 2, 3):
            if cls not in class_to_idx:
                raise ValueError(f"unexpected router4 classes: {classes}")
        p_none = (proba[:, class_to_idx[0]] + proba[:, class_to_idx[1]]).astype(np.float32)
        p_long = proba[:, class_to_idx[2]].astype(np.float32)
        p_short = proba[:, class_to_idx[3]].astype(np.float32)
        return p_none, p_long, p_short

    raise ValueError(f"unsupported router meta type: {router_type}")


def _load_aux_feature_frame(paths: Iterable[Path], extra_cols: list[str]) -> pd.DataFrame:
    frames = []
    keep = ["timestamp", *extra_cols]
    keep = list(dict.fromkeys(keep))
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        cols = pd.read_parquet(p, engine="pyarrow").columns
        use_cols = [c for c in keep if c in cols]
        if "timestamp" not in use_cols:
            continue
        frame = pd.read_parquet(p, columns=use_cols)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=keep)
    aux = pd.concat(frames, ignore_index=True)
    aux = aux.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return aux


def _merge_aux_features(df: pd.DataFrame, aux: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    merged = df.copy()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    if not aux.empty:
        aux_cols = [c for c in aux.columns if c != "timestamp"]
        merged = merged.merge(aux, on="timestamp", how="left", suffixes=("", "__aux"))
        for col in aux_cols:
            aux_col = f"{col}__aux"
            if col in merged.columns and aux_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[aux_col])
                merged = merged.drop(columns=[aux_col])
            elif aux_col in merged.columns:
                merged = merged.rename(columns={aux_col: col})
    for col in feature_cols:
        if col not in merged.columns:
            merged[col] = 0.0
    return merged


def score_router_frame(
    input_csv: Path,
    output_csv: Path,
    prefix: str,
    router_model_path: Path | None = None,
    router_meta_path: Path | None = None,
    aux_paths: Iterable[Path] | None = None,
) -> dict:
    router_model_path = Path(router_model_path) if router_model_path is not None else _default_router_model()
    router_meta_path = Path(router_meta_path) if router_meta_path is not None else _default_router_meta()
    aux_paths = list(aux_paths) if aux_paths is not None else _default_aux_paths()
    meta = _load_router_meta(router_meta_path)
    feature_cols = list(meta["feature_cols"])
    extra_cols = list(feature_cols)
    extra_cols.extend(["regime_whipsaw", "regime_bull", "regime_bear", "regime_chop"])
    extra_cols = list(dict.fromkeys(extra_cols))
    aux = _load_aux_feature_frame(aux_paths, extra_cols)

    df = pd.read_csv(input_csv)
    if "timestamp" not in df.columns:
        raise ValueError(f"timestamp column missing in {input_csv}")
    df = _merge_aux_features(df, aux, feature_cols)
    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    p_none, p_long, p_short = _score_router_proba(meta, X, router_model_path)
    prob_max = np.maximum(p_long, p_short)
    edge = p_long - p_short

    whipsaw_prob = None
    if "clean_regime4_2024_unsup_v1_whipsaw_prob" in df.columns:
        whipsaw_prob = pd.to_numeric(df["clean_regime4_2024_unsup_v1_whipsaw_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    elif "regime_whipsaw" in df.columns:
        whipsaw_prob = pd.to_numeric(df["regime_whipsaw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        whipsaw_prob = np.zeros(len(df), dtype=np.float32)

    available = (whipsaw_prob < 0.55).astype(np.float32)
    none_prob = p_none.astype(np.float32)
    margin = np.abs(edge).astype(np.float32)
    router_type = str(meta.get("type", "router3"))
    if router_type == "router_ensemble":
        router_side = np.where((p_none >= p_long) & (p_none >= p_short), 0, np.where(edge > 0.0, 1, np.where(edge < 0.0, -1, 0))).astype(np.int8)
    elif router_type == "router_ovr_pair":
        long_threshold = float(meta.get("long_threshold", 0.5))
        short_threshold = float(meta.get("short_threshold", 0.5))
        min_margin = float(meta.get("min_margin", 0.0))
        router_side = np.zeros(len(df), dtype=np.int8)
        long_ok = (p_long >= long_threshold) & ((p_long - p_short) >= min_margin)
        short_ok = (p_short >= short_threshold) & ((p_short - p_long) >= min_margin)
        router_side[long_ok & ~short_ok] = 1
        router_side[short_ok & ~long_ok] = -1
        ties = long_ok & short_ok
        router_side[ties & (edge > 0.0)] = 1
        router_side[ties & (edge < 0.0)] = -1
    else:
        router_side = np.where((p_none >= p_long) & (p_none >= p_short), 0, np.where(edge > 0.0, 1, np.where(edge < 0.0, -1, 0))).astype(np.int8)

    df[f"{prefix}_available"] = available
    df[f"{prefix}_none_prob"] = none_prob
    df[f"{prefix}_long_prob"] = p_long
    df[f"{prefix}_short_prob"] = p_short
    df[f"{prefix}_prob_max"] = prob_max.astype(np.float32)
    df[f"{prefix}_edge"] = edge.astype(np.float32)
    df[f"{prefix}_margin"] = margin
    df[f"{prefix}_side"] = router_side
    df[f"{prefix}_whipsaw_prob"] = whipsaw_prob.astype(np.float32)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "router_model": str(router_model_path),
        "router_meta": str(router_meta_path),
        "aux_paths": [str(Path(x)) for x in aux_paths],
        "rows": int(len(df)),
        "feature_cols": feature_cols,
        "prefix": prefix,
        "available_ratio": float(np.mean(available)),
        "long_prob_mean": float(np.mean(p_long)),
        "short_prob_mean": float(np.mean(p_short)),
        "margin_mean": float(np.mean(margin)),
        "input_mtime": float(input_csv.stat().st_mtime) if input_csv.exists() else None,
    }
    with open(output_csv.with_suffix(".router_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score RL CSV with alpha5 direction router and emit event-filter columns.")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--prefix", default="a5dir")
    p.add_argument("--router-model", default=None)
    p.add_argument("--router-meta", default=None)
    p.add_argument("--aux-parquet", action="append", default=None, help="Optional alpha5 parquet source; repeatable.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aux_paths = [Path(x) for x in args.aux_parquet] if args.aux_parquet else None
    summary = score_router_frame(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        prefix=str(args.prefix),
        router_model_path=(Path(args.router_model) if args.router_model else None),
        router_meta_path=(Path(args.router_meta) if args.router_meta else None),
        aux_paths=aux_paths,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
