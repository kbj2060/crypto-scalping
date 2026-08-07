#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_ai_patchmix_direction_core_20260530 as patchmix  # noqa: E402
from scripts.sweep_ai_patchmix_h6_label_params_20260530 import _class_weights, _fit_values, _labels  # noqa: E402


DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/patchtsmixer_channel_profile_compare_20260530"


CORE_FEATURES = (
    *patchmix.BASE_CORE_FEATURES,
    *patchmix.AUDITED_COMPACT_FEATURES,
    *patchmix.LOCAL_REGIME_FEATURES,
)

COMPACT_24 = (
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "atr14_pct",
    "realized_vol_24",
    "compression_ratio",
    "funding_pressure",
    "oi_change_rate",
    "funding_roc_288",
    "long_squeeze_risk",
    "crowding_pressure",
    "crowded_short_squeeze_risk",
    "crowded_long_unwind_risk",
    "smart_money_flow",
    "ofi_acceleration",
    "net_taker_ratio",
    "taker_acceleration",
    "cvp_volume_imbalance",
    "vwap_dev_48",
    "price_cvd_divergence",
    "btc_lead_eth_follow_gap_3",
    "regime_trending",
)

GROUPED_EXTRA_7 = (
    "ret_6",
    "atr14_pct",
    "funding_pressure",
    "oi_change_rate",
    "cvp_volume_imbalance",
    "price_cvd_divergence",
    "regime_trending",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare PatchTSMixer sequence-channel profiles under fixed heads/labels.")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patch-model-id", default="ibm/patchtsmixer-etth1-pretrain")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--iterations", type=int, default=850)
    p.add_argument("--learning-rate", type=float, default=0.025)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--l2-leaf-reg", type=float, default=12.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--random-seed", type=int, default=20260530)
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _core(frame: pd.DataFrame) -> pd.DataFrame:
    patchmix.CORE_FEATURES = CORE_FEATURES
    return patchmix._core_features(frame)


def _channels(core: pd.DataFrame, profile: str) -> pd.DataFrame:
    if profile == "current_13ch":
        return patchmix._patch_channels(core)
    if profile == "raw_core_40ch":
        return core.loc[:, list(CORE_FEATURES)].astype("float32")
    if profile == "compact_24ch":
        return core.loc[:, list(COMPACT_24)].astype("float32")
    if profile == "grouped_20ch":
        current = patchmix._patch_channels(core).reset_index(drop=True)
        extra = core.loc[:, list(GROUPED_EXTRA_7)].reset_index(drop=True).astype("float32")
        extra.columns = [f"raw_{c}" for c in extra.columns]
        return pd.concat([current, extra], axis=1).astype("float32")
    raise ValueError(profile)


def _embeddings(
    frame: pd.DataFrame,
    *,
    profile: str,
    args: argparse.Namespace,
    out_path: Path,
    device: torch.device,
) -> pd.DataFrame:
    if out_path.exists():
        cached = pd.read_csv(out_path)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], errors="raise")
        return cached
    from transformers import PatchTSMixerModel

    model = PatchTSMixerModel.from_pretrained(str(args.patch_model_id), local_files_only=True).eval().to(device)
    core = _core(frame)
    channels = _channels(core, profile)
    values = channels.to_numpy(dtype=np.float32)
    indices = patchmix._refresh_indices(len(frame), int(args.context_length), int(args.stride))
    if indices.size == 0:
        raise ValueError(f"not enough rows for context_length={args.context_length}")
    emb_cols = [f"_patch_emb_{i:02d}" for i in range(int(args.emb_dim))]
    out = pd.DataFrame(np.nan, index=frame.index, columns=emb_cols, dtype="float32")
    with torch.no_grad():
        for start in range(0, len(indices), max(1, int(args.batch_size))):
            batch_idx = indices[start : start + int(args.batch_size)]
            windows = np.stack([values[i - int(args.context_length) : i] for i in batch_idx], axis=0)
            x = torch.as_tensor(windows, dtype=torch.float32, device=device)
            pred = model(past_values=x, return_dict=True).last_hidden_state
            emb = pred.mean(dim=(1, 2)).detach().cpu().numpy()
            out.loc[batch_idx, emb_cols] = emb[:, : int(args.emb_dim)].astype("float32")
    out[emb_cols] = out[emb_cols].ffill().fillna(0.0)
    result = pd.concat([frame[["timestamp"]].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    return result


def _merge_exact(base: pd.DataFrame, feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    got = base[["timestamp"]].merge(feat[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    bad = [c for c in cols if got[c].replace([np.inf, -np.inf], np.nan).isna().any()]
    if bad:
        raise RuntimeError(f"exact timestamp merge produced missing values: {bad}")
    return got[cols].astype("float32")


def _fit_head(
    *,
    profile: str,
    horizon: int,
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_x: pd.DataFrame,
    score_x: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    label_cfg = {"min_edge": 0.0012, "atr_mult": 0.18, "mae_penalty": 0.40, "cost": 0.00055, "margin": 0.00025}
    lab_train = _labels(train, horizon=int(horizon), **label_cfg)
    lab_score = _labels(score, horizon=int(horizon), **label_cfg)
    cols = list(train_x.columns)
    fill = _fit_values(train_x, cols)
    data = pd.concat([train_x.reset_index(drop=True), lab_train.reset_index(drop=True)], axis=1)
    data = data[data["valid"] > 0].reset_index(drop=True)
    split = int(len(data) * 0.82)
    fit_df = data.iloc[:split].reset_index(drop=True)
    hold_df = data.iloc[split:].reset_index(drop=True)
    x_fit = patchmix._matrix(fit_df, cols, fill)
    y_fit = fit_df["label"].to_numpy(dtype=np.int64)
    x_hold = patchmix._matrix(hold_df, cols, fill)
    y_hold = hold_df["label"].to_numpy(dtype=np.int64)
    valid = lab_score["valid"].to_numpy() > 0
    x_score = patchmix._matrix(score_x.loc[valid].reset_index(drop=True), cols, fill)
    y_score = lab_score.loc[valid, "label"].to_numpy(dtype=np.int64)

    params = dict(
        loss_function="MultiClass",
        eval_metric="TotalF1",
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        depth=int(args.depth),
        l2_leaf_reg=float(args.l2_leaf_reg),
        random_seed=int(args.random_seed) + int(horizon) + abs(hash(profile)) % 1000,
        task_type=str(args.task_type),
        class_weights=_class_weights(y_fit, float(args.class_weight_power)),
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
    )
    model = CatBoostClassifier(**params)
    try:
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
    except Exception:
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)

    hold_p = np.asarray(model.predict_proba(x_hold), dtype=np.float64)
    score_p = np.asarray(model.predict_proba(x_score), dtype=np.float64)
    pred = np.argmax(score_p, axis=1)
    model_path = args.out_dir / f"{profile}_h{horizon}.cbm"
    model.save_model(model_path)
    result: dict[str, Any] = {
        "profile": profile,
        "horizon": int(horizon),
        "channel_count": int(train_x.attrs.get("channel_count", -1)),
        "feature_count": int(len(cols)),
        "model_path": str(model_path),
        "best_iteration": int(model.get_best_iteration() or 0),
        "hold_bacc": float(balanced_accuracy_score(y_hold, np.argmax(hold_p, axis=1))),
        "score_bacc": float(balanced_accuracy_score(y_score, pred)),
        "score_pred_counts": np.bincount(pred, minlength=3).astype(int).tolist(),
        "score_confusion": confusion_matrix(y_score, pred, labels=[0, 1, 2]).astype(int).tolist(),
    }
    try:
        result["score_auc"] = float(roc_auc_score(y_score, score_p, multi_class="ovr"))
    except Exception:
        result["score_auc"] = None
    return result


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and str(args.task_type) == "GPU" else "cpu")
    train = patchmix._read_frame(args.train_csv)
    score = patchmix._read_frame(args.score_csv)
    core_train = _core(train).reset_index(drop=True)
    core_score = _core(score).reset_index(drop=True)
    emb_cols = [f"_patch_emb_{i:02d}" for i in range(int(args.emb_dim))]
    profiles = ("current_13ch", "raw_core_40ch", "compact_24ch", "grouped_20ch")
    results = []
    profile_meta: dict[str, Any] = {}
    for profile in profiles:
        train_emb = _embeddings(
            train,
            profile=profile,
            args=args,
            out_path=args.out_dir / profile / "emb_train.csv",
            device=device,
        )
        score_emb = _embeddings(
            score,
            profile=profile,
            args=args,
            out_path=args.out_dir / profile / "emb_score.csv",
            device=device,
        )
        channel_cols = list(_channels(core_train, profile).columns)
        train_x = pd.concat([core_train.reset_index(drop=True), _merge_exact(train, train_emb, emb_cols).reset_index(drop=True)], axis=1)
        score_x = pd.concat([core_score.reset_index(drop=True), _merge_exact(score, score_emb, emb_cols).reset_index(drop=True)], axis=1)
        train_x.attrs["channel_count"] = len(channel_cols)
        score_x.attrs["channel_count"] = len(channel_cols)
        profile_meta[profile] = {"channels": channel_cols, "channel_count": len(channel_cols)}
        for horizon in (6, 12):
            rec = _fit_head(profile=profile, horizon=horizon, train=train, score=score, train_x=train_x, score_x=score_x, args=args)
            results.append(rec)
            print(json.dumps(rec, ensure_ascii=False, default=_json_default), flush=True)
    summary = {
        "type": "patchtsmixer_channel_profile_compare_20260530",
        "contract": "Only PatchTSMixer sequence-channel profile changes; CatBoost head/core/label setup fixed.",
        "label_config": {"min_edge": 0.0012, "atr_mult": 0.18, "mae_penalty": 0.40, "cost": 0.00055, "margin": 0.00025},
        "core_features": list(CORE_FEATURES),
        "profiles": profile_meta,
        "results": sorted(results, key=lambda x: (int(x["horizon"]), -float(x["score_bacc"]))),
        "best_by_horizon": {},
    }
    for horizon in (6, 12):
        got = [r for r in results if int(r["horizon"]) == horizon]
        summary["best_by_horizon"][f"h{horizon}"] = max(got, key=lambda x: float(x["score_bacc"]))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
