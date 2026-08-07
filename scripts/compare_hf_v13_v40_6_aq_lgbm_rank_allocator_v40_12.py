#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos import Chronos2Pipeline  # noqa: E402
from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    prepare_features,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import (  # noqa: E402
    _audit_contract,
    _close,
    _fill_price,
    _feature_cols,
    _json_default,
    _read,
    backtest_policy_frame,
)
from scripts.train_eval_hf_v13_multitrack_foundation_parent_v40 import (  # noqa: E402
    CHRONOS_MODEL,
    KAIROS_MODEL,
    MACRO_COLS,
    MACRO_LEN,
    MICRO_LEN,
    _embedding_cache_path,
    _extract_macro_embeddings,
    _extract_micro_embeddings,
    _parent_cfg,
)
from tsfm.model.kairos import AutoModel as KairosAutoModel  # noqa: E402


MODEL_ID = "hf_v13_tree_vs_foundation_aq_lgbm_rank_allocator_v40_12_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_V40_6_REPORT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_summary.json"
DEFAULT_V40_6_BUNDLE = ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512/target_aware_full_bundle.pkl"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_aq_lgbm_rank_allocator_v40_12_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_aq_lgbm_rank_allocator_v40_12_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_aq_lgbm_rank_allocator_v40_12_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_aq_lgbm_rank_allocator_v40_12_20260512_grid.csv"
V40_6_CACHE_CONSUMER = "hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512"
BUCKET_KEYS = ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")


@dataclass(frozen=True)
class TreeHParams:
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    cash_weight: float

    @property
    def name(self) -> str:
        return (
            f"hgb_mi{self.max_iter}_lr{str(self.learning_rate).replace('.', 'p')}"
            f"_leaf{self.max_leaf_nodes}_l2{str(self.l2_regularization).replace('.', 'p')}"
            f"_cw{str(self.cash_weight).replace('.', 'p')}"
        )


@dataclass(frozen=True)
class BucketHParams:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    reg_alpha: float
    reg_lambda: float
    path_smooth: float
    extra_trees: bool

    @property
    def name(self) -> str:
        return (
            f"lgbm_ne{self.n_estimators}_lr{str(self.learning_rate).replace('.', 'p')}"
            f"_leaves{self.num_leaves}_d{self.max_depth}_child{self.min_child_samples}"
            f"_a{str(self.reg_alpha).replace('.', 'p')}_l{str(self.reg_lambda).replace('.', 'p')}"
            f"_ps{str(self.path_smooth).replace('.', 'p')}_{'et1' if self.extra_trees else 'et0'}"
        )


@dataclass(frozen=True)
class RankHParams:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    reg_alpha: float
    reg_lambda: float
    path_smooth: float
    extra_trees: bool
    adverse_mult: float
    size_mult: float
    hold_mult: float

    @property
    def name(self) -> str:
        return (
            f"rank_ne{self.n_estimators}_lr{str(self.learning_rate).replace('.', 'p')}"
            f"_leaves{self.num_leaves}_d{self.max_depth}_child{self.min_child_samples}"
            f"_a{str(self.reg_alpha).replace('.', 'p')}_l{str(self.reg_lambda).replace('.', 'p')}"
            f"_ps{str(self.path_smooth).replace('.', 'p')}_{'et1' if self.extra_trees else 'et0'}"
            f"_adv{str(self.adverse_mult).replace('.', 'p')}"
            f"_sz{str(self.size_mult).replace('.', 'p')}"
            f"_hold{str(self.hold_mult).replace('.', 'p')}"
        )


def _load_pickle(path: Path) -> dict[str, Any]:
    try:
        obj = joblib.load(path)
    except Exception:
        with path.open("rb") as f:
            obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a dict bundle")
    return obj


def _hgb_classifier(seed: int, hp: TreeHParams) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=int(hp.max_iter),
            learning_rate=float(hp.learning_rate),
            max_leaf_nodes=int(hp.max_leaf_nodes),
            l2_regularization=float(hp.l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _hgb_regressor(seed: int, hp: TreeHParams) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=int(hp.max_iter),
            learning_rate=float(hp.learning_rate),
            max_leaf_nodes=int(hp.max_leaf_nodes),
            l2_regularization=float(hp.l2_regularization),
            early_stopping=False,
            random_state=int(seed),
        ),
    )


def _weighted_fit_classifier(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Any:
    if np.unique(y).size < 2:
        return None
    model.fit(x, y, histgradientboostingclassifier__sample_weight=weights)
    return model


def _train_aq_model(
    x: pd.DataFrame,
    y: dict[str, np.ndarray],
    *,
    cfg: FullyLearnedGovernorConfig,
    hp: TreeHParams,
    random_state: int,
    feature_cols: list[str],
) -> dict[str, Any]:
    action_weights = np.where(np.asarray(y["action"]) == ACTION_CASH, float(hp.cash_weight), 1.0)
    quality_weights = np.clip(np.abs(np.asarray(y["quality"], dtype=np.float64)), 0.03, 1.0)
    weights = np.maximum(action_weights, quality_weights)
    action_model = _weighted_fit_classifier(_hgb_classifier(random_state, hp), x, np.asarray(y["action"]), weights)
    quality_model = _hgb_regressor(random_state + 99, hp)
    quality_model.fit(x, np.asarray(y["quality"], dtype=np.float64), histgradientboostingregressor__sample_weight=weights)
    return {
        "model_type": "v40_6_action_quality_only_v1",
        "feature_cols": list(feature_cols),
        "config": asdict(cfg),
        "action_quality_hparams": asdict(hp),
        "action_model": action_model,
        "quality_model": quality_model,
    }


def _projection_targets(y: dict[str, np.ndarray]) -> np.ndarray:
    action = np.asarray(y["action"], dtype=np.int64)
    side = np.where(action == ACTION_LONG, 1.0, np.where(action == ACTION_SHORT, -1.0, 0.0)).astype(np.float32)
    quality = np.asarray(y["quality"], dtype=np.float32)
    signed_quality = side * np.clip(np.abs(quality), 0.0, None)
    trade_flag = (action != ACTION_CASH).astype(np.float32)
    return np.column_stack([side, signed_quality, trade_flag]).astype(np.float32)


def _fit_pls_scores(train_x: np.ndarray, train_y: np.ndarray, apply_x: np.ndarray, *, n_components: int) -> tuple[np.ndarray, PLSRegression]:
    model = PLSRegression(n_components=int(n_components), scale=True)
    model.fit(train_x, train_y)
    scores = model.transform(apply_x)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores.astype(np.float32), model


def _add_embedding_cols(base: pd.DataFrame, prefix: str, values: np.ndarray) -> pd.DataFrame:
    out = base.reset_index(drop=True).copy()
    for j in range(values.shape[1]):
        out[f"{prefix}_{j:03d}"] = values[:, j].astype(np.float32)
    return out


def _extract_or_load_embeddings(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    eval_feat: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    macro_cols: list[str],
    micro_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    emb_dir = ROOT / "data/ensemble/supervised/hf_v13_multitrack_foundation_parent_v40_20260512" / "embeddings"
    paths = {
        "train_macro": _embedding_cache_path(
            emb_dir,
            prefix="tree_train_macro",
            model_name=CHRONOS_MODEL,
            frame=train_df,
            indices=train_idx,
            cols=[c for c in macro_cols if c in train_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|stride={args.train_stride}|split=train|consumer={V40_6_CACHE_CONSUMER}",
        ),
        "val_macro": _embedding_cache_path(
            emb_dir,
            prefix="val_macro",
            model_name=CHRONOS_MODEL,
            frame=val_df,
            indices=val_idx,
            cols=[c for c in macro_cols if c in val_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|split=val|consumer={V40_6_CACHE_CONSUMER}",
        ),
        "eval_macro": _embedding_cache_path(
            emb_dir,
            prefix="eval_macro",
            model_name=CHRONOS_MODEL,
            frame=eval_df,
            indices=eval_idx,
            cols=[c for c in macro_cols if c in eval_df.columns],
            window_len=MACRO_LEN,
            extra_tag=f"csv={args.eval_csv.name}|split=eval|consumer={V40_6_CACHE_CONSUMER}",
        ),
        "train_micro": _embedding_cache_path(
            emb_dir,
            prefix="tree_train_micro",
            model_name=KAIROS_MODEL,
            frame=train_feat,
            indices=train_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|stride={args.train_stride}|split=train|consumer={V40_6_CACHE_CONSUMER}",
        ),
        "val_micro": _embedding_cache_path(
            emb_dir,
            prefix="val_micro",
            model_name=KAIROS_MODEL,
            frame=val_feat,
            indices=val_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={args.train_csv.name}|split=val|consumer={V40_6_CACHE_CONSUMER}",
        ),
        "eval_micro": _embedding_cache_path(
            emb_dir,
            prefix="eval_micro",
            model_name=KAIROS_MODEL,
            frame=eval_feat,
            indices=eval_idx,
            cols=micro_cols,
            window_len=MICRO_LEN,
            extra_tag=f"csv={args.eval_csv.name}|split=eval|consumer={V40_6_CACHE_CONSUMER}",
        ),
    }
    if all(p.exists() for p in paths.values()):
        return {k: np.load(p) for k, p in paths.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{MODEL_ID}] loading Chronos/Kairos on {device} for missing caches", flush=True)
    chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)
    kairos = KairosAutoModel.from_pretrained(KAIROS_MODEL, trust_remote_code=True).to(device).eval()
    return {
        "train_macro": _extract_macro_embeddings(chronos, train_df, train_idx, cache_path=paths["train_macro"], batch_size=args.embed_batch),
        "val_macro": _extract_macro_embeddings(chronos, val_df, val_idx, cache_path=paths["val_macro"], batch_size=args.embed_batch),
        "eval_macro": _extract_macro_embeddings(chronos, eval_df, eval_idx, cache_path=paths["eval_macro"], batch_size=args.embed_batch),
        "train_micro": _extract_micro_embeddings(kairos, train_feat, train_idx, micro_cols, cache_path=paths["train_micro"], batch_size=args.embed_batch),
        "val_micro": _extract_micro_embeddings(kairos, val_feat, val_idx, micro_cols, cache_path=paths["val_micro"], batch_size=args.embed_batch),
        "eval_micro": _extract_micro_embeddings(kairos, eval_feat, eval_idx, micro_cols, cache_path=paths["eval_micro"], batch_size=args.embed_batch),
    }


def _aq_outputs(aq_bundle: dict[str, Any], encoded_feat: pd.DataFrame) -> pd.DataFrame:
    cols = list(aq_bundle.get("feature_cols") or [])
    x = encoded_feat.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).copy()
    if "side_hint" in x.columns:
        x["side_hint"] = 0.0
    proba = aq_bundle["action_model"].predict_proba(x)
    classes = np.asarray(aq_bundle["action_model"].classes_, dtype=int)
    p_cash = proba[:, np.flatnonzero(classes == ACTION_CASH)[0]] if np.any(classes == ACTION_CASH) else np.zeros(len(x))
    p_long = proba[:, np.flatnonzero(classes == ACTION_LONG)[0]] if np.any(classes == ACTION_LONG) else np.zeros(len(x))
    p_short = proba[:, np.flatnonzero(classes == ACTION_SHORT)[0]] if np.any(classes == ACTION_SHORT) else np.zeros(len(x))
    action_idx = np.argmax(proba, axis=1)
    action = classes[action_idx]
    side = np.where(action == ACTION_LONG, 1.0, np.where(action == ACTION_SHORT, -1.0, 0.0))
    top2 = np.sort(proba, axis=1)[:, -2:] if proba.shape[1] >= 2 else np.column_stack([np.zeros(len(x)), np.max(proba, axis=1)])
    action_conf = np.max(proba, axis=1)
    action_margin = top2[:, 1] - top2[:, 0]
    entropy = -np.sum(np.clip(proba, 1e-12, 1.0) * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)
    quality = aq_bundle["quality_model"].predict(x)
    out = pd.DataFrame(
        {
            "aq_p_cash": p_cash.astype(np.float64),
            "aq_p_long": p_long.astype(np.float64),
            "aq_p_short": p_short.astype(np.float64),
            "aq_trade_prob": (p_long + p_short).astype(np.float64),
            "aq_dir_edge": (p_long - p_short).astype(np.float64),
            "aq_action": action.astype(np.float64),
            "aq_side": side.astype(np.float64),
            "aq_action_conf": action_conf.astype(np.float64),
            "aq_action_margin": action_margin.astype(np.float64),
            "aq_action_entropy": entropy.astype(np.float64),
            "aq_quality": np.asarray(quality, dtype=np.float64),
            "aq_abs_quality": np.abs(quality).astype(np.float64),
            "aq_quality_x_side": (quality * side).astype(np.float64),
            "aq_quality_x_trade_prob": (quality * (p_long + p_short)).astype(np.float64),
        },
        index=encoded_feat.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _bucket_hp_grid() -> list[BucketHParams]:
    return [
        BucketHParams(180, 0.030, 15, 4, 40, 0.20, 1.00, 1.0, True),
        BucketHParams(240, 0.020, 31, -1, 60, 0.50, 2.00, 3.0, True),
        BucketHParams(300, 0.015, 31, -1, 80, 0.80, 3.00, 5.0, True),
        BucketHParams(220, 0.025, 15, 4, 80, 1.00, 3.00, 5.0, False),
    ]


def _lgbm_classifier(seed: int, hp: BucketHParams) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=int(hp.n_estimators),
        learning_rate=float(hp.learning_rate),
        num_leaves=int(hp.num_leaves),
        max_depth=int(hp.max_depth),
        min_child_samples=int(hp.min_child_samples),
        reg_alpha=float(hp.reg_alpha),
        reg_lambda=float(hp.reg_lambda),
        path_smooth=float(hp.path_smooth),
        extra_trees=bool(hp.extra_trees),
        use_missing=True,
        zero_as_missing=False,
        feature_pre_filter=False,
        verbosity=-1,
        n_jobs=-1,
        random_state=int(seed),
    )


def _lgbm_ranker(seed: int, hp: RankHParams) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=int(hp.n_estimators),
        learning_rate=float(hp.learning_rate),
        num_leaves=int(hp.num_leaves),
        max_depth=int(hp.max_depth),
        min_child_samples=int(hp.min_child_samples),
        reg_alpha=float(hp.reg_alpha),
        reg_lambda=float(hp.reg_lambda),
        path_smooth=float(hp.path_smooth),
        extra_trees=bool(hp.extra_trees),
        use_missing=True,
        zero_as_missing=False,
        feature_pre_filter=False,
        verbosity=-1,
        n_jobs=-1,
        random_state=int(seed),
    )


def _rank_hp_grid() -> list[RankHParams]:
    return [
        RankHParams(260, 0.030, 31, -1, 60, 0.20, 1.00, 1.0, True, 1.00, 1.00, 1.00),
        RankHParams(360, 0.020, 31, -1, 80, 0.50, 2.00, 3.0, True, 1.25, 1.00, 1.00),
        RankHParams(420, 0.018, 63, -1, 80, 0.60, 2.50, 5.0, True, 1.50, 1.25, 1.00),
        RankHParams(320, 0.025, 31, 5, 60, 0.50, 2.00, 3.0, False, 1.15, 0.80, 1.30),
    ]


def _stage2_context_cols(encoded_cols: list[str]) -> list[str]:
    keep_prefixes = ("macro_factor_", "micro_factor_", "m7_", "clean_regime_2024_unsup_v4_")
    keep_exact = {
        "side_hint",
        "log_return",
        "volatility_z",
        "rogers_satchell_vol",
        "amihud_illiquidity_z",
        "funding_pressure",
        "funding_abs",
        "smart_money_flow",
        "net_taker_ratio",
        "taker_acceleration",
        "ofi_acceleration",
        "ai_dir_edge",
        "ai_dir_p_up",
        "ai_dir_p_down",
        "ai_dir_entropy",
        "ai_adverse_risk",
        "ai_reward_risk",
        "ai_vol_regime_pct",
        "ai_flow_pressure",
        "ai_flow_exhaustion",
        "ai_flow_flip_prob",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "rsi",
        "trade_intensity",
        "big_trade_ratio",
        "whale_retail_ratio",
        "squeeze_power",
        "breakout_strength",
    }
    out: list[str] = []
    for col in encoded_cols:
        if col in keep_exact or any(col.startswith(p) for p in keep_prefixes):
            out.append(col)
    return out


def _stage1_from_full_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": "v40_6_original_action_quality_only_view",
        "feature_cols": list(bundle.get("feature_cols") or []),
        "config": dict(bundle.get("config") or {}),
        "source_model_type": bundle.get("model_type"),
        "source_hparams": bundle.get("tuned_tree_hparams"),
        "action_model": bundle["action_model"],
        "quality_model": bundle["quality_model"],
    }


def _candidate_grid(cfg: FullyLearnedGovernorConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tp_values = [v for v in cfg.take_profit_buckets if v <= 0.45]
    hold_values = [v for v in cfg.max_hold_buckets if 6 <= int(v) <= 288]
    for ni, notional in enumerate(cfg.notional_buckets):
        for li, leverage in enumerate(cfg.leverage_buckets):
            margin = float(notional) / max(float(leverage), 1e-12)
            if margin > float(cfg.max_margin_fraction) + 1e-9:
                continue
            for ti, tp in enumerate(cfg.take_profit_buckets):
                if float(tp) not in tp_values:
                    continue
                for si, sl in enumerate(cfg.stop_loss_buckets):
                    for hi, hold in enumerate(cfg.max_hold_buckets):
                        if int(hold) not in hold_values:
                            continue
                        rows.append(
                            {
                                "notional_idx": int(ni),
                                "leverage_idx": int(li),
                                "take_profit_idx": int(ti),
                                "stop_loss_idx": int(si),
                                "max_hold_idx": int(hi),
                                "cand_notional": float(notional),
                                "cand_leverage": float(leverage),
                                "cand_take_profit": float(tp),
                                "cand_stop_loss": float(sl),
                                "cand_max_hold": float(hold),
                                "cand_margin": float(margin),
                                "cand_log_hold": float(np.log1p(float(hold))),
                                "cand_risk_per_margin": float(float(sl) * float(notional) / max(margin, 1e-12)),
                            }
                        )
    return pd.DataFrame(rows)


def _future_return_matrix(frame: pd.DataFrame, indices: np.ndarray, horizon: int) -> np.ndarray:
    close = _close(frame)
    idx = np.asarray(indices, dtype=np.int64)
    steps = np.arange(1, int(horizon) + 1, dtype=np.int64)
    fut_idx = np.minimum(idx[:, None] + steps[None, :], len(close) - 1)
    return close[fut_idx] / np.maximum(close[idx][:, None], 1e-12) - 1.0


def _score_candidate_grid(raw_ret: np.ndarray, side: float, candidates: pd.DataFrame, cfg: FullyLearnedGovernorConfig, hp: RankHParams) -> np.ndarray:
    side_ret = np.nan_to_num(np.asarray(raw_ret, dtype=np.float64) * float(side), nan=0.0, posinf=0.0, neginf=0.0)
    notional = candidates["cand_notional"].to_numpy(dtype=np.float64)
    leverage = candidates["cand_leverage"].to_numpy(dtype=np.float64)
    tp = candidates["cand_take_profit"].to_numpy(dtype=np.float64)
    sl = candidates["cand_stop_loss"].to_numpy(dtype=np.float64)
    hold = candidates["cand_max_hold"].to_numpy(dtype=np.int64)
    exp_path = side_ret[None, :] * notional[:, None]
    hit = (exp_path >= tp[:, None]) | (exp_path <= -np.abs(sl[:, None]))
    has_hit = hit.any(axis=1)
    first_hit = np.where(has_hit, hit.argmax(axis=1), exp_path.shape[1] - 1).astype(np.int64)
    hold_i = np.minimum(np.maximum(hold - 1, 0), exp_path.shape[1] - 1).astype(np.int64)
    exit_i = np.minimum(first_hit, hold_i)
    row_idx = np.arange(len(candidates))
    cum_min = np.minimum.accumulate(exp_path, axis=1)
    pnl = exp_path[row_idx, exit_i] - 2.0 * float(cfg.fee + cfg.slip) * notional
    adverse = np.maximum(0.0, -cum_min[row_idx, exit_i])
    liq_buffer = 0.70 / np.maximum(leverage, 1.0)
    liq_penalty = 2.5 * np.maximum(0.0, adverse - liq_buffer)
    hold_frac = (exit_i.astype(np.float64) + 1.0) / max(float(cfg.max_train_horizon_bars), 1.0)
    max_notional = max(float(v) for v in cfg.notional_buckets)
    score = (
        pnl
        - float(cfg.adverse_penalty) * float(hp.adverse_mult) * adverse
        - float(cfg.size_penalty) * float(hp.size_mult) * (notional / max_notional) ** 2
        - float(cfg.hold_penalty) * float(hp.hold_mult) * hold_frac
        - liq_penalty
        + float(cfg.turnover_bonus) / np.maximum(exit_i.astype(np.float64) + 1.0, 1.0) ** 0.35
    )
    return score.astype(np.float32)


def _rank_feature_frame(base: pd.DataFrame, candidates: pd.DataFrame, *, sample_idx: np.ndarray | None = None) -> pd.DataFrame:
    cand = candidates if sample_idx is None else candidates.iloc[np.asarray(sample_idx, dtype=np.int64)]
    base_rep = pd.DataFrame(np.repeat(base.to_numpy(dtype=np.float32), len(cand), axis=0), columns=list(base.columns))
    cand_rep = pd.concat([cand.reset_index(drop=True)] * len(base), ignore_index=True)
    out = pd.concat([base_rep.reset_index(drop=True), cand_rep.reset_index(drop=True)], axis=1)
    out["rank_quality_x_notional"] = out["aq_quality"].astype(float) * out["cand_notional"].astype(float)
    out["rank_trade_prob_x_notional"] = out["aq_trade_prob"].astype(float) * out["cand_notional"].astype(float)
    out["rank_margin_x_conf"] = out["aq_action_margin"].astype(float) * out["cand_margin"].astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def _fit_cooldown_model(x: pd.DataFrame, y: dict[str, np.ndarray], *, mask: np.ndarray, hp: RankHParams, seed: int) -> Any:
    yy = np.asarray(y["cooldown"])[mask]
    if np.unique(yy).size < 2:
        return None
    weights = np.clip(np.abs(np.asarray(y["quality"], dtype=np.float64))[mask], 0.03, 1.0)
    model = LGBMClassifier(
        n_estimators=max(120, int(hp.n_estimators // 2)),
        learning_rate=float(hp.learning_rate),
        num_leaves=int(hp.num_leaves),
        max_depth=int(hp.max_depth),
        min_child_samples=int(hp.min_child_samples),
        reg_alpha=float(hp.reg_alpha),
        reg_lambda=float(hp.reg_lambda),
        path_smooth=float(hp.path_smooth),
        extra_trees=bool(hp.extra_trees),
        use_missing=True,
        zero_as_missing=False,
        verbosity=-1,
        n_jobs=-1,
        random_state=int(seed),
    )
    model.fit(x.loc[mask], yy, sample_weight=weights)
    return model

def _stage2_feature_frame(
    aq: pd.DataFrame,
    encoded_feat: pd.DataFrame,
    *,
    mode: str,
    context_cols: list[str],
) -> pd.DataFrame:
    aq_reset = aq.reset_index(drop=True).copy()
    if mode == "aq_only":
        return aq_reset
    if mode != "aq_plus_context":
        raise ValueError(f"unknown stage2 feature mode: {mode}")
    ctx = encoded_feat.reset_index(drop=True).reindex(columns=context_cols).replace([np.inf, -np.inf], np.nan).copy()
    if "side_hint" in context_cols and "side_hint" in ctx.columns:
        ctx["side_hint"] = aq_reset["aq_side"].to_numpy(dtype=np.float64)
    ctx.columns = [f"ctx_{i:03d}" for i in range(len(context_cols))]
    side_ctx = ctx.multiply(aq_reset["aq_side"].to_numpy(dtype=np.float64), axis=0)
    side_ctx.columns = [f"sidectx_{i:03d}" for i in range(len(context_cols))]
    return pd.concat([aq_reset, ctx, side_ctx], axis=1)


def _fit_bucket_allocator(
    x: pd.DataFrame,
    y: dict[str, np.ndarray],
    *,
    cfg: FullyLearnedGovernorConfig,
    hp: BucketHParams,
    random_state: int,
    feature_mode: str,
    context_cols: list[str],
) -> dict[str, Any]:
    trade_mask = np.asarray(y["action"]) != ACTION_CASH
    x_trade = x.loc[trade_mask].copy()
    weights = np.clip(np.abs(np.asarray(y["quality"], dtype=np.float64))[trade_mask], 0.03, 1.0)
    default_bucket_indexes = {
        key: int(pd.Series(np.asarray(y[key])[trade_mask]).mode().iloc[0]) if np.any(trade_mask) else 0
        for key in BUCKET_KEYS
    }
    bundle: dict[str, Any] = {
        "model_type": "action_quality_lightgbm_bucket_allocator_v1",
        "feature_cols": list(x.columns),
        "feature_mode": str(feature_mode),
        "context_cols": list(context_cols),
        "config": asdict(cfg),
        "bucket_hparams": asdict(hp),
        "default_bucket_indexes": default_bucket_indexes,
        "label_distribution": {
            key: pd.Series(vals).value_counts().sort_index().to_dict()
            for key, vals in y.items()
            if key != "quality"
        },
    }
    for offset, key in enumerate(BUCKET_KEYS, start=1):
        yy = np.asarray(y[key])[trade_mask]
        if np.unique(yy).size < 2:
            continue
        model = _lgbm_classifier(random_state + offset, hp)
        model.fit(x_trade, yy, sample_weight=weights)
        bundle[f"{key}_model"] = model
    return bundle


def _fit_rank_allocator(
    base_x: pd.DataFrame,
    y: dict[str, np.ndarray],
    raw_ret: np.ndarray,
    *,
    cfg: FullyLearnedGovernorConfig,
    hp: RankHParams,
    candidates: pd.DataFrame,
    context_cols: list[str],
    random_state: int,
    samples_per_row: int,
) -> dict[str, Any]:
    pred_trade_mask = base_x["aq_action"].to_numpy(dtype=np.int64) != ACTION_CASH
    pred_side = base_x["aq_side"].to_numpy(dtype=np.float64)
    trade_idx = np.flatnonzero(pred_trade_mask & (pred_side != 0.0))
    if trade_idx.size < 10:
        raise RuntimeError(f"too few OOF predicted trade rows for rank allocator: {trade_idx.size}")
    rng = np.random.default_rng(int(random_state))
    train_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    full_n = len(candidates)
    for row_pos in trade_idx:
        sample_n = min(int(samples_per_row), full_n)
        sampled = rng.choice(full_n, size=sample_n, replace=False)
        label_combo = {
            "notional_idx": int(y["notional"][row_pos]),
            "leverage_idx": int(y["leverage"][row_pos]),
            "take_profit_idx": int(y["take_profit"][row_pos]),
            "stop_loss_idx": int(y["stop_loss"][row_pos]),
            "max_hold_idx": int(y["max_hold"][row_pos]),
        }
        label_match = np.flatnonzero(
            (candidates["notional_idx"].to_numpy() == label_combo["notional_idx"])
            & (candidates["leverage_idx"].to_numpy() == label_combo["leverage_idx"])
            & (candidates["take_profit_idx"].to_numpy() == label_combo["take_profit_idx"])
            & (candidates["stop_loss_idx"].to_numpy() == label_combo["stop_loss_idx"])
            & (candidates["max_hold_idx"].to_numpy() == label_combo["max_hold_idx"])
        )
        if label_match.size:
            sampled = np.unique(np.concatenate([sampled, label_match[:1]])).astype(np.int64)
        scores = _score_candidate_grid(raw_ret[row_pos], pred_side[row_pos], candidates.iloc[sampled].reset_index(drop=True), cfg, hp)
        bx = base_x.iloc[[row_pos]].reset_index(drop=True)
        train_parts.append(_rank_feature_frame(bx, candidates, sample_idx=sampled))
        target_parts.append(scores)
        row_w = np.full(len(scores), max(0.03, min(1.0, abs(float(y["quality"][row_pos])))), dtype=np.float32)
        if len(scores):
            top_cut = np.quantile(scores, 0.90)
            row_w = np.where(scores >= top_cut, row_w * 1.8, row_w)
        weight_parts.append(row_w.astype(np.float32))
    train_x = pd.concat(train_parts, ignore_index=True)
    train_y = np.concatenate(target_parts).astype(np.float32)
    weights = np.concatenate(weight_parts).astype(np.float32)
    model = _lgbm_ranker(random_state, hp)
    model.fit(train_x, train_y, sample_weight=weights)
    cooldown_model = _fit_cooldown_model(base_x, y, mask=pred_trade_mask, hp=hp, seed=random_state + 99)
    default_cooldown_idx = int(pd.Series(np.asarray(y["cooldown"])[pred_trade_mask]).mode().iloc[0]) if np.any(pred_trade_mask) else 0
    return {
        "model_type": "action_quality_side_aware_lightgbm_rank_allocator_v1",
        "feature_cols": list(train_x.columns),
        "base_feature_cols": list(base_x.columns),
        "feature_mode": "aq_plus_context",
        "context_cols": list(context_cols),
        "config": asdict(cfg),
        "rank_hparams": asdict(hp),
        "rank_model": model,
        "cooldown_model": cooldown_model,
        "default_cooldown_idx": default_cooldown_idx,
        "candidate_grid": candidates.to_dict(orient="list"),
        "stage2_train_trade_rows": int(trade_idx.size),
        "stage2_train_candidate_rows": int(len(train_x)),
        "samples_per_row": int(samples_per_row),
    }


def _bucket_expectation(model: Any, x: pd.DataFrame, buckets: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    vals = np.asarray([buckets[int(c)] for c in classes], dtype=np.float64)
    return proba @ vals, np.max(proba, axis=1)


def _bucket_or_default(bucket_bundle: dict[str, Any], key: str, x: pd.DataFrame, buckets: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    model = bucket_bundle.get(f"{key}_model")
    if model is not None:
        return _bucket_expectation(model, x, buckets)
    default_idx = int(dict(bucket_bundle.get("default_bucket_indexes", {})).get(key, 0))
    default = float(buckets[int(np.clip(default_idx, 0, len(buckets) - 1))])
    return np.full(len(x), default, dtype=np.float64), np.ones(len(x), dtype=np.float64)


def _predict_two_stage(aq_bundle: dict[str, Any], bucket_bundle: dict[str, Any], encoded_feat: pd.DataFrame) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(aq_bundle.get("config", {})))
    aq = _aq_outputs(aq_bundle, encoded_feat)
    x_bucket = _stage2_feature_frame(
        aq,
        encoded_feat,
        mode=str(bucket_bundle.get("feature_mode") or "aq_only"),
        context_cols=list(bucket_bundle.get("context_cols") or []),
    )
    x_bucket = x_bucket.reindex(columns=list(bucket_bundle.get("feature_cols") or [])).replace([np.inf, -np.inf], np.nan).copy()
    action = aq["aq_action"].to_numpy(dtype=np.int64)
    side = aq["aq_side"].to_numpy(dtype=np.int64)
    notional, c1 = _bucket_or_default(bucket_bundle, "notional", x_bucket, cfg.notional_buckets)
    leverage, c2 = _bucket_or_default(bucket_bundle, "leverage", x_bucket, cfg.leverage_buckets)
    take_profit, c3 = _bucket_or_default(bucket_bundle, "take_profit", x_bucket, cfg.take_profit_buckets)
    stop_loss, c4 = _bucket_or_default(bucket_bundle, "stop_loss", x_bucket, cfg.stop_loss_buckets)
    max_hold, c5 = _bucket_or_default(bucket_bundle, "max_hold", x_bucket, tuple(float(v) for v in cfg.max_hold_buckets))
    cooldown, c6 = _bucket_or_default(bucket_bundle, "cooldown", x_bucket, tuple(float(v) for v in cfg.cooldown_buckets))
    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, min(cfg.notional_buckets), max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    confidence = np.mean(np.vstack([aq["aq_action_conf"].to_numpy(), c1, c2, c3, c4, c5, c6]), axis=0)
    cash = action == ACTION_CASH
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": take_profit.astype(np.float64),
            "stop_loss": stop_loss.astype(np.float64),
            "max_hold_bars": np.rint(max_hold).astype(np.int64),
            "cooldown_bars": np.rint(cooldown).astype(np.int64),
            "quality_score": aq["aq_quality"].to_numpy(dtype=np.float64),
            "confidence": confidence.astype(np.float64),
        },
        index=encoded_feat.index,
    )
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _predict_rank_two_stage(aq_bundle: dict[str, Any], rank_bundle: dict[str, Any], encoded_feat: pd.DataFrame) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(aq_bundle.get("config", {})))
    aq = _aq_outputs(aq_bundle, encoded_feat)
    base_x = _stage2_feature_frame(
        aq,
        encoded_feat,
        mode="aq_plus_context",
        context_cols=list(rank_bundle.get("context_cols") or []),
    )
    base_x = base_x.reindex(columns=list(rank_bundle.get("base_feature_cols") or [])).replace([np.inf, -np.inf], np.nan).copy()
    candidates = pd.DataFrame(dict(rank_bundle["candidate_grid"]))
    rank_cols = list(rank_bundle.get("feature_cols") or [])
    action = aq["aq_action"].to_numpy(dtype=np.int64)
    side = aq["aq_side"].to_numpy(dtype=np.int64)
    n = len(encoded_feat)
    notional = np.zeros(n, dtype=np.float64)
    leverage = np.ones(n, dtype=np.float64)
    take_profit = np.zeros(n, dtype=np.float64)
    stop_loss = np.zeros(n, dtype=np.float64)
    max_hold = np.zeros(n, dtype=np.int64)
    rank_score = np.zeros(n, dtype=np.float64)
    rank_conf = np.zeros(n, dtype=np.float64)
    model = rank_bundle["rank_model"]
    trade_rows = np.flatnonzero((action != ACTION_CASH) & (side != 0))
    for row in trade_rows:
        x_rank = _rank_feature_frame(base_x.iloc[[row]].reset_index(drop=True), candidates)
        x_rank = x_rank.reindex(columns=rank_cols).replace([np.inf, -np.inf], np.nan)
        pred = np.asarray(model.predict(x_rank), dtype=np.float64)
        best_i = int(np.argmax(pred))
        best = candidates.iloc[best_i]
        notional[row] = float(best["cand_notional"])
        leverage[row] = float(best["cand_leverage"])
        take_profit[row] = float(best["cand_take_profit"])
        stop_loss[row] = float(best["cand_stop_loss"])
        max_hold[row] = int(best["cand_max_hold"])
        rank_score[row] = float(pred[best_i])
        if pred.size > 1:
            rank_conf[row] = float(pred[best_i] - np.partition(pred, -2)[-2])
    cooldown_model = rank_bundle.get("cooldown_model")
    if cooldown_model is not None:
        cooldown_pred = cooldown_model.predict(base_x)
    else:
        cooldown_pred = np.full(n, int(rank_bundle.get("default_cooldown_idx", 0)), dtype=np.int64)
    cooldown_vals = np.asarray(cfg.cooldown_buckets, dtype=np.int64)
    cooldown_idx = np.clip(np.asarray(cooldown_pred, dtype=np.int64), 0, len(cooldown_vals) - 1)
    cooldown = cooldown_vals[cooldown_idx]
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    cash = action == ACTION_CASH
    confidence = np.mean(
        np.vstack(
            [
                aq["aq_action_conf"].to_numpy(dtype=np.float64),
                np.clip(aq["aq_trade_prob"].to_numpy(dtype=np.float64), 0.0, 1.0),
                1.0 / (1.0 + np.exp(-np.clip(rank_score, -20.0, 20.0))),
            ]
        ),
        axis=0,
    )
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": take_profit.astype(np.float64),
            "stop_loss": stop_loss.astype(np.float64),
            "max_hold_bars": max_hold.astype(np.int64),
            "cooldown_bars": cooldown.astype(np.int64),
            "quality_score": aq["aq_quality"].to_numpy(dtype=np.float64),
            "confidence": confidence.astype(np.float64),
            "rank_score": rank_score.astype(np.float64),
            "rank_margin": rank_conf.astype(np.float64),
        },
        index=encoded_feat.index,
    )
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars", "rank_score", "rank_margin"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _backtest_decisions(df: pd.DataFrame, decisions: pd.DataFrame, *, fee: float, slip: float) -> dict[str, Any]:
    close = _close(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    next_cooldown = 0
    cooldown_left = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    notional_sum = 0.0
    leverage_sum = 0.0

    def mark_equity(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark_equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                pos = 0
                notional = 0.0
                leverage = 1.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
                action_counts["cash"] += 1
                continue
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                action_counts["cash"] += 1
                continue
            action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
            fill_idx = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(dec.notional_exposure)
            leverage = float(dec.leverage)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / max(((pd.to_datetime(df["timestamp"].iloc[-1]) - pd.to_datetime(df["timestamp"].iloc[0])).total_seconds() / 86400.0), 1e-6)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": float(leverage_sum / entries),
        "action_counts": action_counts,
        "exits": exits,
    }


def _score(cost1: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    return (
        float(cost1["pnl"])
        + 0.35 * float(cost2["pnl"])
        + 0.15 * float(cost3["pnl"])
        - 0.45 * abs(float(cost1["mdd"]))
        - 0.15 * abs(float(cost2["mdd"]))
    )


def _forward_oof_aq_features(
    x: pd.DataFrame,
    y: dict[str, np.ndarray],
    *,
    cfg: FullyLearnedGovernorConfig,
    hp: TreeHParams,
    feature_cols: list[str],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    n = len(x)
    ranges = [
        (int(n * 0.40), int(n * 0.60)),
        (int(n * 0.60), int(n * 0.80)),
        (int(n * 0.80), n),
    ]
    frames: list[pd.DataFrame] = []
    idxs: list[np.ndarray] = []
    for fold, (train_end, pred_end) in enumerate(ranges, start=1):
        if train_end <= 50 or pred_end <= train_end:
            continue
        sub_y = {k: np.asarray(v)[:train_end] for k, v in y.items()}
        aq = _train_aq_model(x.iloc[:train_end].copy(), sub_y, cfg=cfg, hp=hp, random_state=seed + 1000 * fold, feature_cols=feature_cols)
        take = np.arange(train_end, pred_end, dtype=np.int64)
        frames.append(_aq_outputs(aq, x.iloc[take].copy()))
        idxs.append(take)
    if not frames:
        raise RuntimeError("no forward OOF AQ features")
    return pd.concat(frames, ignore_index=True), np.concatenate(idxs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "V40.12: original v40.6 action/quality parent + side-aware LightGBM risk-adjusted rank allocator. "
            "Stage2 uses OOF stage1 predictions and scores candidate bucket tuples instead of classifying bucket labels."
        )
    )
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--v40-6-report", type=Path, default=DEFAULT_V40_6_REPORT)
    p.add_argument("--v40-6-bundle", type=Path, default=DEFAULT_V40_6_BUNDLE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--rank-samples-per-row", type=int, default=768)
    p.add_argument("--seed", type=int, default=2051)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"[{MODEL_ID}] loading data", flush=True)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    feature_audit = _audit_contract(train_all, eval_df, feature_cols)
    cfg = _parent_cfg()
    with args.v40_6_report.open(encoding="utf-8") as f:
        v40_6_report = json.load(f)
    v40_6_bundle = _load_pickle(args.v40_6_bundle)

    print(f"[{MODEL_ID}] building labels and encoded feature matrix", flush=True)
    x_train_base, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    train_idx = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(train_idx) != len(x_train_base):
        raise RuntimeError(f"train_idx/x mismatch: {len(train_idx)} vs {len(x_train_base)}")
    val_idx = np.arange(len(val_df), dtype=np.int64)
    eval_idx = np.arange(len(eval_df), dtype=np.int64)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    comp = v40_6_report["comparison"]
    spec = dict(comp["selected_projection_spec"])
    macro_cols = list(v40_6_report.get("macro_cols") or MACRO_COLS)
    micro_cols = list(v40_6_report.get("micro_cols") or [])
    emb = _extract_or_load_embeddings(
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        train_feat=train_feat,
        val_feat=val_feat,
        eval_feat=eval_feat,
        train_idx=train_idx,
        val_idx=val_idx,
        eval_idx=eval_idx,
        macro_cols=macro_cols,
        micro_cols=micro_cols,
        args=args,
    )
    proj_targets = _projection_targets(y)
    train_macro_f, macro_pls = _fit_pls_scores(emb["train_macro"], proj_targets, emb["train_macro"], n_components=int(spec["macro_dim"]))
    val_macro_f = macro_pls.transform(emb["val_macro"]).astype(np.float32)
    eval_macro_f = macro_pls.transform(emb["eval_macro"]).astype(np.float32)
    train_micro_f, micro_pls = _fit_pls_scores(emb["train_micro"], proj_targets, emb["train_micro"], n_components=int(spec["micro_dim"]))
    val_micro_f = micro_pls.transform(emb["val_micro"]).astype(np.float32)
    eval_micro_f = micro_pls.transform(emb["eval_micro"]).astype(np.float32)
    raw_cols = [c for c in list(v40_6_bundle.get("feature_cols") or []) if not c.startswith("macro_factor_") and not c.startswith("micro_factor_")]
    x_train_full = x_train_base.reset_index(drop=True).reindex(columns=raw_cols).copy()
    x_train_full = _add_embedding_cols(x_train_full, "macro_factor", train_macro_f)
    x_train_full = _add_embedding_cols(x_train_full, "micro_factor", train_micro_f)
    val_full = val_df.reset_index(drop=True).copy()
    val_full = _add_embedding_cols(val_full, "macro_factor", val_macro_f)
    val_full = _add_embedding_cols(val_full, "micro_factor", val_micro_f)
    eval_full = eval_df.reset_index(drop=True).copy()
    eval_full = _add_embedding_cols(eval_full, "macro_factor", eval_macro_f)
    eval_full = _add_embedding_cols(eval_full, "micro_factor", eval_micro_f)

    selected_hp = dict(comp.get("selected_full_hparams") or {})
    aq_hp = TreeHParams(
        int(selected_hp.get("max_iter", 220)),
        float(selected_hp.get("learning_rate", 0.040)),
        int(selected_hp.get("max_leaf_nodes", 31)),
        float(selected_hp.get("l2_regularization", 0.08)),
        float(selected_hp.get("cash_weight", 0.35)),
    )
    print(f"[{MODEL_ID}] building OOF action/quality predictions; final AQ reuses original v40.6 heads", flush=True)
    final_aq_bundle = _stage1_from_full_bundle(v40_6_bundle)
    oof_aq, oof_idx = _forward_oof_aq_features(x_train_full, y, cfg=cfg, hp=aq_hp, feature_cols=list(x_train_full.columns), seed=int(args.seed) + 500)
    y_oof = {k: np.asarray(v)[oof_idx] for k, v in y.items()}
    context_cols = _stage2_context_cols(list(x_train_full.columns))
    candidates = _candidate_grid(cfg)
    oof_base = _stage2_feature_frame(
        oof_aq,
        x_train_full.iloc[oof_idx].reset_index(drop=True),
        mode="aq_plus_context",
        context_cols=context_cols,
    )
    oof_raw_ret = _future_return_matrix(train_df, train_idx[oof_idx], cfg.max_train_horizon_bars)

    print(f"[{MODEL_ID}] selecting side-aware risk-adjusted rank allocator on 2025 Q4", flush=True)
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for idx, hp in enumerate(_rank_hp_grid(), start=1):
        print(f"[{MODEL_ID}] rank candidate {idx}/{len(_rank_hp_grid())}: {hp.name}", flush=True)
        rank_bundle = _fit_rank_allocator(
            oof_base,
            y_oof,
            oof_raw_ret,
            cfg=cfg,
            hp=hp,
            candidates=candidates,
            context_cols=context_cols,
            random_state=int(args.seed) + idx * 100,
            samples_per_row=int(args.rank_samples_per_row),
        )
        val_dec = _predict_rank_two_stage(final_aq_bundle, rank_bundle, val_full)
        val_cost1 = _backtest_decisions(val_df, val_dec, fee=cfg.fee, slip=cfg.slip)
        val_cost2 = _backtest_decisions(val_df, val_dec, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0)
        val_cost3 = _backtest_decisions(val_df, val_dec, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0)
        score = _score(val_cost1, val_cost2, val_cost3)
        row = {
            "rank_name": hp.name,
            **asdict(hp),
            "validation_score": score,
            "val_cost1_pnl": val_cost1["pnl"],
            "val_cost1_mdd": val_cost1["mdd"],
            "val_cost1_trades": val_cost1["trades"],
            "val_cost1_trades_day": val_cost1["trades_per_day"],
            "val_cost2_pnl": val_cost2["pnl"],
            "val_cost3_pnl": val_cost3["pnl"],
            "stage2_oof_rows": int(len(oof_base)),
            "stage2_oof_pred_trade_rows": int(np.sum(oof_base["aq_action"].to_numpy(dtype=np.int64) != ACTION_CASH)),
            "stage2_train_candidate_rows": int(rank_bundle["stage2_train_candidate_rows"]),
            "stage2_feature_count": int(len(rank_bundle["feature_cols"])),
            "candidate_grid_size": int(len(candidates)),
        }
        grid_rows.append(row)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "hp": hp,
                "rank_bundle": rank_bundle,
                "validation": {"cost1": val_cost1, "cost2": val_cost2, "cost3": val_cost3},
            }
    if best is None:
        raise RuntimeError("no rank candidate selected")

    print(f"[{MODEL_ID}] evaluating 2026 OOS", flush=True)
    selected_rank = best["rank_bundle"]
    eval_dec = _predict_rank_two_stage(final_aq_bundle, selected_rank, eval_full)
    metrics = {f"cost{k}": _backtest_decisions(eval_df, eval_dec, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}
    baseline_eval = v40_6_report.get("comparison", {}).get("oos_2026", {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aq_path = args.out_dir / "action_quality_parent.pkl"
    rank_path = args.out_dir / "lgbm_rank_allocator.pkl"
    with aq_path.open("wb") as f:
        pickle.dump(final_aq_bundle, f)
    with rank_path.open("wb") as f:
        pickle.dump(selected_rank, f)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).sort_values("validation_score", ascending=False).to_csv(args.grid_out, index=False)

    blocking = list(feature_audit.get("blocking", []))
    warnings = list(feature_audit.get("warnings", []))
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025 Q4 rank allocator HP selection",
        "oos_window": "2026 fixed OOS only after selection",
        "stage1": "original v40_6 encoded action/quality HGB heads reused for final inference",
        "stage2": "side-aware LightGBM rank allocator over candidate bucket tuples",
        "stage2_forward_oof": True,
        "stage2_context_feature_count": int(len(context_cols)),
        "stage2_candidate_grid_size": int(len(candidates)),
        "feature_audit": feature_audit,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Two-stage v40.6 parent split. Stage 1 reuses the original v40.6 Chronos/Kairos PLS encoded HGB action and quality heads for final inference. Stage 2 does not classify bucket labels independently; it trains a side-aware LightGBM regressor to rank candidate notional/leverage/TP/SL/max_hold tuples by risk-adjusted utility. Stage 2 training uses forward OOF action/quality predictions to avoid same-row stacking leakage.",
        "split_policy": "Train=2025 Jan-Sep, selection=2025 Q4, OOS=2026 fixed",
        "projection_spec": spec,
        "stage1_hparams": asdict(aq_hp),
        "selected_rank_hparams": asdict(best["hp"]),
        "stage2_feature_cols": list(selected_rank.get("feature_cols") or []),
        "stage2_context_cols": list(context_cols),
        "candidate_grid_size": int(len(candidates)),
        "stage2_oof_rows": int(len(oof_aq)),
        "stage2_oof_pred_trade_rows": int(np.sum(oof_base["aq_action"].to_numpy(dtype=np.int64) != ACTION_CASH)),
        "stage2_train_candidate_rows": int(selected_rank["stage2_train_candidate_rows"]),
        "validation": best["validation"],
        "metrics": metrics,
        "baseline_v40_6_oos": baseline_eval,
        "audit": audit,
        "artifacts": {
            "action_quality_parent": str(aq_path),
            "rank_allocator": str(rank_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "selected_rank": asdict(best["hp"]), "metrics": metrics, "audit_status": audit["status"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
