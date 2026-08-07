#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chronos import Chronos2Pipeline  # noqa: E402
from tsfm.model.kairos import AutoModel as KairosAutoModel  # noqa: E402

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    _bucket_or_default_batch,
    build_training_set,
    prepare_features,
    train_policy,
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
    MICRO_PRIORITY,
    _extract_macro_embeddings,
    _extract_micro_embeddings,
    _parent_cfg,
)


MODEL_ID = "hf_v13_tree_vs_foundation_summary_action_quality_v40_4_20260512"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tree_vs_foundation_summary_action_quality_v40_4_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_summary_action_quality_v40_4_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_summary_action_quality_v40_4_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_tree_vs_foundation_summary_action_quality_v40_4_20260512_grid.csv"


def _add_embedding_cols(base: pd.DataFrame, prefix: str, values: np.ndarray) -> pd.DataFrame:
    cols = [f"{prefix}_{j:03d}" for j in range(values.shape[1])]
    extra = pd.DataFrame(values, columns=cols, index=base.index)
    return pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


@dataclass(frozen=True)
class SummarySpec:
    macro_dim: int
    micro_dim: int
    drop_raw_micro: bool

    @property
    def name(self) -> str:
        return f"macro{self.macro_dim}_micro{self.micro_dim}_{'dropmicro' if self.drop_raw_micro else 'keepmicro'}"


@dataclass(frozen=True)
class TreeHParams:
    max_iter: int
    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float
    cash_weight: float

    @property
    def name(self) -> str:
        lr = str(self.learning_rate).replace(".", "p")
        l2 = str(self.l2_regularization).replace(".", "p")
        return f"mi{self.max_iter}_lr{lr}_leaf{self.max_leaf_nodes}_l2{l2}_cw{str(self.cash_weight).replace('.', 'p')}"


def _classifier_hp(seed: int, hp: TreeHParams) -> Any:
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


def _regressor_hp(seed: int, hp: TreeHParams) -> Any:
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


def train_action_quality_with_hparams(
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
    bundle: dict[str, Any] = {
        "model_type": "fully_learned_governor_policy_action_quality_only_v1",
        "feature_cols": list(feature_cols),
        "config": asdict(cfg),
        "tuned_tree_hparams": asdict(hp),
        "action_model": _weighted_fit_classifier(_classifier_hp(random_state, hp), x, np.asarray(y["action"]), weights),
        "quality_model": _regressor_hp(random_state + 99, hp),
    }
    bundle["quality_model"].fit(x, np.asarray(y["quality"], dtype=np.float64), histgradientboostingregressor__sample_weight=weights)
    return bundle


def predict_policy_frame_mixed(
    aq_bundle: dict[str, Any],
    bucket_bundle: dict[str, Any],
    base_feat: pd.DataFrame,
    encoded_feat: pd.DataFrame,
) -> pd.DataFrame:
    cfg = FullyLearnedGovernorConfig(**dict(bucket_bundle.get("config", {})))
    enc_cols = list(aq_bundle.get("feature_cols") or [])
    base_cols = list(bucket_bundle.get("feature_cols") or [])
    x_enc = encoded_feat.reindex(columns=enc_cols).replace([np.inf, -np.inf], np.nan).copy()
    x_base = base_feat.reindex(columns=base_cols).replace([np.inf, -np.inf], np.nan).copy()
    if "side_hint" in x_enc.columns:
        x_enc["side_hint"] = 0.0
    if "side_hint" in x_base.columns:
        x_base["side_hint"] = 0.0

    action_proba = aq_bundle["action_model"].predict_proba(x_enc)
    action_classes = np.asarray(aq_bundle["action_model"].classes_, dtype=int)
    action_idx = np.argmax(action_proba, axis=1)
    action = action_classes[action_idx]
    action_conf = np.max(action_proba, axis=1)
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    quality = aq_bundle["quality_model"].predict(x_enc) if "quality_model" in aq_bundle else np.zeros(len(x_enc), dtype=np.float64)

    x_side = x_base.copy()
    x_side["side_hint"] = side.astype(np.float64)
    notional, c1 = _bucket_or_default_batch(bucket_bundle, "notional", x_side, cfg.notional_buckets)
    leverage, c2 = _bucket_or_default_batch(bucket_bundle, "leverage", x_side, cfg.leverage_buckets)
    take_profit, c3 = _bucket_or_default_batch(bucket_bundle, "take_profit", x_side, cfg.take_profit_buckets)
    stop_loss, c4 = _bucket_or_default_batch(bucket_bundle, "stop_loss", x_side, cfg.stop_loss_buckets)
    max_hold, c5 = _bucket_or_default_batch(bucket_bundle, "max_hold", x_side, tuple(float(v) for v in cfg.max_hold_buckets))
    cooldown, c6 = _bucket_or_default_batch(bucket_bundle, "cooldown", x_side, tuple(float(v) for v in cfg.cooldown_buckets))

    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    notional = np.clip(notional, min(cfg.notional_buckets), max(cfg.notional_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    confidence = np.mean(np.vstack([action_conf, c1, c2, c3, c4, c5, c6]), axis=0)
    cash = action == ACTION_CASH
    out = pd.DataFrame(
        {
            "action": action.astype(np.int64),
            "side": side.astype(np.int64),
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": take_profit.astype(np.float64),
            "stop_loss": stop_loss.astype(np.float64),
            "max_hold_bars": np.rint(max_hold).astype(np.int64),
            "cooldown_bars": np.rint(cooldown).astype(np.int64),
            "quality_score": np.asarray(quality, dtype=np.float64),
            "confidence": confidence.astype(np.float64),
        },
        index=base_feat.index,
    )
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def backtest_decisions(df: pd.DataFrame, decisions: pd.DataFrame, *, fee: float, slip: float) -> dict[str, Any]:
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
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
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
    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / max(((pd.to_datetime(df["timestamp"].iloc[-1]) - pd.to_datetime(df["timestamp"].iloc[0])).total_seconds() / 86400.0), 1e-6)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "action_counts": action_counts,
        "exits": exits,
    }


def _score_result(cost1: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    return (
        float(cost1["pnl"])
        + 0.35 * float(cost2["pnl"])
        + 0.15 * float(cost3["pnl"])
        - 0.45 * abs(float(cost1["mdd"]))
        - 0.15 * abs(float(cost2["mdd"]))
    )


def _encoded_hparam_grid() -> list[TreeHParams]:
    return [
        TreeHParams(320, 0.020, 31, 0.80, 0.50),
        TreeHParams(220, 0.040, 31, 0.08, 0.35),
        TreeHParams(360, 0.020, 63, 0.50, 0.25),
        TreeHParams(220, 0.030, 63, 0.20, 0.35),
    ]


def _summary_grid() -> list[SummarySpec]:
    return [
        SummarySpec(4, 4, True),
        SummarySpec(8, 4, True),
        SummarySpec(8, 8, True),
        SummarySpec(12, 6, True),
        SummarySpec(8, 4, False),
        SummarySpec(8, 8, False),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base HGB parent vs Chronos/Kairos/KAN encoded HGB parent.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--train-stride", type=int, default=48)
    p.add_argument("--kan-epochs", type=int, default=25)
    p.add_argument("--embed-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=2041)
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
    audit = _audit_contract(train_all, eval_df, feature_cols)
    cfg: FullyLearnedGovernorConfig = _parent_cfg()

    print(f"[{MODEL_ID}] building baseline labels", flush=True)
    x_train_base, y, meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.train_stride), batch_size=512, feature_cols=feature_cols)
    valid = np.arange(0, max(0, len(train_df) - cfg.max_train_horizon_bars - 1), max(1, int(args.train_stride)), dtype=np.int64)
    if len(valid) != len(x_train_base):
        raise RuntimeError(f"valid/train mismatch: {len(valid)} vs {len(x_train_base)}")

    print(f"[{MODEL_ID}] preparing full features", flush=True)
    train_feat = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    val_feat = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_feat = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)

    print(f"[{MODEL_ID}] training baseline tree parent", flush=True)
    baseline_bundle = train_policy(x_train_base, y, cfg=cfg, random_state=int(args.seed), feature_cols=list(x_train_base.columns))

    micro_cols = [c for c in MICRO_PRIORITY if c in train_feat.columns]
    if not micro_cols:
        raise RuntimeError("no microstructure columns available for Kairos track")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{MODEL_ID}] loading Chronos-2 and Kairos_23m on {device}", flush=True)
    chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map=device)
    kairos = KairosAutoModel.from_pretrained(KAIROS_MODEL, trust_remote_code=True).to(device).eval()

    emb_dir = ROOT / "data/ensemble/supervised/hf_v13_multitrack_foundation_parent_v40_20260512" / "embeddings"
    print(f"[{MODEL_ID}] extracting foundation embeddings", flush=True)
    train_macro = _extract_macro_embeddings(chronos, train_df, valid, cache_path=emb_dir / f"tree_train_macro_s{args.train_stride}.npy", batch_size=args.embed_batch)
    val_macro = _extract_macro_embeddings(chronos, val_df, np.arange(len(val_df), dtype=np.int64), cache_path=emb_dir / "val_macro.npy", batch_size=args.embed_batch)
    eval_macro = _extract_macro_embeddings(chronos, eval_df, np.arange(len(eval_df), dtype=np.int64), cache_path=emb_dir / "eval_macro.npy", batch_size=args.embed_batch)
    train_micro = _extract_micro_embeddings(kairos, train_feat, valid, micro_cols, cache_path=emb_dir / f"tree_train_micro_s{args.train_stride}.npy", batch_size=args.embed_batch)
    val_micro = _extract_micro_embeddings(kairos, val_feat, np.arange(len(val_feat), dtype=np.int64), micro_cols, cache_path=emb_dir / "val_micro.npy", batch_size=args.embed_batch)
    eval_micro = _extract_micro_embeddings(kairos, eval_feat, np.arange(len(eval_feat), dtype=np.int64), micro_cols, cache_path=emb_dir / "eval_micro.npy", batch_size=args.embed_batch)

    print(f"[{MODEL_ID}] preparing summary-factor action-quality tuning set", flush=True)
    residual_keep_cols = [c for c in x_train_base.columns if c not in {"side_hint"}]

    baseline_val = {f"cost{k}": backtest_policy_frame(val_df, baseline_bundle, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}
    baseline_eval = {f"cost{k}": backtest_policy_frame(eval_df, baseline_bundle, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}

    print(f"[{MODEL_ID}] tuning Chronos/Kairos summary-factor action-quality heads on 2025 Q4 validation", flush=True)
    hparams_grid = _encoded_hparam_grid()
    summary_grid = _summary_grid()
    grid_rows: list[dict[str, Any]] = []
    best_score = -1e18
    best_bundle: dict[str, Any] | None = None
    best_hp: TreeHParams | None = None
    best_spec: SummarySpec | None = None
    best_val: dict[str, Any] | None = None
    candidate_idx = 0
    for spec in summary_grid:
        macro_pca = PCA(n_components=int(spec.macro_dim), random_state=int(args.seed))
        micro_pca = PCA(n_components=int(spec.micro_dim), random_state=int(args.seed))
        train_macro_f = macro_pca.fit_transform(train_macro).astype(np.float32)
        val_macro_f = macro_pca.transform(val_macro).astype(np.float32)
        eval_macro_f = macro_pca.transform(eval_macro).astype(np.float32)
        train_micro_f = micro_pca.fit_transform(train_micro).astype(np.float32)
        val_micro_f = micro_pca.transform(val_micro).astype(np.float32)
        eval_micro_f = micro_pca.transform(eval_micro).astype(np.float32)

        raw_cols = [c for c in residual_keep_cols if (not spec.drop_raw_micro or c not in micro_cols)]
        x_train_aq = x_train_base.reset_index(drop=True).reindex(columns=raw_cols).copy()
        x_train_aq = _add_embedding_cols(x_train_aq, "macro_factor", train_macro_f)
        x_train_aq = _add_embedding_cols(x_train_aq, "micro_factor", train_micro_f)
        val_aq = val_feat.reset_index(drop=True).reindex(columns=raw_cols).copy()
        val_aq = _add_embedding_cols(val_aq, "macro_factor", val_macro_f)
        val_aq = _add_embedding_cols(val_aq, "micro_factor", val_micro_f)
        eval_aq = eval_feat.reset_index(drop=True).reindex(columns=raw_cols).copy()
        eval_aq = _add_embedding_cols(eval_aq, "macro_factor", eval_macro_f)
        eval_aq = _add_embedding_cols(eval_aq, "micro_factor", eval_micro_f)

        for hp in hparams_grid:
            candidate_idx += 1
            print(f"[{MODEL_ID}] candidate {candidate_idx}/{len(summary_grid) * len(hparams_grid)}: {spec.name} | {hp.name}", flush=True)
            bundle = train_action_quality_with_hparams(
                x_train_aq,
                y,
                cfg=cfg,
                hp=hp,
                random_state=int(args.seed) + candidate_idx,
                feature_cols=list(x_train_aq.columns),
            )
            val_decisions = predict_policy_frame_mixed(bundle, baseline_bundle, val_feat, val_aq)
            val_cost1 = backtest_decisions(val_df, val_decisions, fee=cfg.fee, slip=cfg.slip)
            val_cost2 = backtest_decisions(val_df, val_decisions, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0)
            val_cost3 = backtest_decisions(val_df, val_decisions, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0)
            score = _score_result(val_cost1, val_cost2, val_cost3)
            grid_rows.append(
                {
                    "summary_name": spec.name,
                    **asdict(spec),
                    "tree_name": hp.name,
                    **asdict(hp),
                    "validation_score": float(score),
                    "val_cost1_pnl": float(val_cost1["pnl"]),
                    "val_cost1_mdd": float(val_cost1["mdd"]),
                    "val_cost1_trades": int(val_cost1["trades"]),
                    "val_cost1_trades_day": float(val_cost1["trades_per_day"]),
                    "val_cost2_pnl": float(val_cost2["pnl"]),
                    "val_cost3_pnl": float(val_cost3["pnl"]),
                    "aq_feature_count": int(x_train_aq.shape[1]),
                    "raw_feature_count": int(len(raw_cols)),
                }
            )
            if score > best_score:
                best_score = float(score)
                best_bundle = bundle
                best_hp = hp
                best_spec = spec
                best_val = {"cost1": val_cost1, "cost2": val_cost2, "cost3": val_cost3}
                best_eval_aq = eval_aq.copy()

    if best_bundle is None or best_hp is None or best_spec is None or best_val is None:
        raise RuntimeError("summary-factor action-quality tuning produced no valid candidate")

    print(f"[{MODEL_ID}] backtesting baseline vs tuned mixed selection", flush=True)
    mixed_val = best_val
    eval_decisions = predict_policy_frame_mixed(best_bundle, baseline_bundle, eval_feat, best_eval_aq)
    mixed_eval = {f"cost{k}": backtest_decisions(eval_df, eval_decisions, fee=cfg.fee * k, slip=cfg.slip * k) for k in (1, 2, 3)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "baseline_tree_parent.pkl").open("wb") as f:
        pickle.dump(baseline_bundle, f)
    with (args.out_dir / "mixed_action_quality_bundle.pkl").open("wb") as f:
        pickle.dump(best_bundle, f)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(grid_rows).sort_values("validation_score", ascending=False).to_csv(args.grid_out, index=False)

    comparison = {
        "validation": {
            "baseline": baseline_val,
            "mixed_action_quality": mixed_val,
            "delta_cost1_pnl": float(mixed_val["cost1"]["pnl"] - baseline_val["cost1"]["pnl"]),
            "delta_cost1_mdd": float(mixed_val["cost1"]["mdd"] - baseline_val["cost1"]["mdd"]),
        },
        "oos_2026": {
            "baseline": baseline_eval,
            "mixed_action_quality": mixed_eval,
            "delta_cost1_pnl": float(mixed_eval["cost1"]["pnl"] - baseline_eval["cost1"]["pnl"]),
            "delta_cost1_mdd": float(mixed_eval["cost1"]["mdd"] - baseline_eval["cost1"]["mdd"]),
            "delta_cost2_pnl": float(mixed_eval["cost2"]["pnl"] - baseline_eval["cost2"]["pnl"]),
            "delta_cost3_pnl": float(mixed_eval["cost3"]["pnl"] - baseline_eval["cost3"]["pnl"]),
        },
        "selected_summary_name": best_spec.name,
        "selected_mixed_name": best_hp.name,
        "selected_summary_spec": asdict(best_spec),
        "selected_mixed_hparams": asdict(best_hp),
        "baseline_tree_hparams": {
            "max_iter": 220,
            "learning_rate": 0.040,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.08,
            "cash_weight": 0.35,
        },
    }

    blocking = list(audit.get("blocking", []))
    warnings = list(audit.get("warnings", []))
    final_audit = {
        "status": "pass" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "feature_audit": audit,
        "comparison": comparison,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Baseline tree versus summary-factor mixed parent. Baseline uses the original fixed HGB parent settings. The challenger removes KAN, compresses Chronos-2 and Kairos_23m embeddings into low-dimensional summary factors, feeds those only into action/quality heads, and keeps the baseline tree bucket heads for notional, leverage, TP, SL, hold, and cooldown. Summary spec and action-quality HGB hyperparameters are selected on 2025 Q4 validation.",
        "split_policy": "train=2025 Jan-Sep, validation=2025 Q4, OOS=2026 fixed",
        "chronos_model": CHRONOS_MODEL,
        "kairos_model": KAIROS_MODEL,
        "macro_cols": ["open", "high", "low", "close", "volume"],
        "micro_cols": micro_cols,
        "residual_keep_cols": residual_keep_cols,
        "train_stride": int(args.train_stride),
        "training_meta": meta,
        "grid_size": len(summary_grid) * len(hparams_grid),
        "comparison": comparison,
        "audit": final_audit,
        "artifacts": {
            "baseline_bundle": str(args.out_dir / "baseline_tree_parent.pkl"),
            "mixed_action_quality_bundle": str(args.out_dir / "mixed_action_quality_bundle.pkl"),
            "grid_csv": str(args.grid_out),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "comparison": comparison}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
