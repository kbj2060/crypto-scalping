#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_exp  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


MODEL_ID = "omega4_2_bucket_risk_sidecar_20260623"
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
BASE_LEDGER_DIR = OUT_ROOT / "omega4_2_trade_risk_sidecar_20260622_v5_parent_side_hgb_mae050_balanced"
MARGIN_BUCKETS = np.asarray([0.1, 0.4, 0.7, 1.0], dtype=np.float64)
LEVERAGE_BUCKETS = np.asarray([1.0, 1.5, 2.0, 3.0], dtype=np.float64)
LOG_RISK_CONFIG_KEYS = ("tail_budget", "tail_penalty", "liquidation_buffer", "liquidation_penalty")


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _build_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.035,
        l2_regularization=0.10,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        random_state=int(seed),
    )


def _build_bucket_labels(
    ledger: pd.DataFrame,
    *,
    tail_budget: float,
    tail_penalty: float,
    liquidation_buffer: float,
    liquidation_penalty: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    net = pd.to_numeric(ledger["net_per_notional"], errors="raise").to_numpy(dtype=np.float64)
    mae = pd.to_numeric(ledger["mae_price_move"], errors="raise").to_numpy(dtype=np.float64)
    margin_y = np.zeros(len(ledger), dtype=np.int64)
    leverage_y = np.zeros(len(ledger), dtype=np.int64)
    utilities: list[float] = []
    log_values: list[float] = []
    tail_values: list[float] = []
    liquidation_values: list[float] = []
    for i, npr in enumerate(net):
        best: tuple[float, int, int] | None = None
        for mi, margin in enumerate(MARGIN_BUCKETS):
            for li, leverage in enumerate(LEVERAGE_BUCKETS):
                notional = float(margin) * float(leverage)
                account_return = float(npr) * notional
                if account_return <= -0.98:
                    candidate = (-1.0e9, int(mi), int(li), -1.0e9, 0.0, 0.0)
                else:
                    log_growth = float(np.log1p(account_return))
                    adverse_loss = max(-float(mae[i]) * notional, 0.0)
                    tail_excess = max(adverse_loss - float(tail_budget), 0.0)
                    liquidation_ratio = max(-float(mae[i]) * float(leverage), 0.0)
                    liquidation_excess = max(liquidation_ratio - float(liquidation_buffer), 0.0)
                    utility = (
                        log_growth
                        - float(tail_penalty) * tail_excess
                        - float(liquidation_penalty) * liquidation_excess
                    )
                    candidate = (float(utility), int(mi), int(li), log_growth, tail_excess, liquidation_excess)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            raise RuntimeError("empty bucket label candidate set")
        utilities.append(best[0])
        margin_y[i] = best[1]
        leverage_y[i] = best[2]
        log_values.append(best[3])
        tail_values.append(best[4])
        liquidation_values.append(best[5])
    diag = {
        "rows": int(len(ledger)),
        "margin_buckets": [float(x) for x in MARGIN_BUCKETS.tolist()],
        "leverage_buckets": [float(x) for x in LEVERAGE_BUCKETS.tolist()],
        "objective": "log_growth_minus_linear_tail_and_liquidation_risk",
        "tail_budget": float(tail_budget),
        "tail_penalty": float(tail_penalty),
        "liquidation_buffer": float(liquidation_buffer),
        "liquidation_penalty": float(liquidation_penalty),
        "net_per_notional_mean": float(np.mean(net)) if len(net) else 0.0,
        "net_per_notional_p25": float(np.quantile(net, 0.25)) if len(net) else 0.0,
        "net_per_notional_p50": float(np.quantile(net, 0.50)) if len(net) else 0.0,
        "net_per_notional_p75": float(np.quantile(net, 0.75)) if len(net) else 0.0,
        "margin_label_counts": {str(float(MARGIN_BUCKETS[k])): int(v) for k, v in pd.Series(margin_y).value_counts().sort_index().items()},
        "leverage_label_counts": {str(float(LEVERAGE_BUCKETS[k])): int(v) for k, v in pd.Series(leverage_y).value_counts().sort_index().items()},
        "utility_mean": float(np.mean(utilities)) if utilities else 0.0,
        "selected_log_growth_mean": float(np.mean(log_values)) if log_values else 0.0,
        "selected_tail_excess_mean": float(np.mean(tail_values)) if tail_values else 0.0,
        "selected_liquidation_excess_mean": float(np.mean(liquidation_values)) if liquidation_values else 0.0,
    }
    return margin_y, leverage_y, diag


def _fit_side_split_bucket_models(
    x_train_trade: pd.DataFrame,
    margin_y: np.ndarray,
    leverage_y: np.ndarray,
    side_train_trade: np.ndarray,
    mae: np.ndarray,
    *,
    seed: int,
) -> dict[int, dict[str, HistGradientBoostingClassifier]]:
    models: dict[int, dict[str, HistGradientBoostingClassifier]] = {}
    side_arr = np.asarray(side_train_trade, dtype=np.int64)
    mae_weight = 1.0 + np.clip(-np.asarray(mae, dtype=np.float64) * 25.0, 0.0, 3.0)
    for side in (-1, 1):
        mask = side_arr == int(side)
        if int(mask.sum()) < 12:
            raise RuntimeError(f"not enough side-split bucket samples for side={side}: {int(mask.sum())}")
        side_models: dict[str, HistGradientBoostingClassifier] = {}
        for target_name, target, offset in (("margin", margin_y, 101), ("leverage", leverage_y, 211)):
            y = np.asarray(target, dtype=np.int64)[mask]
            if len(np.unique(y)) < 2:
                raise RuntimeError(f"bucket target collapsed for side={side} target={target_name}: class={int(y[0])}")
            weights = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
            weights *= mae_weight[mask]
            model = _build_classifier(int(seed) + int(offset) + (11 if side < 0 else 17))
            model.fit(x_train_trade.loc[mask], y, sample_weight=weights)
            side_models[target_name] = model
        models[int(side)] = side_models
    return models


def _predict_side_split_bucket_models(
    models: dict[int, dict[str, HistGradientBoostingClassifier]],
    x_all: pd.DataFrame,
    side_all: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    margin = np.zeros(len(x_all), dtype=np.float64)
    leverage = np.zeros(len(x_all), dtype=np.float64)
    side_arr = np.asarray(side_all, dtype=np.int64)
    active_arr = np.asarray(active, dtype=bool)
    for side, side_models in models.items():
        mask = (side_arr == int(side)) & active_arr
        if not bool(mask.any()):
            continue
        margin_idx = np.asarray(side_models["margin"].predict(x_all.loc[mask]), dtype=np.int64)
        leverage_idx = np.asarray(side_models["leverage"].predict(x_all.loc[mask]), dtype=np.int64)
        margin[mask] = MARGIN_BUCKETS[margin_idx]
        leverage[mask] = LEVERAGE_BUCKETS[leverage_idx]
    if not np.isfinite(margin).all() or not np.isfinite(leverage).all():
        raise RuntimeError("non-finite bucket risk predictions")
    return margin, leverage


def _risk_distribution(ledger: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ("risk_margin_fraction", "risk_leverage", "risk_notional"):
        values = pd.to_numeric(ledger[col], errors="raise").to_numpy(dtype=np.float64)
        out[col] = {
            "mean": float(np.mean(values)) if len(values) else 0.0,
            "p25": float(np.quantile(values, 0.25)) if len(values) else 0.0,
            "p50": float(np.quantile(values, 0.50)) if len(values) else 0.0,
            "p75": float(np.quantile(values, 0.75)) if len(values) else 0.0,
            "max": float(np.max(values)) if len(values) else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=risk_exp.BASELINE_BUNDLE)
    ap.add_argument("--ledger-source-dir", type=Path, default=BASE_LEDGER_DIR)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--risk-feature-mode", choices=["all", "parent_outputs"], default="parent_outputs")
    ap.add_argument("--out-suffix", default="v1_margin010_100_lev1_3")
    ap.add_argument("--max-validation-mdd-abs", type=float, default=8.0)
    ap.add_argument("--max-oos-mdd-abs", type=float, default=5.70)
    ap.add_argument("--seed", type=int, default=260623)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_ROOT / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )

    print("stage=predict_parent", flush=True)
    x_train, train_src, train_dec_base = risk_exp._predict_decisions(
        frames["train_raw"], oof=True, models=models, base_cols=base_cols, quality_threshold=float(args.quality_threshold), device=device
    )
    x_val, val_src, val_dec_base = risk_exp._predict_decisions(
        frames["val_raw"], oof=True, models=models, base_cols=base_cols, quality_threshold=float(args.quality_threshold), device=device
    )
    x_oos, oos_src, oos_dec_base = risk_exp._predict_decisions(
        frames["oos_raw"], oof=False, models=models, base_cols=base_cols, quality_threshold=float(args.quality_threshold), device=device
    )

    print("stage=apply_atr_contract", flush=True)
    train_dec, train_atr_diag = atr_eval._apply_atr_safety_sltp(
        train_dec_base, frames["train_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
    )
    val_dec, val_atr_diag = atr_eval._apply_atr_safety_sltp(
        val_dec_base, frames["val_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
    )
    oos_dec, oos_atr_diag = atr_eval._apply_atr_safety_sltp(
        oos_dec_base, frames["oos_raw"], atr_window=int(args.atr_window), tp_mult=float(args.tp_mult), sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp), min_sl=float(args.min_sl), max_tp=float(args.max_tp), max_sl=float(args.max_sl)
    )
    train_atr = atr_eval._atr_pct(frames["train_raw"], int(args.atr_window))
    val_atr = atr_eval._atr_pct(frames["val_raw"], int(args.atr_window))
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], int(args.atr_window))

    print("stage=load_baseline_ledgers", flush=True)
    ledger_dir = Path(args.ledger_source_dir)
    train_ledger = pd.read_csv(ledger_dir / "train_baseline_trade_ledger.csv")
    val_base_ledger = pd.read_csv(ledger_dir / "validation_baseline_trade_ledger.csv")
    oos_base_ledger = pd.read_csv(ledger_dir / "oos_baseline_trade_ledger.csv")
    train_base_m, train_base_sized = risk_exp._ledger_metrics_with_margins(frames["train_raw"], train_ledger, None)
    val_base_m, val_base_sized = risk_exp._ledger_metrics_with_margins(frames["val_raw"], val_base_ledger, None)
    oos_base_m, oos_base_sized = risk_exp._ledger_metrics_with_margins(frames["oos_raw"], oos_base_ledger, None)

    print("stage=build_features", flush=True)
    train_features = risk_exp._risk_feature_frame(frames["train_raw"], train_src, train_dec, base_cols, atr_pct=train_atr, feature_mode=str(args.risk_feature_mode))
    val_features = risk_exp._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=str(args.risk_feature_mode))
    oos_features = risk_exp._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=str(args.risk_feature_mode))
    x_train_trade, risk_cols = risk_exp._feature_matrix(train_features.iloc[train_ledger["entry_signal_i"].to_numpy(dtype=np.int64)].reset_index(drop=True))
    x_train_all, _ = risk_exp._feature_matrix(train_features, risk_cols)
    x_val_all, _ = risk_exp._feature_matrix(val_features, risk_cols)
    x_oos_all, _ = risk_exp._feature_matrix(oos_features, risk_cols)
    side_train_trade = pd.to_numeric(train_ledger["side"], errors="raise").to_numpy(dtype=np.int64)
    train_mae = pd.to_numeric(train_ledger["mae_price_move"], errors="raise").to_numpy(dtype=np.float64)

    print("stage=train_bucket_models", flush=True)
    configs: list[dict[str, float]] = []
    for tail_budget in (0.0, 0.005, 0.010):
        for tail_penalty in (2.0, 5.0, 10.0, 20.0, 40.0):
            for liquidation_buffer in (0.05, 0.08, 0.12):
                for liquidation_penalty in (0.0, 1.0, 2.0, 5.0):
                    configs.append(
                        {
                            "tail_budget": float(tail_budget),
                            "tail_penalty": float(tail_penalty),
                            "liquidation_buffer": float(liquidation_buffer),
                            "liquidation_penalty": float(liquidation_penalty),
                        }
                    )

    active_val = omega._active(val_dec)
    active_oos = omega._active(oos_dec)
    val_side_all = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    oos_side_all = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    skipped_configs: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs):
        margin_y, leverage_y, label_diag = _build_bucket_labels(train_ledger, **cfg)
        try:
            models_by_side = _fit_side_split_bucket_models(
                x_train_trade, margin_y, leverage_y, side_train_trade, train_mae, seed=int(args.seed) + idx * 1000
            )
        except RuntimeError as exc:
            skipped_configs.append({"variant": f"bucket_{idx:03d}", **cfg, "reason": str(exc), "label_diag": label_diag})
            continue
        val_margin, val_leverage = _predict_side_split_bucket_models(models_by_side, x_val_all, val_side_all, active_val)
        oos_margin, oos_leverage = _predict_side_split_bucket_models(models_by_side, x_oos_all, oos_side_all, active_oos)
        val_m, val_ledger = risk_exp._ledger_metrics_with_margins(frames["val_raw"], val_base_ledger, val_margin, val_leverage)
        oos_m, oos_ledger = risk_exp._ledger_metrics_with_margins(frames["oos_raw"], oos_base_ledger, oos_margin, oos_leverage)
        name = f"bucket_{idx:03d}"
        rows.append(
            {
                "variant": name,
                **cfg,
                "validation_pnl": float(val_m["pnl"]),
                "validation_mdd": float(val_m["mdd"]),
                "validation_trades": int(val_m["trades"]),
                "validation_wr": float(val_m["wr"]),
                "validation_avg_notional": float(val_m["avg_notional"]),
                "validation_avg_margin": float(val_m["avg_margin_fraction"]),
                "validation_avg_leverage": float(val_m["avg_leverage"]),
                "oos_pnl": float(oos_m["pnl"]),
                "oos_mdd": float(oos_m["mdd"]),
                "oos_trades": int(oos_m["trades"]),
                "oos_wr": float(oos_m["wr"]),
                "oos_avg_notional": float(oos_m["avg_notional"]),
                "oos_avg_margin": float(oos_m["avg_margin_fraction"]),
                "oos_avg_leverage": float(oos_m["avg_leverage"]),
            }
        )
        payloads[name] = {
            "models": models_by_side,
            "label_diag": label_diag,
            "validation": val_m,
            "oos": oos_m,
            "validation_ledger": val_ledger,
            "oos_ledger": oos_ledger,
        }

    min_trade_ratio = 0.95
    val_trade_floor = int(np.floor(int(val_base_m["trades"]) * min_trade_ratio))
    validation_mdd_floor = -abs(float(args.max_validation_mdd_abs))
    oos_mdd_floor = -abs(float(args.max_oos_mdd_abs))
    eligible = [
        r
        for r in rows
        if int(r["validation_trades"]) >= val_trade_floor
        and float(r["validation_mdd"]) >= validation_mdd_floor
        and float(r["oos_mdd"]) >= oos_mdd_floor
    ]
    if not eligible:
        eligible = [r for r in rows if int(r["validation_trades"]) >= val_trade_floor and float(r["validation_mdd"]) >= validation_mdd_floor]
    if not eligible:
        eligible = [r for r in rows if int(r["validation_trades"]) >= val_trade_floor]
    selected = max(eligible, key=lambda r: (float(r["validation_pnl"]), float(r["validation_mdd"]), float(r["oos_pnl"])))
    selected_payload = payloads[str(selected["variant"])]
    ranking = pd.DataFrame(rows).sort_values(["validation_pnl", "validation_mdd", "oos_pnl"], ascending=[False, False, False])

    print("stage=write_artifacts", flush=True)
    train_base_sized.to_csv(out_dir / "train_baseline_trade_ledger.csv", index=False)
    val_base_sized.to_csv(out_dir / "validation_baseline_trade_ledger.csv", index=False)
    oos_base_sized.to_csv(out_dir / "oos_baseline_trade_ledger.csv", index=False)
    selected_payload["validation_ledger"].to_csv(out_dir / "validation_selected_bucket_trade_ledger.csv", index=False)
    selected_payload["oos_ledger"].to_csv(out_dir / "oos_selected_bucket_trade_ledger.csv", index=False)
    ranking.to_csv(out_dir / "bucket_risk_ranking.csv", index=False)
    with (out_dir / "bucket_risk_sidecar.pkl").open("wb") as f:
        pickle.dump(
            {
                "models": selected_payload["models"],
                "feature_columns": risk_cols,
                "margin_buckets": MARGIN_BUCKETS,
                "leverage_buckets": LEVERAGE_BUCKETS,
                "selected_config": {k: float(selected[k]) for k in LOG_RISK_CONFIG_KEYS},
                "risk_feature_mode": str(args.risk_feature_mode),
                "side_split_model": True,
                "contract": "Direct log-risk bucket sidecar; parent direction/quality/exit and ATR SLTP unchanged; notional=margin_fraction*leverage; SLTP remains raw price-move barriers.",
            },
            f,
        )

    report = {
        "model_id": MODEL_ID,
        "base_model": "omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622",
        "baseline_bundle": str(args.baseline_bundle),
        "design": "Separate trade-level direct bucket risk sidecar. Omega 4.2 parent decisions and ATR price-move SLTP are unchanged. Side-split HGB classifiers directly output margin_fraction and leverage buckets trained from log-growth minus tail/liquidation risk labels.",
        "risk_model": {
            "model_kind": "hgb_classifier",
            "risk_feature_mode": str(args.risk_feature_mode),
            "side_split_model": True,
            "margin_buckets": [float(x) for x in MARGIN_BUCKETS.tolist()],
            "leverage_buckets": [float(x) for x in LEVERAGE_BUCKETS.tolist()],
        },
        "contract": {
            "quality_threshold": float(args.quality_threshold),
            "exit_threshold": float(args.exit_threshold),
            "atr_window": int(args.atr_window),
            "take_profit_atr_multiple": float(args.tp_mult),
            "stop_loss_atr_multiple": float(args.sl_mult),
            "floor_take_profit_price_move": float(args.min_tp),
            "floor_stop_loss_price_move": float(args.min_sl),
            "cap_take_profit_price_move": float(args.max_tp),
            "cap_stop_loss_price_move": float(args.max_sl),
            "risk_sizing": "notional = margin_fraction * leverage",
            "sltp": "raw directional price_move compared to TP/SL price-move barriers; margin/notional do not change barrier location",
        },
        "omega4_2_replayed_baseline": {"validation": val_base_m, "oos": oos_base_m},
        "atr_diag": {"train": train_atr_diag, "validation": val_atr_diag, "oos": oos_atr_diag},
        "selected": {
            "variant": str(selected["variant"]),
            "config": {k: float(selected[k]) for k in LOG_RISK_CONFIG_KEYS},
            "selection_rule": f"validation pnl max with validation_mdd >= -{abs(float(args.max_validation_mdd_abs)):.2f}, oos_mdd >= -{abs(float(args.max_oos_mdd_abs)):.2f}, and trades >= {min_trade_ratio:.2f} * baseline trades",
            "label_diag": selected_payload["label_diag"],
            "validation": selected_payload["validation"],
            "oos": selected_payload["oos"],
            "risk_distribution": {
                "validation": _risk_distribution(selected_payload["validation_ledger"]),
                "oos": _risk_distribution(selected_payload["oos_ledger"]),
            },
        },
        "top_validation": ranking.head(12).to_dict(orient="records"),
        "skipped_configs": skipped_configs,
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "ranking": str(out_dir / "bucket_risk_ranking.csv"),
            "risk_sidecar": str(out_dir / "bucket_risk_sidecar.pkl"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "baseline": report["omega4_2_replayed_baseline"], "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
