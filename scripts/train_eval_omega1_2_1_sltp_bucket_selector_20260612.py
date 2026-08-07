#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402


MODEL_ID = "omega1_2_1_sltp_bucket_selector_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class Bucket:
    bucket_id: int
    tp_mult: float
    sl_mult: float


BUCKETS = tuple(
    Bucket(i, tp, sl)
    for i, (tp, sl) in enumerate(
        [
            (0.45, 0.65),
            (0.55, 0.65),
            (0.70, 0.65),
            (0.85, 0.80),
            (1.00, 1.00),
            (0.70, 1.00),
            (0.55, 1.00),
        ]
    )
)


FEATURE_COLS = [
    "bar_range_pct",
    "atr14_pct",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_vol_12",
    "ret_vol_24",
    "ema9_21_gap",
    "tod_sin",
    "tod_cos",
    "tabm_router_confidence",
    "tabm_router_margin",
    "tabm_dir_confidence",
    "tabm_dir_side_edge",
    "tabm_dir_trade_prob",
    "tabm_quality_for_action",
    "dec_side",
    "dec_notional_exposure",
    "dec_leverage",
    "dec_rr",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _apply_bucket_to_row(dec: pd.DataFrame, i: int, bucket: Bucket) -> pd.DataFrame:
    out = dec.copy()
    out.loc[int(i), "take_profit"] = float(out.loc[int(i), "take_profit"]) * float(bucket.tp_mult)
    out.loc[int(i), "stop_loss"] = abs(float(out.loc[int(i), "stop_loss"])) * float(bucket.sl_mult)
    return out


def _single_trade_return(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    i: int,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[float, str]:
    arrays = runner.base._arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash, pos, entered = runner.base._enter(1.0, arrays, dec, int(i), fee_eff, slip_eff)
    if not entered:
        return -1e9, "no_fill"
    for j in range(int(pos.entry_i), len(frame) - 1):
        unreal = runner.base._unreal(arrays, pos, j, slip_eff)
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        reason = runner.base._hit_reason(unreal, pos)
        if reason:
            cash, _pos, _ = runner.base._close_fraction(cash, arrays, pos, j, 1.0, fee_eff, slip_eff)
            return float(cash - 1.0), reason
    cash, _pos, _ = runner.base._close_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
    return float(cash - 1.0), "forced_end"


def _build_dataset(data: dict[str, dict[str, Any]], split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = data[split]
    frame = payload["frame"]
    dec = payload["dec"].reset_index(drop=True)
    state = payload["state"].reset_index(drop=True)
    active = np.flatnonzero(runner.base.omega._active(dec))
    rows: list[dict[str, Any]] = []
    for i in active:
        if int(i) >= len(frame) - 2:
            continue
        rewards: list[float] = []
        reasons: list[str] = []
        for bucket in BUCKETS:
            bdec = _apply_bucket_to_row(dec, int(i), bucket)
            r, reason = _single_trade_return(
                frame,
                bdec,
                int(i),
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
            )
            rewards.append(float(r))
            reasons.append(reason)
        best = int(np.argmax(rewards))
        row = {c: float(state.iloc[int(i)].get(c, 0.0)) for c in FEATURE_COLS}
        row.update(
            {
                "i": int(i),
                "best_bucket": best,
                "best_reward": float(rewards[best]),
                "base_reward": float(rewards[4]),
                "best_reason": reasons[best],
            }
        )
        for bucket, reward in zip(BUCKETS, rewards):
            row[f"reward_b{bucket.bucket_id}"] = float(reward)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"empty selector dataset: {split}")
    diag = {
        "split": split,
        "rows": int(len(df)),
        "best_bucket_counts": {str(k): int(v) for k, v in df["best_bucket"].value_counts().sort_index().items()},
        "best_reward_mean": float(df["best_reward"].mean()),
        "base_reward_mean": float(df["base_reward"].mean()),
    }
    return df, diag


def _predict_bucket(clf: Any | None, state: pd.DataFrame, i: int, *, fallback_bucket: int) -> Bucket:
    if clf is None:
        return BUCKETS[int(fallback_bucket)]
    x = np.asarray([[float(state.iloc[int(i)].get(c, 0.0)) for c in FEATURE_COLS]], dtype=np.float64)
    pred = int(clf.predict(x)[0])
    pred = int(np.clip(pred, 0, len(BUCKETS) - 1))
    return BUCKETS[pred]


def _simulate_selector(
    payload: dict[str, Any],
    *,
    clf: Any | None,
    fallback_bucket: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = payload["frame"]
    dec = payload["dec"].reset_index(drop=True)
    state = payload["state"].reset_index(drop=True)
    arrays = runner.base._arrays(frame)
    active = np.asarray(runner.base.omega._active(dec), dtype=bool)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    cash = 1.0
    equity_curve = [cash]
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    bucket_counts: dict[int, int] = {}
    pos = runner.base.Position()
    long_entries = short_entries = 0
    current_bucket = -1
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = runner.base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            equity_curve.append(cash * (1.0 + unreal))
            reason = runner.base._hit_reason(unreal, pos)
            if reason:
                close_pos = runner.base.Position(**pos.__dict__)
                cash, pos, _ = runner.base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                item = runner._ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason, 0)
                item["sltp_bucket"] = int(current_bucket)
                rows.append(item)
                current_bucket = -1
            continue
        equity_curve.append(cash)
        if not bool(active[i]):
            continue
        bucket = _predict_bucket(clf, state, i, fallback_bucket=int(fallback_bucket))
        bdec = _apply_bucket_to_row(dec, i, bucket)
        side = int(bdec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = runner.base._enter(cash, arrays, bdec, i, fee_eff, slip_eff)
        if entered:
            current_bucket = int(bucket.bucket_id)
            bucket_counts[current_bucket] = bucket_counts.get(current_bucket, 0) + 1
            long_entries += int(side > 0)
            short_entries += int(side < 0)
    if pos.side != 0:
        close_pos = runner.base.Position(**pos.__dict__)
        cash, pos, _ = runner.base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        item = runner._ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end", 0)
        item["sltp_bucket"] = int(current_bucket)
        rows.append(item)
    metrics = runner._metric(cash, equity_curve, trades, reasons, long_entries, short_entries)
    metrics["bucket_counts"] = {str(k): int(v) for k, v in sorted(bucket_counts.items())}
    return metrics, pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_bucket_counts": metrics["bucket_counts"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    train_df, train_diag = _build_dataset(data, "validation")
    oos_label_df, oos_label_diag = _build_dataset(data, "oos")
    train_df.to_csv(OUT_DIR / "validation_sltp_bucket_training.csv", index=False)
    oos_label_df.to_csv(OUT_DIR / "oos_sltp_bucket_oracle_labels_reference.csv", index=False)

    x = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = train_df["best_bucket"].astype(int).to_numpy()
    clf = HistGradientBoostingClassifier(
        max_iter=80,
        max_leaf_nodes=8,
        min_samples_leaf=12,
        l2_regularization=1.0,
        learning_rate=0.04,
        random_state=260612,
    )
    clf.fit(x, y)

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    variants = [
        ("fixed_tp_runner_only", None, 4),
        ("fixed_narrow_best_tradecount", None, 1),
        ("hgb_sltp_bucket_selector", clf, 4),
    ]
    for name, model, fallback_bucket in variants:
        row: dict[str, Any] = {"variant": name, "fallback_bucket": int(fallback_bucket)}
        for split in ("validation", "oos"):
            metrics, ledger = _simulate_selector(data[split], clf=model, fallback_bucket=int(fallback_bucket))
            row.update(_row("val" if split == "validation" else "oos", metrics))
            ledgers[f"{split}_{name}"] = ledger
        rows.append(row)
    ranking = pd.DataFrame(rows)
    base = ranking[ranking["variant"].eq("fixed_tp_runner_only")].iloc[0]
    ranking["delta_val_vs_tp_runner"] = ranking["val_pnl"] - float(base["val_pnl"])
    ranking["delta_oos_vs_tp_runner"] = ranking["oos_pnl"] - float(base["oos_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(base["oos_trades"])
    ranking = ranking.sort_values(["oos_pnl", "val_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "sltp_bucket_selector_ranking.csv", index=False)
    for name, ledger in ledgers.items():
        ledger.to_csv(OUT_DIR / f"{name}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "buckets": [bucket.__dict__ for bucket in BUCKETS],
        "feature_cols": FEATURE_COLS,
        "train_diag": train_diag,
        "oos_label_diag_reference_only": oos_label_diag,
        "top": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "sltp_bucket_selector_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "ranking": ranking.to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
