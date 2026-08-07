#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega3_margin_cap1_bucket_20260618 as exp  # noqa: E402


MODEL_ID = "regime_volatility_ranges_omega3_inputs_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REGIME_PROBS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
]
REGIME_LABELS = ["bull", "bear", "chop"]
HORIZONS = (1, 3, 6, 12, 24, 48)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _features(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    df = frame.reset_index(drop=True).copy()
    close = pd.to_numeric(df["close"], errors="raise")
    high = pd.to_numeric(df["high"], errors="raise")
    low = pd.to_numeric(df["low"], errors="raise")
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    out = pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"], errors="raise"), "split": split})
    out["ret_1_abs"] = close.pct_change().abs().replace([np.inf, -np.inf], np.nan)
    out["bar_range_pct"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    out["atr14_pct"] = (atr14 / close).replace([np.inf, -np.inf], np.nan)
    out["realized_vol_24"] = close.pct_change().rolling(24, min_periods=6).std().replace([np.inf, -np.inf], np.nan)
    for horizon in HORIZONS:
        out[f"fwd_abs_ret_{horizon}"] = (close.shift(-horizon) / close - 1.0).abs().replace([np.inf, -np.inf], np.nan)
        out[f"fwd_range_{horizon}"] = (
            (high.shift(-horizon + 1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1)) - low.shift(-horizon + 1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1)))
            / close
        ).replace([np.inf, -np.inf], np.nan)

    missing = [c for c in REGIME_PROBS if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing active regime columns: {missing}")
    probs = df[REGIME_PROBS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    out["regime3_current"] = np.asarray(REGIME_LABELS, dtype=object)[np.argmax(probs, axis=1)]
    out["regime3_confidence"] = np.max(probs, axis=1)
    out["regime3_margin"] = pd.to_numeric(df.get("regime3_current_sensitive_wide24_margin", 0.0), errors="coerce").fillna(0.0)
    out["regime3_stability_h6_score"] = pd.to_numeric(df.get("regime3_stability_h6_score", 0.0), errors="coerce").fillna(0.0)
    out["regime3_transition_h6_risk_prob"] = pd.to_numeric(df.get("regime3_transition_h6_risk_prob", 0.0), errors="coerce").fillna(0.0)
    out["vol_bucket_atr14"] = pd.qcut(out["atr14_pct"].rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"])
    return out.replace([np.inf, -np.inf], np.nan)


def _summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = ["bar_range_pct", "atr14_pct", "realized_vol_24", "ret_1_abs"]
    metrics += [f"fwd_abs_ret_{h}" for h in HORIZONS]
    metrics += [f"fwd_range_{h}" for h in HORIZONS]
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: str(val) for col, val in zip(group_cols, key)}
        row["rows"] = int(len(g))
        row["start"] = str(g["timestamp"].min())
        row["end"] = str(g["timestamp"].max())
        row["regime3_confidence_median"] = float(g["regime3_confidence"].median())
        row["transition_risk_median"] = float(g["regime3_transition_h6_risk_prob"].median())
        for metric in metrics:
            s = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(s) == 0:
                row[f"{metric}_mean"] = 0.0
                row[f"{metric}_p50"] = 0.0
                row[f"{metric}_p75"] = 0.0
                row[f"{metric}_p90"] = 0.0
                row[f"{metric}_p95"] = 0.0
                continue
            row[f"{metric}_mean"] = float(s.mean())
            row[f"{metric}_p50"] = float(s.quantile(0.50))
            row[f"{metric}_p75"] = float(s.quantile(0.75))
            row[f"{metric}_p90"] = float(s.quantile(0.90))
            row[f"{metric}_p95"] = float(s.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _markdown(summary: dict[str, Any], by_regime: pd.DataFrame, by_split_regime: pd.DataFrame, by_regime_vol: pd.DataFrame) -> str:
    def table(frame: pd.DataFrame, cols: list[str]) -> str:
        data = frame[cols].copy()
        for col in data.columns:
            if pd.api.types.is_float_dtype(data[col]):
                data[col] = data[col].map(lambda x: f"{float(x):.6f}")
            else:
                data[col] = data[col].map(str)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in data.to_numpy(dtype=str)]
        return "\n".join([header, sep, *body])

    lines = [
        "# Regime Volatility Range Audit",
        "",
        f"- model_id: `{MODEL_ID}`",
        f"- generated_at: `{pd.Timestamp.now(tz='Asia/Seoul').isoformat()}`",
        f"- source: Omega3/full-retrain parent validation+OOS frames",
        f"- primary regime key: `regime3_current_sensitive_wide24_[bull,bear,chop]_prob` argmax",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        "```",
        "",
        "## By Regime",
        "",
        table(
            by_regime,
            [
                "regime3_current",
                "rows",
                "bar_range_pct_p50",
                "bar_range_pct_p90",
                "atr14_pct_p50",
                "atr14_pct_p90",
                "fwd_abs_ret_24_p50",
                "fwd_abs_ret_24_p90",
                "fwd_range_24_p50",
                "fwd_range_24_p90",
            ],
        ),
        "",
        "## By Split And Regime",
        "",
        table(
            by_split_regime,
            [
                "split",
                "regime3_current",
                "rows",
                "bar_range_pct_p50",
                "bar_range_pct_p90",
                "atr14_pct_p50",
                "atr14_pct_p90",
                "fwd_abs_ret_24_p50",
                "fwd_abs_ret_24_p90",
            ],
        ),
        "",
        "## By Regime And ATR Vol Bucket",
        "",
        table(
            by_regime_vol,
            [
                "regime3_current",
                "vol_bucket_atr14",
                "rows",
                "atr14_pct_p50",
                "atr14_pct_p90",
                "fwd_abs_ret_24_p50",
                "fwd_abs_ret_24_p90",
                "fwd_range_24_p50",
                "fwd_range_24_p90",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_payload, oos_payload, _meta = exp._build_payloads()
    val = _features(val_payload["frame"], "validation")
    oos = _features(oos_payload["frame"], "oos")
    df = pd.concat([val, oos], ignore_index=True)
    by_regime = _summarize(df, ["regime3_current"])
    by_split_regime = _summarize(df, ["split", "regime3_current"])
    by_regime_vol = _summarize(df, ["regime3_current", "vol_bucket_atr14"])
    summary = {
        "rows": int(len(df)),
        "validation_rows": int(len(val)),
        "oos_rows": int(len(oos)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "regime_counts": {str(k): int(v) for k, v in df["regime3_current"].value_counts().sort_index().items()},
        "vol_bucket_counts": {str(k): int(v) for k, v in df["vol_bucket_atr14"].value_counts().sort_index().items()},
    }
    df.to_parquet(OUT_DIR / "regime_volatility_rows.parquet", index=False)
    by_regime.to_csv(OUT_DIR / "by_regime.csv", index=False)
    by_split_regime.to_csv(OUT_DIR / "by_split_regime.csv", index=False)
    by_regime_vol.to_csv(OUT_DIR / "by_regime_vol_bucket.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "summary": summary,
        "artifacts": {
            "rows": str(OUT_DIR / "regime_volatility_rows.parquet"),
            "by_regime": str(OUT_DIR / "by_regime.csv"),
            "by_split_regime": str(OUT_DIR / "by_split_regime.csv"),
            "by_regime_vol_bucket": str(OUT_DIR / "by_regime_vol_bucket.csv"),
            "markdown": str(OUT_DIR / "report.md"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    (OUT_DIR / "report.md").write_text(_markdown(summary, by_regime, by_split_regime, by_regime_vol), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
