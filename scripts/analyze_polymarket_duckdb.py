import argparse
import json
import math
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


RANGE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")
LT_RE = re.compile(r"^\s*<\s*([0-9]+(?:\.[0-9]+)?)\s*$")
GT_RE = re.compile(r"^\s*>\s*([0-9]+(?:\.[0-9]+)?)\s*$")
NUM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _safe_float(v, default=0.0):
    try:
        x = float(v)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _parse_label_center(label: str) -> float | None:
    text = str(label or "").strip()
    if not text:
        return None
    m = RANGE_RE.match(text)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = LT_RE.match(text)
    if m:
        return float(m.group(1))
    m = GT_RE.match(text)
    if m:
        return float(m.group(1))
    m = NUM_RE.match(text)
    if m:
        return float(m.group(1))
    return None


def _norm_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if len(probs) <= 1:
        return 0.0
    ent = -float(np.sum(probs * np.log(probs)))
    return float(ent / max(np.log(len(probs)), 1e-12))


def _locate_prev_values(ts_series: pd.Series, value_series: pd.Series, sec: float) -> pd.Series:
    out = []
    times = ts_series.tolist()
    vals = value_series.tolist()
    j = 0
    for i, now_ts in enumerate(times):
        cutoff = now_ts - pd.Timedelta(seconds=float(sec))
        while j + 1 < i and times[j + 1] <= cutoff:
            j += 1
        if i == 0 or times[j] > cutoff:
            out.append(np.nan)
        else:
            out.append(vals[j])
    return pd.Series(out, index=ts_series.index, dtype=float)


def _load_raw_df(db_path: str, table: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        tables = {x[0] for x in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            raise ValueError(f"table not found: {table}; available={sorted(tables)}")
        df = con.execute(f"SELECT ts, markets_json FROM {table} ORDER BY ts").df()
    finally:
        con.close()
    if df.empty:
        raise ValueError("no rows in table")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def _expand_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in df.itertuples(index=False):
        try:
            markets = json.loads(rec.markets_json)
        except Exception:
            markets = []
        compact = []
        for item in list(markets or []):
            label = str((item or {}).get("label", "") or "").strip()
            prob = _safe_float((item or {}).get("prob", 0.0), 0.0)
            center = _parse_label_center(label)
            compact.append({"label": label, "prob": prob, "center": center})
        compact = [x for x in compact if x["label"]]
        compact.sort(key=lambda x: x["prob"], reverse=True)
        probs = np.array([x["prob"] for x in compact], dtype=float)
        probs = np.clip(probs, 0.0, None)
        prob_sum = float(np.sum(probs))
        if prob_sum > 1e-12:
            probs = probs / prob_sum
        centers = np.array([x["center"] if x["center"] is not None else np.nan for x in compact], dtype=float)
        center_ok = np.isfinite(centers)
        weighted_target = (
            float(np.sum(centers[center_ok] * probs[center_ok]) / max(np.sum(probs[center_ok]), 1e-12))
            if np.any(center_ok)
            else np.nan
        )
        weighted_std = (
            float(np.sqrt(np.sum(((centers[center_ok] - weighted_target) ** 2) * probs[center_ok]) / max(np.sum(probs[center_ok]), 1e-12)))
            if np.any(center_ok)
            else np.nan
        )
        mode = compact[0] if compact else {"label": "", "prob": 0.0, "center": np.nan}
        second_prob = compact[1]["prob"] if len(compact) > 1 else 0.0
        top_labels = [x["label"] for x in compact[:5]]
        top_probs = [float(x["prob"]) for x in compact[:5]]
        support_mask = probs >= 0.10
        if np.any(center_ok & support_mask):
            support_low = float(np.nanmin(centers[center_ok & support_mask]))
            support_high = float(np.nanmax(centers[center_ok & support_mask]))
        elif np.any(center_ok):
            support_low = float(np.nanmin(centers[center_ok]))
            support_high = float(np.nanmax(centers[center_ok]))
        else:
            support_low = np.nan
            support_high = np.nan
        up_mass = 0.0
        down_mass = 0.0
        if np.any(center_ok) and np.isfinite(weighted_target):
            up_mass = float(np.sum(probs[center_ok & (centers > weighted_target)]))
            down_mass = float(np.sum(probs[center_ok & (centers < weighted_target)]))
        rows.append(
            {
                "ts": rec.ts,
                "bucket_count": int(len(compact)),
                "mode_label": str(mode["label"]),
                "mode_prob": float(mode["prob"]),
                "mode_center": _safe_float(mode["center"], np.nan),
                "top2_gap": float(float(mode["prob"]) - float(second_prob)),
                "weighted_target": _safe_float(weighted_target, np.nan),
                "weighted_std": _safe_float(weighted_std, np.nan),
                "entropy_norm": _norm_entropy(probs),
                "concentration_hhi": float(np.sum(probs ** 2)) if len(probs) else 0.0,
                "support_low": _safe_float(support_low, np.nan),
                "support_high": _safe_float(support_high, np.nan),
                "support_width": _safe_float(support_high - support_low, np.nan),
                "up_mass": up_mass,
                "down_mass": down_mass,
                "skew_up_minus_down": float(up_mass - down_mass),
                "markets_json": rec.markets_json,
                "top_labels_json": json.dumps(top_labels, ensure_ascii=False),
                "top_probs_json": json.dumps(top_probs, ensure_ascii=False),
            }
        )
    feat = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    feat["dt_sec"] = feat["ts"].diff().dt.total_seconds()
    feat["mode_changed"] = (feat["mode_label"] != feat["mode_label"].shift(1)).astype(int)
    for sec, tag in ((60.0, "1m"), (180.0, "3m"), (600.0, "10m")):
        prev_target = _locate_prev_values(feat["ts"], feat["weighted_target"], sec)
        prev_mode_prob = _locate_prev_values(feat["ts"], feat["mode_prob"], sec)
        feat[f"target_delta_{tag}"] = feat["weighted_target"] - prev_target
        feat[f"mode_prob_delta_{tag}"] = feat["mode_prob"] - prev_mode_prob
    feat["future_abs_target_delta_1m"] = feat["target_delta_1m"].shift(-1).abs()
    feat["future_abs_target_delta_3m"] = feat["target_delta_3m"].shift(-1).abs()
    return feat


def _corr(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 20:
        return 0.0
    return _safe_float(pair.iloc[:, 0].corr(pair.iloc[:, 1]), 0.0)


def _write_markdown_report(out_path: Path, summary: dict, ideas: list[str]) -> None:
    lines = [
        "# Polymarket DuckDB Analysis",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- UTC range: `{summary['ts_start_utc']}` -> `{summary['ts_end_utc']}`",
        f"- Median interval: `{summary['interval_seconds']['median_sec']:.2f}s`",
        f"- Mode change ratio: `{summary['mode_change_ratio_pct']:.2f}%`",
        f"- Avg top1 prob: `{summary['avg_mode_prob']:.4f}`",
        f"- Avg top2 gap: `{summary['avg_top2_gap']:.4f}`",
        f"- Avg entropy(norm): `{summary['avg_entropy_norm']:.4f}`",
        f"- Avg weighted std: `{summary['avg_weighted_std']:.2f}`",
        "",
        "## Suggested Real-Time Uses",
        "",
    ]
    lines.extend([f"- {idea}" for idea in ideas])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(db_path: str, table: str, out_dir: str, top_n: int) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_df = _load_raw_df(db_path, table)
    feat = _expand_features(raw_df)

    diffs = feat["dt_sec"].dropna()
    mode_counts = feat["mode_label"].value_counts().head(10).to_dict()
    top_target = feat.assign(abs_move=feat["target_delta_1m"].abs()).sort_values("abs_move", ascending=False).head(max(1, int(top_n)))
    top_flips = feat[feat["mode_changed"] == 1].head(0) if feat.empty else feat[feat["mode_changed"] == 1].tail(max(1, int(top_n)))

    summary = {
        "db_path": str(Path(db_path).resolve()),
        "table": table,
        "rows": int(len(feat)),
        "ts_start_utc": str(feat["ts"].min()),
        "ts_end_utc": str(feat["ts"].max()),
        "interval_seconds": {
            "median_sec": _safe_float(diffs.median(), 0.0) if len(diffs) else 0.0,
            "mean_sec": _safe_float(diffs.mean(), 0.0) if len(diffs) else 0.0,
            "p90_sec": _safe_float(diffs.quantile(0.9), 0.0) if len(diffs) else 0.0,
            "max_sec": _safe_float(diffs.max(), 0.0) if len(diffs) else 0.0,
        },
        "mode_counts_top10": mode_counts,
        "mode_change_ratio_pct": _safe_float(feat["mode_changed"].mean() * 100.0, 0.0),
        "avg_mode_prob": _safe_float(feat["mode_prob"].mean(), 0.0),
        "avg_top2_gap": _safe_float(feat["top2_gap"].mean(), 0.0),
        "avg_entropy_norm": _safe_float(feat["entropy_norm"].mean(), 0.0),
        "avg_weighted_std": _safe_float(feat["weighted_std"].mean(), 0.0),
        "avg_support_width": _safe_float(feat["support_width"].mean(), 0.0),
        "corr_top1_future_abs_target_move_1m": _corr(feat["mode_prob"], feat["future_abs_target_delta_1m"]),
        "corr_entropy_future_abs_target_move_1m": _corr(feat["entropy_norm"], feat["future_abs_target_delta_1m"]),
        "corr_top2gap_future_abs_target_move_1m": _corr(feat["top2_gap"], feat["future_abs_target_delta_1m"]),
        "corr_weighted_std_future_abs_target_move_1m": _corr(feat["weighted_std"], feat["future_abs_target_delta_1m"]),
        "corr_skew_future_target_delta_1m": _corr(feat["skew_up_minus_down"], feat["target_delta_1m"].shift(-1)),
    }

    ideas = [
        "Use `weighted_target` as a smooth fair-value anchor and trade the gap versus live ETH price rather than using only the top bucket label.",
        "Use `target_delta_1m` and `target_delta_3m` as event repricing momentum. These are cleaner than raw bucket flips for fast overlays.",
        "Use `mode_prob`, `top2_gap`, and `entropy_norm` as confidence filters. High top1 and low entropy mean the market is concentrated enough to trust.",
        "Use `weighted_std` or `support_width` as uncertainty bands. When the distribution is wide, reduce leverage or veto entries.",
        "Use `mode_changed` bursts as shock markers. Rapid mode flips indicate regime instability even when the top bucket itself does not move far.",
        "Join these Polymarket features with live ETH price so you can compute `price_vs_weighted_target`, `price_vs_support_low/high`, and lead-lag statistics.",
        "Promote Polymarket from an exit-only guard to a sizing overlay: increase conviction only when ETH direction aligns with rising weighted target and strong top1 concentration.",
    ]

    summary_path = out / "polymarket_duckdb_summary.json"
    feature_path = out / "polymarket_duckdb_features.csv"
    move_path = out / "polymarket_duckdb_top_target_moves.csv"
    flip_path = out / "polymarket_duckdb_recent_mode_flips.csv"
    report_path = out / "polymarket_duckdb_report.md"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    feat.drop(columns=["markets_json"]).to_csv(feature_path, index=False)
    top_target.drop(columns=["markets_json"]).to_csv(move_path, index=False)
    top_flips.drop(columns=["markets_json"]).to_csv(flip_path, index=False)
    _write_markdown_report(report_path, summary, ideas)

    print("[OK] polymarket duckdb analysis complete")
    print(f"- rows: {summary['rows']}")
    print(f"- range(UTC): {summary['ts_start_utc']} -> {summary['ts_end_utc']}")
    print(f"- median interval sec: {summary['interval_seconds']['median_sec']:.2f}")
    print(f"- avg top1 prob: {summary['avg_mode_prob']:.4f}")
    print(f"- avg entropy(norm): {summary['avg_entropy_norm']:.4f}")
    print(f"- corr(top1, future_abs_move_1m): {summary['corr_top1_future_abs_target_move_1m']:.4f}")
    print(f"- corr(entropy, future_abs_move_1m): {summary['corr_entropy_future_abs_target_move_1m']:.4f}")
    print(f"- saved: {summary_path}")
    print(f"- saved: {feature_path}")
    print(f"- saved: {move_path}")
    print(f"- saved: {flip_path}")
    print(f"- saved: {report_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Analyze raw Polymarket DuckDB market-distribution snapshots.")
    parser.add_argument("--db", default="data/live/polymarket.duckdb", help="DuckDB path")
    parser.add_argument("--table", default="polymarket_markets_10s_json", help="Raw table name")
    parser.add_argument("--out", default="data/live", help="Output directory")
    parser.add_argument("--top-n", type=int, default=50, help="Number of large moves / flips to export")
    args = parser.parse_args()
    raise SystemExit(analyze(args.db, args.table, args.out, args.top_n))


if __name__ == "__main__":
    main()
