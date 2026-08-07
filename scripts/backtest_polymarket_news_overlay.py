#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


KST = "Asia/Seoul"
SYMBOL = "ETHUSDT"


@dataclass
class Trade:
    side: str
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    open_price: float
    close_price: float
    realized_pct: float


@dataclass(frozen=True)
class OverlayConfig:
    entry_gap_th: float
    exit_gap_th: float
    shock_th: float
    tail_th: float
    aftershock_cap: float
    toxicity_cap: float
    neutral_gap_th: float
    entropy_cap: float

    @property
    def name(self) -> str:
        return (
            f"g{self.entry_gap_th:.4f}_x{self.exit_gap_th:.4f}_s{self.shock_th:.2f}"
            f"_t{self.tail_th:.2f}_a{self.aftershock_cap:.2f}_z{self.toxicity_cap:.2f}"
        )


def _parse_ts_kst(v) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(v)
        if ts.tzinfo is None:
            ts = ts.tz_localize(KST)
        return ts.tz_convert("UTC")
    except Exception:
        return None


def _parse_poly_value(label: str, mode: str = "mid", tilted_alpha: float = 0.75) -> float | None:
    nums: list[float] = []
    for token in str(label).replace(">", "").replace("<", "").split("-"):
        token = token.strip().replace(",", "")
        try:
            nums.append(float(token))
        except Exception:
            continue
    if not nums:
        return None
    if len(nums) == 1:
        return float(nums[0])
    lo = float(min(nums))
    hi = float(max(nums))
    if mode == "upper":
        return hi
    if mode == "tilted_upper":
        alpha = float(np.clip(tilted_alpha, 0.0, 1.0))
        return lo + ((hi - lo) * alpha)
    return float((lo + hi) / 2.0)


def _load_trades(events_path: str, start_utc: pd.Timestamp | None = None, end_utc: pd.Timestamp | None = None) -> list[Trade]:
    rows: list[dict] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    rows.sort(key=lambda x: str(x.get("ts", "")))

    out: list[Trade] = []
    open_pos: dict | None = None
    for row in rows:
        ts = _parse_ts_kst(row.get("ts"))
        if ts is None:
            continue
        px = float(row.get("price", 0.0) or 0.0)
        frm = str(row.get("from", "") or "").upper()
        to = str(row.get("to", "") or "").upper()
        pnl = float(row.get("pnl_pct", 0.0) or 0.0)
        if frm in {"LONG", "SHORT"} and open_pos and open_pos.get("side") == frm and px > 0.0:
            trade = Trade(
                side=frm,
                open_ts=open_pos["ts"],
                close_ts=ts,
                open_price=float(open_pos["price"]),
                close_price=px,
                realized_pct=pnl,
            )
            if (start_utc is None or trade.close_ts >= start_utc) and (end_utc is None or trade.open_ts <= end_utc):
                out.append(trade)
            open_pos = None
        if to in {"LONG", "SHORT"} and px > 0.0:
            open_pos = {"side": to, "ts": ts, "price": px}
    return out


def _net_frac(side: str, entry: float, exitp: float, lev: float, fee: float, slip: float) -> float:
    if side == "LONG":
        en = entry * (1.0 + slip)
        ex = exitp * (1.0 - slip)
        gross = (ex - en) / max(en, 1e-12)
    else:
        en = entry * (1.0 - slip)
        ex = exitp * (1.0 + slip)
        gross = (en - ex) / max(abs(en), 1e-12)
    return float((gross * lev) - (2.0 * fee * lev))


def _est_lev(tr: Trade, fee: float, slip: float) -> float:
    if tr.side == "LONG":
        en = tr.open_price * (1.0 + slip)
        ex = tr.close_price * (1.0 - slip)
        gross = (ex - en) / max(en, 1e-12)
    else:
        en = tr.open_price * (1.0 - slip)
        ex = tr.close_price * (1.0 + slip)
        gross = (en - ex) / max(abs(en), 1e-12)
    denom = gross - (2.0 * fee)
    if abs(denom) <= 1e-10:
        return 0.0
    lev = (tr.realized_pct / 100.0) / denom
    if not np.isfinite(lev):
        return 0.0
    return float(np.clip(lev, 0.0, 3.0))


def _mdd(eq_curve: list[float]) -> float:
    if not eq_curve:
        return 0.0
    eq = np.array(eq_curve, dtype=np.float64)
    run_max = np.maximum.accumulate(eq)
    dd = eq / np.maximum(run_max, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _sharpe(returns_pct: list[float]) -> float:
    if len(returns_pct) < 2:
        return 0.0
    arr = np.asarray(returns_pct, dtype=np.float64) / 100.0
    sd = float(arr.std())
    if sd <= 1e-12:
        return 0.0
    return float(arr.mean() / sd * math.sqrt(len(arr)))


def _fetch_binance_1m(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    import json as _json
    from urllib.request import urlopen

    start_ms = int(start_utc.floor("min").timestamp() * 1000)
    end_ms = int(end_utc.ceil("min").timestamp() * 1000)
    cursor = start_ms
    rows: list[list] = []
    while cursor <= end_ms:
        url = (
            "https://fapi.binance.com/fapi/v1/klines"
            f"?symbol={SYMBOL}&interval=1m&startTime={cursor}&endTime={end_ms}&limit=1500"
        )
        raw = _json.loads(urlopen(url, timeout=20).read().decode("utf-8"))
        if not isinstance(raw, list) or not raw:
            break
        rows.extend(raw)
        nxt = int(raw[-1][0]) + 60_000
        if nxt <= cursor:
            break
        cursor = nxt
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    px = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4]]
    px.columns = ["open_time", "open", "high", "low", "close"]
    px["ts"] = pd.to_datetime(px["open_time"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close"):
        px[col] = pd.to_numeric(px[col], errors="coerce")
    px = px.dropna(subset=["ts", "open", "high", "low", "close"]).drop_duplicates(subset=["ts"]).sort_values("ts")
    return px[["ts", "open", "high", "low", "close"]].reset_index(drop=True)


def _load_duckdb_features(bucket_mode: str = "mid", tilted_alpha: float = 0.75) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    con = duckdb.connect("data/live/polymarket.duckdb", read_only=True)
    poly = con.execute("select ts, markets_json from polymarket_markets_10s_json order by ts").fetchdf()
    con.close()

    con = duckdb.connect("data/live/microstructure.duckdb", read_only=True)
    ms = con.execute("select * from microstructure_1m order by ts").fetchdf()
    con.close()

    con = duckdb.connect("data/live/tail_risk.duckdb", read_only=True)
    tr = con.execute("select * from tail_risk_1m order by ts").fetchdf()
    con.close()

    start_ts = pd.Timestamp(poly["ts"].min()).tz_convert("UTC")
    end_ts = pd.Timestamp(poly["ts"].max()).tz_convert("UTC")

    px = _fetch_binance_1m(start_ts - pd.Timedelta(minutes=5), end_ts + pd.Timedelta(minutes=5))
    px_kst = px.copy()
    px_kst["ts"] = px_kst["ts"].dt.tz_convert(KST)
    price_lookup = px_kst.set_index("ts")["close"]

    records: list[dict] = []
    iterator = poly.itertuples(index=False)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(poly), desc="poly-10s parse", ncols=100)
    for row in iterator:
        ts = pd.Timestamp(row.ts)
        close_px = float(price_lookup.asof(ts.floor("s")) if len(price_lookup) else np.nan)
        if not np.isfinite(close_px):
            continue
        try:
            arr = json.loads(row.markets_json)
        except Exception:
            continue
        centers: list[float] = []
        probs: list[float] = []
        for item in arr:
            center = _parse_poly_value(item.get("label", ""), mode=bucket_mode, tilted_alpha=tilted_alpha)
            prob = float(item.get("prob", 0.0) or 0.0)
            if center is None or not np.isfinite(prob):
                continue
            centers.append(center)
            probs.append(prob)
        if not probs:
            continue
        p = np.asarray(probs, dtype=np.float64)
        c = np.asarray(centers, dtype=np.float64)
        ps = float(p.sum())
        w = p / max(ps, 1e-12)
        weighted_target = float((w * c).sum())
        order = np.argsort(p)[::-1]
        top_prob = float(p[order[0]])
        second_prob = float(p[order[1]]) if len(order) > 1 else 0.0
        entropy = float(-(w * np.log(np.maximum(w, 1e-12))).sum() / np.log(max(len(w), 2)))
        up_mask = c > close_px
        down_mask = c < close_px
        tail_up = float(p[up_mask].sum())
        tail_down = float(p[down_mask].sum())
        mode_center = float(c[order[0]])
        records.append(
            {
                "ts": ts.floor("min"),
                "close_ref": close_px,
                "weighted_target": weighted_target,
                "mode_center": mode_center,
                "mode_prob": top_prob,
                "mode_spread": top_prob - second_prob,
                "entropy": entropy,
                "tail_up_prob": tail_up,
                "tail_down_prob": tail_down,
            }
        )
    poly_1m = pd.DataFrame.from_records(records)
    poly_1m = poly_1m.sort_values("ts").groupby("ts").last().reset_index()

    ms["ts"] = pd.to_datetime(ms["ts"])
    tr["ts"] = pd.to_datetime(tr["ts"])
    feat = ms.merge(tr, on="ts", how="left").merge(poly_1m, on="ts", how="inner")
    feat = feat.merge(px_kst[["ts", "open", "high", "low", "close"]], on="ts", how="inner")
    feat = feat.sort_values("ts").reset_index(drop=True)

    feat["target_gap"] = (feat["weighted_target"] - feat["close"]) / feat["close"].clip(lower=1e-8)
    feat["mode_gap"] = (feat["mode_center"] - feat["close"]) / feat["close"].clip(lower=1e-8)
    feat["tail_bias"] = feat["tail_up_prob"] - feat["tail_down_prob"]
    feat["prob_mom_1m"] = feat["mode_prob"].diff().fillna(0.0)
    feat["target_gap_delta_1m"] = feat["target_gap"].diff().fillna(0.0)
    feat["mode_gap_delta_1m"] = feat["mode_gap"].diff().fillna(0.0)
    feat["toxicity"] = pd.to_numeric(feat["shadow_toxicity_score"], errors="coerce").fillna(0.0).clip(0.0, 2.0)
    feat["aftershock"] = pd.to_numeric(feat["shadow_aftershock_prob"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    feat["obi_norm"] = np.tanh(pd.to_numeric(feat["obi"], errors="coerce").fillna(0.0))
    feat["signal_bias_norm"] = pd.to_numeric(feat["signal_bias"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    feat["news_impulse"] = (
        0.38 * np.tanh(feat["target_gap_delta_1m"] / 0.0030)
        + 0.22 * np.tanh(feat["prob_mom_1m"] / 0.0300)
        + 0.18 * feat["tail_bias"]
        + 0.12 * feat["obi_norm"]
        + 0.10 * feat["signal_bias_norm"]
    )
    feat["shock_score"] = (
        0.32 * np.abs(np.tanh(feat["target_gap_delta_1m"] / 0.0030))
        + 0.22 * np.abs(np.tanh(feat["prob_mom_1m"] / 0.0300))
        + 0.16 * np.abs(feat["tail_bias"])
        + 0.16 * feat["aftershock"]
        + 0.14 * np.clip(feat["toxicity"] / 1.2, 0.0, 1.0)
    ).clip(0.0, 1.5)
    return feat, start_ts, end_ts


def _side_signal(row: pd.Series, gap_th: float, tail_th: float) -> str:
    if float(row["target_gap"]) >= gap_th and float(row["tail_up_prob"]) >= tail_th:
        return "LONG"
    if float(row["target_gap"]) <= -gap_th and float(row["tail_down_prob"]) >= tail_th:
        return "SHORT"
    return "HOLD"


def _entry_block(row: pd.Series, side: str, cfg: OverlayConfig) -> tuple[bool, str]:
    target_gap = float(row["target_gap"])
    entropy = float(row["entropy"])
    shock = float(row["shock_score"])
    tail_up = float(row["tail_up_prob"])
    tail_down = float(row["tail_down_prob"])
    if side == "LONG" and target_gap <= -cfg.entry_gap_th and tail_down >= cfg.tail_th:
        return True, "entry_adverse_poly"
    if side == "SHORT" and target_gap >= cfg.entry_gap_th and tail_up >= cfg.tail_th:
        return True, "entry_adverse_poly"
    if shock >= cfg.shock_th and entropy >= cfg.entropy_cap and abs(target_gap) <= cfg.neutral_gap_th:
        return True, "entry_uncertain_news"
    return False, ""


def _force_exit(row: pd.Series, side: str, cfg: OverlayConfig) -> tuple[bool, str]:
    target_gap = float(row["target_gap"])
    shock = float(row["shock_score"])
    tail_up = float(row["tail_up_prob"])
    tail_down = float(row["tail_down_prob"])
    aftershock = float(row["aftershock"])
    toxicity = float(row["toxicity"])
    adverse = (
        (side == "LONG" and target_gap <= -cfg.exit_gap_th and tail_down >= cfg.tail_th)
        or (side == "SHORT" and target_gap >= cfg.exit_gap_th and tail_up >= cfg.tail_th)
    )
    if adverse and shock >= cfg.shock_th and (aftershock >= cfg.aftershock_cap or toxicity >= cfg.toxicity_cap):
        return True, "exit_news_shock"
    return False, ""


def _asof_price(px: pd.DataFrame, ts_utc: pd.Timestamp) -> float | None:
    s = px.set_index("ts")["close"]
    try:
        val = s.asof(ts_utc)
    except Exception:
        return None
    if val is None:
        return None
    fv = float(val)
    if not np.isfinite(fv):
        return None
    return fv


def run_backtest(feat_kst: pd.DataFrame, px_utc: pd.DataFrame, trades: list[Trade], cfg: OverlayConfig, fee: float, slip: float) -> dict:
    feat = feat_kst.copy()
    feat["ts_utc"] = feat["ts"].dt.tz_convert("UTC")
    feat = feat.set_index("ts_utc")
    baseline_sum = float(sum(float(t.realized_pct) for t in trades))
    base_wins = int(sum(1 for t in trades if float(t.realized_pct) > 0.0))

    new_returns: list[float] = []
    eq_curve = [1.0]
    skip_count = 0
    exit_count = 0
    improved = 0
    worsened = 0

    for tr in trades:
        lev = _est_lev(tr, fee=fee, slip=slip)
        open_row = feat.loc[: tr.open_ts]
        if len(open_row) == 0:
            pnl = float(tr.realized_pct)
            new_returns.append(pnl)
            eq_curve.append(eq_curve[-1] * (1.0 + pnl / 100.0))
            continue
        open_state = open_row.iloc[-1]
        blocked, _ = _entry_block(open_state, tr.side, cfg)
        if blocked:
            pnl = 0.0
            skip_count += 1
        else:
            pnl = float(tr.realized_pct)
            path = feat.loc[(feat.index > tr.open_ts) & (feat.index <= tr.close_ts)]
            for _, row in path.iterrows():
                should_exit, _ = _force_exit(row, tr.side, cfg)
                if not should_exit:
                    continue
                exit_px = _asof_price(px_utc, row.name)
                if exit_px is None:
                    continue
                pnl = _net_frac(tr.side, tr.open_price, exit_px, lev, fee=fee, slip=slip) * 100.0
                exit_count += 1
                break
        new_returns.append(float(pnl))
        if pnl > tr.realized_pct:
            improved += 1
        elif pnl < tr.realized_pct:
            worsened += 1
        eq_curve.append(eq_curve[-1] * (1.0 + pnl / 100.0))

    total = float(sum(new_returns))
    wins = int(sum(1 for x in new_returns if x > 0.0))
    return {
        "config": asdict(cfg),
        "name": cfg.name,
        "trades": len(trades),
        "baseline_sum_pct": baseline_sum,
        "overlay_sum_pct": total,
        "delta_pct": total - baseline_sum,
        "baseline_wr": (100.0 * base_wins / max(len(trades), 1)),
        "overlay_wr": (100.0 * wins / max(len(trades), 1)),
        "skip_count": skip_count,
        "exit_count": exit_count,
        "improved_trades": improved,
        "worsened_trades": worsened,
        "overlay_mdd_pct": _mdd(eq_curve),
        "overlay_sharpe": _sharpe(new_returns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest polymarket news shock overlay on actual DSAC live trades.")
    ap.add_argument("--events-path", default="data/live/dashboard_events.jsonl")
    ap.add_argument("--out-json", default="data/ensemble/reports/polymarket_news_overlay_backtest_20260424.json")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--slip", type=float, default=0.0002)
    ap.add_argument("--bucket-mode", choices=["mid", "upper", "tilted_upper"], default="mid")
    ap.add_argument("--tilted-alpha", type=float, default=0.75)
    args = ap.parse_args()

    feat_kst, start_utc, end_utc = _load_duckdb_features(bucket_mode=args.bucket_mode, tilted_alpha=args.tilted_alpha)
    trades = _load_trades(args.events_path, start_utc=start_utc, end_utc=end_utc)
    if not trades:
        raise SystemExit("No overlapping trades found for duckdb window.")

    px_utc = _fetch_binance_1m(start_utc - pd.Timedelta(minutes=5), end_utc + pd.Timedelta(minutes=5))

    grid = [
        OverlayConfig(*vals)
        for vals in product(
            [0.0030, 0.0045, 0.0060],
            [0.0035, 0.0050, 0.0065],
            [0.18, 0.24, 0.30],
            [0.52, 0.58],
            [0.45, 0.60],
            [0.80, 1.00],
            [0.0015, 0.0025],
            [0.78, 0.86],
        )
    ]

    results: list[dict] = []
    iterator = grid
    if tqdm is not None:
        iterator = tqdm(grid, desc="overlay-grid", ncols=100)
    for cfg in iterator:
        results.append(run_backtest(feat_kst, px_utc, trades, cfg, fee=args.fee, slip=args.slip))

    results.sort(key=lambda x: (x["delta_pct"], x["overlay_sum_pct"], -x["overlay_mdd_pct"]), reverse=True)
    baseline_sum = float(results[0]["baseline_sum_pct"])
    summary = {
        "bucket_mode": args.bucket_mode,
        "tilted_alpha": args.tilted_alpha,
        "window": {
            "duckdb_start_kst": str(start_utc.tz_convert(KST)),
            "duckdb_end_kst": str(end_utc.tz_convert(KST)),
        },
        "trade_count": len(trades),
        "baseline_sum_pct": baseline_sum,
        "top5": results[:5],
        "all_results": results,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    best = results[0]
    print("=== Polymarket News Overlay Backtest ===")
    print(f"window={summary['window']['duckdb_start_kst']} -> {summary['window']['duckdb_end_kst']}")
    print(f"trades={len(trades)} baseline_sum={baseline_sum:+.4f}%")
    print(
        "best="
        f"{best['name']} overlay_sum={best['overlay_sum_pct']:+.4f}% "
        f"delta={best['delta_pct']:+.4f}%p wr={best['overlay_wr']:.2f}% "
        f"mdd={best['overlay_mdd_pct']:.2f}% exits={best['exit_count']} skips={best['skip_count']}"
    )
    print(f"report={out_path}")


if __name__ == "__main__":
    main()
