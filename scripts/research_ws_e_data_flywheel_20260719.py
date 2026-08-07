"""WS-E: 데이터 플라이휠 - T-E1(round-trip 자가검증) + T-E2(짧은 파일럿 간격/커버리지 체크).

격리 원칙: 프로덕션 data/live/*.duckdb를 절대 건드리지 않는다. 신규 격리 DB
(data/research/ws_e_orderbook_raw_pilot.duckdb)에만 기록. 실제 라이브 recorder에
배선하는 것은 별도 승인이 필요한 프로덕션 변경이라 이번 세션 범위 밖.

T-E2의 72시간 소크테스트는 이번 세션에서 불가능 -- 대신 실제 라이브 거래소에서
짧은 파일럿(수 분)을 실행해 패턴이 동작하는지만 확인하고, 72h 기준은 명시적으로
미충족 상태로 남긴다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import ccxt
import duckdb
import numpy as np
import pandas as pd

OUT_DIR = Path("docs/test_designs_duckdb_live_20260719/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DB = "data/research/ws_e_orderbook_raw_pilot.duckdb"
Path("data/research").mkdir(parents=True, exist_ok=True)

PILOT_DURATION_SEC = 360  # 6 minutes -- short pilot, NOT the 72h soak in the design doc
INTERVAL_SEC = 10


def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def depth_notional(levels, n):
    qty = notional = 0.0
    for price, amount in levels[:n]:
        p, a = safe_float(price), safe_float(amount)
        qty += a
        notional += p * a
    return qty, notional


def imbalance(bid_notional, ask_notional):
    denom = abs(bid_notional) + abs(ask_notional)
    return 0.0 if denom <= 1e-12 else (bid_notional - ask_notional) / denom


def summarize(orderbook, symbol):
    bids = [[safe_float(x[0]), safe_float(x[1])] for x in orderbook.get("bids", []) if len(x) >= 2]
    asks = [[safe_float(x[0]), safe_float(x[1])] for x in orderbook.get("asks", []) if len(x) >= 2]
    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    row = {
        "recorded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": symbol,
        "exchange_timestamp": orderbook.get("timestamp"),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "bids_json": json.dumps(bids),
        "asks_json": json.dumps(asks),
    }
    for n in (1, 5, 10, 20):
        bq, bn = depth_notional(bids, n)
        aq, an = depth_notional(asks, n)
        row[f"bid_notional_{n}"] = bn
        row[f"ask_notional_{n}"] = an
        row[f"imbalance_{n}"] = imbalance(bn, an)
    return row


def recompute_from_raw(bids_json, asks_json):
    """Recompute summary stats purely from stored raw JSON -- this is the T-E1 self-check."""
    bids = json.loads(bids_json)
    asks = json.loads(asks_json)
    out = {}
    for n in (1, 5, 10, 20):
        bq, bn = depth_notional(bids, n)
        aq, an = depth_notional(asks, n)
        out[f"bid_notional_{n}"] = bn
        out[f"ask_notional_{n}"] = an
        out[f"imbalance_{n}"] = imbalance(bn, an)
    return out


def run_pilot():
    ex = ccxt.binance({"options": {"defaultType": "future"}})
    con = duckdb.connect(PILOT_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS orderbook_periodic_snapshots_eth_pilot (
            recorded_at_utc TIMESTAMPTZ, symbol VARCHAR, exchange_timestamp BIGINT,
            best_bid DOUBLE, best_ask DOUBLE, mid DOUBLE,
            bids_json VARCHAR, asks_json VARCHAR,
            bid_notional_1 DOUBLE, ask_notional_1 DOUBLE, imbalance_1 DOUBLE,
            bid_notional_5 DOUBLE, ask_notional_5 DOUBLE, imbalance_5 DOUBLE,
            bid_notional_10 DOUBLE, ask_notional_10 DOUBLE, imbalance_10 DOUBLE,
            bid_notional_20 DOUBLE, ask_notional_20 DOUBLE, imbalance_20 DOUBLE
        )
        """
    )
    con.execute("DELETE FROM orderbook_periodic_snapshots_eth_pilot")  # fresh pilot run

    timestamps = []
    t_start = time.monotonic()
    n_captured = 0
    n_errors = 0
    while time.monotonic() - t_start < PILOT_DURATION_SEC:
        loop_t0 = time.monotonic()
        try:
            ob = ex.fetch_order_book("ETH/USDT", limit=20)
            row = summarize(ob, "ETH/USDT")
            con.execute(
                """
                INSERT INTO orderbook_periodic_snapshots_eth_pilot VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    row["recorded_at_utc"], row["symbol"], row["exchange_timestamp"],
                    row["best_bid"], row["best_ask"], row["mid"],
                    row["bids_json"], row["asks_json"],
                    row["bid_notional_1"], row["ask_notional_1"], row["imbalance_1"],
                    row["bid_notional_5"], row["ask_notional_5"], row["imbalance_5"],
                    row["bid_notional_10"], row["ask_notional_10"], row["imbalance_10"],
                    row["bid_notional_20"], row["ask_notional_20"], row["imbalance_20"],
                ],
            )
            timestamps.append(pd.Timestamp(row["recorded_at_utc"]))
            n_captured += 1
        except Exception as exc:
            n_errors += 1
            print("capture error:", exc)
        elapsed = time.monotonic() - loop_t0
        sleep_left = INTERVAL_SEC - elapsed
        if sleep_left > 0:
            time.sleep(sleep_left)
    con.close()
    return timestamps, n_captured, n_errors


def run_t_e1_roundtrip():
    con = duckdb.connect(PILOT_DB)
    df = con.execute("select * from orderbook_periodic_snapshots_eth_pilot").df()
    con.close()
    mismatches = []
    for _, r in df.iterrows():
        recomputed = recompute_from_raw(r["bids_json"], r["asks_json"])
        for k, v in recomputed.items():
            stored = r[k]
            denom = max(abs(stored), abs(v), 1e-9)
            rel_err = abs(stored - v) / denom
            if rel_err > 1e-9:
                mismatches.append({"row_ts": str(r["recorded_at_utc"]), "field": k,
                                    "stored": float(stored), "recomputed": float(v), "rel_err": float(rel_err)})
    return {"n_rows_checked": int(len(df)), "n_mismatches": len(mismatches), "mismatches_sample": mismatches[:10]}


def main():
    report = {"stage": "WS-E", "generated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat()}
    report["design_deviation_note"] = (
        f"Pilot capture duration = {PILOT_DURATION_SEC}s (~{PILOT_DURATION_SEC/60:.0f} min), "
        "NOT the 72h soak test specified in the design doc. This validates the mechanism "
        "(round-trip integrity, interval discipline) works correctly; it does NOT satisfy "
        "the T-E2 coverage/soak acceptance criteria, which require multi-day monitoring."
    )
    print(f"Capturing live ETH/USDT orderbook every {INTERVAL_SEC}s for {PILOT_DURATION_SEC}s...")
    timestamps, n_captured, n_errors = run_pilot()
    report["T_E2_pilot_capture"] = {
        "n_captured": n_captured,
        "n_errors": n_errors,
        "target_duration_sec": PILOT_DURATION_SEC,
        "target_interval_sec": INTERVAL_SEC,
    }
    if len(timestamps) > 1:
        ts_series = pd.Series(timestamps).sort_values()
        gaps = ts_series.diff().dt.total_seconds().dropna()
        report["T_E2_pilot_capture"]["gap_p50_sec"] = float(gaps.median())
        report["T_E2_pilot_capture"]["gap_p99_sec"] = float(gaps.quantile(0.99))
        report["T_E2_pilot_capture"]["gap_max_sec"] = float(gaps.max())
        report["T_E2_pilot_capture"]["coverage_ratio_vs_target"] = float(
            n_captured / (PILOT_DURATION_SEC / INTERVAL_SEC)
        )

    print("Running T-E1 round-trip self-verification...")
    report["T_E1_roundtrip"] = run_t_e1_roundtrip()
    report["T_E1_verdict"] = (
        "PASS" if report["T_E1_roundtrip"]["n_mismatches"] == 0 else "FAIL"
    )

    out_json = OUT_DIR / "ws_e_data_flywheel_20260719.json"
    out_json.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print("WROTE", out_json)
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
