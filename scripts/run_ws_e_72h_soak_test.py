"""WS-E T-E2: 72시간 원시 L2 캡처 소크테스트 (격리 환경, 무인 장기 실행용).

data/research/ws_e_orderbook_raw_pilot.duckdb 6분 파일럿(2026-07-19)의 후속.
프로덕션 data/live/*.duckdb는 절대 건드리지 않는다 -- 신규 격리 DB에만 기록.
채팅 세션과 무관하게 nohup으로 독립 실행되도록 설계 (에러 1건에 전체가 죽지 않게
try/except로 감싸고, 주기적으로 상태 파일을 갱신해 언제든 진행상황을 확인할 수 있게 함).

수락 기준 (설계 문서 WS-E T-E2):
  - 커버리지 >= 95% (목표 캡처수 대비)
  - 간격 p99 <= 15초
  - round-trip 불일치 0건 (주기적으로 재검증)
  - 오류 발생률 낮음 (기록)

중단하려면: data/research/ws_e_soak_stop.sentinel 파일을 생성하면 다음 루프에서 정지.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import ccxt
import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/research/ws_e_orderbook_raw_pilot.duckdb"
TABLE = "orderbook_periodic_snapshots_eth_soak_20260719"
STATUS_PATH = Path("data/research/ws_e_soak_status.json")
STOP_SENTINEL = Path("data/research/ws_e_soak_stop.sentinel")
Path("data/research").mkdir(parents=True, exist_ok=True)

TARGET_DURATION_SEC = 72 * 3600
INTERVAL_SEC = 10
STATUS_UPDATE_EVERY_N = 30       # ~every 5 min
ROUNDTRIP_CHECK_EVERY_N = 180    # ~every 30 min, verify last N rows


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
        "best_bid": best_bid, "best_ask": best_ask, "mid": mid,
        "bids_json": json.dumps(bids), "asks_json": json.dumps(asks),
    }
    for n in (1, 5, 10, 20):
        bq, bn = depth_notional(bids, n)
        aq, an = depth_notional(asks, n)
        row[f"bid_notional_{n}"] = bn
        row[f"ask_notional_{n}"] = an
        row[f"imbalance_{n}"] = imbalance(bn, an)
    return row


def recompute_from_raw(bids_json, asks_json):
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


def ensure_table(con):
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
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


def insert_row(con, row):
    con.execute(
        f"""INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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


def roundtrip_check_recent(con, n=180):
    df = con.execute(f"select * from {TABLE} order by recorded_at_utc desc limit {n}").df()
    mismatches = 0
    for _, r in df.iterrows():
        recomputed = recompute_from_raw(r["bids_json"], r["asks_json"])
        for k, v in recomputed.items():
            stored = r[k]
            denom = max(abs(stored), abs(v), 1e-9)
            if abs(stored - v) / denom > 1e-9:
                mismatches += 1
    return {"n_checked": int(len(df)), "n_mismatches": int(mismatches)}


def write_status(status: dict):
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str, ensure_ascii=False))


def main():
    ex = ccxt.binance({"options": {"defaultType": "future"}})
    con = duckdb.connect(DB_PATH)
    ensure_table(con)
    con.close()

    t_start = time.monotonic()
    start_wall = pd.Timestamp.now(tz="UTC")
    n_captured = 0
    n_errors = 0
    error_samples = []
    last_roundtrip_result = None
    gap_history = []
    last_capture_monotonic = None

    print(f"[soak] started {start_wall.isoformat()}, target {TARGET_DURATION_SEC/3600:.0f}h, "
          f"table={TABLE}, db={DB_PATH}")

    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= TARGET_DURATION_SEC:
            print("[soak] target duration reached, stopping")
            break
        if STOP_SENTINEL.exists():
            print("[soak] stop sentinel found, stopping")
            STOP_SENTINEL.unlink(missing_ok=True)
            break

        loop_t0 = time.monotonic()
        try:
            ob = ex.fetch_order_book("ETH/USDT", limit=20)
            row = summarize(ob, "ETH/USDT")
            con = duckdb.connect(DB_PATH)
            insert_row(con, row)
            con.close()
            n_captured += 1
            if last_capture_monotonic is not None:
                gap_history.append(loop_t0 - last_capture_monotonic)
                if len(gap_history) > 5000:
                    gap_history = gap_history[-5000:]
            last_capture_monotonic = loop_t0
        except Exception as exc:
            n_errors += 1
            if len(error_samples) < 20:
                error_samples.append({"t": pd.Timestamp.now(tz="UTC").isoformat(), "error": str(exc)})
            print(f"[soak] capture error #{n_errors}: {exc}")

        if n_captured > 0 and n_captured % ROUNDTRIP_CHECK_EVERY_N == 0:
            try:
                con = duckdb.connect(DB_PATH)
                last_roundtrip_result = roundtrip_check_recent(con, n=ROUNDTRIP_CHECK_EVERY_N)
                con.close()
                print(f"[soak] roundtrip check @ n={n_captured}: {last_roundtrip_result}")
            except Exception as exc:
                print(f"[soak] roundtrip check error: {exc}")

        if n_captured % STATUS_UPDATE_EVERY_N == 0 or n_errors > 0:
            gaps = np.array(gap_history) if gap_history else np.array([])
            elapsed_h = (time.monotonic() - t_start) / 3600.0
            target_captures = elapsed / INTERVAL_SEC if elapsed > 0 else 1
            status = {
                "started_utc": start_wall.isoformat(),
                "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "elapsed_hours": elapsed_h,
                "target_hours": TARGET_DURATION_SEC / 3600.0,
                "progress_pct": min(100.0, elapsed_h / (TARGET_DURATION_SEC / 3600.0) * 100.0),
                "n_captured": n_captured,
                "n_errors": n_errors,
                "error_rate": n_errors / max(n_captured + n_errors, 1),
                "coverage_ratio_vs_target": n_captured / max(target_captures, 1),
                "gap_p50_sec": float(np.median(gaps)) if len(gaps) else None,
                "gap_p99_sec": float(np.percentile(gaps, 99)) if len(gaps) else None,
                "gap_max_sec": float(np.max(gaps)) if len(gaps) else None,
                "last_roundtrip_check": last_roundtrip_result,
                "error_samples": error_samples,
                "acceptance_criteria_so_far": {
                    "coverage_gte_95pct": (n_captured / max(target_captures, 1)) >= 0.95,
                    "gap_p99_lte_15sec": (float(np.percentile(gaps, 99)) <= 15.0) if len(gaps) else None,
                    "roundtrip_zero_mismatches": (last_roundtrip_result or {}).get("n_mismatches") == 0
                        if last_roundtrip_result else None,
                },
            }
            write_status(status)

        sleep_left = INTERVAL_SEC - (time.monotonic() - loop_t0)
        if sleep_left > 0:
            time.sleep(sleep_left)

    # final status write
    con = duckdb.connect(DB_PATH)
    final_roundtrip = roundtrip_check_recent(con, n=min(n_captured, 2000))
    total_rows = con.execute(f"select count(*) from {TABLE}").fetchone()[0]
    con.close()
    gaps = np.array(gap_history) if gap_history else np.array([])
    elapsed_h = (time.monotonic() - t_start) / 3600.0
    final_status = {
        "started_utc": start_wall.isoformat(),
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_hours": elapsed_h,
        "n_captured": n_captured,
        "n_errors": n_errors,
        "total_rows_in_table": int(total_rows),
        "gap_p50_sec": float(np.median(gaps)) if len(gaps) else None,
        "gap_p99_sec": float(np.percentile(gaps, 99)) if len(gaps) else None,
        "final_roundtrip_check": final_roundtrip,
        "finished": True,
    }
    write_status(final_status)
    print("[soak] FINISHED", json.dumps(final_status, default=str))


if __name__ == "__main__":
    main()
