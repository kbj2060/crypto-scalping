#!/usr/bin/env python3
"""oi_lsratio 테이블에 ttc/tkv(+누락 구간 전 채널)를 보존 한계까지 소급해 채운다.

`/futures/data/*` 는 `startTime` 을 무시하지만 `endTime` 은 존중한다(2026-09-12 전수 실측:
5코인 x 4채널 20/20). 보존 한계는 약 30.9일 -- 그보다 앞은 빈 배열이라 더 소급할 수 없다.
그래서 endTime 을 뒤로 밀며 역방향으로 페이징한다. 같은 관용구를
scripts/live_regime_wide24_signal_20260826.py 와 test/test_futures_data_backward_paging_20260912.py
가 쓴다.

쓰기는 수집기 자신의 `_db_upsert_rows`(read-merge-delete-insert)에 맡긴다 -- 새 upsert 를
만들면 라이브와 채움 규칙이 갈라진다. 기존 행은 COALESCE 로 보존되고 빠진 컬럼만 채워진다.

    python3 scripts/backfill_oi_lsratio_ttc_tkv_20260912.py [--dry-run] [--symbols ethusdt,btcusdt]
"""
from __future__ import annotations

import argparse, os, sys, time, urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oi_lsratio_collector import OiLsRatioCollector  # noqa: E402

STEP_MS = 5 * 60 * 1000
LIMIT = 500
# 심볼 -> 그 심볼의 DB 를 가리키는 환경변수 값. supervisor_*.sh 의 배선과 같아야 한다.
DB_FOR = {
    "ethusdt": "data/live/oi_lsratio.duckdb",
    "btcusdt": "data/live/oi_lsratio.duckdb",
    "solusdt": "data/live/oi_lsratio.duckdb",
    "xrpusdt": "data/live/oi_lsratio_xrp.duckdb",
    "hypeusdt": "data/live/oi_lsratio_hype.duckdb",
}
SOURCES = (
    ("/futures/data/openInterestHist", lambda r: {
        "sum_open_interest": float(r["sumOpenInterest"]),
        "sum_open_interest_value": float(r["sumOpenInterestValue"])}),
    ("/futures/data/globalLongShortAccountRatio", lambda r: {
        "global_ls_ratio": float(r["longShortRatio"]),
        "global_ls_long_account": float(r["longAccount"]),
        "global_ls_short_account": float(r["shortAccount"])}),
    ("/futures/data/topLongShortPositionRatio", lambda r: {
        "top_pos_ls_ratio": float(r["longShortRatio"]),
        "top_pos_ls_long_account": float(r["longAccount"]),
        "top_pos_ls_short_account": float(r["shortAccount"])}),
    ("/futures/data/topLongShortAccountRatio", lambda r: {
        "top_acct_ls_ratio": float(r["longShortRatio"]),
        "top_acct_ls_long_account": float(r["longAccount"]),
        "top_acct_ls_short_account": float(r["shortAccount"])}),
    ("/futures/data/takerlongshortRatio", lambda r: {
        "taker_ls_ratio": float(r["buySellRatio"]),
        "taker_buy_vol": float(r["buyVol"]),
        "taker_sell_vol": float(r["sellVol"])}),
)


def page_back(coll: OiLsRatioCollector, path: str, end_ms: int) -> list[dict]:
    """endTime 을 뒤로 밀며 보존 한계까지. 같은 창이 반복되면 그게 한계다."""
    rows: list[dict] = []
    seen: set[int] = set()
    cursor = end_ms
    while True:
        url = (f"{coll._BASE_URL if hasattr(coll, '_BASE_URL') else 'https://fapi.binance.com'}"
               f"{path}?symbol={coll._api_symbol}&period=5m&limit={LIMIT}&endTime={cursor}")
        try:
            data = coll._http_get_json(url)
        except urllib.error.HTTPError as e:
            # 보존 한계를 넘겨 요청하면 400 이 온다 -- 실패가 아니라 정상 종료다.
            if e.code != 400:
                print(f"    {path}: HTTP {e.code}")
            break
        except Exception as e:  # noqa: BLE001 -- 한 채널 실패가 나머지를 막으면 안 된다
            print(f"    {path}: 요청 실패 {e}")
            break
        if not isinstance(data, list) or not data:
            break
        fresh = [r for r in data if int(r["timestamp"]) not in seen]
        if not fresh:
            break
        rows.extend(fresh)
        seen.update(int(r["timestamp"]) for r in fresh)
        cursor = min(int(r["timestamp"]) for r in data) - 1
        time.sleep(0.2)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DB_FOR))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    now_ms = int(time.time() * 1000)
    rc = 0
    for sym in [s.strip().lower() for s in a.symbols.split(",") if s.strip()]:
        if sym not in DB_FOR:
            print(f"{sym}: 알 수 없는 심볼, 건너뜀"); rc = 1; continue
        os.environ["QUANT_OI_LSRATIO_DB_PATH"] = DB_FOR[sym]
        import importlib, oi_lsratio_collector
        importlib.reload(oi_lsratio_collector)          # 모듈 상수 _DB_PATH 를 새 환경변수로
        coll = oi_lsratio_collector.OiLsRatioCollector(symbol=sym)
        coll._db_init()
        by_ts: dict[int, dict] = {}
        print(f"\n[{sym.upper()}] db={DB_FOR[sym]} table={coll._table}")
        for path, mapper in SOURCES:
            got = page_back(coll, path, now_ms)
            for r in got:
                ts = int(r["timestamp"])
                e = by_ts.setdefault(ts, {"sources_ok": 0})
                e.update(mapper(r)); e["sources_ok"] += 1
            span = ((max(int(r["timestamp"]) for r in got) - min(int(r["timestamp"]) for r in got))
                    / 86400000) if got else 0
            print(f"    {path.rsplit('/', 1)[-1]:28s} {len(got):>6,}행 {span:5.1f}일")
        rows = []
        for ts, e in sorted(by_ts.items()):
            e["ts"] = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            rows.append(e)
        if not rows:
            print("    받은 행 없음"); rc = 1; continue
        print(f"  병합 {len(rows):,}행 {rows[0]['ts']} ~ {rows[-1]['ts']}")
        if a.dry_run:
            print("  --dry-run: 쓰지 않음"); continue
        coll._db_upsert_rows(rows)
        print(f"  upsert 완료")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
