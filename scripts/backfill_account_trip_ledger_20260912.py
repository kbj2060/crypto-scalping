#!/usr/bin/env python3
"""왕복 원장 백필 — 거래소엔 있는데 원장에 없는 종료 왕복을 채운다 (2026-09-12).

사용자: *"49건 백필해줘."*  선행: `repair_account_trip_ledger_20260912.py`(진입/청산가 VWAP 수리).

## 왜 빠졌나
대시보드는 **켜져 있는 동안 본 왕복만** 적는데 거래소 조회 창은 7일이다. 꺼져 있던 구간이
통째로 빠진다. 실측: 거래소 재계산 68건 vs 원장 28줄, **빠진 종료 왕복 49건**(2026-08-21~09-04).
표본 수가 지금 이 저장소의 구속조건이라(탐지 벽 a* = 0.5 + 1/√N) 28→77 은 가장 값싼 개선이다.

## 중복을 막는 두 겹
1. **키**(`symbol|side|entry_time`)가 이미 있으면 건너뛴다.
2. ⭐**시간 겹침**: 같은 symbol+side 의 기존 줄과 [진입,청산] 구간이 겹치면 건너뛴다.
   한 방향 포지션은 같은 시각에 둘일 수 없으므로 **겹침 = 같은 왕복**이다. 이 가드가 필요한 이유는
   09-11 스냅샷에서 백필된 4줄의 `entry_time` 이 재구성값이라 키가 안 맞기 때문이다 --
   키만 보면 그 4건이 두 번 들어간다.

## 안전
- **덧붙이기만 한다**(기존 줄을 다시 쓰지 않는다). 대시보드가 같은 파일에 append 중이라,
  통째로 다시 쓰면 그 사이 들어온 줄을 잃는다. `O_APPEND` 한 줄 쓰기는 그 위험이 없다.
- 기본 dry-run. `--apply` 전에 `.bak_backfill_<타임스탬프>` 복사.
- 회계 항등식(`pnl_check_bp`)이 1bp 를 넘는 후보는 **넣지 않는다** -- 못 믿을 줄을 표본에 섞지 않는다.
- 대시보드 재시작 불필요: 그 프로세스의 `seen` 은 메모리에 있고, 지금 조회 창(09-05~)이
  이 49건(~09-04)과 겹치지 않아 다시 붙이지 않는다. 실측으로 확인하고 넣는다.

실행  python3 scripts/backfill_account_trip_ledger_20260912.py [--apply]
자체점검 --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import live_binance_account_20260910 as A  # noqa: E402
import repair_account_trip_ledger_20260912 as RP  # noqa: E402

TAG = "exchange_replay_20260912"
CHECK_TOL_BP = 1.0


def key(t: dict) -> str:
    return f"{t.get('symbol')}|{t.get('side')}|{t.get('entry_time')}"


def span(t: dict) -> tuple[int, int]:
    e = int(t["entry_time"])
    x = t.get("exit_time")
    return e, int(x) if x else e


def overlaps(cand: dict, rows: list[dict]) -> dict | None:
    """같은 symbol+side 의 기존 줄과 구간이 겹치면 그 줄을 돌려준다(= 같은 왕복)."""
    c0, c1 = span(cand)
    for r in rows:
        if r.get("symbol") != cand.get("symbol") or r.get("side") != cand.get("side"):
            continue
        r0, r1 = span(r)
        if c0 <= r1 and r0 <= c1:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rows = [{"symbol": "ETHUSDT", "side": "LONG", "entry_time": 100, "exit_time": 200}]
        assert overlaps({"symbol": "ETHUSDT", "side": "LONG", "entry_time": 150, "exit_time": 250}, rows)
        assert overlaps({"symbol": "ETHUSDT", "side": "LONG", "entry_time": 50, "exit_time": 120}, rows)
        assert overlaps({"symbol": "ETHUSDT", "side": "LONG", "entry_time": 120, "exit_time": 180}, rows)
        assert not overlaps({"symbol": "ETHUSDT", "side": "LONG", "entry_time": 201, "exit_time": 300}, rows)
        assert not overlaps({"symbol": "ETHUSDT", "side": "SHORT", "entry_time": 150, "exit_time": 250}, rows), \
            "헤지 모드라 반대 측면은 동시에 열려 있어도 다른 왕복이다"
        assert not overlaps({"symbol": "BTCUSDT", "side": "LONG", "entry_time": 150, "exit_time": 250}, rows)
        print("selftest OK — 겹침 가드(같은 측면만·경계 포함)")
        return 0

    rows = RP.load_ledger()
    have = {key(r) for r in rows}
    symbols = sorted({r["symbol"] for r in rows})
    start = min(int(r["entry_time"]) for r in rows) - a.lookback_days * 86400_000
    print(f"원장 {len(rows)}줄 · {symbols} · 체결 재수신 시작 {A._iso(start)}")
    trips = RP.fetch_trips(symbols, start) if not asyncio.iscoroutinefunction(RP.fetch_trips) \
        else asyncio.run(RP.fetch_trips(symbols, start))
    closed = [t for t in trips if t.get("closed")]
    print(f"거래소 종료 왕복 {len(closed)}건")

    add, skip_key, skip_ov, skip_chk = [], 0, [], []
    for t in closed:
        if key(t) in have:
            skip_key += 1
            continue
        dup = overlaps(t, rows)
        if dup is not None:
            skip_ov.append((t, dup))
            continue
        chk = t.get("pnl_check_bp")
        if chk is None or abs(chk) > CHECK_TOL_BP:
            skip_chk.append(t)
            continue
        add.append({**t, "backfill": TAG, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())})

    print(f"\n키 중복 {skip_key}건 · 시간 겹침 {len(skip_ov)}건 · 항등식 미달 {len(skip_chk)}건 → **추가 {len(add)}건**")
    for t, d in skip_ov:
        print(f"   겹침 제외: {A._iso(t['entry_time'])} {t['side']} ↔ 기존 {d.get('entry_at') or d['entry_time']}"
              f" (기존 손익 {d.get('net_pnl')}, 후보 {round(t['net_pnl'], 2)})")
    for t in skip_chk:
        print(f"   항등식 제외: {A._iso(t['entry_time'])} {t['side']} check={t.get('pnl_check_bp')}bp")
    if add:
        lo, hi = min(int(t["entry_time"]) for t in add), max(int(t["entry_time"]) for t in add)
        print(f"\n추가 구간 {A._iso(lo)} ~ {A._iso(hi)} · 순손익 합 {sum(t['net_pnl'] for t in add):+.2f} USDT")
        print(f"원장 {len(rows)} → **{len(rows) + len(add)}줄**")
    if not a.apply:
        print("\n미리보기만 했다. 반영하려면 --apply")
        return 0
    if not add:
        return 0
    shutil.copy2(RP.LEDGER, RP.LEDGER.with_suffix(f".jsonl.bak_backfill_{time.strftime('%Y%m%d_%H%M%S')}"))
    with RP.LEDGER.open("a") as handle:          # 덧붙이기만 -- 동시 append 를 잃지 않는다
        for t in add:
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    total = len(RP.LEDGER.read_text().splitlines())
    print(f"\n반영 완료 — {RP.LEDGER} 현재 {total}줄 (원본은 .bak_backfill_* 로 보관)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
