#!/usr/bin/env python3
"""왕복 원장 무결성 수리 — 진입가/청산가를 **다리별 VWAP** 으로 다시 계산한다 (2026-09-12).

사용자: *"원장 무결성부터 수리해줘."*

## 무엇이 틀렸나
`live_binance_account_20260910._fold` 가 왕복의 `entry_price` 로 **첫 체결가**, `exit_price` 로
**마지막 체결가**를 그대로 실었다. 물타기나 분할청산이 섞이면 그 둘은 평균단가가 아니다.
실계좌 19왕복 중 **6건**에서 `(청산가−진입가)×방향` 의 부호가 `realizedPnl` 과 어긋났다
(예: 2026-09-03 LONG 2489.79→2459.06 인데 손익 +26.30).
`_fold` 는 고쳤다(다리별 VWAP + `pnl_check_bp` 자기검증). 이 스크립트는 **이미 적힌 줄**을 고친다.

## 어떻게
거래소 `userTrades` 를 7일 창으로 나눠 다시 받아 같은 폴딩을 돌리고, `symbol|side|entry_time`
으로 원장 줄과 맞춰 가격 필드만 갈아끼운다. `recorded_at` · `backfill` · `caveat` 같은
**원장 고유 필드는 보존**한다. API 창 밖이라 못 맞춘 줄은 지우지 않고 `price_basis` 를
`first_last_fill_unverified` 로 **표시만** 한다 — 못 고친 걸 고친 척하지 않는다.

## 안전
- 기본 **dry-run**. 실제 반영은 `--apply`.
- 쓰기는 임시파일 + `os.replace` **원자 교체**. 대시보드가 같은 파일에 붙이고 있다.
- 줄 수는 **절대 줄지 않는다**(맞춘 줄만 교체). 줄어들면 쓰지 않고 중단한다.
- `--apply` 전에 원본을 `.bak_repair_<타임스탬프>` 로 복사한다.
- 자격증명이 필요하므로 **서버에서** 돈다(로컬 .env 에는 키가 없다).

실행  python3 scripts/repair_account_trip_ledger_20260912.py            # 미리보기
      python3 scripts/repair_account_trip_ledger_20260912.py --apply
자체점검 --selftest (네트워크 없이 병합 규칙만)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import live_binance_account_20260910 as A  # noqa: E402

LEDGER = ROOT / "data/live/account_round_trips.jsonl"
WINDOW_MS = 7 * 24 * 3600 * 1000 - 60_000      # Binance: startTime~endTime 은 7일 이내
KEEP = ("recorded_at", "backfill", "caveat")   # 원장에만 있는 필드 — 거래소 값이 덮으면 안 된다


def merge(old: dict, new: dict) -> dict:
    """거래소 재계산본을 기준으로 하되 원장 고유 필드는 남긴다."""
    out = dict(new)
    for k in KEEP:
        if k in old and old[k] is not None:
            out[k] = old[k]
    return out


def load_ledger() -> list[dict]:
    rows, bad = [], 0
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
    if bad:
        print(f"⚠️읽지 못한 줄 {bad}개 — 그 줄은 건드리지 않고 그대로 둘 수 없으므로 중단한다")
        raise SystemExit(2)
    return rows


async def fetch_trips(symbols: list[str], start_ms: int) -> list[dict]:
    import aiohttp
    key, secret = os.getenv("BINANCE_API_KEY", ""), os.getenv("BINANCE_SECRET_KEY", "")
    if not (key and secret):
        print("BINANCE_API_KEY/BINANCE_SECRET_KEY 가 없다. 서버에서 실행해야 한다.")
        raise SystemExit(3)
    fills: list[dict] = []
    async with aiohttp.ClientSession() as session:
        offset = await A._clock_offset(session)
        now = int(time.time() * 1000) + offset
        for symbol in symbols:
            cursor = start_ms
            while cursor < now:
                end = min(cursor + WINDOW_MS, now)
                page = await A._get(session, "/fapi/v1/userTrades",
                                    {"symbol": symbol, "startTime": cursor, "endTime": end,
                                     "limit": A.DEFAULT_TRADE_LIMIT}, key, secret, offset)
                if isinstance(page, dict) and "__error__" in page:
                    print(f"  {symbol} {cursor}: {page['__error__']}")
                    raise SystemExit(4)
                fills.extend(page)
                print(f"  {symbol} {cursor} ~ {end}: 체결 {len(page)}건", flush=True)
                if len(page) >= A.DEFAULT_TRADE_LIMIT:
                    cursor = int(page[-1]["time"]) + 1     # 잘렸다 — 마지막 체결부터 이어 받는다
                else:
                    cursor = end + 1
    return A.round_trips(fills)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--lookback-days", type=int, default=14,
                    help="첫 왕복보다 이만큼 앞에서 체결을 받기 시작한다(스트림 절단 방지)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        old = {"symbol": "ETHUSDT", "side": "LONG", "entry_time": 1, "entry_price": 9.0,
               "recorded_at": "T", "backfill": "b", "caveat": "c", "net_pnl": 5.0}
        new = {"symbol": "ETHUSDT", "side": "LONG", "entry_time": 1, "entry_price": 7.5,
               "price_basis": "leg_vwap", "pnl_check_bp": 0.0, "net_pnl": 5.0}
        m = merge(old, new)
        assert m["entry_price"] == 7.5 and m["price_basis"] == "leg_vwap"
        assert (m["recorded_at"], m["backfill"], m["caveat"]) == ("T", "b", "c"), m
        assert A.trip_key_of(m) == A.trip_key_of(old) if hasattr(A, "trip_key_of") else True
        print("selftest OK — 가격은 교체되고 원장 고유 필드는 보존된다")
        return 0

    rows = load_ledger()
    key = lambda t: f"{t.get('symbol')}|{t.get('side')}|{t.get('entry_time')}"   # noqa: E731
    symbols = sorted({r["symbol"] for r in rows})
    # 🔴체결 스트림의 **시작이 잘리면** 첫 왕복이 창 이전에 열린 포지션의 꼬리가 되어 진입/청산
    # 다리가 뒤섞인다(1시간만 앞서 받았던 첫 실행: 항등식 이탈 49bp). 넉넉히 앞에서 시작해
    # 창 안에서 포지션이 0 으로 돌아가는 지점부터 접히게 한다.
    start = min(int(r["entry_time"]) for r in rows) - a.lookback_days * 86400_000
    print(f"원장 {len(rows)}줄 · 심볼 {symbols} · {start} 부터 체결 재수신")
    trips = asyncio.run(fetch_trips(symbols, start))
    by = {key(t): t for t in trips}
    print(f"거래소 재계산 왕복 {len(trips)}건")

    out, fixed, unmatched = [], 0, 0
    for r in rows:
        t = by.get(key(r))
        if t is None:
            r = {**r, "price_basis": "first_last_fill_unverified"}
            unmatched += 1
        else:
            before = (r.get("entry_price"), r.get("exit_price"))
            r = merge(r, t)
            if (r.get("entry_price"), r.get("exit_price")) != before:
                fixed += 1
        out.append(r)
    assert len(out) == len(rows), "줄 수가 변했다"

    def mismatch(rs):
        """가격 방향과 **realizedPnl** 의 부호가 어긋나는 줄 수.

        ⚠️`net_pnl`(= realized − 수수료)로 재면 안 된다. 소액 승리는 수수료가 먹어 순손익만
        음수가 되는데(예: LONG 2446.40→2448.82 · 1.359개 → 총이익 3.29, 수수료 ~2.7 → 순 −0.04)
        그건 기록 오류가 아니라 정상이다. 첫 판이 이걸 «불일치»로 세어 4건을 헛되이 남겼다.
        """
        n = 0
        for r in rs:
            ep, xp, pnl = r.get("entry_price"), r.get("exit_price"), r.get("realized_pnl")
            if ep and xp and pnl:
                s = 1 if r.get("side") == "LONG" else -1
                if (xp - ep) * s * pnl < 0:
                    n += 1
        return n

    ver = [r for r in out if r.get("pnl_check_bp") is not None]
    off = [r for r in ver if abs(r["pnl_check_bp"]) > 1.0]
    print(f"\n가격 갱신 {fixed}줄 · API 창 밖 {unmatched}줄")
    print(f"부호 불일치: {mismatch(rows)} → **{mismatch(out)}**")
    print(f"회계 항등식 검증 가능 {len(ver)}/{len(out)}줄 · 1bp 초과 이탈 {len(off)}줄")
    for r in off[:8]:
        print(f"   ⚠️이탈 {r['pnl_check_bp']:>10.2f}bp  {r.get('entry_at') or r.get('entry_time')} {r.get('side')}")
    bad = [r for r in out
           if r.get("entry_price") and r.get("exit_price") and r.get("realized_pnl")
           and (r["exit_price"] - r["entry_price"]) * (1 if r.get("side") == "LONG" else -1) * r["realized_pnl"] < 0]
    if bad:
        print("\n남은 부호 불일치 — 어떤 줄이고 왜 못 고쳤나")
        for r in bad:
            print(f"   {r.get('entry_at') or r.get('entry_time')} {r.get('side'):<5} "
                  f"진입 {r['entry_price']:>9.2f} 청산 {r['exit_price']:>9.2f} 총이익 {r.get('realized_pnl'):>9.2f} "
                  f"basis={r.get('price_basis')} check={r.get('pnl_check_bp')}")
    if not a.apply:
        print("\n미리보기만 했다. 반영하려면 --apply")
        return 0
    shutil.copy2(LEDGER, LEDGER.with_suffix(f".jsonl.bak_repair_{time.strftime('%Y%m%d_%H%M%S')}"))
    tmp = LEDGER.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    os.replace(tmp, LEDGER)
    print(f"\n반영 완료 — {LEDGER} ({len(out)}줄, 원본은 .bak_repair_* 로 보관)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
