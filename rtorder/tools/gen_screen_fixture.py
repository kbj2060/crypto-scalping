"""계좌 카드 화면 몫 정답지(2026-11-23) -- 원본 dashboard/orders.py 함수를 그대로 돌려 tests/fixtures/screen_fixture.json 을 만든다.
대조: rtorder/tests/screen_fixture.rs. 실행: python rtorder/tools/gen_screen_fixture.py (저장소 뿌리에서)."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dashboard import orders  # noqa: E402

NOW = 1_763_830_000.0          # 2025-11-22 16:46:40 UTC = 11-23 01:46 KST
NOW_MS = int(NOW * 1000)


def trip(sym, side, entry, exit_, check=0.0, closed=True, **kw):
    return {"symbol": sym, "side": side, "entry_time": entry, "max_qty": 0.5, "qty_in": 0.5, "qty_out": 0.5, "realized_pnl": 3.0, "commission": 0.2,
            "fills": 2, "adds": 0, "peak_notional": 1000.0, "legs": 2, "exit_time": exit_, "exit_price": 2010.0, "closed": closed, "entry_price": 2000.0,
            "net_pnl": 2.8, "price_basis": "leg_vwap", "pnl_check_bp": check, **kw}


def daily(start_close, step, n=130, end0=1_752_000_000_000):
    closes = [start_close * step ** i for i in range(n)]
    ends = [end0 + i * 86_400_000 for i in range(n)]
    return closes, ends


def main() -> None:
    out = {}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        led = d / "account_round_trips.jsonl"
        hab = d / "account_trip_habits.json"
        with mock.patch.multiple(orders, ACCOUNT_TRIP_LEDGER_PATH=led, ACCOUNT_TRIP_HABITS_PATH=hab), mock.patch("builtins.print") as pr:
            # ── 원장 기록 ──
            old = [trip("ETHUSDC", "LONG", 1000, 2000), trip("ETHUSDC", "SHORT", 5000, 9000)]
            led.write_text("".join(json.dumps(t) + "\n" for t in old) + "{깨진 줄\n", "utf-8")
            pre = led.read_text("utf-8")
            seen = orders.load_account_trip_keys()
            trades = [trip("ETHUSDC", "LONG", 1000, 2000),                        # 이미 있음(키)
                      trip("ETHUSDC", "SHORT", 6000, 8500),                       # 겹침
                      trip("SOLUSDC", "LONG", 3000, 4000, truncated_open=True),   # 잘림
                      trip("SOLUSDC", "LONG", 4100, 4200, check=1.2),             # 항등식 불일치
                      trip("SOLUSDC", "LONG", 4300, 4400, check=None),            # 항등식 없음
                      trip("XRPUSDC", "SHORT", 7000, None, closed=False),         # 미청산
                      trip("XRPUSDC", "SHORT", 7100, 7900, check=-0.3),           # 적힘
                      trip("BTCUSDC", "LONG", 8000, 9500)]                        # 적힘
            payload = {"ok": True, "balance": {"margin": 1958.5}, "trades": trades}
            n = orders.record_account_trips(payload, seen)
            again = orders.record_account_trips(payload, seen)
            rows = [json.loads(x) for x in led.read_text("utf-8")[len(pre):].splitlines()]
            out["record"] = {"pre": pre, "payload": payload, "n": n, "again": again, "appended": rows,
                             "logs": [c.args[0] for c in pr.call_args_list], "seen": sorted(seen)}
            pr.reset_mock()
            # ── 화면 원장 꼬리 · 습관 · 추세 ──
            led.write_text("".join(json.dumps(t) + "\n" for t in [
                {**trip("ETHUSDT", "LONG", 1_763_000_000_000, 1_763_100_000_000), "peak_notional": None, "adds": None},            # 백필에서
                {**trip("ETHUSDC", "SHORT", 1_752_000_000_000 + 95 * 86_400_000 + 5, 1_763_200_000_000), "equity_at_record": 2100.0},
                {**trip("SOLUSDC", "LONG", 1_763_300_000_000, 1_763_400_000_000), "peak_notional": None, "equity_at_record": 900.0, "net_pnl": None},
                {**trip("DOGEUSDC", "LONG", 1_763_300_000_000, 1_763_400_000_000)},                                                  # 일봉 없음
                {**trip("BTCUSDC", "LONG", 1_752_000_000_000 + 50 * 86_400_000, 1_763_400_000_000)},                                # 일봉 모자람
            ]) + "{\"쓰다 만", "utf-8")
            hab.write_text(json.dumps({"ETHUSDT|LONG|1763000000000": {"adds": 2, "peak_notional": 3000.0, "equity_entry": 1500.0}}), "utf-8")
            ledger_rows = orders.load_account_trip_rows()
            dl = {"eth": daily(2000.0, 1.004), "sol": daily(200.0, 0.995), "btc": daily(90000.0, 1.001)}
            orders.annotate_trip_trend(ledger_rows, dl)
            out["ledger"] = {"file": led.read_text("utf-8"), "habits": hab.read_text("utf-8"), "daily": dl, "rows": ledger_rows}
        # ── 0시 순자산 ──
        ds = d / "account_day_start.json"
        cases = []
        for name, payload, pre_file, now in [
            ("pools_new", {"ok": True, "balance": {"wallet": 900.0, "unrealized": 5.0}, "balances": {"USDT": {"wallet": 1000.0, "unrealized": -3.0}, "USDC": {"wallet": 900.0, "unrealized": 5.0}}}, None, NOW),
            ("same_day", {"ok": True, "balance": {"wallet": 1.0}}, {"date": "2025-11-23", "equity": 7.0}, NOW),
            ("multi", {"ok": True, "multi_assets_margin": True, "balance": {"wallet": 500.0, "unrealized": None}, "balances": {"USDT": {"wallet": 9.0}}}, {"date": "2025-11-22"}, NOW),
            ("not_ok", {"ok": False, "error": "x"}, None, NOW),
            ("no_wallet", {"ok": True, "balance": {"margin": 1.0}}, None, NOW),
        ]:
            ds.unlink(missing_ok=True)
            if pre_file is not None:
                ds.write_text(json.dumps(pre_file))
            r = orders.account_day_start(payload, now=now, path=ds)
            cases.append({"name": name, "payload": payload, "pre": pre_file, "now": now, "out": r, "file": json.loads(ds.read_text()) if ds.exists() else None})
        out["day_start"] = cases
    # ── income ──
    base = int(NOW * 1000) - 88 * 86_400_000
    master = []
    for i in range(1300):     # 첫 창에 1000 줄 넘게 -- 반으로 줄여 다시 묻는다
        master.append({"tranId": 10_000 + i, "incomeType": "COMMISSION", "asset": "USDC", "income": f"-{0.01 * (i % 7):.2f}", "time": base + 3_600_000 + i * 60_000})
    master += [{"tranId": 1, "incomeType": "REALIZED_PNL", "asset": "USDT", "income": "12.5", "time": NOW_MS - 3 * 86_400_000},
               {"tranId": 2, "incomeType": "FUNDING_FEE", "asset": "USDC", "income": "-0.75", "time": NOW_MS - 2 * 86_400_000},
               {"tranId": 3, "incomeType": "TRANSFER", "asset": "USDC", "income": "500", "time": NOW_MS - 2 * 86_400_000 + 5},
               {"tranId": 4, "incomeType": "COMMISSION", "asset": "BNB", "income": "-0.001", "time": NOW_MS - 86_400_000},
               {"tranId": 5, "incomeType": "INSURANCE_CLEAR", "asset": "USDT", "income": "0.3", "time": NOW_MS - 1000}]
    calls = []

    async def fetch(start, end):
        calls.append([start, end])
        return [r for r in master if start <= r["time"] <= end][:orders.INCOME_LIMIT]

    async def run(max_calls):
        st: dict = {}
        await orders.income_refresh(st, fetch, NOW_MS, max_calls=max_calls)
        return st
    full = asyncio.run(run(40))     # 따라잡을 때까지(synced_at)
    c1 = list(calls)
    calls.clear()
    part = asyncio.run(run(3))
    out["income"] = {"now_ms": NOW_MS, "master": master, "full": full, "full_calls": c1, "part": part, "part_calls": list(calls)}
    dst = ROOT / "rtorder" / "tests" / "fixtures" / "screen_fixture.json"
    dst.write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {dst} · 기록 {out['record']['n']}건 · 원장 줄 {len(out['ledger']['rows'])} · income 호출 {len(c1)}/{len(out['income']['part_calls'])}")


if __name__ == "__main__":
    main()
