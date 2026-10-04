"""계좌 카드 «습관 분해»의 과거 왕복 채우기 (2026-10-05, 사용자 «3번과 4번 모두 진행»).

원장(data/live/account_round_trips.jsonl)은 10-05 전까지 «추가 횟수»·«최대 명목»을 안 적었다. 그 값은 체결 경로에서만
나온다 -- 연구(research_eth_avgdown_gate_notional_cap_20261004 --fetch)가 받아 둔 체결·income 으로 한 번 채운다.
폴딩은 대시보드와 같은 함수(live_binance_account_20260910.round_trips)라 왕복 키(심볼|측면|진입ms)가 원장과 같다.

자본(진입 시) = income 전 유형 누적(USDT+USDC, 입출금 포함) -- 연구의 «지갑» 정의(kelly 재현: 현재 잔고와 0.22 차이).
산출: data/live/account_trip_habits.json  {trip_key: {"adds", "peak_notional", "equity_entry"}}  (서버 원장 옆에 둔다)

  python scripts/backfill_account_trip_habits_20261005.py FILLS.jsonl INCOME.jsonl [OUT.json]
  python scripts/backfill_account_trip_habits_20261005.py --selftest
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_binance_account_20260910 import round_trips  # noqa: E402


def build(fills: list[dict], income: list[dict]) -> dict[str, dict]:
    inc = sorted((int(r["time"]), float(r["income"])) for r in income)
    times, cum, run = [t for t, _ in inc], [], 0.0
    for _, v in inc:
        run += v
        cum.append(run)
    out = {}
    for t in round_trips(fills):
        if not t.get("closed"):
            continue
        i = bisect.bisect_left(times, int(t["entry_time"])) - 1     # 진입 **전**까지의 누적
        out[f"{t['symbol']}|{t['side']}|{t['entry_time']}"] = {
            "adds": t["adds"], "peak_notional": round(t["peak_notional"], 2),
            "equity_entry": round(cum[i], 2) if i >= 0 else None}
    return out


def selftest() -> None:
    f = [{"symbol": "ETHUSDC", "id": 1, "time": 1_000_000, "side": "BUY", "price": "2000", "qty": "1", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"},
         {"symbol": "ETHUSDC", "id": 2, "time": 1_400_000, "side": "BUY", "price": "1900", "qty": "1", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"},
         {"symbol": "ETHUSDC", "id": 3, "time": 1_500_000, "side": "SELL", "price": "1950", "qty": "2", "realizedPnl": "0", "commission": "0", "positionSide": "LONG"}]
    inc = [{"time": 500_000, "income": "1000"}, {"time": 900_000, "income": "-10"}, {"time": 1_000_000, "income": "-1"}]
    h = build(f, inc)["ETHUSDC|LONG|1000000"]
    assert h == {"adds": 1, "peak_notional": 3800.0, "equity_entry": 990.0}, h   # 진입 시각과 같은 ms 의 income 은 뺀다
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        sys.exit(0)
    fills = [json.loads(x) for x in open(sys.argv[1])]
    income = [json.loads(x) for x in open(sys.argv[2])]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "data/live/account_trip_habits.json"
    res = build(fills, income)
    out.write_text(json.dumps(res, ensure_ascii=False))
    print(f"{len(res)}건 -> {out}")
