"""계좌 카드 날별 손익 = /fapi/v1/income KST 날 합(2026-10-20). 바이낸스는 안 부른다 -- fetch 는 가짜.
python test/test_account_income_daily_20261020.py  (또는 pytest)"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.orders import INCOME_LIMIT, INCOME_OVERLAP_MS, income_fold, income_refresh  # noqa: E402

KST = timezone(timedelta(hours=9))
ms = lambda *a: int(datetime(*a, tzinfo=KST).timestamp() * 1000)  # noqa: E731
row = lambda t, typ, amt, tid, asset="USDC": {"time": t, "incomeType": typ, "income": str(amt), "asset": asset, "tranId": tid}  # noqa: E731


def test_fold_buckets_classifies_dedupes():
    st = {}
    rows = [row(ms(2026, 10, 19, 23, 46), "REALIZED_PNL", 13.33, 1), row(ms(2026, 10, 19, 23, 46), "COMMISSION", -0.48, 1),
            row(ms(2026, 10, 20, 0, 0), "FUNDING_FEE", -0.2, 2), row(ms(2026, 10, 20, 9, 0), "TRANSFER", 500, 3),
            row(ms(2026, 10, 20, 9, 1), "COMMISSION", -0.01, 4, "BNB"), row(ms(2026, 10, 20, 12, 55), "REALIZED_PNL", 32.7, 5)]
    income_fold(st, rows)
    income_fold(st, rows[-2:])                      # 같은 줄 다시 = 무시
    d19, d20 = st["days"]["2026-10-19"], st["days"]["2026-10-20"]
    assert abs(d19["pnl"] - 12.85) < 1e-9 and d19["commission"] == -0.48      # 같은 tranId 라도 type 이 다르면 다른 줄
    assert abs(d20["pnl"] - 32.5) < 1e-9 and d20["funding"] == -0.2 and d20["transfer"] == 500
    assert d20["other_assets"] == {"BNB": -0.01}                               # $ 칸에 안 섞임
    try:
        income_fold(st, [{"time": 1}])               # 깨진 줄 = 아무것도 안 바뀜
    except KeyError:
        pass
    assert abs(st["days"]["2026-10-20"]["pnl"] - 32.5) < 1e-9


def test_refresh_backfill_halving_overlap_and_error():
    now = ms(2026, 10, 20, 15, 0)
    hist = [row(ms(2026, 9, 1, 10) + i * 60_000, "FUNDING_FEE", -0.01, 100 + i) for i in range(1500)]  # 한 창에 1500줄
    hist.append(row(now - 60_000, "REALIZED_PNL", 5, 9))
    calls = []

    async def fetch(s, e):
        calls.append((s, e))
        got = [r for r in hist if s <= r["time"] <= e]
        return got[:INCOME_LIMIT]                     # 꽉 차면 잘린다(정렬 방향은 모른다고 본다)

    st = {}
    asyncio.run(income_refresh(st, fetch, now, max_calls=100))
    assert st["from"] == "2026-07-24" and st["synced_at"] == now and st["cursor"] == now - INCOME_OVERLAP_MS
    assert all(e - s <= 7 * 86400_000 for s, e in calls)
    tot = sum(d["funding"] for d in st["days"].values())
    assert abs(tot + 15.0) < 1e-6, tot                # 1500줄 하나도 안 빠지고 한 번씩
    assert st["days"]["2026-10-20"]["pnl"] == 5
    asyncio.run(income_refresh(st, fetch, now + 60_000))   # 겹친 10분을 다시 물어도 두 번 안 센다
    assert st["days"]["2026-10-20"]["pnl"] == 5

    async def boom(s, e):
        raise RuntimeError("ban")
    before = st["cursor"]
    try:
        asyncio.run(income_refresh(st, boom, now + 120_000))
    except RuntimeError:
        pass
    assert st["cursor"] == before and st["days"]["2026-10-20"]["pnl"] == 5


def test_partial_backfill_not_synced():
    async def fetch(s, e):
        return []
    st = {}
    asyncio.run(income_refresh(st, fetch, ms(2026, 10, 20, 15), max_calls=3))
    assert "synced_at" not in st and st["cursor"] < ms(2026, 10, 1)      # 밀린 창은 다음 갱신이


if __name__ == "__main__":
    test_fold_buckets_classifies_dedupes()
    test_refresh_backfill_halving_overlap_and_error()
    test_partial_backfill_not_synced()
    print("ok")
