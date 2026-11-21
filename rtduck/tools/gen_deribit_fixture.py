"""Deribit 체결·블록 첫 단계 정답지: 원본 live_deribit_block_trade_collector_20260928 의 write()(세 번) · backfill()(가짜 requests) 를 가짜 시계로.

블록 = 구조 13종 + single·other·future_only · 선물 헤지(역·USDC 선형 · 값 0 = ZeroDivision → 구조 칸 없음) · iv 0·지수 없음 · 24시간 창 밖 · 중복 trade_id.
흐름 = 코인 넷 · 강제청산 · 블록 베가 · 창 밖. 서버 판 venv(duckdb 1.5.1)로 돌린다:   $PY rtduck/tools/gen_deribit_fixture.py
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.modules.setdefault("requests", types.ModuleType("requests"))   # GEX 모듈이 맨 위에서 import -- 이 정답지는 REST 를 가짜로
import live_deribit_block_trade_collector_20260928 as S  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
NOW = dt.datetime(2026, 11, 21, 3, 20, 0, 123456, tzinfo=dt.timezone.utc)
T = int(NOW.timestamp() * 1000)
rnd = random.Random(1121)
seq = [0]


def tr(inst, d, a, ts, iv=55.0, ix=2600.0, block=None, legs=None, rfq=None, combo=None, liq=None):
    seq[0] += 1
    t = {"timestamp": ts, "trade_id": f"ETH-{300000000 + seq[0]}", "trade_seq": seq[0], "instrument_name": inst, "direction": d, "amount": a,
         "price": round(rnd.uniform(0.001, 0.2), 4), "mark_price": round(rnd.uniform(0.001, 0.2), 5), "index_price": ix, "iv": iv, "tick_direction": seq[0] % 4}
    if liq:
        t["liquidation"] = liq
    if block:
        t.update(block_trade_id=block, block_trade_leg_count=legs)
        if rfq is not None:
            t["block_rfq_id"] = rfq
        if combo:
            t.update(combo_id=combo, combo_trade_id=seq[0] * 7)
    return t


def blocks() -> tuple[list, list]:
    E = "ETH-26DEC26"
    out, fut = [], []
    k = [0]

    def blk(legs, **kw):
        k[0] += 1
        bid = f"BLOCK-{900 + k[0]}"
        ts = T - rnd.randrange(60_000, 20 * 3_600_000)
        for inst, d, a in legs:
            out.append(tr(inst, d, a, ts, block=bid, legs=len(legs), rfq=k[0] * 11 if k[0] % 2 else None, combo=f"combo-{k[0]}" if k[0] % 3 == 0 else None, **kw))
        return bid, ts
    blk([(f"{E}-3000-C", "buy", 100)])                                                            # single
    blk([(f"{E}-3000-C", "sell", 50), (f"{E}-3000-P", "sell", 50)])                                # straddle
    blk([(f"{E}-3200-C", "buy", 25), (f"{E}-2800-P", "buy", 25)])                                  # strangle
    blk([(f"{E}-3000-C", "buy", 10), (f"{E}-3000-P", "sell", 10)])                                 # synthetic
    blk([(f"{E}-3400-C", "buy", 40), (f"{E}-2600-P", "sell", 40)])                                 # risk_reversal
    blk([(f"{E}-3000-C", "buy", 30), (f"{E}-3200-C", "sell", 30)])                                 # vertical
    blk([(f"{E}-3000-C", "buy", 10), (f"{E}-3200-C", "sell", 20)])                                 # ratio_spread
    blk([(f"{E}-3000-C", "buy", 10), (f"{E}-3200-C", "buy", 20)])                                  # other(같은 쪽)
    blk([("ETH-30JAN27-3000-C", "buy", 15), (f"{E}-3000-C", "sell", 15)])                          # calendar
    blk([("ETH-30JAN27-3200-C", "buy", 15), (f"{E}-3000-C", "sell", 15)])                          # diagonal
    blk([(f"{E}-2800-C", "buy", 10), (f"{E}-3000-C", "sell", 20), (f"{E}-3200-C", "buy", 10)])     # butterfly
    blk([(f"{E}-2800-P", "buy", 10), (f"{E}-2600-P", "sell", 10), (f"{E}-2400-P", "sell", 10)])    # ladder
    blk([(f"{E}-2600-C", "buy", 5), (f"{E}-2800-C", "sell", 5), (f"{E}-3000-C", "sell", 5), (f"{E}-3200-C", "buy", 5)])   # condor
    blk([(f"{E}-2800-C", "buy", 8), (f"{E}-2800-P", "sell", 8), (f"{E}-3200-C", "sell", 8), (f"{E}-3200-P", "buy", 8)])  # box
    blk([(f"{E}-2600-P", "buy", 6), (f"{E}-3000-P", "sell", 6), (f"{E}-3000-C", "sell", 6), (f"{E}-3400-C", "buy", 6)])  # iron_butterfly
    blk([(f"{E}-2600-P", "buy", 6), (f"{E}-2800-P", "sell", 6), (f"{E}-3200-C", "sell", 6), (f"{E}-3400-C", "buy", 6)])  # iron_condor
    blk([(f"{E}-3000-C", "buy", 10), (f"{E}-3000-C", "sell", 10)])                                 # future_only(순 0)
    b1, t1 = blk([(f"{E}-3000-C", "buy", 20), (f"{E}-3000-P", "buy", 20)], iv=0.0)                  # 헤지(역선물) · iv 0
    fut.append((f"FUT-{b1}-1", t1, "ETH-PERPETUAL", "sell", 52000.0, 2600.0, 2600.0, b1))
    b2, t2 = blk([("SOL_USDC-26DEC26-200-C", "buy", 300), ("SOL_USDC-26DEC26-180-P", "sell", 300)], ix=190.0)   # USDC 선형 헤지
    fut.append((f"FUT-{b2}-1", t2, "SOL_USDC-PERPETUAL", "buy", 120.0, 190.0, 190.0, b2))
    b3, t3 = blk([("BTC-26DEC26-100000-C", "sell", 2), ("BTC-26DEC26-90000-P", "sell", 2)], ix=None)    # 지수 없음 · 선물 값 0 = 구조 칸 없음
    fut.append((f"FUT-{b3}-1", t3, "BTC-PERPETUAL", "buy", 1000.0, 0.0, 0.0, b3))
    blk([("XRP_USDC-26DEC26-1d55-C", "buy", 5000)], ix=2.1)
    old = tr(f"{E}-3000-C", "buy", 1, T - 25 * 3_600_000, block="BLOCK-OLD", legs=1)                 # 24시간 창 밖
    out.append(old)
    return out, fut


def singles(n: int) -> list:
    insts = ["ETH-26DEC26-3000-C", "ETH-26DEC26-2600-P", "BTC-26DEC26-100000-C", "BTC-30JAN27-80000-P", "SOL_USDC-26DEC26-200-C", "XRP_USDC-26DEC26-2d5-P",
             "ETH-21NOV26-2600-C"]   # 마지막 = 오늘 08:00 만기(남은 해 > 0 · 곧 0)
    out = []
    for _ in range(n):
        inst = rnd.choice(insts)
        ix = {"E": 2600.0, "B": 95000.0, "S": 190.0, "X": 2.1}[inst[0]]
        out.append(tr(inst, rnd.choice(["buy", "sell"]), round(rnd.uniform(0.1, 50), 1), T - rnd.randrange(0, 26 * 3_600_000),
                      iv=rnd.choice([0.0, 45.5, 61.25, 80.0]), ix=rnd.choice([ix, ix, None]), liq=rnd.choice([None, None, None, "T", "M", ""])))
    return out


def main() -> None:
    bl, fut = blocks()
    sg = singles(400)
    b1, b2 = sg[:200] + bl[:30], sg[200:] + bl[30:] + sg[5:15]     # 둘째 = 중복 10건 포함
    with tempfile.TemporaryDirectory() as td:
        S.LIVE = Path(td)
        S.DB, S.STATE_PATH = S.LIVE / "deribit_options.duckdb", S.LIVE / "deribit_block_trades_state.json"

        class FakeDT(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return NOW
        S.datetime = FakeDT
        S.time = types.SimpleNamespace(time=lambda: NOW.timestamp(), sleep=lambda s: None)
        con = S._connect()
        con.executemany("INSERT INTO block_future_legs VALUES (?,?,?,?,?,?,?,?)", fut)
        con.close()
        gold = []
        for rows, gaps in ((b1, [(T - 600_000, T - 300_000, "ConnectionClosedError", 12)]), (b2, []), ([], [(T - 90_000_000, T - 86_400_000, "unrecoverable ETH", 0)])):
            added = S.write([S.row(t) for t in rows], gaps)
            gold.append({"added": added, "state": S.STATE_PATH.read_text()})
        import duckdb
        c = duckdb.connect(str(S.DB))
        dump = {t: [list(r) for r in c.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()] for t in ("option_trades", "gaps")}
        c.close()
        # backfill: ETH 두 쪽(has_more) · BTC 빈 쪽 · USDC 한 쪽(코인 밖 종목 섞임)
        pages = {"ETH": [{"trades": sg[:3], "has_more": True}, {"trades": sg[3:5], "has_more": False}], "BTC": [{"trades": [], "has_more": False}],
                 "USDC": [{"trades": [sg[5], dict(sg[6], instrument_name="AVAX_USDC-26DEC26-30-C")], "has_more": True}, {"trades": [], "has_more": True}]}
        calls, served = [], {k: 0 for k in pages}

        class R:
            def __init__(self, body):
                self.b = body

            def raise_for_status(self):
                pass

            def json(self):
                return self.b

        def get(url, timeout=None, params=None):
            calls.append([url, dict(params), timeout])
            api = params["currency"]
            p = pages[api][min(served[api], len(pages[api]) - 1)]
            served[api] += 1
            return R({"result": p})
        sys.modules["requests"].get = get
        bf = S.backfill({"ETH": T - 3_600_000, "BTC": T - 7_200_000, "USDC": T - 100}, T)
    (FIX / "deribit_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "iso_now": NOW.isoformat(), "batches": [b1, b2, []], "gaps": [
        [[T - 600_000, T - 300_000, "ConnectionClosedError", 12]], [], [[T - 90_000_000, T - 86_400_000, "unrecoverable ETH", 0]]], "fut": fut, "pages": pages,
        "since": [T - 3_600_000, T - 7_200_000, T - 100]}, ensure_ascii=False))
    (FIX / "deribit_golden.json").write_text(json.dumps({"writes": gold, "dump": dump, "backfill": {"calls": calls, "rows": [list(r) for r in bf]}}, ensure_ascii=False))
    st = json.loads(gold[-1]["state"])
    print("새 행", [g["added"] for g in gold], "· 블록", st["n_blocks_by_coin"], "· 구조", sorted({b.get("structure", "-") for v in st["blocks_by_coin"].values() for b in v}),
          "· backfill 요청", len(calls), "행", len(bf))


if __name__ == "__main__":
    main()
