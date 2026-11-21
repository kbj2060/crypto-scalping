"""Deribit ③ 정답지: 원본 reconcile()(history 체결 다시 받기 · 강제청산 표시 · 헤지 다리 · 블록 RFQ) → write()(상태 JSON: 헤지 다리 · ΔOI) 를 가짜 requests 로.
ΔOI = 체인 스냅숏 28시간(10분마다 · 사라지는 종목 하나) -- 🔴DuckDB now() 는 진짜 시계라 NOW = 지금 · 정시 경계에서 멀리.
  $PY rtduck/tools/gen_deribit_recon_fixture.py   (서버 판 venv)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
fake = types.ModuleType("requests")
sys.modules["requests"] = fake
import live_deribit_block_trade_collector_20260928 as S  # noqa: E402
import duckdb  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
NOW = dt.datetime.fromtimestamp(int(time.time()) + 0.25, dt.timezone.utc)
T = int(NOW.timestamp() * 1000)
E = "ETH-26DEC26"


def tr(tid, inst, ts, d="buy", a=1.0, block=None, legs=None, liq=None, ix=2600.0):
    t = {"timestamp": ts, "trade_id": tid, "trade_seq": 1, "instrument_name": inst, "direction": d, "amount": a, "price": 0.05, "mark_price": 0.05,
         "index_price": ix, "iv": 55.0, "tick_direction": 1}
    if block:
        t.update(block_trade_id=block, block_trade_leg_count=legs)
    if liq:
        t["liquidation"] = liq
    return t


SEED = [  # 미리 있는 체결(WS 로 받은 것) -- 강제청산 표시는 비어 있다
    tr("ETH-1", f"{E}-3000-C", T - 3_000_000), tr("ETH-2", f"{E}-3000-P", T - 2_000_000),
    tr("ETH-A1", f"{E}-3000-C", T - 3_600_000, block="BLOCK-A", legs=3), tr("ETH-A2", f"{E}-3200-C", T - 3_600_000, "sell", block="BLOCK-A", legs=3),
    tr("BTC-B1", "BTC-26DEC26-100000-C", T - 5_400_000, block="BLOCK-B", legs=2, ix=95000.0),
    tr("ETH-C1", f"{E}-2800-P", T - 1_800_000, block="BLOCK-C", legs=3), tr("ETH-C2", f"{E}-2600-P", T - 1_800_000, block="BLOCK-C", legs=3),
    tr("SOL-D1", "SOL_USDC-26DEC26-200-C", T - 900_000, block="BLOCK-D", legs=2, ix=190.0),
]
FUT0 = [("FUT-C", T - 1_800_000, "ETH-PERPETUAL", "sell", 1000.0, 2600.0, 2600.0, "BLOCK-C")]
RFQ0 = [("RFQ-OLD", T - 5 * 86_400_000, "ETH", "buy", 10.0, 0.1, 0.1, None, "[]", None, "{}")]


def chain_sql() -> list[str]:
    out = []
    t0 = (T // 3_600_000) * 3_600_000 - 28 * 3_600_000 + 5 * 60_000
    for k in range(28 * 6):
        t = t0 + k * 600_000
        if abs((T - 26 * 3_600_000) - t) < 120_000:
            continue   # 26시간 경계 가까이는 두지 않는다
        vals = []
        for cur, insts in (("ETH", [f"{E}-3000-C", f"{E}-2600-P", "ETH-28NOV26-2500-C"]), ("BTC", ["BTC-26DEC26-100000-C"])):
            for j, inst in enumerate(insts):
                if inst.startswith("ETH-28NOV26") and k > 100:
                    continue
                vals.append(f"(to_timestamp({t // 1000}), '{cur}', '{inst}', 'call', 1.0, to_timestamp({(T + 86_400_000 * 30) // 1000}), 30.0, "   # 🔴make_timestamp::TIMESTAMPTZ 는 ICU 가 있으면 지역 시각으로 읽힌다(9시간 밀림) -- 절대 시각으로
                            f"{1000 + 3.5 * k + 17 * j + (k % 7) * 0.3}, 55.0, 2600.0, 0.05, 1.0, 0.0001)")
        out.append("INSERT INTO option_chain_snapshot VALUES " + ", ".join(vals))
    return out


PAGES = {
    ("hist", "ETH"): [{"trades": [tr("ETH-1", f"{E}-3000-C", T - 3_000_000, liq="T"), tr("ETH-3", f"{E}-3200-P", T - 2_500_000, liq="M"),
                                  tr("AVAX-1", "AVAX_USDC-26DEC26-30-C", T - 2_400_000)], "has_more": True},
                      {"trades": [tr("ETH-4", f"{E}-3000-C", T - 1_000_000, "sell", 2.5), tr("ETH-A1", f"{E}-3000-C", T - 3_600_000, block="BLOCK-A", legs=3, liq="")], "has_more": False}],
    ("hist", "BTC"): [{"trades": [tr("BTC-1", "BTC-26DEC26-90000-P", T - 600_000, ix=95000.0)], "has_more": False}],
    ("hist", "USDC"): [{"trades": [], "has_more": False}],
}
FUTS = {"ETH": {"trades": [{"trade_id": "ETH-F1", "timestamp": T - 3_600_000, "instrument_name": "ETH-PERPETUAL", "direction": "buy", "amount": 5200.0, "price": 2601.5,
                            "index_price": 2600.0, "block_trade_id": "BLOCK-A"},
                           {"trade_id": 77, "timestamp": T - 3_600_000, "instrument_name": "ETH-26DEC26", "direction": "sell", "amount": 100.0, "price": 2650.0, "block_trade_id": "BLOCK-A"},
                           {"trade_id": "ETH-F9", "timestamp": T - 3_600_000, "instrument_name": "ETH-PERPETUAL", "direction": "sell", "amount": 1.0, "price": 2601.0}]},
        "BTC": {"trades": [{"trade_id": "BTC-F1", "timestamp": T - 5_400_000, "instrument_name": "BTC-PERPETUAL", "direction": "sell", "amount": 10.0, "price": 1.0, "block_trade_id": "BLOCK-Z"}]},
        "USDC": {"trades": [{"trade_id": "SOL-F1", "timestamp": T - 900_000, "instrument_name": "SOL_USDC-PERPETUAL", "direction": "buy", "amount": 3.0, "price": 190.5,
                             "index_price": 190.0, "block_trade_id": "BLOCK-D"}]}}
RFQS = {"ETH": [{"block_rfqs": [{"id": 501, "timestamp": T - 86_400_000, "direction": "sell", "amount": 25, "trades": [{"price": 0.07}], "mark_price": 0.071, "combo_id": "c-1",
                                 "legs": [{"instrument_name": f"{E}-3000-C", "ratio": 1, "direction": "sell"}], "hedge": {"amount": 1000, "instrument_name": "ETH-PERPETUAL"},
                                 "index_prices": {"eth_usd": 2600.5}}], "continuation": "abc"},
                {"block_rfqs": [{"id": 499, "timestamp": T - 6 * 86_400_000, "direction": "buy", "amount": 5, "trades": [], "legs": [], "index_prices": {}}], "continuation": "def"}],
        "BTC": [{"block_rfqs": [{"id": "BR-1", "timestamp": T - 3_600_000, "amount": 1.5, "legs": []}], "continuation": None}]}
CALLS: list = []


class R:
    def __init__(self, b, st=200):
        self.b, self.status_code = b, st

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTPError {self.status_code}")

    def json(self):
        return self.b


served: dict = {}


def get(url, params=None, timeout=None):
    p = dict(params or {})
    if url.endswith("get_last_trades_by_currency_and_time") and p["kind"] == "option":
        key = ("hist", p["currency"])
        k = served.get(key, 0); served[key] = k + 1
        r = R({"result": PAGES[key][min(k, len(PAGES[key]) - 1)]})
    elif url.endswith("get_last_trades_by_currency_and_time"):
        r = R({"result": FUTS[p["currency"]]})
    elif url.endswith("get_block_rfq_trades"):
        if p["currency"] == "USDC":
            r = R({"error": "x"}, 400)
        else:
            key = ("rfq", p["currency"])
            k = served.get(key, 0); served[key] = k + 1
            r = R({"result": RFQS[p["currency"]][min(k, len(RFQS[p["currency"]]) - 1)]})
    else:
        raise AssertionError(url)
    CALLS.append([url, p, timeout, r.status_code, r.b])
    return r


fake.get = get


class FakeDT(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        S.LIVE = Path(td)
        S.DB, S.STATE_PATH = S.LIVE / "deribit_options.duckdb", S.LIVE / "deribit_block_trades_state.json"
        S.datetime = FakeDT
        S.time = types.SimpleNamespace(time=lambda: NOW.timestamp(), sleep=lambda s: None)
        import collect_deribit_option_gex_20260815 as G
        con = S._connect()
        G.ensure_tables(con)
        con.executemany(f"INSERT INTO option_trades VALUES ({','.join('?' * len(S.COLS))})", [S.row(t) for t in SEED])
        con.executemany("INSERT INTO block_future_legs VALUES (?,?,?,?,?,?,?,?)", FUT0)
        con.executemany("INSERT INTO block_rfq_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", RFQ0)
        chain = chain_sql()
        for q in chain:
            con.execute(q)
        con.close()
        res = S.reconcile()
        S.write([], [])
        c = duckdb.connect(str(S.DB))
        db_now = c.execute("SELECT epoch_us(now())").fetchone()[0]
        dump = {t: [list(r) for r in c.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()] for t in ("option_trades", "block_future_legs", "block_rfq_trades")}
        c.close()
        state = S.STATE_PATH.read_text()
    (FIX / "deribit_recon_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "iso_now": NOW.isoformat(), "seed": SEED, "fut0": FUT0, "rfq0": RFQ0, "chain": chain}, ensure_ascii=False))
    (FIX / "deribit_recon_golden.json").write_text(json.dumps({"calls": CALLS, "result": res, "dump": dump, "state": state, "db_now_us": db_now}, ensure_ascii=False))
    st = json.loads(state)
    print("대조", res, "· 요청", len(CALLS), "· ETH 블록 헤지", [(b["block_trade_id"], b.get("hedge")) for b in st["blocks_by_coin"]["ETH"]],
          "· doi 칸", sum(1 for c in st["flow_by_coin"].values() for b in c if b["doi"] is not None), "· db now 차이", db_now / 1e6 - NOW.timestamp())


if __name__ == "__main__":
    main()
