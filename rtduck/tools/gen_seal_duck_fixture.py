"""seal_to_lake DuckDB 원천 나머지 정답지: binance/micro_1m · binance/tail_risk_1m · deribit 5(option_trades·chain·option_summary·block_rfq·block_future_legs).
표 = rtduck 소스의 CREATE(+ ALTER 로 붙는 열) 글자 · 원본 수집기와 같은 열·형(BOOLEAN·INTEGER 포함). 원본 seal() 두 번(둘째 = 최근 행 추가 → 다시 봉인).
서버 판 venv(duckdb 1.5.1)로:   $PY rtduck/tools/gen_seal_duck_fixture.py   ·   $PY rtduck/tools/gen_seal_duck_fixture.py --check D
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.seal_to_lake as S  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
NOW = dt.datetime(2026, 11, 21, 3, 0, tzinfo=dt.timezone.utc)
DAY0 = dt.datetime(2026, 11, 8, tzinfo=dt.timezone.utc)
L, A = "data/live", "data/archive/live_retired_20261001"
MICRO = ("ts TIMESTAMPTZ, obi DOUBLE, taker_buy_ratio DOUBLE, spoofing_score DOUBLE, nif_whale DOUBLE, nif_retail DOUBLE, eai DOUBLE, oi_delta_pct DOUBLE, "
         "funding_rate DOUBLE, kelly_mult DOUBLE, signal_bias INTEGER, shadow_toxicity_score DOUBLE, shadow_toxicity_regime VARCHAR, shadow_queue_collapse DOUBLE, "
         "shadow_absorption_score DOUBLE, shadow_queue_bias INTEGER, shadow_regime_tag VARCHAR, shadow_regime_conf DOUBLE, data_stale BOOLEAN, depth_connected BOOLEAN, "
         "trade_connected BOOLEAN, poll_connected BOOLEAN, depth_age_sec DOUBLE, trade_age_sec DOUBLE, poll_age_sec DOUBLE, recent_trade_count_5m INTEGER, "
         "recent_trade_notional_5m DOUBLE, recent_whale_count_5m INTEGER, valid_taker_flow BOOLEAN, valid_nif BOOLEAN, warmup_30m_ready BOOLEAN, schema_version INTEGER, "
         "mark_price DOUBLE, whale_position_score DOUBLE")
TAIL = ("ts TIMESTAMPTZ, long_usd_1m DOUBLE, short_usd_1m DOUBLE, mu_long DOUBLE, sigma_long DOUBLE, mu_short DOUBLE, sigma_short DOUBLE, shadow_aftershock_prob DOUBLE, "
        "shadow_decay_half_life DOUBLE, shadow_risk_bucket VARCHAR, ws_connected BOOLEAN, ws_stale BOOLEAN, ws_age_sec DOUBLE, liq_event_count_1m INTEGER, "
        "valid_liq_stream BOOLEAN, schema_version INTEGER")
TRADES = ("ts_ms BIGINT, trade_id VARCHAR PRIMARY KEY, trade_seq BIGINT, instrument_name VARCHAR, direction VARCHAR, amount DOUBLE, price DOUBLE, mark_price DOUBLE, "
          "index_price DOUBLE, iv DOUBLE, tick_direction INTEGER, liquidation VARCHAR, is_block BOOLEAN, block_trade_id VARCHAR, block_trade_leg_count INTEGER, "
          "block_rfq_id VARCHAR, combo_id VARCHAR, combo_trade_id VARCHAR")
RFQ = "rfq_id VARCHAR PRIMARY KEY, ts_ms BIGINT, api VARCHAR, direction VARCHAR, amount DOUBLE, price DOUBLE, mark_price DOUBLE, combo_id VARCHAR, legs VARCHAR, hedge VARCHAR, index_prices VARCHAR"
FUT = "trade_id VARCHAR PRIMARY KEY, ts_ms BIGINT, instrument_name VARCHAR, direction VARCHAR, amount DOUBLE, price DOUBLE, index_price DOUBLE, block_trade_id VARCHAR"
CHAIN = ("recorded_at_utc TIMESTAMPTZ, currency VARCHAR, instrument_name VARCHAR, option_type VARCHAR, strike DOUBLE, expiration_ts TIMESTAMPTZ, days_to_expiry DOUBLE, "
         "open_interest DOUBLE, mark_iv DOUBLE, underlying_price DOUBLE, mark_price DOUBLE, volume DOUBLE, gamma_bs DOUBLE")
SUMM = ("recorded_at_utc TIMESTAMPTZ, currency VARCHAR, index_price DOUBLE, dvol DOUBLE, rv7 DOUBLE, front_atm_iv DOUBLE, front_rr25 DOUBLE, front_bf25 DOUBLE, "
        "gamma_flip DOUBLE, payload VARCHAR")
INSTS = ["ETH-26DEC26-3000-C", "BTC-26DEC26-100000-P", "SOL_USDC-26DEC26-200-C", "XRP_USDC-26DEC26-2d5-P"]


def cols(spec: str) -> list[tuple[str, str]]:
    return [(c.split()[0], c.split()[1]) for c in spec.split(", ")]


def lit(name, ty, i, t, rnd, cur):
    if ty == "TIMESTAMPTZ":
        return f"TIMESTAMPTZ '{(t if name in ('ts', 'recorded_at_utc') else t + dt.timedelta(days=30)).isoformat()}'"
    if name == "ts_ms":
        return str(int(t.timestamp() * 1000) + i % 997)
    if name in ("trade_id", "rfq_id"):
        return f"'{cur}-{i}'"
    if name == "instrument_name":
        return f"'{cur}'"
    if name == "currency":
        return f"'{cur}'"
    if ty == "BOOLEAN":
        return rnd.choice(["true", "false", "NULL"])
    if ty in ("INTEGER", "BIGINT"):
        return str(rnd.randrange(-5, 10**6))
    if ty == "DOUBLE":
        return "NULL" if rnd.random() < 0.05 else repr(round(rnd.uniform(-50, 5000), 5))
    return rnd.choice(["'a'", "'calm'", "'[{\"x\": 1}]'", "NULL"])


def build() -> list:
    rnd = random.Random(20261121)
    stmts = []
    def table(rel, tbl, spec, coins, per_day=3):
        stmts.append((rel, f"CREATE TABLE {tbl} ({spec})"))
        vals, i = [], 0
        for cur in coins:
            for d in range(14):
                for k in range(per_day):
                    i += 1
                    t = DAY0 + dt.timedelta(days=d, hours=2 + 6 * k, seconds=rnd.randrange(3600))
                    vals.append("(" + ", ".join(lit(n, ty, i, t, rnd, cur) for n, ty in cols(spec.replace(" PRIMARY KEY", ""))) + ")")
        stmts.append((rel, f"INSERT INTO {tbl} VALUES " + ", ".join(vals)))
    for c in ("", "_btc", "_sol"):
        table(f"{L}/microstructure.duckdb", f"microstructure_1m{c}", MICRO, ["x"])
    table(f"{A}/microstructure_xrp.duckdb", "microstructure_1m_xrp", MICRO, ["x"])
    table(f"{L}/tail_risk.duckdb", "tail_risk_1m", TAIL, ["x"])
    for c in ("btc", "sol"):
        table(f"{A}/tail_risk_btc_sol.duckdb", f"tail_risk_1m_{c}", TAIL, ["x"])
    d = f"{L}/deribit_options.duckdb"
    table(d, "option_trades", TRADES, INSTS)
    table(d, "block_rfq_trades", RFQ, ["r"])
    table(d, "block_future_legs", FUT, ["ETH-PERPETUAL", "SOL_USDC-PERPETUAL"])
    table(d, "option_chain_snapshot", CHAIN, ["ETH", "BTC"], per_day=2)
    table(d, "option_summary", SUMM, ["ETH", "SOL"], per_day=2)
    return stmts


MODS = [(f"{L}/tail_risk.duckdb", "INSERT INTO tail_risk_1m (ts, long_usd_1m, valid_liq_stream) VALUES (TIMESTAMPTZ '2026-11-18 05:00:00+00', 1.5, true)"),
        (f"{L}/deribit_options.duckdb", "INSERT INTO option_trades (ts_ms, trade_id, instrument_name, is_block) VALUES (1795320000000, 'NEW-1', 'BTC-26DEC26-90000-C', true)")]


def run_sql(root: Path, stmts) -> None:
    for rel, sql in stmts:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        c = duckdb.connect(str(root / rel))
        c.execute(sql)
        c.close()


def describe(lake: Path) -> dict:
    con = duckdb.connect()
    out = {}
    for p in sorted(lake.rglob("*.parquet")):
        q = f"read_parquet('{p.as_posix()}', hive_partitioning = false)"
        cs = [list(r[:2]) for r in con.execute(f"describe select * from {q}").fetchall()]
        sel = ", ".join(f'epoch_us("{n}")' if "TIMESTAMP" in t else f'"{n}"' for n, t in cs)
        out[str(p.relative_to(lake))] = {"cols": cs, "rows": [list(r) for r in con.execute(f"select {sel} from {q} order by all").fetchall()]}
    return out


SPEC = [s for s in S.SPECS if f"{s[0]}/{s[1]}" in ("binance/micro_1m", "binance/tail_risk_1m", "deribit/option_trades", "deribit/chain", "deribit/gex_summary",
                                                    "deribit/option_summary", "deribit/block_rfq", "deribit/block_future_legs")]


def main() -> None:
    stmts = build()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run_sql(root, stmts)
        lake = root / "lake"
        st1 = S.seal(specs=SPEC, root=root, lake=lake, now=NOW)
        run_sql(root, MODS)
        st2 = S.seal(specs=SPEC, root=root, lake=lake, now=NOW)
        d2 = describe(lake)
    (FIX / "seal_duck_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "build": stmts, "mods": MODS, "streams": [f"{s[0]}/{s[1]}" for s in SPEC], "duckdb": duckdb.__version__}))
    (FIX / "seal_duck_golden.json").write_text(json.dumps({"run1": st1, "run2": st2, "lake": d2}, ensure_ascii=False))
    print("1회", st1, "\n2회", st2, "\n파일", len(d2), "행", sum(len(v["rows"]) for v in d2.values()))


def check(d: Path) -> None:
    gold = json.loads((FIX / "seal_duck_golden.json").read_text())
    got = describe(d / "lake")
    bad = [k for k in set(got) | set(gold["lake"]) if got.get(k) != gold["lake"].get(k)]
    print(f"값 대조: 파일 {len(got)} · 다른 파일 {len(bad)}")
    ok = not bad
    for v, s_, _, _ in SPEC:
        st = S.seal(specs=SPEC, root=d, lake=d / "lake", now=NOW, dry_run=True, only=f"{v}/{s_}")
        same = st["written"] == 0 and st["resealed"] == 0 and st["failed"] == 0
        ok &= same
        print(f"{'같음' if same else '🔴다름'} {v}/{s_}: {st}")
    assert ok


if __name__ == "__main__":
    check(Path(sys.argv[2])) if "--check" in sys.argv else main()
