"""seal_to_lake 전 스트림 정답지: 원본 seal() 을 SPECS 전부로 · 합성 hot SQLite(표 스키마 = rtcollect·rtduck 소스의 CREATE 글자 -- 원본 수집기와 같은 글자)
+ 합성 DuckDB 원천(oi_lsratio · HL from_pi)으로 두 번 돌리고, 두 번째는 원본 __main__ 처럼 hot 정리 + seal_status 글자까지.

🔴sqlite 확장: 서버 판 duckdb 1.5.1 용 sqlite 확장은 개발기에 없다(설치 = 바깥 접속). 개발기 python3 의 duckdb 1.5.5 + 이미 깔린 sqlite 확장으로 뜬다.
  hash(row) 지문은 1.5.1(Rust)과 1.5.5 가 같다(oi_lsratio 46파일 실측) -- 1.5.1 확장의 형 대응은 서버에서 `--check` 로 확인할 것.
  python3 rtduck/tools/gen_seal_full_fixture.py              # 정답지
  python3 rtduck/tools/gen_seal_full_fixture.py --check D    # Rust 가 쓴 D/lake 위에서 원본 seal 을 스트림마다 dry-run → 전부 «건너뜀»인지 + 값 대조
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import random
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.seal_to_lake as S  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
NOW = dt.datetime(2026, 11, 21, 3, 0, tzinfo=dt.timezone.utc)
DAY0 = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)
N_DAYS = 20
COINS = {  # 표 → 코인 열 값들(coin SQL 이 바꾸는 꼴 그대로)
    "trade_tape_1s@binance_tape": ["ethusdt", "BTCUSDT"], "trade_tape_1s@binance_spot_tape": ["ETHUSDT"], "trade_tape_1s@okx_tape": ["ETH-USDT-SWAP", "BTC-USDT-SWAP"],
    "trade_tape_1s@bybit_tape": ["ETHUSDT", "SOLUSDT"], "oi_1s": ["ethusdt", "btcusdt"], "mark_price_1s": ["ethusdt"], "liquidations": ["ETHUSDT", "BTCUSDT"],
    "okx_oi": ["ETH-USDT-SWAP"], "okx_mark": ["ETH-USDT-SWAP", "BTC-USDT-SWAP"], "okx_funding": ["ETH-USDT-SWAP"], "okx_liquidations": ["ETH-USDT-SWAP", "DOGE-USDT-SWAP"],
    "hl_asset_ctx": ["ETH", "BTC"], "bybit_oi": ["ETHUSDT"], "bybit_mark": ["ETHUSDT"], "bybit_funding": ["SOLUSDT"], "treasury_fut_1m": ["ZN", "ZB"],
    "hl_xyz_1m": ["xyz:XYZ100"], "hl_xyz_ctx_1s": ["xyz:XYZ100"], "yahoo_5m": ["ES", "USDJPY"], "hl_positions": ["ETH"], "hl_liquidations": ["ETH", "BTC"],
}
COIN_COL = {"okx_liquidations": "inst_id", "okx_oi": "inst", "okx_mark": "inst", "okx_funding": "inst", "bybit_oi": "inst", "bybit_mark": "inst", "bybit_funding": "inst",
            "hl_asset_ctx": "coin", "hl_xyz_1m": "coin", "hl_xyz_ctx_1s": "coin", "hl_positions": "coin", "hl_liquidations": "coin"}
TS_COL = {"trade_tape_1s": ("ts_sec", 1), "oi_1s": ("ts_ms", 1000), "mark_price_1s": ("ts_ms", 1000), "liquidations": ("ts_ms", 1000), "okx_oi": ("ts_ms", 1000),
          "okx_mark": ("ts_ms", 1000), "okx_funding": ("ts_ms", 1000), "okx_liquidations": ("ts_ms", 1000), "hl_asset_ctx": ("recv_ms", 1000), "bybit_oi": ("ts_ms", 1000),
          "bybit_mark": ("ts_ms", 1000), "bybit_funding": ("ts_ms", 1000), "treasury_fut_1m": ("ts_sec", 1), "hl_xyz_1m": ("ts_sec", 1), "hl_xyz_ctx_1s": ("recv_sec", 1),
          "yahoo_5m": ("ts_sec", 1), "hl_positions": ("ts_ms", 1000), "hl_liquidations": ("detected_ms", 1000)}


def ddls() -> dict[str, str]:
    """rtcollect·rtduck 소스의 CREATE TABLE 글자(원본 수집기와 같은 글자 -- 각 수집기 시험이 스키마 글자를 대조한다)."""
    out = {}
    for f in glob.glob(str(ROOT / "rtcollect/src/*.rs")) + glob.glob(str(ROOT / "rtduck/src/*.rs")):
        for m in re.finditer(r'"(CREATE TABLE IF NOT EXISTS (\w+)\s*\((?:[^"\\]|\\.)*?)"', open(f).read(), re.S):
            if m.group(2) in TS_COL:
                out[m.group(2)] = m.group(1).replace("\\n", "\n")
    return out


def cols_of(ddl: str) -> list[tuple[str, str]]:
    body = ddl[ddl.index("(") + 1:ddl.rindex(")")]
    out = []
    for part in re.split(r",(?![^()]*\))", body):
        w = part.split()
        if w and w[0].upper() not in ("PRIMARY", "UNIQUE"):
            out.append((w[0], w[1] if len(w) > 1 else ""))
    return out


def build() -> tuple[dict, list]:
    rnd = random.Random(20261121)
    D = ddls()
    files: dict[str, list] = {}       # 상대 경로 → [(표, DDL, 행들)]
    for venue, stream, sources, _ in S.SPECS:
        for rel, table, _ in sources:
            if not rel.endswith(".sqlite") or table not in TS_COL:
                continue
            key = f"{table}@{Path(rel).stem}" if table == "trade_tape_1s" else table
            if any(t == table for t, _, _ in files.get(rel, [])):
                continue
            cols = cols_of(D[table])
            tcol, mult = TS_COL[table]
            ccol = COIN_COL.get(table, "symbol")
            rows = []
            for c in COINS[key]:
                for d in range(N_DAYS):
                    for k in range(2):
                        t = DAY0 + dt.timedelta(days=d, hours=3 + 7 * k, seconds=rnd.randrange(3600))
                        r = []
                        for name, ty in cols:
                            u = ty.upper()
                            if name == tcol:
                                r.append(int(t.timestamp()) * mult + (rnd.randrange(1000) if mult == 1000 else 0))
                            elif name == ccol:
                                r.append(c)
                            elif "INT" in u:
                                r.append(rnd.randrange(1, 10**6) if name != "price_bin" else rnd.randrange(240000, 260000))
                            elif any(x in u for x in ("REAL", "DOUB")):
                                r.append(None if rnd.random() < 0.05 else round(rnd.uniform(-100, 3000), 4))
                            else:
                                r.append(rnd.choice(["long", "short", "buy", "sell", "isolated", "cross", "liquidation"]))
                        rows.append(r)
            files.setdefault(rel, []).append((table, D[table], rows))
    # DuckDB 원천: HL from_pi(같은 표 둘 · DuckDB 형) -- 만드는 SQL 로 넘긴다
    duck = []
    fp = "data/live/hyperliquid_positions_btc_sol_xrp_hype.from_pi.duckdb"
    for table in ("hl_positions", "hl_liquidations"):
        duck.append((fp, D[table].replace("IF NOT EXISTS ", "")))
        cols = cols_of(D[table])
        vals = []
        for d in range(0, 6):
            for c in ("SOL", "XRP"):
                t = DAY0 + dt.timedelta(days=d, hours=5)
                vals.append("(" + ", ".join(
                    str(int(t.timestamp()) * 1000 + d) if n in ("ts_ms", "detected_ms") else (f"'{c}'" if n == "coin" else
                    (str(rnd.randrange(1, 10**6)) if "INT" in ty.upper() else (f"{round(rnd.uniform(0, 100), 3)}" if "DOUB" in ty.upper() else "'cross'")))
                    for n, ty in cols) + ")")
        duck.append((fp, f"INSERT INTO {table} VALUES " + ", ".join(vals)))
    return files, duck


def make(root: Path, files: dict, duck: list) -> None:
    for rel, tables in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(p)
        for table, ddl, rows in tables:
            c.execute(ddl)
            c.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)
        c.commit()
        c.close()
    for rel, sql in duck:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        c = duckdb.connect(str(p))
        c.execute(sql)
        c.close()


MODS = [  # 두 번째 전에 원천에 거는 SQL(sqlite) -- 최근 날짜 행 추가(다시 봉인) · 최근 날짜 행 삭제(거부) · 오래된 날짜 값 바꿈(건너뜀)
    ("data/hot/binance_tape.sqlite", "INSERT INTO trade_tape_1s(symbol, ts_sec, price_bin, buy_qty) VALUES ('ethusdt', 1763434800, 250001, 1.25)"),     # 2025-11-18? -> 아래에서 고침
    ("data/hot/okx_ctx.sqlite", "DELETE FROM okx_mark WHERE rowid IN (SELECT rowid FROM okx_mark WHERE ts_ms >= 1795132800000 AND ts_ms < 1795219200000 LIMIT 1)"),
    ("data/hot/bybit_ctx.sqlite", "UPDATE bybit_oi SET oi_base = oi_base + 1 WHERE ts_ms < 1794268800000"),
]
MODS[0] = ("data/hot/binance_tape.sqlite", f"INSERT INTO trade_tape_1s(symbol, ts_sec, price_bin, buy_qty) VALUES ('ethusdt', {int(dt.datetime(2026, 11, 17, 12, tzinfo=dt.timezone.utc).timestamp())}, 250001, 1.25)")


def run_mods(root: Path) -> None:
    for rel, sql in MODS:
        c = sqlite3.connect(root / rel)
        c.execute(sql)
        c.commit()
        c.close()


def describe(lake: Path) -> dict:
    con = duckdb.connect()
    out = {}
    for p in sorted(lake.rglob("*.parquet")):
        q = f"read_parquet('{p.as_posix()}', hive_partitioning = false)"
        cols = [list(r[:2]) for r in con.execute(f"describe select * from {q}").fetchall()]
        sel = ", ".join(f'epoch_ms("{n}")' if "TIMESTAMP" in t else f'"{n}"' for n, t in cols)
        rows = [list(r) for r in con.execute(f"select {sel} from {q} order by all").fetchall()]
        out[str(p.relative_to(lake))] = {"cols": cols, "rows": rows}
    return out


def hot_counts(root: Path) -> dict:
    out = {}
    for f in sorted((root / "data/hot").glob("*.sqlite")):
        c = sqlite3.connect(f)
        for (t,) in c.execute("select name from sqlite_master where type='table' order by name"):
            out[f"{f.name}:{t}"] = c.execute(f"select count(*) from {t}").fetchone()[0]
        c.close()
    return out


def prune_table(root: Path):
    """원본 HOT_PRUNE 을 root 기준으로(경로만 바꾼다)."""
    return [(root / Path(h).relative_to(S.ROOT), t, ts, cc, v, st, fn, k) for h, t, ts, cc, v, st, fn, k in S.HOT_PRUNE]


def main() -> None:
    files, duck = build()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make(root, files, duck)
        lake = root / "lake"
        st1 = S.seal(specs=S.SPECS, root=root, lake=lake, now=NOW)
        d1 = describe(lake)
        run_mods(root)
        st2 = S.seal(specs=S.SPECS, root=root, lake=lake, now=NOW)
        assert not st2["failed"], st2
        st2["hot_pruned"] = {f"{v}/{s_}": S.prune_hot(h, t, ts, cc, v, s_, fn, k, now=NOW, lake=lake) for h, t, ts, cc, v, s_, fn, k in prune_table(root)}
        status = json.dumps({"finished_at": NOW.timestamp(), **st2}, ensure_ascii=False)
        d2 = describe(lake)
        hc = hot_counts(root)
    specs = [[v, s_, [list(x) for x in src], ts] for v, s_, src, ts in S.SPECS]
    hp = [[str(Path(h).relative_to(S.ROOT)), t, ts, cc, v, st, getattr(fn, "__name__", "?"), k] for h, t, ts, cc, v, st, fn, k in S.HOT_PRUNE]
    (FIX / "seal_full_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "files": files, "duck": duck, "mods": MODS, "specs": specs, "hot_prune": hp,
                                                          "duckdb": duckdb.__version__}, ensure_ascii=False))
    (FIX / "seal_full_golden.json").write_text(json.dumps({"run1": {"stats": st1, "lake": d1}, "run2": {"stats": st2, "lake": d2, "status": status, "hot": hc}}, ensure_ascii=False))
    print("1회", st1, len(d1), "파일")
    print("2회", st2, len(d2), "파일")
    print("hot 남은 행", hc)


def check(d: Path) -> None:
    """Rust lake 위에서 원본 seal 을 스트림마다 dry-run(원천 = Rust 시험이 남긴 같은 원천)."""
    gold = json.loads((FIX / "seal_full_golden.json").read_text())["run2"]
    got = describe(d / "lake")
    bad = [k for k in set(got) | set(gold["lake"]) if got.get(k) != gold["lake"].get(k)]
    print(f"값 대조: 파일 {len(got)} · 다른 파일 {len(bad)} {bad[:3]}")
    res = {}
    for v, s_, _, _ in S.SPECS:
        st = S.seal(specs=S.SPECS, root=d, lake=d / "lake", now=NOW, dry_run=True, only=f"{v}/{s_}")
        res[f"{v}/{s_}"] = st
        ok = st["written"] == 0 and st["resealed"] == 0 and st["failed"] == 0
        print(f"{'같음' if ok else '🔴다름'} {v}/{s_}: {st}")
    assert not bad and all(r["written"] == 0 and r["resealed"] == 0 and r["failed"] == 0 for r in res.values())


if __name__ == "__main__":
    check(Path(sys.argv[2])) if "--check" in sys.argv else main()
