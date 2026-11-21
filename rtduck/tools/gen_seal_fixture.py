"""seal_to_lake 첫 표 정답지: 원본 seal(specs=[binance/oi_lsratio_5m]) 를 합성 DuckDB 원천 둘(data/live/oi_lsratio.duckdb 4표 · 은퇴 보관 xrp 1표)로 두 번.

  1회: 전부 봉인(끝난 날짜만 · 오늘·어제(LAG 2시간 안)는 안 씀)
  2회: 원천을 고친 뒤 -- 최근(RESEAL 안) 날짜에 행 추가 = 다시 봉인 · 최근 날짜 행 삭제 = 거부 · 오래된 날짜 변경 = 건너뜀
서버 판 venv(duckdb 1.5.1 -- rtmodel/tools/server_env.sh + pip duckdb==1.5.1)로 돌린다.
  $PY rtduck/tools/gen_seal_fixture.py            # 정답지
  $PY rtduck/tools/gen_seal_fixture.py --check D  # Rust 가 쓴 lake 폴더 D 를 이 파이썬으로 읽어 정답지와 대조
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
NOW = dt.datetime(2026, 11, 21, 1, 30, tzinfo=dt.timezone.utc)    # LAG 2시간 → 마지막 끝난 날 = 11-19
SPEC = [s for s in S.SPECS if (s[0], s[1]) == ("binance", "oi_lsratio_5m")]
DDL = ("CREATE TABLE {t} (ts TIMESTAMPTZ, symbol VARCHAR, sum_open_interest DOUBLE, sum_open_interest_value DOUBLE, global_ls_ratio DOUBLE, "
       "global_ls_long_account DOUBLE, collected_at TIMESTAMPTZ)")
MODS = [  # 2회 전에 원천에 거는 SQL(Rust 시험도 같은 것을 건다)
    ("data/live/oi_lsratio.duckdb", "INSERT INTO oi_lsratio_5m VALUES (TIMESTAMPTZ '2026-11-18 12:02:30+00', 'ETHUSDT', 1.5, 2.5, 3.5, 0.5, TIMESTAMPTZ '2026-11-18 12:03:00+00')"),
    ("data/live/oi_lsratio.duckdb", "DELETE FROM oi_lsratio_5m_btc WHERE ts >= TIMESTAMPTZ '2026-11-17 00:00:00+00' AND ts < TIMESTAMPTZ '2026-11-17 01:00:00+00'"),
    ("data/live/oi_lsratio.duckdb", "UPDATE oi_lsratio_5m_sol SET global_ls_ratio = global_ls_ratio + 1 WHERE ts < TIMESTAMPTZ '2026-11-05 00:00:00+00'"),
]


def rows_sql(n: int, day0: dt.datetime, sym: str, rnd: random.Random) -> list[str]:
    out = []
    for i in range(n):
        t = day0 + dt.timedelta(minutes=20 * i)   # 20분 간격(크기)
        v = [round(rnd.uniform(1e5, 9e5), 3), round(rnd.uniform(1e8, 9e8), 2), round(rnd.uniform(0.5, 3.0), 4), round(rnd.uniform(0.3, 0.7), 4)]
        if i % 97 == 3:
            v[2] = "NULL"
        out.append(f"(TIMESTAMPTZ '{t.isoformat()}', '{sym}', {v[0]}, {v[1]}, {v[2]}, {v[3]}, TIMESTAMPTZ '{(t + dt.timedelta(seconds=30)).isoformat()}')")
    return out


def build_sql() -> list[tuple[str, str]]:
    """원천을 만드는 SQL(원천 상대 경로, 문장) -- Rust 시험도 같은 문장으로 만든다(같은 판 DuckDB 가 같은 값으로 읽는다)."""
    rnd = random.Random(1121)
    L, A = "data/live/oi_lsratio.duckdb", "data/archive/live_retired_20261001/oi_lsratio_xrp.duckdb"
    start = dt.datetime(2026, 11, 2, tzinfo=dt.timezone.utc)
    out = []
    for tbl, sym, days in (("oi_lsratio_5m", "ETHUSDT", 20), ("oi_lsratio_5m_btc", "BTCUSDT", 19), ("oi_lsratio_5m_sol", "SOLUSDT", 6)):
        out.append((L, DDL.format(t=tbl)))
        rows = rows_sql(days * 288 // 4, start, sym, rnd)
        out += [(L, f"INSERT INTO {tbl} VALUES " + ",".join(rows[k:k + 500])) for k in range(0, len(rows), 500)]
    out.append((A, DDL.format(t="oi_lsratio_5m_xrp")))
    out.append((A, "INSERT INTO oi_lsratio_5m_xrp VALUES " + ",".join(rows_sql(300, dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc), "XRPUSDT", rnd))))
    return out


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
        rel = str(p.relative_to(lake))
        q = f"read_parquet('{p.as_posix()}', hive_partitioning = false)"
        cols = [list(r[:2]) for r in con.execute(f"describe select * from {q}").fetchall()]
        rows = con.execute(f"select epoch_ms(ts), symbol, sum_open_interest, sum_open_interest_value, global_ls_ratio, global_ls_long_account, epoch_ms(collected_at) "
                           f"from {q} order by all").fetchall()
        out[rel] = {"cols": cols, "rows": [list(r) for r in rows]}
    return out


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = build_sql()
        run_sql(root, src)
        lake = root / "lake"
        runs = []
        for k in range(2):
            if k == 1:
                run_sql(root, MODS)
            st = S.seal(specs=SPEC, root=root, lake=lake, now=NOW)
            runs.append({"stats": st, "lake": describe(lake)})
    (FIX / "seal_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "build": src, "mods": MODS, "duckdb": duckdb.__version__}))
    (FIX / "seal_golden.json").write_text(json.dumps({"runs": runs}))
    for r in runs:
        print(r["stats"], len(r["lake"]), sum(len(v["rows"]) for v in r["lake"].values()))


def check(lake: Path) -> None:
    gold = json.loads((FIX / "seal_golden.json").read_text())["runs"][-1]["lake"]
    got = describe(lake)
    assert got == gold, [k for k in set(got) | set(gold) if got.get(k) != gold.get(k)][:5]
    print(f"파이썬 duckdb {duckdb.__version__} 가 Rust 가 쓴 lake {len(got)}파일·{sum(len(v['rows']) for v in got.values())}행을 정답지와 같은 열·형·값으로 읽음")


if __name__ == "__main__":
    check(Path(sys.argv[2])) if "--check" in sys.argv else main()
