#!/usr/bin/env python3
"""oi_lsratio(롱숏비 워커 = duckdb_persist_worker.py COLLECT_OI_LSRATIO 모드 → oi_lsratio_collector.OiLsRatioCollector) 일치 시험 (2026-11-21).

원본 `_db_init`·`_poll_once_blocking`·`_db_upsert_rows` 를 그대로 돌린다. 바뀌는 것은 바깥뿐: `_http_get_json` = 정해 둔 응답(요청 URL·머리글 기록) ·
`datetime.now` = 폴 시각 · `_DB_PATH` = 임시 파일. 폴 루프(`_poll_loop`)의 «경계 + 90초 잠» 과 «예외면 15초 쉼» 은 Rust 단위 시험이 식으로 본다.
합성 응답만 · 네트워크 없음(소켓 connect 를 막는다).

    python3 rtduck/tools/gen_oi_fixture.py            # tests/fixtures/oi_{input,golden}.json
    python3 rtduck/tools/gen_oi_fixture.py --check    # Rust 가 쓴 DuckDB ↔ 파이썬 duckdb 서로 읽기 · 서버 SQL · seal_to_lake 봉인 결과 대조
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtduck" / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))


def _no_net(*_a, **_k):
    raise RuntimeError("네트워크 금지(시험)")


socket.socket.connect = _no_net
socket.create_connection = _no_net

import duckdb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from duckdump import dump  # noqa: E402

SYMS = ["ethusdt", "btcusdt"]
PATHS = ["/futures/data/openInterestHist", "/futures/data/globalLongShortAccountRatio", "/futures/data/topLongShortPositionRatio",
         "/futures/data/topLongShortAccountRatio", "/futures/data/takerlongshortRatio"]
T0 = 1_790_000_100      # 첫 폴의 벽시계(합성 · 5분 경계 + 90초 언저리)


def body_rows(rng, sym, path, last_ms, n, style="str"):
    """한 원천의 최근 n 봉(5분 격자 · 마지막 = last_ms). style: str(거래소 그대로 글자) · num(숫자) · long(긴 가수·지수)."""
    out = []
    for k in range(n):
        ts = last_ms - (n - 1 - k) * 300_000
        v = lambda a, b: (f"{rng.uniform(a, b):.8f}" if style == "str" else rng.uniform(a, b) if style == "num"   # noqa: E731
                          else repr(rng.uniform(a, b) * 10 ** rng.randint(-9, 9)))
        if path.endswith("openInterestHist"):
            out.append({"symbol": sym.upper(), "sumOpenInterest": v(1e5, 3e6), "sumOpenInterestValue": v(1e8, 9e9), "CMCCirculatingSupply": "1", "timestamp": ts})
        elif path.endswith("takerlongshortRatio"):
            out.append({"buySellRatio": v(0.5, 2), "sellVol": v(10, 9e3), "buyVol": v(10, 9e3), "timestamp": ts})
        else:
            la = rng.uniform(0.3, 0.7)
            out.append({"symbol": sym.upper(), "longAccount": f"{la:.4f}", "longShortRatio": v(0.4, 2.5), "shortAccount": f"{1 - la:.4f}", "timestamp": str(ts)})
    return out


def gen(rng) -> dict:
    """폴 5번 × 종목 둘. 응답 = {"body": 글자} | {"status": 4xx/5xx} | {"err": "URLError"}."""
    polls = []
    for p in range(5):
        t = T0 + p * 300
        last = (t - 90) // 300 * 300 * 1000 - 300_000          # 경계 + 90초에 묻는다 → 끝난 마지막 봉
        resp = {}
        for sym in SYMS:
            r = {}
            for path in PATHS:
                n = 12 - (path.endswith("takerlongshortRatio"))                          # 테이커 비는 한 봉 늦다(원본 주석)
                rows = body_rows(rng, sym, path, last - 300_000 * path.endswith("takerlongshortRatio"), n,
                                 style=("num" if (p == 3 and sym == "btcusdt") else "long" if p == 4 else "str"))
                r[path] = {"body": json.dumps(rows)}
            if p == 1:      # 원천 실패 셋: HTTP 오류 · 목록 아닌 응답 · 연결 실패 -- 그 칸은 이전 값(COALESCE)
                r[PATHS[1]] = {"status": 503}
                r[PATHS[3]] = {"body": json.dumps({"code": -1121, "msg": "Invalid symbol."})}
                r[PATHS[4]] = {"err": "URLError"}
            if p == 2 and sym == "btcusdt":   # 숫자로 못 바꾸는 칸 = 그 폴 전체 예외(아무것도 안 씀)
                rows = json.loads(r[PATHS[0]]["body"])
                rows[3]["sumOpenInterest"] = "abc"
                r[PATHS[0]] = {"body": json.dumps(rows)}
            if p == 2 and sym == "ethusdt":   # 같은 ts 두 번(sources_ok 가 두 번 센다) · 깨진 JSON(그 원천만 None)
                rows = json.loads(r[PATHS[2]]["body"])
                rows.append(dict(rows[-1], longShortRatio="1.23456789"))
                r[PATHS[2]] = {"body": json.dumps(rows)}
                r[PATHS[3]] = {"body": "[{\"timestamp\": 1"}
            if p == 3 and sym == "ethusdt":   # 다섯 다 실패 → 행 없음 → 쓰지 않음
                r = {path: {"err": "URLError"} for path in PATHS}
            if p == 4 and sym == "btcusdt":   # 빈 목록(닿았지만 새 점 없음)
                r[PATHS[0]] = {"body": "[]"}
            resp[sym] = r
        polls.append({"t": t, "resp": resp})
    return {"symbols": SYMS, "fetch_limit": 12, "polls": polls}


def python_reference(inp, db: Path) -> dict:
    m = __import__("oi_lsratio_collector")
    m._DB_PATH = str(db)
    clock = [0.0]

    class FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromtimestamp(clock[0], tz=tz)

    m.datetime = FakeDT
    reqs = []
    cur = {}

    def get(url, timeout=15.0):
        reqs.append(url)
        path = url.split("?")[0].removeprefix(m._BASE_URL)
        r = cur[path]
        if "err" in r:
            raise urllib.error.URLError("가짜")
        if r.get("status", 200) >= 400:
            raise urllib.error.HTTPError(url, r["status"], "가짜", {}, None)
        return json.loads(r["body"])

    os.environ["OI_FETCH_LIMIT"] = str(inp["fetch_limit"])
    cs = [m.OiLsRatioCollector(s) for s in inp["symbols"]]
    for c in cs:
        c._http_get_json = get
        c._db_init()
    log = []
    for p in inp["polls"]:
        clock[0] = p["t"]
        for c in cs:
            cur.clear()
            cur.update(p["resp"][c.symbol])
            try:
                rows = c._poll_once_blocking()
                if rows:
                    c._db_upsert_rows(rows)
                log.append([c.symbol, "ok", len(rows), c._last_poll_ok_sources])
            except Exception as e:  # noqa: BLE001 -- _poll_loop 의 except 그대로(그 폴은 버린다)
                log.append([c.symbol, type(e).__name__, 0, 0])
    assert m._HEADERS == {"User-Agent": "crypto-scalping-oi-lsratio-collector/1.0"}
    return {"requests": reqs, "log": log}


def server_sql(db: Path, now_s: float) -> dict:
    """서버 _mc_collectors 의 롱숏비 SQL(dashboard/server.py:2535) -- now() 만 고정 시각으로."""
    con = duckdb.connect(str(db), read_only=True)
    out = {}
    for asset in ("eth", "btc"):
        t = "oi_lsratio_5m" if asset == "eth" else f"oi_lsratio_5m_{asset}"
        out[asset] = [[repr(x) if isinstance(x, float) else x for x in r] for r in con.execute(
            f"""SELECT epoch(ts), global_ls_ratio, top_pos_ls_ratio, taker_ls_ratio FROM {t}
                WHERE ts >= to_timestamp(?) - INTERVAL 25 HOUR ORDER BY ts""", [now_s]).fetchall()]
    con.close()
    return out


def seal_files(db: Path, work: Path) -> dict:
    """seal_to_lake.seal 을 롱숏비 명세 하나로(원천 = 이 파일) -- 쓴 parquet 의 행(파이썬 duckdb 로 읽음)."""
    import scripts.seal_to_lake as sl
    root = work / "root"
    (root / "data/live").mkdir(parents=True, exist_ok=True)
    (root / "data/live/oi_lsratio.duckdb").write_bytes(db.read_bytes())
    spec = [s for s in sl.SPECS if s[1] == "oi_lsratio_5m"]
    assert len(spec) == 1
    st = sl.seal(specs=spec, root=root, lake=work / "lake", now=dt.datetime(2026, 9, 30, 6, tzinfo=dt.timezone.utc))
    out = {"stats": st}
    con = duckdb.connect()
    for p in sorted((work / "lake").rglob("*.parquet")):
        out[str(p.relative_to(work / "lake"))] = [[repr(x) if isinstance(x, float) else str(x) for x in r]
                                                  for r in con.execute(f"SELECT * FROM read_parquet('{p}', hive_partitioning = false) ORDER BY ts").fetchall()]
    return out


def cargo(*args) -> str:
    env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
    r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 300 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release", "--example", *args],
                       cwd=ROOT / "rtduck", env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"cargo {args} 실패:\n{r.stderr}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        inp = json.loads((FIX / "oi_input.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            python_reference(inp, td / "py.duckdb")
            cargo("replay_oi", str(FIX / "oi_input.json"), str(td / "rs.duckdb"))
            x, y = dump(td / "py.duckdb"), dump(td / "rs.duckdb")
            print(f"  파이썬 duckdb {duckdb.__version__} 이 Rust 파일을 읽음: {'같음' if x == y else '다름'} (표 {len(y)} · 행 {sum(len(v['rows']) for v in y.values())})")
            assert x == y
            rs_reads_py = json.loads(cargo("dump_duck", str(td / "py.duckdb")))
            same = rs_reads_py == json.loads(cargo("dump_duck", str(td / "rs.duckdb")))
            print(f"  Rust(DuckDB 1.5.1) 가 파이썬 {duckdb.__version__} 파일을 읽음: {'같음' if same else '다름'}")
            assert same
            now = inp["polls"][-1]["t"] + 60
            s1, s2 = server_sql(td / "py.duckdb", now), server_sql(td / "rs.duckdb", now)
            print(f"  서버 롱숏비 SQL: {'같음' if s1 == s2 else '다름'} (eth {len(s1['eth'])}행 · btc {len(s1['btc'])}행)")
            assert s1 == s2 and s1["eth"]
            k1, k2 = seal_files(td / "py.duckdb", td / "w1"), seal_files(td / "rs.duckdb", td / "w2")
            print(f"  seal_to_lake 봉인: {'같음' if k1 == k2 else '다름'} ({k1['stats']} · 파일 {len(k1) - 1})")
            assert k1 == k2 and len(k1) > 1
        print("check OK")
        return
    inp = gen(random.Random(20261128))
    with tempfile.TemporaryDirectory() as td:
        ref = python_reference(inp, Path(td) / "py.duckdb")
        golden = dict(ref, tables=dump(Path(td) / "py.duckdb"), duckdb_version=duckdb.__version__)
    (FIX / "oi_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "oi_golden.json").write_text(json.dumps(golden, separators=(",", ":")))
    print("log", golden["log"])
    print({t: len(v["rows"]) for t, v in golden["tables"].items()}, f"요청 {len(ref['requests'])}",
          f"{(FIX / 'oi_input.json').stat().st_size // 1024}+{(FIX / 'oi_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
