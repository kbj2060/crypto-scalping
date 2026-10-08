"""ETH 일봉 풋프린트 소급 빌더 (2026-10-08, 사용자 «일봉 풋프린트를 최대치로 · 30일 이상 · 델타·거래대금·OI·청산»).

라이브 테이프(trade_tape_1s)는 2026-09-15 부터라 그 전은 공개 아카이브 data.binance.vision 의 USDⓈ-M aggTrades
(ETHUSDT, 2020-01~)로 다시 만든다. 바이낸스 REST 가 아니다(정적 파일 — IP 한도와 무관).

산출 (data/footprint_daily/, gitignore):
  ETHUSDT_cells.parquet  day(date) · bin(int, 가격 하한 $1) · buy · sell (ETH 수량, 공격자 기준)
  ETHUSDT_days.parquet   day · open/high/low/close · buy_qty/sell_qty · buy_quote/sell_quote · n ·
                         oi(ETH, 그날 마지막 스냅샷) · oi_usd · src
  날짜 = UTC 자정(KST 09:00) — 바이낸스 일봉과 같은 경계. 공격자: is_buyer_maker=false 면 매수.
OI: 로컬 패널(5분, 2022-01-01~2026-09-14) + 일별 metrics 아카이브(그 뒤). 그 전은 비움(«모름»).
월 파일 하나 ≈ 0.3~0.6GB zip → 압축 풀어 duckdb 로 묶고 원본은 바로 지운다(재개 가능: 끝난 기간은 건너뜀).
점검: 겹치는 날의 거래량을 로컬 1분봉(klines1m) 합과 대조해 최대 상대오차를 출력한다.
🔴아카이브 aggTrades 에 같은 agg_trade_id 가 여러 번 든 날이 있다(2022-09-11~13 거래량 2~4배) -- id 로 하나만 남긴다.
  월 파일에 통째로 빠진 날(2022-08-28~30·09-01·10-29·11-07·11-14)은 일 파일로 채운다.
  재실행: --redo YYYY-MM[,YYYY-MM…] 는 그 달 조각을 지우고 다시 받는다.

실행: nice python scripts/build_eth_footprint_daily_20261008.py [--selftest]
"""
from __future__ import annotations

import io
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

MAIN = Path("/home/kbj20/crypto-scalping")
OUT = MAIN / "data/footprint_daily"
WORK = MAIN / "tmp/fp_daily_build"
SYM = "ETHUSDT"
BASE = "https://data.binance.vision/data/futures/um"
FIRST_MONTH = "2020-01"

AGG_SQL = """
WITH r AS (SELECT * FROM read_csv('{f}', header={h}, columns={cols})
           QUALIFY row_number() OVER (PARTITION BY agg_trade_id ORDER BY transact_time) = 1),   -- 아카이브 중복 행(2022-09-11~13 실측 2~4배)
     t AS (SELECT CAST(epoch_ms(transact_time) AS DATE) AS day, price, quantity AS q, transact_time AS ts,
                  NOT is_buyer_maker AS buy FROM r)
SELECT {sel} FROM t GROUP BY {grp}"""
CELL_SEL = ("day, CAST(floor(price) AS INTEGER) AS bin, sum(CASE WHEN buy THEN q ELSE 0 END) AS buy, "
            "sum(CASE WHEN buy THEN 0 ELSE q END) AS sell", "day, bin")
DAY_SEL = ("day, arg_min(price, ts) AS open, max(price) AS high, min(price) AS low, arg_max(price, ts) AS close, "
           "sum(CASE WHEN buy THEN q ELSE 0 END) AS buy_qty, sum(CASE WHEN buy THEN 0 ELSE q END) AS sell_qty, "
           "sum(CASE WHEN buy THEN q * price ELSE 0 END) AS buy_quote, sum(CASE WHEN buy THEN 0 ELSE q * price END) AS sell_quote, "
           "count(*) AS n", "day")
DUCK_COLS = ("{'agg_trade_id': 'BIGINT', 'price': 'DOUBLE', 'quantity': 'DOUBLE', 'first_trade_id': 'BIGINT', "
             "'last_trade_id': 'BIGINT', 'transact_time': 'BIGINT', 'is_buyer_maker': 'BOOLEAN'}")


def _get(url: str, dest: Path) -> bool:
    """curl 로 받는다(큰 파일 · 재시도). 없으면(404) False."""
    r = subprocess.run(["curl", "-sfL", "--retry", "5", "--retry-delay", "5", "-o", str(dest), url])
    if r.returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return True


def aggregate(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """aggTrades CSV 하나 → (cells, days). 헤더 유무는 첫 줄로 판단(2022 전 파일은 헤더 없음)."""
    with open(csv_path) as fh:
        h = "true" if not fh.readline()[:1].isdigit() else "false"
    con = duckdb.connect()
    out = [con.execute(AGG_SQL.format(f=csv_path, h=h, cols=DUCK_COLS, sel=sel, grp=grp)).df()
           for sel, grp in (CELL_SEL, DAY_SEL)]
    con.close()
    for df in out:
        df["day"] = pd.to_datetime(df.day).dt.date            # duckdb DATE → datetime64 로 온다
    return out[0], out[1]


def _ingest(name: str, url: str) -> bool:
    """한 기간(월/일) zip → parts/{name}_cells|days.parquet. 이미 있으면 건너뜀. 아카이브에 없으면 False."""
    pc, pdy = WORK / "parts" / f"{name}_cells.parquet", WORK / "parts" / f"{name}_days.parquet"
    if pc.exists() and pdy.exists():
        return True
    z, csv = WORK / f"{name}.zip", WORK / f"{name}.csv"
    if not _get(url, z):
        return False
    with zipfile.ZipFile(z) as zf, open(csv, "wb") as fo, zf.open(zf.namelist()[0]) as fi:
        while chunk := fi.read(1 << 24):
            fo.write(chunk)
    z.unlink()
    cells, days = aggregate(csv)
    csv.unlink()
    cells.to_parquet(pc); days.to_parquet(pdy)
    print(name, "일", len(days), "칸", len(cells), flush=True)
    return True


def build_parts() -> None:
    """2020-01 부터 월 파일 · 월 파일이 아직 없는 달은 일 파일(어제까지)."""
    (WORK / "parts").mkdir(parents=True, exist_ok=True)
    m, last = pd.Period(FIRST_MONTH, "M"), pd.Period(date.today(), "M")
    while m <= last:
        if m != last and _ingest(str(m), f"{BASE}/monthly/aggTrades/{SYM}/{SYM}-aggTrades-{m}.zip"):
            got = set(pd.read_parquet(WORK / "parts" / f"{m}_days.parquet").day.map(str))
            for d in pd.date_range(m.start_time, m.end_time.normalize()).date:   # 월 파일에 빠진 날(2022-08-28 등 7일 실측) → 일 파일
                if str(d) not in got:
                    _ingest(str(d), f"{BASE}/daily/aggTrades/{SYM}/{SYM}-aggTrades-{d}.zip")
        else:
            d = m.start_time.date()
            while d < min(m.end_time.date() + timedelta(days=1), date.today()):
                if not _ingest(str(d), f"{BASE}/daily/aggTrades/{SYM}/{SYM}-aggTrades-{d}.zip"):
                    print(d, "아카이브 없음", flush=True)
                d += timedelta(days=1)
        m += 1


def oi_daily(days: pd.Series) -> pd.DataFrame:
    """그날 마지막 OI 스냅샷(ETH)과 달러값. 패널(5분) → 그 뒤 날은 일별 metrics 아카이브."""
    p = pd.read_parquet(MAIN / "data/binance_vision/panel/ETHUSDT.parquet", columns=["timestamp", "close", "sum_open_interest"])
    p = p.dropna(subset=["sum_open_interest"])
    p["day"] = pd.to_datetime(p.timestamp).dt.date
    last = p.groupby("day").tail(1)
    rows = [(r.day, float(r.sum_open_interest), float(r.sum_open_interest * r.close)) for r in last.itertuples()]
    have = {r[0] for r in rows}
    for d in sorted(set(days) - have):
        if d < date(2026, 9, 15):
            continue
        try:
            raw = urllib.request.urlopen(f"{BASE}/daily/metrics/{SYM}/{SYM}-metrics-{d}.zip", timeout=60).read()
        except urllib.error.URLError:
            continue
        m = pd.read_csv(io.BytesIO(zipfile.ZipFile(io.BytesIO(raw)).read(f"{SYM}-metrics-{d}.csv")))
        r = m.sort_values("create_time").iloc[-1]
        rows.append((d, float(r.sum_open_interest), float(r.sum_open_interest_value)))
    return pd.DataFrame(rows, columns=["day", "oi", "oi_usd"])


def finalize() -> None:
    parts = sorted((WORK / "parts").glob("*_days.parquet"))
    days = pd.concat([pd.read_parquet(p) for p in parts]).sort_values("day").drop_duplicates("day")
    cells = pd.concat([pd.read_parquet(str(p).replace("_days", "_cells")) for p in parts])
    cells = cells.groupby(["day", "bin"], as_index=False)[["buy", "sell"]].sum().sort_values(["day", "bin"])
    days = days.merge(oi_daily(days.day), on="day", how="left")
    days["src"] = "aggTrades"
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(OUT / f"{SYM}_cells.parquet", index=False)
    days.to_parquet(OUT / f"{SYM}_days.parquet", index=False)
    print("완료 일", len(days), days.day.min(), "~", days.day.max(), "칸", len(cells),
          "OI 있는 날", int(days.oi.notna().sum()), flush=True)
    check(days, cells)


def check(days: pd.DataFrame, cells: pd.DataFrame) -> None:
    """칸 합 == 일 합 · 거래량 == 로컬 1분봉 합(겹치는 날)."""
    cs = cells.groupby("day")[["buy", "sell"]].sum()
    d = days.set_index("day")
    assert ((cs.buy - d.buy_qty).abs() / d.buy_qty).max() < 1e-9 and ((cs.sell - d.sell_qty).abs() / d.sell_qty).max() < 1e-9
    k = pd.concat([pd.read_parquet(f, columns=["t", "v", "tb"])
                   for f in sorted((MAIN / "data/binance_vision/klines1m").glob(f"{SYM}-1m-*.parquet"))])
    k["day"] = pd.to_datetime(k.t, unit="ms").dt.date
    j = k.groupby("day")[["v", "tb"]].sum().join(d[["buy_qty", "sell_qty"]], how="inner")
    rel = (j.buy_qty + j.sell_qty - j.v).abs() / j.v
    relb = (j.buy_qty - j.tb).abs() / j.tb
    print(f"1분봉 대조 {len(j)}일 · 거래량 상대오차 최대 {rel.max():.2e} 중앙 {rel.median():.2e} · "
          f"테이커 매수 최대 {relb.max():.2e}", flush=True)


def selftest() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    f = WORK / "selftest.csv"
    t0 = 1_700_000_000_000                     # 2023-11-14 22:13 UTC
    f.write_text("\n".join([
        "1,100.4,2,1,1,%d,false" % t0,             # 매수 2 @ bin 100
        "2,100.9,1,2,2,%d,true" % (t0 + 1000),     # 매도 1 @ bin 100
        "3,101.2,3,3,3,%d,false" % (t0 + 2000),    # 매수 3 @ bin 101 (마지막 = 종가)
        "4,99.5,1,4,4,%d,true" % (t0 + 7_200_000),  # 다음 날 00:13 UTC → 다른 날
        "2,100.9,1,2,2,%d,true" % (t0 + 1000),     # 아카이브 중복 행 → 한 번만 센다
    ]) + "\n")
    cells, days = aggregate(f)
    c = cells.set_index(["day", "bin"])
    d0 = date(2023, 11, 14)
    assert c.loc[(d0, 100)].buy == 2 and c.loc[(d0, 100)].sell == 1 and c.loc[(d0, 101)].buy == 3
    r = days.set_index("day").loc[d0]
    assert r.open == 100.4 and r.close == 101.2 and r.high == 101.2 and r.low == 100.4 and r.n == 3
    assert abs(r.buy_quote - (100.4 * 2 + 101.2 * 3)) < 1e-9 and abs(r.sell_quote - 100.9) < 1e-9
    assert len(days) == 2
    f.write_text("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
                 "1,100.4,2,1,1,%d,false\n" % t0)               # 헤더 있는 파일
    assert len(aggregate(f)[1]) == 1
    f.unlink()
    print("selftest OK -- 공격자 방향·$1 칸·OHLC·거래대금·UTC 날짜·헤더 유무")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        if "--redo" in sys.argv:
            for m in sys.argv[sys.argv.index("--redo") + 1].split(","):
                for f in (WORK / "parts").glob(f"{m}*.parquet"):
                    f.unlink()
        build_parts()
        finalize()
