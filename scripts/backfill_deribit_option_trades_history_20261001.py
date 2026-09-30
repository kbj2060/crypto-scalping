"""Deribit 옵션 체결 이력 백필 → 체결 기반 딜러 포지션 커버 100% (2026-10-01, 사용자 «과거 체결 기록을 한 번 받아 와서 커버율 100%로»).

왜: 옵션 카드의 딜러 감마·DEX·플립·charm 은 «종목 첫 체결부터 누적한 −(테이커 순매수)»다. 수집기는 09-27 에 시작해
  그 전에 상장된 종목은 처음 재고를 몰라 뺐다(09-30 커버: 가까운 만기 3.6% · 7일 11.9% · 전 만기 1.1%).
  history.deribit.com 은 현 상장 종목의 체결을 첫 건부터 끊김 없이 준다(09-30 연구: 633종목 전부 trade_seq 1 부터, max = 건수).
무엇: 현 상장 옵션 종목 중 우리 DB 의 체결 이력이 **완결되지 않은** 종목(trade_seq 가 1 부터 끊김 없이 이어지지 않음)만
  history 에서 상장 시각부터 지금까지 받아 option_trades 에 넣는다(INSERT OR IGNORE -- trade_id PK 가 겹침을 버린다).
  행 모양은 수집기 row() 그대로. 받는 단계(수 분)와 쓰는 단계(한 트랜잭션, 수 초)를 나눠 상주 수집기의 쓰기를 오래 막지 않는다.
실행(서버, 수집기가 도는 채로):
  python scripts/backfill_deribit_option_trades_history_20261001.py            # 받고 쓰고 커버 보고
  python scripts/backfill_deribit_option_trades_history_20261001.py --dry-run  # 받기만(parquet)
  python scripts/backfill_deribit_option_trades_history_20261001.py --selftest
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_deribit_block_trade_collector_20260928 as C  # noqa: E402  row · COLS · coin_of · API_CURRENCIES · DB

HIST = "https://history.deribit.com/api/v2/public/"
API = "https://www.deribit.com/api/v2/public/"
PAUSE = 0.15            # 초당 ~6회 -- 공개 한도 안에서 넉넉히


def get(base: str, method: str, **params) -> dict:
    for i in range(6):
        try:
            r = requests.get(base + method, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** i); continue
            r.raise_for_status()
            return r.json()["result"]
        except requests.RequestException:
            if i == 5:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def complete(seqs: pd.Series) -> bool:
    """체결 이력이 첫 건부터 끊김 없이 있나: trade_seq 가 1 부터 시작하고 고유값 개수 = 최댓값."""
    s = pd.Series(seqs).dropna().astype("int64")
    return len(s) > 0 and int(s.min()) == 1 and int(s.nunique()) == int(s.max())


def incomplete_instruments(con, listed: dict[str, int]) -> list[str]:
    """현 상장 종목 중 DB 이력이 완결되지 않은 것. 체결이 하나도 없는 종목도 넣는다(history 에도 없으면 0건으로 끝)."""
    have = con.execute("SELECT instrument_name, list(trade_seq) FROM option_trades GROUP BY 1").fetchall()
    ok = {n for n, seqs in have if complete(pd.Series(seqs))}
    return sorted(n for n in listed if n not in ok)


def fetch_instrument(name: str, start_ms: int, end_ms: int) -> list[tuple]:
    out, t = [], start_ms
    while True:
        res = get(HIST, "get_last_trades_by_instrument_and_time", instrument_name=name, start_timestamp=t,
                  end_timestamp=end_ms, count=1000, sorting="asc", include_old="true")
        trades = res.get("trades") or []
        out += [C.row(x) for x in trades]
        nxt = max((x["timestamp"] for x in trades), default=t)
        time.sleep(PAUSE)
        # ponytail: 같은 ms 에 1,000건 넘게 몰리면 그 ms 의 나머지를 놓친다 -- 종목 하나에서 그런 밀도는 없다(PK 가 겹침만 버림)
        if not res.get("has_more") or nxt <= t:
            return out
        t = nxt


def coverage(con) -> pd.DataFrame:
    """최신 체인 스냅샷의 미결제 가중 커버(완결 종목 비중)를 코인·범위별로."""
    snap = con.execute("""SELECT currency, instrument_name, open_interest, days_to_expiry FROM option_chain_snapshot
                          WHERE recorded_at_utc = (SELECT max(recorded_at_utc) FROM option_chain_snapshot s2
                                                   WHERE s2.currency = option_chain_snapshot.currency)""").df()
    have = con.execute("SELECT instrument_name, list(trade_seq) AS seqs FROM option_trades GROUP BY 1").df()
    ok = set(have.loc[have["seqs"].map(lambda s: complete(pd.Series(s))), "instrument_name"])
    snap["ok"] = snap["instrument_name"].isin(ok)
    rows = []
    for cur, g in snap.groupby("currency"):
        front = g[g["days_to_expiry"] == g["days_to_expiry"].min()]
        for nm, x in (("가까운 만기", front), ("7일 안", g[g["days_to_expiry"] <= 7]), ("전 만기", g)):
            oi = x["open_interest"].sum()
            rows.append({"coin": cur, "범위": nm, "커버": round(float(x.loc[x["ok"], "open_interest"].sum() / oi), 4) if oi else None,
                         "종목": len(x), "완결": int(x["ok"].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("tmp/deribit_option_backfill_20261001"))
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    import duckdb
    listed: dict[str, int] = {}
    for api in C.API_CURRENCIES:
        for x in get(API, "get_instruments", currency=api, kind="option", expired="false"):
            if C.coin_of(x["instrument_name"]):
                listed[x["instrument_name"]] = int(x["creation_timestamp"])
    con = duckdb.connect(str(C.DB), read_only=True)
    try:
        todo = incomplete_instruments(con, listed)
    finally:
        con.close()
    print(f"현 상장 {len(listed):,}종목 · 이력 미완결 {len(todo):,}종목 → history 에서 받는다", flush=True)
    now = int(time.time() * 1000)
    rows: list[tuple] = []
    for i, n in enumerate(todo, 1):
        rows += fetch_instrument(n, listed[n] - 60_000, now)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} · 누적 {len(rows):,}건", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=list(C.COLS))
    df.to_parquet(a.out / "hist_option_trades.parquet")
    print(f"받음 {len(df):,}건 · {df['instrument_name'].nunique() if len(df) else 0}종목 → {a.out}", flush=True)
    if a.dry_run or df.empty:
        return
    for i in range(40):                          # 상주 수집기가 20초마다 잠깐 쓴다 -- 비는 틈에 한 번에
        try:
            con = duckdb.connect(str(C.DB))
            break
        except duckdb.IOException:
            time.sleep(0.5)
    else:
        raise SystemExit("DB 잠금이 안 풀린다 -- parquet 는 남았으니 다시 실행하면 쓴다")
    try:
        before = con.execute("SELECT count(*) FROM option_trades").fetchone()[0]
        con.begin()
        con.register("h", df)
        con.execute(f"INSERT OR IGNORE INTO option_trades SELECT {','.join(C.COLS)} FROM h")
        con.commit()
        added = con.execute("SELECT count(*) FROM option_trades").fetchone()[0] - before
        print(f"DB 에 새 {added:,}건", flush=True)
        print(coverage(con).to_string(index=False), flush=True)
    finally:
        con.close()


def selftest() -> None:
    assert complete(pd.Series([1, 2, 3])) and complete(pd.Series([3, 1, 2, 2]))       # 순서·중복 무관
    assert not complete(pd.Series([2, 3])), "첫 건이 없다(수집 시작 뒤부터만 봄)"
    assert not complete(pd.Series([1, 2, 4])), "중간이 빠졌다"
    assert not complete(pd.Series([], dtype="int64")) and not complete(pd.Series([None, None]))
    t = {"timestamp": 5, "trade_id": "ETH-9", "trade_seq": 1, "instrument_name": "ETH-9OCT26-2550-P", "direction": "buy",
         "amount": 1.0, "price": 0.01, "mark_price": 0.01, "index_price": 2600.0, "iv": 50.0, "tick_direction": 0}
    assert len(C.row(t)) == len(C.COLS), "행 모양이 수집기와 같다"
    print("selftest ok")


if __name__ == "__main__":
    main()
