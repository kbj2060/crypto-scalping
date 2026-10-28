#!/usr/bin/env python3
"""매크로 국면·4주 편향 2라운드 **전진 기록기** -- 사전등록 docs/prereg/macro_regime_bias_r2_20261028.md (2026-10-28).

매주 결정 시각 τ(직전 금요일, 가격 = 바이낸스 현물 1일봉 종가 = 토 00:00 UTC)의 Z 를 **그 시각에 받은 값으로** 계산해
data/live/macro_bias_forward.jsonl 에 한 줄씩 남긴다(기록 시각 = 알 수 있던 값 · 사후에 메우지 않는다).
연구와 같은 함수(scripts/research_macro_regime_bias_20261028.py 의 sample_week·feats_for·z_combo·z91)를 부른다.
원천: FRED 정식 API(.env FRED_API_KEY) · KOSPI = Yahoo ^KS11 · S&P = FRED SP500 · 코인 = 바이낸스 현물 1d. 주문 호출 없음.
  python scripts/live_macro_bias_forward_20261028.py               # cron(서버, 토·일 10:30 KST -- 일요일은 토요일이 실패했을 때만 쓴다)
  python scripts/live_macro_bias_forward_20261028.py --parity CSV  # 2026 주마다 Z 이력(패널 Z 와 대조용, 기록 안 함)
  python scripts/live_macro_bias_forward_20261028.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.research_macro_regime_bias_20261028 as rb  # noqa: E402

LEDGER = ROOT / "data/live/macro_bias_forward.jsonl"
START = "2022-06-01"      # 104주 sd(최소 52) + 13주 변화 + 1주 미룸 + Z91 730일 sd(최소 365) 를 채우는 길이
UA = {"User-Agent": "Mozilla/5.0"}
FROZEN_SHA = "e4b0cb86a2d345f9c2c85ecd59b3bd4aec998de9c1d26f5e74e6cbf3622666ff"   # 1라운드 동결 사양(7b7dcfbd) -- 다르면 기록하지 않는다
LATE_DAYS = 2           # 결정 금요일 일봉이 닫힌 뒤 이보다 늦으면 기록 거부(늦은 빈티지로 과거 τ 를 메우지 않게)


def binance_close(sym: str) -> pd.Series:
    """현물 1일봉 종가, 라벨 = 봉 시작 날짜(종가 = 다음 날 00:00 UTC = 연구의 Yahoo Open[D+1]). 닫힌 봉만."""
    out, t0 = [], int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    while True:
        k = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": sym, "interval": "1d", "startTime": t0, "limit": 1000}, timeout=30).json()
        k = [x for x in k if x[6] < time.time() * 1000]
        out += k
        if len(k) < 1000:
            break
        t0 = k[-1][0] + 86_400_000
        time.sleep(0.3)
    return pd.Series([float(x[4]) for x in out], index=pd.to_datetime([x[0] for x in out], unit="ms"))


def yahoo_close(ticker: str) -> pd.Series:
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", params={"range": "5y", "interval": "1d"}, headers=UA, timeout=30)
    res = r.json()["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=pd.to_datetime(res["timestamp"], unit="s") + pd.Timedelta(hours=9))   # 한국 장 날짜
    s.index = s.index.normalize()
    return s.dropna()


def fetch_panel() -> pd.DataFrame:
    """연구 패널과 같은 열 이름의 일별 프레임(필요한 열만) -- 순유동성은 패널 생성 스크립트와 같은 식."""
    from dotenv import load_dotenv
    from dashboard.macro_trend import fetch_fred
    load_dotenv(ROOT / ".env")
    fz = json.loads(rb.FROZEN.read_text())
    v = rb._verdict()
    t = {**v.SERIES, **v.EXTRA}
    names = {n for a in ("ETH", "BTC") for m in ("B4", "A") for n in rb.selected(fz, a, m)}
    cols = {t[n][0] for n in names} - {"SPX", "KOSPI", "NETLIQ"} | {"DGS10", "DTWEXBGS", "WALCL", "WTREGEN", "RRPONTSYD"}
    d = {sid: fetch_fred(sid, START, timeout=60) for sid in sorted(cols)}
    d["SPX"] = fetch_fred("SP500", START, timeout=60)
    d["KOSPI"] = yahoo_close("%5EKS11")
    d["BTC"], d["ETH"] = binance_close("BTCUSDT"), binance_close("ETHUSDT")
    p = pd.concat(d, axis=1).sort_index()
    w = p[["WALCL", "WTREGEN"]].dropna()   # research_macro_crypto_panel_20261027.py 와 같은 식(수요일 격자, 백만$)
    p["NETLIQ"] = w["WALCL"] - w["WTREGEN"] - p["RRPONTSYD"].ffill(limit=5).reindex(w.index).fillna(0.0) * 1000.0
    return p


def z_rows(p: pd.DataFrame, end=None) -> pd.DataFrame:
    fz = json.loads(rb.FROZEN.read_text())
    w = rb.sample_week(p, end)
    zz = rb.z91(p, w.index)
    rows = []
    for a in ("ETH", "BTC"):
        zb, nb = rb.z_combo(rb.feats_for(w, rb.selected(fz, a, "B4")), rb.selected(fz, a, "B4"))
        za, na = rb.z_combo(rb.feats_for(w, rb.selected(fz, a, "A")), rb.selected(fz, a, "A"))
        rows.append(pd.DataFrame({"asset": a, "z_b4": zb, "n_b4": nb, "z_a": za, "n_a": na, "z91": zz, "price_tau": w[a]}))
    return pd.concat(rows)


def too_late(now: dt.datetime, tau: pd.Timestamp) -> bool:
    return now - (tau.tz_localize("UTC") + pd.Timedelta(days=1)).to_pydatetime() > dt.timedelta(days=LATE_DAYS)


def decision_tau(now: dt.datetime) -> pd.Timestamp:
    """직전 금요일 -- 그 금요일 일봉이 닫힌(토 00:00 UTC 지난) 것."""
    d = (now - dt.timedelta(days=1)).date()
    return pd.Timestamp(d - dt.timedelta(days=(d.weekday() - 4) % 7))


def append(rows: list[dict], ledger: Path = LEDGER) -> int:
    """(tau, asset) 가 이미 있으면 안 쓴다(덮어쓰기·사후 메움 금지)."""
    have = set()
    if ledger.exists():
        have = {(r["tau"], r["asset"]) for r in map(json.loads, ledger.read_text().splitlines()) if r}
    new = [r for r in rows if (r["tau"], r["asset"]) not in have]
    if new:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(new)


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)
    tau = decision_tau(now)
    sha = hashlib.sha256(rb.FROZEN.read_bytes()).hexdigest()
    if sha != FROZEN_SHA:
        print(f"{now:%F %T} 동결 사양 해시 불일치 {sha[:12]} ≠ {FROZEN_SHA[:12]} -- 기록 안 함", file=sys.stderr)
        return 2
    if too_late(now, tau):
        print(f"{now:%F %T} τ {tau.date()} 는 {LATE_DAYS}일 넘게 지났다 -- 사후 메움 금지, 기록 안 함", file=sys.stderr)
        return 1
    p = fetch_panel()
    z = z_rows(p, tau)
    fz = json.loads(rb.FROZEN.read_text())
    if tau not in z.index:
        print(f"{now:%F %T} τ {tau.date()} 행 없음 -- 기록 안 함", file=sys.stderr)
        return 1
    zt = z.loc[[tau]]
    short = [(a, m, int(zt[zt.asset == a][f"n_{m.lower()}"].iloc[0]), len(rb.selected(fz, a, m)))
             for a in ("ETH", "BTC") for m in ("B4", "A")]
    short = [x for x in short if x[2] != x[3]]
    if short:   # 지표가 빠진 채 평균하면 구성이 바뀐다 -- 기록하지 않고 알린다(10-28 리뷰)
        print(f"{now:%F %T} τ {tau.date()} 지표 빠짐 {short} -- 기록 안 함", file=sys.stderr)
        return 2
    last = {c: str(p[c].last_valid_index().date()) for c in p.columns if p[c].notna().any()}
    rows = []
    f = lambda x: None if pd.isna(x) else float(x)   # noqa: E731
    for a, r in z.loc[[tau]].groupby("asset"):
        r = r.iloc[0]
        rows.append({"tau": str(tau.date()), "recorded_utc": now.isoformat(timespec="seconds"), "asset": a, "z_b4": f(r.z_b4), "n_b4": int(r.n_b4),
                     "z_a": f(r.z_a), "n_a": int(r.n_a), "z91": f(r.z91), "price_tau": f(r.price_tau), "last_obs": last, "frozen_sha256": sha})
    n = append(rows)
    print(f"{now:%F %T} τ {tau.date()} 새 기록 {n}줄 (n_b4 {[r['n_b4'] for r in rows]} · n_a {[r['n_a'] for r in rows]})")   # 값은 판정 전 안 찍는다
    return 0


def selftest() -> None:
    import tempfile
    assert decision_tau(dt.datetime(2026, 10, 31, 1, 30, tzinfo=dt.timezone.utc)) == pd.Timestamp("2026-10-30")   # 토 → 그 전날 금
    assert decision_tau(dt.datetime(2026, 11, 1, 1, 30, tzinfo=dt.timezone.utc)) == pd.Timestamp("2026-10-30")    # 일 재시도 → 같은 금
    assert decision_tau(dt.datetime(2026, 10, 30, 23, 0, tzinfo=dt.timezone.utc)) == pd.Timestamp("2026-10-23")   # 금 종가 전 → 지난주
    sat, sun, wed = (dt.datetime(2026, m, d, 1, 30, tzinfo=dt.timezone.utc) for m, d in ((10, 31), (11, 1), (11, 4)))
    tau = pd.Timestamp("2026-10-30")
    assert not too_late(sat, tau) and not too_late(sun, tau) and too_late(wed, tau)        # 토·일 재시도 허용 · 수요일은 거부
    # DGS10 금요일 값이 없어도 τ 행이 생긴다(표본기 절단 우회) -- 합성 패널
    ix = pd.date_range("2023-01-01", "2026-10-30", freq="D")
    fake = pd.DataFrame({"DGS10": 4.0, "BTC": 1.0, "ETH": 1.0}, index=ix)
    fake.loc["2026-10-30", "DGS10"] = float("nan")
    assert tau not in rb.sample_week(fake).index and tau in rb.sample_week(fake, tau).index
    with tempfile.TemporaryDirectory() as t:
        L = Path(t) / "l.jsonl"
        r = {"tau": "2026-10-30", "asset": "ETH", "z_b4": 0.1}
        assert append([r], L) == 1 and append([{**r, "z_b4": 9.9}], L) == 0                # 같은 (τ, 자산)은 덮지 않는다
        assert json.loads(L.read_text().splitlines()[0])["z_b4"] == 0.1
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--parity", help="2026 주별 Z 이력을 이 CSV 로(기록 안 함)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.parity:
        z = z_rows(fetch_panel())
        z[z.index >= "2026-01-01"].to_csv(a.parity, float_format="%.5f")   # 두 자산을 이어 붙여 인덱스가 정렬돼 있지 않다 -- 구간 자르기 대신 마스크
        print("parity csv", a.parity)
    else:
        sys.exit(main())
