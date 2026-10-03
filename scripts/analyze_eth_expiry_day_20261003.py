"""ETH 옵션 만기 하루 분석(2026-10-03, 사용자 «오늘 만기도 기다렸다가 분석») -- 만기 날짜 하나를 받아 같은 표를 낸다.

내용: max pain 추이(24·12·6·3·2·1h 전) · 정산가(Deribit delivery) · max pain 1시간 규칙(07:00 UTC 스냅샷 → 07:59 종가,
  사전 규칙 options_2026only_revalidation_20260929) · 보유자 지급액(정산가 vs max pain) · 꼬리확률·DVOL 1σ 대 실제 ·
  실현 감마(12h 5분 acf1, 07:00 UTC) 대 만기 뒤 4h 실현 · 만기 종목 마지막 체결.
가격 = Deribit ETH-PERPETUAL 1분봉(로컬 바이낸스 REST 금지 · vision 은 다음 날 게시). 서버 DB 는 read_only 로 COPY.
  python scripts/analyze_eth_expiry_day_20261003.py 2026-10-03
  python scripts/analyze_eth_expiry_day_20261003.py --selftest
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/llewyn/miniconda3/envs/quant_ai/bin/python"


def max_pain(g: pd.DataFrame) -> float:
    ks = np.sort(g.strike.unique()); c = g[g.option_type == "call"]; q = g[g.option_type == "put"]
    return float(ks[int(np.argmin([(c.open_interest * np.maximum(K - c.strike, 0)).sum()
                                   + (q.open_interest * np.maximum(q.strike - K, 0)).sum() for K in ks]))])


def payout(g: pd.DataFrame, px: float) -> float:
    """만기 보유자에게 줄 내재가치 합(기초자산 수량 × 가격차, USD)."""
    c = g[g.option_type == "call"]; q = g[g.option_type == "put"]
    return float((c.open_interest * np.maximum(px - c.strike, 0)).sum() + (q.open_interest * np.maximum(q.strike - px, 0)).sum())


def fetch(day: str, d: Path) -> None:
    exp = pd.Timestamp(day + " 08:00", tz="UTC")
    lo, hi = exp - pd.Timedelta(hours=25), exp + pd.Timedelta(hours=4)   # 만기 뒤 4h(돌리는 시각까지만 채워진다)
    rel_d = d.relative_to(ROOT)
    prefix = f"ETH-{exp.day}{exp:%b%y}".upper()            # 예: ETH-3OCT26
    code = f'''import duckdb, pathlib
o = pathlib.Path("{rel_d}"); o.mkdir(parents=True, exist_ok=True)
c = duckdb.connect("data/live/deribit_options.duckdb", read_only=True)
w = "currency='ETH' AND recorded_at_utc BETWEEN TIMESTAMPTZ '{lo}' AND TIMESTAMPTZ '{hi}'"
c.execute(f"COPY (SELECT * FROM option_chain_snapshot WHERE {{w}} AND expiration_ts = TIMESTAMPTZ '{exp}') TO '{{o}}/chain_exp.parquet' (FORMAT parquet)")
c.execute(f"COPY (SELECT recorded_at_utc, index_price, dvol, payload FROM option_summary WHERE {{w}}) TO '{{o}}/summary.parquet' (FORMAT parquet)")
c.execute(f"COPY (SELECT * FROM option_trades WHERE starts_with(instrument_name, '{prefix}-')) TO '{{o}}/trades_exp.parquet' (FORMAT parquet)")
print("export ok")'''
    f = ROOT / "tmp" / f"expiry_export_{day}.py"
    f.write_text(code, encoding="utf-8")
    h = ["bash", str(ROOT / "scripts/ops/handoff.sh")]
    rel = str(f.relative_to(ROOT)); job = f"expiry_{day.replace('-', '')}"
    subprocess.run(h + ["push", "server", rel], cwd=ROOT, capture_output=True, check=True)
    subprocess.run(h + ["launch", "server", job, "--", PY, rel], cwd=ROOT, capture_output=True)
    out = ""
    for _ in range(20):                                   # 30초 간격(handoff 분당 호출 상한 안)
        time.sleep(30)
        out = subprocess.run(h + ["logs", "server", job], cwd=ROOT, capture_output=True, text=True).stdout
        if "export ok" in out or "STOPPED" in out:
            break
    if "export ok" not in out:
        raise SystemExit("서버 추출 실패:\n" + out[-800:])
    subprocess.run(h + ["pull", "server", str(rel_d)], cwd=ROOT, capture_output=True, check=True)
    rows = []
    for a in pd.date_range(lo, hi, freq="12h"):
        b = min(a + pd.Timedelta(hours=12), hi)
        r = requests.get("https://www.deribit.com/api/v2/public/get_tradingview_chart_data", timeout=30, params=dict(
            instrument_name="ETH-PERPETUAL", start_timestamp=int(a.timestamp() * 1000), end_timestamp=int(b.timestamp() * 1000), resolution="1")).json()["result"]
        rows.append(pd.DataFrame({"o": r["open"], "h": r["high"], "l": r["low"], "c": r["close"]}, index=pd.to_datetime(r["ticks"], unit="ms", utc=True)))
    p = pd.concat(rows); p[~p.index.duplicated()].sort_index().to_parquet(d / "deribit_perp_1m.parquet")


def analyze(day: str, d: Path) -> None:
    exp = pd.Timestamp(day + " 08:00", tz="UTC")
    ch = pd.read_parquet(d / "chain_exp.parquet"); sm = pd.read_parquet(d / "summary.parquet")
    tr = pd.read_parquet(d / "trades_exp.parquet"); p = pd.read_parquet(d / "deribit_perp_1m.parquet")
    ch["t"] = pd.to_datetime(ch.recorded_at_utc, utc=True); sm["t"] = pd.to_datetime(sm.recorded_at_utc, utc=True)
    dp = requests.get("https://www.deribit.com/api/v2/public/get_delivery_prices", params={"index_name": "eth_usd", "count": 5}, timeout=30).json()["result"]["data"]
    st = {x["date"]: x["delivery_price"] for x in dp}.get(day)
    if st is None:
        raise SystemExit(f"정산가 아직 없음({day}) -- 08:00 UTC 뒤 몇 분 기다렸다 다시")
    ts = np.sort(ch.t.unique())
    snap = lambda T: ch[ch.t == ts[max(0, np.searchsorted(ts, T, side="right") - 1)]]   # noqa: E731  T 이전 마지막 스냅샷
    print(f"== {day} 17:00 KST 만기 · 정산가 {st}")
    for h in (24, 12, 6, 3, 2, 1):
        g = snap(exp - pd.Timedelta(hours=h)); t = g.t.iloc[0]; px = p.c.asof(t); M = max_pain(g); oi = g.open_interest.sum()
        co, po = g[g.option_type == "call"].open_interest.sum(), g[g.option_type == "put"].open_interest.sum()
        print(f"{h:>2}h 전 {t.tz_convert('Asia/Seoul'):%m-%d %H:%M} KST 가격 {px:.1f} · max pain {M:.0f} (가격 − pain {(px / M - 1) * 100:+.2f}%) · OI ${oi * px / 1e6:.0f}M · P/C {po / co:.2f}")
    g = ch[(ch.t >= exp - pd.Timedelta(hours=1)) & (ch.t < exp - pd.Timedelta(minutes=50))]
    g = g[g.t == g.t.min()]; t1 = g.t.iloc[0]; M = max_pain(g)
    a, b = t1.floor("min") + pd.Timedelta(minutes=1), exp - pd.Timedelta(minutes=1)
    P0, P1 = p.o.asof(a), p.c.asof(b); side = np.sign(M - P0); seg = p.loc[a:b]
    mae = ((seg.l.min() / P0 - 1) if side > 0 else -(seg.h.max() / P0 - 1)) * 1e4
    print(f"1시간 규칙: max pain {M:.0f} · 진입 {P0:.2f} {'롱' if side > 0 else '숏'} → 청산 {P1:.2f} = {side * (P1 / P0 - 1) * 1e4:+.1f}bp · 최대 역행 {mae:+.1f}bp")
    print(f"정산가 − max pain {(st / M - 1) * 100:+.2f}% (1시간 전 {(P0 / M - 1) * 100:+.2f}%) · ±1% 안 {'예' if abs(st / M - 1) <= 0.01 else '아니오'}")
    print(f"보유자 지급 ${payout(g, st) / 1e6:.2f}M (max pain 에서 끝났으면 ${payout(g, M) / 1e6:.2f}M)")
    for lab, m in (("-60", -60), ("-30", -30), ("0", 0), ("+60", 60), ("+120", 120)):
        print(f"  만기{lab}분 {p.c.asof(exp + pd.Timedelta(minutes=m)):.2f}")
    s0 = sm[sm.t <= t1].iloc[-1]
    print(f"DVOL {s0.dvol:.1f} → 1h 1σ ±{P0 * s0.dvol / 100 * np.sqrt(1 / 8760):.1f}$ · 실제 1h {P1 - P0:+.1f}$")
    for _, r in sm.iterrows():
        fr = (json.loads(r.payload).get("surface") or {}).get("front") or {}
        if fr.get("exp_ms") == int(exp.timestamp() * 1000) and fr.get("hours", 0) >= 12:
            P = p.c.asof(r.t)
            print(f"꼬리확률({fr['hours']:.0f}h 전) ±2% 밖 {fr['tail']['2']:.1%} · ±3% {fr['tail']['3']:.1%} · 실제 {(st / P - 1) * 100:+.2f}%")
            break
    c5 = p.c.resample("5min", label="right", closed="right").last().dropna()
    lp = np.log(c5[c5.index <= exp - pd.Timedelta(hours=1)].iloc[-145:].values)
    if len(lp) == 145:
        r = np.diff(lp); x = r - r.mean(); rg = (x[1:] * x[:-1]).sum() / (x * x).sum()
        side_rg = "되돌림" if rg <= -0.0769 else "추세" if rg >= 0.0124 else "보통"
        nx = np.diff(np.log(c5[(c5.index >= exp) & (c5.index <= exp + pd.Timedelta(hours=4))].values))
        print(f"실현 감마(16:00 KST) {rg:+.3f} {side_rg} 쪽 · 만기 뒤 실현 {np.sqrt((nx ** 2).sum()) * 100:.2f}%({len(nx) * 5}분) vs DVOL 4h 1σ {s0.dvol / 100 * np.sqrt(4 / 8760) * 100:.2f}%")
    tr["t"] = pd.to_datetime(tr.ts_ms, unit="ms", utc=True)
    for hh in (6, 1):
        x = tr[(tr.t >= exp - pd.Timedelta(hours=hh)) & (tr.t < exp)]
        print(f"마지막 {hh}h 만기 종목 체결 {len(x)}건 · {x.amount.sum():,.0f} ETH · 테이커 순매수 {(np.where(x.direction == 'buy', 1, -1) * x.amount).sum():+,.0f}")


def selftest() -> None:
    g = pd.DataFrame({"strike": [90, 100, 110, 90, 100, 110], "option_type": ["call"] * 3 + ["put"] * 3,
                      "open_interest": [0, 10, 0, 0, 10, 0]})
    assert max_pain(g) == 100 and payout(g, 100) == 0 and payout(g, 110) == 100
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        day = sys.argv[1]; d = ROOT / "tmp" / f"expiry_{day}"
        if not (d / "deribit_perp_1m.parquet").exists() or "--refetch" in sys.argv:
            fetch(day, d)
        analyze(day, d)
