"""사전등록 전진 검정: ETH 1~4주 추세 + 변동성 사이징 trend4_vs (동결 2026-10-07 · 판정 = 2026-10-08 00:00 UTC 이후 결정한 포지션).

설계 원문: docs/experiments/eth_trend4_fwd_prereg_20261007.md (이 스크립트와 같은 커밋 = 동결).
정의·비용·펀딩·샤프는 scripts/research_trend4_poi1h_solxrp_20261004.py 를 그대로 불러 쓴다(엔진 rl_1s_agent.py 와의 일별 포지션 일치 assert 는 거기 있다).
  --status  경과 일수·판정 창 거래일 수만(손익 안 봄)        --look N  N번째 점검일(365·730일) 이후에만 판정
  --parity  양성 대조(선물 22~26 샤프 0.62±0.1) + 전진 경로 = 연구 경로(겹치는 과거)        --selftest
🔴fapi/api.binance.com REST 는 쓰지 않는다 — data.binance.vision 정적 아카이브만.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_trend4_poi1h_solxrp_20261004 as R  # noqa: E402

LOOKS = R.ROOT / "docs/experiments/eth_trend4_fwd_looks.json"
DOC = R.ROOT / "docs/experiments/eth_trend4_fwd_prereg_20261007.md"
SYM, WARM, FREEZE = "ETHUSDT", "2022-01", pd.Timestamp("2026-10-08")   # WARM = 연구 패널과 같은 시작(k 경로까지 같게)
LOOK_DAYS, REJECT1 = (365, 730), -1.0
ARMS = ("trend4_vs", "trend4", "always_long", "vol_long")
DAY = pd.Timedelta(days=1)


def closes(today: pd.Timestamp) -> pd.Series:
    """선물 완결 일봉 종가(날짜 = 봉 시작일). 끝난 달은 월별, 아직 안 올라온 달은 일별 파일. 첫 빈 날에서 자른다."""
    parts = []
    for m in R.months(WARM, f"{today:%Y-%m}"):
        f = R.fetch(f"futures/um/monthly/klines/{SYM}/1d/{SYM}-1d-{m}.zip")
        days = pd.date_range(f"{m}-01", min(pd.Period(m).end_time.normalize(), today - DAY))
        fs = [f] if f else [R.fetch(f"futures/um/daily/klines/{SYM}/1d/{SYM}-1d-{d:%Y-%m-%d}.zip") for d in days]
        parts += [pd.read_csv(x) for x in fs if x]
    k = pd.concat(parts)
    c = pd.Series(k.close.astype(float).to_numpy(), index=pd.to_datetime(k.open_time.astype("int64"), unit="ms")).sort_index()
    c = c[~c.index.duplicated(keep="last")]
    c = c.reindex(pd.date_range(c.index[0], c.index[-1]))
    return c.iloc[:int(c.isna().to_numpy().argmax())] if c.isna().any() else c


def funding(days: pd.DatetimeIndex) -> tuple[pd.Series, pd.Timestamp]:
    """R.funding_daily 와 같은 집계(그날 00:00 초과~다음 00:00 이하 정산 합)를 전진 달까지. 반환 둘째 = 펀딩이 다 있는 마지막 날."""
    fs = []
    for m in R.months(f"{days[0]:%Y-%m}", f"{days[-1]:%Y-%m}"):
        f = R.fetch(f"futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{m}.zip")
        if f is None:                                    # 월별만 있다(일별 fundingRate 없음) -- 안 올라온 달부터 끊는다
            break
        fs.append(f)
    f = pd.concat([pd.read_csv(x) for x in fs])
    t = pd.to_datetime(f.calc_time, unit="ms") - pd.Timedelta("1s")
    s = f.last_funding_rate.groupby(t.dt.floor("D").to_numpy()).sum().reindex(days).fillna(0.0)
    return s, t.max().floor("D") - DAY                   # 마지막 정산일의 전날까지만 «다음 00:00 정산»이 확실히 들어 있다


def decide(li: int, s_vs: float, s_long: float) -> str:
    if li == 0:
        return "기각" if s_vs < REJECT1 else "계속"
    return "통과" if s_vs > 0 and s_vs > s_long else "기각"


def gate(li: int, today: pd.Timestamp, n_close: int | None = None, n_fund: int | None = None) -> str | None:
    need = LOOK_DAYS[li]
    if today < FREEZE + need * DAY:
        return f"점검 {li + 1} 은 {FREEZE + need * DAY:%Y-%m-%d} 이후에만 — 지금 {today:%Y-%m-%d}. 결과를 보지 않는다."
    if n_close is not None and n_close < need:
        return f"점검 {li + 1}: 일봉 아카이브가 {n_close}/{need}일뿐. 결과를 보지 않는다."
    if n_fund is not None and n_fund < need:
        return f"점검 {li + 1}: 펀딩 월별 아카이브가 {n_fund}/{need}일뿐(달이 끝나고 며칠 뒤 올라온다). 결과를 보지 않는다."
    return None


def evaluate(c: pd.Series, fund: pd.Series, li: int) -> dict:
    """판정 창 = 연구 trend_block 의 «주 구간» 자리를 전진 시작으로 옮긴 것. 포지션·비용은 2022 부터 이어서 계산(창 첫날 비용 = 실제 노출 변화)."""
    end = FREEZE + (LOOK_DAYS[li] - 1) * DAY
    R.MAIN = FREEZE
    t = R.trend_block(c[c.index <= end], fund)
    s = {a: t[a]["main"] for a in ARMS}
    res = decide(li, s["trend4_vs"]["net"]["sharpe"], s["always_long"]["net"]["sharpe"])
    return {"look": li + 1, "window": [str(FREEZE.date()), str(end.date())], "result": res, **s}


def today_utc() -> pd.Timestamp:
    return pd.Timestamp.now("UTC").tz_localize(None).normalize()


def main() -> None:
    today = today_utc()
    if "--status" in sys.argv:
        c = closes(today)
        print(f"경과 {max(0, (today - FREEZE).days)}일 · 판정 창 거래일 {int((c.index >= FREEZE).sum())}일")
        return
    li = int(sys.argv[sys.argv.index("--look") + 1]) - 1
    looks = json.loads(LOOKS.read_text()) if LOOKS.exists() else []
    if any(r["result"] in ("통과", "기각") for r in looks):
        print("판정 완료:", json.dumps(looks[-1], ensure_ascii=False)); return
    if li != len(looks):
        print(f"다음 점검은 {len(looks) + 1}번이다."); return
    msg = gate(li, today)
    if msg:
        print(msg); return
    c = closes(today)
    fund, cov = funding(c.index)
    msg = gate(li, today, int((c.index >= FREEZE).sum()), int(((c.index >= FREEZE) & (c.index <= cov)).sum()))
    if msg:
        print(msg); return
    r = evaluate(c[c.index <= cov], fund, li)
    looks.append(r)
    LOOKS.write_text(json.dumps(looks, ensure_ascii=False, indent=1))
    n = {a: r[a]["net"]["sharpe"] for a in ARMS}
    DOC.write_text(DOC.read_text().replace("## 점검 기록\n(없음)\n", "## 점검 기록\n")
                   + f"- 점검 {r['look']} ({today:%Y-%m-%d}, 창 {r['window'][0]}~{r['window'][1]}): **{r['result']}** · 순 샤프 {n}\n")
    print(json.dumps(r, ensure_ascii=False, indent=1))


def parity() -> None:
    d = R.load_panel(SYM)                                     # 연구 경로: 5분 패널 23:55 봉 = 일 종가
    c = d.close[(d.index + pd.Timedelta("5min")).normalize() == d.index + pd.Timedelta("5min")]
    c.index = c.index.normalize()
    sp = R.eth_spot_closes()
    cw = pd.concat([sp[(sp.index >= "2021-11-01") & (sp.index < c.index[0])], c])
    pc = R.research_vs(cw, R.funding_daily(SYM, cw.index), "2022-01-01")["long_short"]["sharpe"]
    fr = R.funding_daily(SYM, c.index)
    t = R.trend_block(c, fr)
    print(f"양성 대조: 선물 2022-01~{c.index[-1]:%Y-%m-%d} 원 연구 변동성 롱숏 샤프 {pc} (기준 0.62±0.1) · "
          f"엔진 정의 전 기간 순 샤프 trend4_vs {t['trend4_vs']['all']['net']['sharpe']} · trend4 {t['trend4']['all']['net']['sharpe']} · "
          f"항상 롱 {t['always_long']['all']['net']['sharpe']}")
    assert abs(pc - 0.62) <= 0.1
    cf = closes(today_utc())                                  # 전진 경로: 선물 1d 아카이브
    ff, cov = funding(cf.index)
    ov = c.index[c.index.isin(cf.index)]
    P, Pf = R.trend_positions(c), R.trend_positions(cf)
    dc = float(np.abs(cf[ov] / c[ov] - 1).max())
    dp = float(np.abs(Pf.loc[ov, ["trend4", "trend4_vs"]].to_numpy() - P.loc[ov, ["trend4", "trend4_vs"]].to_numpy()).max())
    df = float(np.abs(ff[ov] - fr[ov]).max())
    print(f"전진=연구 겹침 {ov[0]:%Y-%m-%d}~{ov[-1]:%Y-%m-%d} {len(ov)}일: 종가 최대 상대차 {dc:.2e} · 포지션 최대차 {dp:.2e} · 일 펀딩 최대차 {df:.2e} · "
          f"전진 일봉 끝 {cf.index[-1]:%Y-%m-%d} · 펀딩 확정 끝 {cov:%Y-%m-%d}")
    w = ov[(ov >= "2026-06-01") & (ov <= "2026-09-14")]
    for a in ("trend4_vs", "trend4"):
        x = R.book(P[a].astype(float), c.pct_change(), fr)[w]
        y = R.book(Pf[a].astype(float), cf.pct_change(), ff)[w]
        print(f"  {a} 2026-06-01~09-14 ({len(w)}일): 연구 순 샤프 {R.daily_stats(x)['sharpe']} · 전진 {R.daily_stats(y)['sharpe']} · 일 손익 최대차 {float((x - y).abs().max()):.2e}")
        assert float((x - y).abs().max()) < 1e-6
    assert dp == 0 and df < 1e-12 and dc < 1e-9


def selftest() -> None:
    assert decide(0, -1.01, 0) == "기각" and decide(0, -0.99, 5) == "계속"
    assert decide(1, 0.3, 0.2) == "통과" and decide(1, 0.3, 0.4) == "기각" and decide(1, -0.1, -0.5) == "기각"
    assert gate(0, pd.Timestamp("2027-10-07")) and gate(0, pd.Timestamp("2027-10-08")) is None
    assert gate(1, pd.Timestamp("2028-10-06")) and gate(1, pd.Timestamp("2028-10-07")) is None
    assert gate(0, pd.Timestamp("2027-11-05"), 365, 364) and gate(0, pd.Timestamp("2027-11-05"), 365, 365) is None
    days = pd.date_range("2026-07-01", "2027-11-30")                     # 합성 랜덤워크: 창이 정확히 365일이고 샤프 = 손으로 잰 값
    c = pd.Series(2000 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.03, len(days)))), index=days)
    f = pd.Series(0.0001, index=days)
    r = evaluate(c, f, 0)
    pos, m = R.trend_positions(c).trend4_vs, (days >= FREEZE) & (days <= pd.Timestamp("2027-10-07"))
    pnl = R.book(pos[pos.index >= pos.index[29]], c.pct_change(), f)[m]
    assert r["trend4_vs"]["net"]["n_days"] == 365 and r["window"] == ["2026-10-08", "2027-10-07"]
    assert abs(r["trend4_vs"]["net"]["sharpe"] - round(pnl.mean() / pnl.std() * np.sqrt(365), 3)) < 1e-9
    z = np.random.default_rng(0).standard_normal((2, 400_000))           # 검정력: 연 샤프 SE ≈ 1/√년(정규 근사)
    for mu in (0.62, 0.413, 0.0):                                        # 0.413 = 엔진 정의 trend4_vs 22~26 순 샤프(--parity)
        s1, s2 = mu + z[0], mu + (z[0] + z[1]) / 2
        print(f"참 샤프 {mu}: P(1년 추정>0) {np.mean(s1 > 0):.3f} · P(2년>0) {np.mean(s2 > 0):.3f} · "
              f"점검1 기각 {np.mean(s1 < REJECT1):.3f} · 최종 통과(조건 a) {np.mean((s1 >= REJECT1) & (s2 > 0)):.3f}")
    print("selftest OK")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else parity() if "--parity" in sys.argv else main()
