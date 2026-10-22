"""SOXLUSDT 에 ETH 추세·지정가 규칙(10-16 backtest_trend_limit_ladder_b) 그대로 -- 미국장/장외/주말 분해 (2026-10-22, 사용자 요청).

결과 전 고정. 규칙·엔진은 ETH 판과 같다(5분봉 순차 전진, 각 봉은 그 봉까지 확정된 값만):
- 방향 T5 = mean(sign(ln C/C₋ₖ)), k = 7·14·28·56·90 캘린더일, UTC 하루 마지막 종가 → 부호, D 방향 = D−1 종가까지.
  상장(2026-05-15) 전 날짜는 SOXL ETF 일봉 종가(20:00 UTC 확정, 휴장일은 직전값)로 채운다 -- 90일 되돌아보기 때문.
- 진입: 델타 불균형 (2·테이커매수−거래량)/거래량 이 직전 288봉 10분위 이하 & 음봉(롱) / 90분위 이상 & 양봉(숏),
  지정가 = 그 봉 종가, 다음 3봉 안 관통 체결 · 첫 N/3 + 첫 진입가 ∓0.75% 지정가 2N/3 · ∓5% 손절 · 추세 바뀐 날 0시 시가 청산 · 손절 뒤 그날 쉼.
- 크기 N = L × 순자산. 주 L=1(ETH 사후 권고) · 보조 L=3(권장 B).
- 비용 두 벌: «ETH 판»(지정가 0 · 시장가 4bp) · «USDT 기본»(지정가 2 · 시장가 5bp, TradFi 수수료표 미확인). 실제 8h 펀딩 × 명목(롱 지불).
- 세션(뉴욕 시각): 미국장 = 평일 09:30~16:00 · 주말·휴장 = 토·일·ETF 휴장 평일 · 장외 = 나머지 평일 시간.
  ① 손익 귀속: 봉별 순자산 로그 변화를 그 봉 세션에 합산(보유 그대로와 대조) · ② 진입 제한: 그 세션에서 난 신호만 진입(청산·손절은 언제나).
- 귀무: 일 단위 방향 무작위 200번, 같은 규칙·같은 세션 제한 → A 샤프 이상 비율(p).
- 한계: 161일 · 주 판정 구간 하나뿐(표본 밖 없음) → 통과해도 «후보», 불통과면 이 기간 기준 기각.
fresh_forward_bar_by_bar=true · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false
사용: python scripts/research_soxl_trend_limit_sessions_20261022.py [--selftest]
"""
import json, sys
import numba
import numpy as np, pandas as pd

D = "tmp/soxl_trend_limit_20261022/"
STOP, ADD = 0.05, 0.0075
COSTS = {"ETH 판(0/4bp)": (0.0, 0.0004), "USDT 기본(2/5bp)": (0.0002, 0.0005)}
SESS = ("미국장", "장외", "주말·휴장")


@numba.njit(cache=True)
def sim(o, h, l, c, trig_l, trig_s, day_start, dir_bar, fund, lcap, maker, taker):
    """ETH 판 sim(mode_timing=1, use_stop=True, 하루 판) + 지정가 수수료. → (순자산 곡선, 왕복 손익, 손절 수, 신호 수, 체결 수)."""
    n = len(c); E = 1.0; eq = np.empty(n)
    side = 0; avg = 0.0; qty = 0.0; stop_px = 0.0; add_px = 0.0; add_q = 0.0; e_in = 1.0
    pend = 0; lim = 0.0; exp_i = -1; cool = False
    rt = np.zeros(20000); nrt = 0; nstop = 0; nsig = 0; nfill = 0
    for i in range(n):
        d = dir_bar[i]
        if day_start[i]:
            cool = False
            if side != 0 and d != side and np.isfinite(o[i]):
                px = o[i] * (1 - taker * side)
                E += side * qty * (px - avg); rt[nrt] = E / e_in - 1; nrt += 1
                side = 0; qty = 0.0; add_q = 0.0
            pend = 0
        if not (np.isfinite(c[i]) and np.isfinite(o[i])):
            eq[i] = eq[i - 1] if i > 0 else E
            continue
        if side != 0:
            E -= side * fund[i] * qty * o[i]
            if add_q > 0 and ((side == 1 and l[i] < add_px) or (side == -1 and h[i] > add_px)):
                E -= maker * add_q * add_px
                avg = (avg * qty + add_px * add_q) / (qty + add_q); qty += add_q; add_q = 0.0
            if (side == 1 and l[i] <= stop_px) or (side == -1 and h[i] >= stop_px):
                px0 = min(o[i], stop_px) if side == 1 else max(o[i], stop_px)
                px = px0 * (1 - taker * side)
                E += side * qty * (px - avg); rt[nrt] = E / e_in - 1; nrt += 1; nstop += 1
                side = 0; qty = 0.0; add_q = 0.0; cool = True
        if side == 0:
            if pend != 0:
                if i > exp_i:
                    pend = 0
                elif (pend == 1 and l[i] < lim) or (pend == -1 and h[i] > lim):
                    N = lcap * E
                    side = pend; e_in = E; qty = N / 3 / lim; avg = lim; E -= maker * N / 3
                    stop_px = lim * (1 - STOP * side); add_px = lim * (1 - ADD * side); add_q = 2 * N / 3 / add_px
                    pend = 0; nfill += 1
            if side == 0 and pend == 0 and not cool and d != 0:
                if (d == 1 and trig_l[i]) or (d == -1 and trig_s[i]):
                    pend = d; lim = c[i]; exp_i = i + 3; nsig += 1
        eq[i] = E + (side * qty * (c[i] - avg) if side != 0 else 0.0)
    return eq, rt[:nrt], nstop, nsig, nfill


def prepare():
    p = pd.read_parquet(D + "soxl_5m.parquet").sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
    p = p.reindex(pd.date_range(p.index[0], p.index[-1], freq="5min"))
    imb = (2 * p.taker_buy_base - p.volume) / p.volume.replace(0, np.nan)
    q10 = imb.rolling(288, min_periods=280).quantile(.1); q90 = imb.rolling(288, min_periods=280).quantile(.9)
    trig_l = ((imb <= q10) & (p.close < p.open)).fillna(False).to_numpy()
    trig_s = ((imb >= q90) & (p.close > p.open)).fillna(False).to_numpy()
    day = p.index.floor("D")
    perp_d = p.close.groupby(day).last()
    etf = pd.read_csv(D + "soxl_etf_1d.csv", parse_dates=["timestamp"]).dropna(subset=["raw"])
    etf_d = etf.set_index(etf.timestamp.dt.floor("D")).raw                    # 20:00 UTC 종가 → 그 UTC 날
    cal = pd.date_range(etf_d.index[0], perp_d.index[-1], freq="D")
    lc = np.log(perp_d.reindex(cal).combine_first(etf_d.reindex(cal).ffill()))   # 상장 뒤 = 무기한 종가
    t5 = sum(np.sign(lc - lc.shift(k)) for k in (7, 14, 28, 56, 90)) / 5
    d_next = np.sign(t5).shift(1)
    dir_bar = d_next.reindex(day).fillna(0).to_numpy().astype(np.int64)
    day_start = np.asarray(p.index == day, dtype=bool)
    f = pd.read_csv(D + "soxl_funding.csv", parse_dates=["timestamp"])
    fund = f.groupby(f.timestamp.dt.floor("5min")).last_funding_rate.sum().reindex(p.index).fillna(0.0).to_numpy()
    ny = p.index.tz_localize("UTC").tz_convert("America/New_York")
    hol = set(pd.bdate_range(etf.timestamp.min(), etf.timestamp.max()).date) - set(etf.timestamp.dt.date)
    mins = ny.hour * 60 + ny.minute
    wknd = (ny.weekday >= 5) | pd.Index(ny.date).isin(hol)
    sess = np.where(wknd, 2, np.where((mins >= 570) & (mins < 960), 0, 1))
    return p, trig_l, trig_s, day_start, dir_bar, fund, sess, d_next


def metrics(eq, idx):
    d = pd.Series(eq, idx).groupby(idx.floor("D")).last()
    r = d.pct_change().dropna()
    return {"총": round(float(d.iloc[-1] - 1), 3), "샤프": round(float(r.mean() / r.std() * np.sqrt(365)), 2) if r.std() > 0 else 0.0,
            "최대낙폭": round(float((d / d.cummax() - 1).min()), 3)}


def by_session(eq, sess):
    lr = np.diff(np.log(np.r_[1.0, eq]))
    return {SESS[s]: round(float(lr[sess == s].sum()), 3) for s in range(3)}


def run():
    p, tl, ts, ds, dirb, fund, sess, d_next = prepare()
    o, h, l, c = (p[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    idx = p.index
    bh = np.diff(np.log(np.r_[o[0], c]))
    out = {"기간": f"{idx[0]} ~ {idx[-1]}", "일수": int(len(idx.floor('D').unique())),
           "세션 봉 비율": {SESS[s]: round(float((sess == s).mean()), 3) for s in range(3)},
           "그냥 보유 로그수익 세션 귀속": {SESS[s]: round(float(np.nansum(bh[sess == s])), 3) for s in range(3)},
           "세션별 5분 |수익| 평균 bp": {SESS[s]: round(float(np.nanmean(np.abs(bh[sess == s])) * 1e4), 1) for s in range(3)},
           "방향 롱 비율": round(float((dirb[ds] > 0).mean()), 2), "방향 전환": int((np.diff(dirb[ds]) != 0).sum())}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(20261022)
    days = idx.floor("D"); ud = days.unique()
    rand_dirs = [pd.Series(rng.choice([-1, 1], len(ud)), ud).reindex(days).to_numpy().astype(np.int64) for _ in range(200)]
    allows = {"전체": np.ones(len(c), bool), **{f"{SESS[s]}만 진입": sess == s for s in range(3)}}
    for cname, (mk, tk) in COSTS.items():
        for lcap in (1.0, 3.0):
            for aname, al in allows.items():
                a_tl, a_ts = tl & al, ts & al
                eq, rt, nst, nsig, nfill = sim(o, h, l, c, a_tl, a_ts, ds, dirb, fund, lcap, mk, tk)
                row = metrics(eq, idx)
                row.update({"왕복": int(len(rt)), "승률": round(float((rt > 0).mean()), 2) if len(rt) else None,
                            "왕복 평균 %": round(float(rt.mean() * 100), 2) if len(rt) else None, "손절": int(nst),
                            "신호→체결": f"{nfill}/{nsig}", "손익 세션 귀속(로그)": by_session(eq, sess)})
                if lcap == 1.0:
                    rs = [metrics(sim(o, h, l, c, a_tl, a_ts, ds, rd, fund, lcap, mk, tk)[0], idx)["샤프"] for rd in rand_dirs]
                    row["무작위 방향 200"] = {"샤프 중앙": round(float(np.median(rs)), 2), "95분위": round(float(np.quantile(rs, .95)), 2),
                                         "p(A 이상)": round(float(np.mean(np.array(rs) >= row["샤프"])), 3)}
                key = f"{cname} L={lcap:g} {aname}"
                out[key] = row
                print(key, json.dumps(row, ensure_ascii=False), flush=True)
    out["flags"] = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
                    "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    with open(D + "report.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def selftest():
    # ETH 판 selftest 와 같은 경로: 신호 → 체결 100 → 물타기 99.25 → 손절 95. 수수료 0/4bp 면 ETH 판과 같은 값, 지정가 m 이면 정확히 −3m.
    o = np.array([100, 100, 100, 99.9, 99.5, 96.]); c = np.array([100, 100, 99.9, 99.5, 96, 96.])
    h = np.maximum(o, c) + .01; l = np.array([99.95, 99.95, 99.9, 99.2, 94, 95.9])
    tl = np.array([False, True, False, False, False, False]); ts = np.zeros(6, bool)
    ds = np.array([True, False, False, False, False, False]); dirb = np.ones(6, np.int64)
    eq, rt, nst, nsig, nfill = sim(o, h, l, c, tl, ts, ds, dirb, np.zeros(6), 3.0, 0.0, 0.0004)
    assert (nsig, nfill, nst, len(rt)) == (1, 1, 1, 1)
    q = 1 / 100 + 2 / 99.25; avg = (100 * (1 / 100) + 99.25 * (2 / 99.25)) / q
    assert abs(eq[-1] - (1 + q * (95 * (1 - 0.0004) - avg))) < 1e-9, eq[-1]
    eq2 = sim(o, h, l, c, tl, ts, ds, dirb, np.zeros(6), 3.0, 0.0002, 0.0004)[0]
    assert abs(eq2[-1] - (eq[-1] - 3 * 0.0002)) < 1e-12, (eq2[-1], eq[-1])
    print("selftest ok", round(eq[-1], 4))


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run()
