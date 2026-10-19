"""ETH «바탕 스윙» S4 재검증 (2026-10-19, 10-19 mle-reviewer 지적 후속). 원 연구 =
4hour-forecast-accuracy-5c6776/scripts/research_eth_multihorizon_structure_20261019.py (S4 = sgn(추세+7일+4-13주) 1.52 vs T .69 «통과»).

사전등록(계산 전 고정):
  R0 재현  S4·T·L(4-13주 단독) 2025~ 샤프가 원 연구 1.52·.69·1.53 와 ±.01 → 아니면 이하 «무효».
  R1 펀딩  실제 바이낸스 ETHUSDT 펀딩(data.binance.vision 월간, 8h) 을 그 시각 들고 있던 포지션에 부과(롱이 양수 펀딩 냄).
           하루 d 포지션(UTC 0시 갱신) = (d 0시, d+1 0시] 정산분 → 0시 정산은 전날 포지션 몫.
  R2 원 판정 재적용(펀딩 포함): S4 vs T 2025·2026 각각 샤프 ↑ + 5일 블록 P ≥ .90.
  R3 다중검정: 원 사전등록 5쌍(S1·S2·S3 vs S0, S4 vs T, S5 vs S0+T) p = 1 − P_better(2025~), Holm, α = .10(P≥.90 과 같은 수준).
  R4 위약: 4-13주 상태열(2008~ 전 구간 일 시계열)을 원형 밀기, |밀기| ≤ 60일 제외 전부. S4·L 실제 샤프의 위약 분위.
  R5 증분: S4 − L 일 손익 차 평균 5·20일 블록 CI · 샤프 P(S4 > L).
  판정 «S4 통과»:  유지 = R2 ∧ Holm 보정 p ≤ .10 ∧ S4 위약 분위 ≥ .95 · 하나라도 실패 = «미통과로 강등» · R0 실패 = «무효».
  «힘은 4-13주 규칙» 유지 = 펀딩 포함 L 샤프 ≥ .9 × S4 ∧ P(S4 > L) < .90 · 단 L 위약 분위 < .95 면 «L 자체가 우연 타이밍과 구분 안 됨»으로 고쳐 적는다.
  R6 선택 밖 기간: 4-13주 규칙(W91·|Z|≥1)은 2018~26 에서 골랐다 → 선택 밖 = 2018 이전뿐(현물, 펀딩 없음):
           ETH 2016-17 · BTC 2014-17 (CoinMetrics 일 종가). 지지 = L 샤프 > 0 ∧ L 위약 분위 ≥ .90. S4 는 7일 원소가 2018-08 전 계산 불가 → 불가.

결과(2025-01-01~2026-09-14, 일 손익 bp, 4bp/회전, 펀딩 포함 · 원본 = report.json):
  R0 재현 OK S4 1.52 · T .69 · L 1.53.  R1 펀딩 영향 미미(S4 −.05 · T −.22 · L +.06 bp/일) → S4 1.52 · T .68 · L 1.53.
  R2 원 규칙 펀딩 후에도 통과(P .957 · 2025 1.78>.75 · 2026 1.08>.55).  R3 Holm: S4 p .043 → 보정 .215 > .10 → 탈락(5쌍 전부 비유의).
  R4 위약(2005~ 상태열 7,005 밀기): S4 분위 .988 · L .983(보조 2018~ .991/.973 · 2022~ .989/.952) → 통과.
  R5 S4 − L +2.1bp/일 CI5[−16.5,22.1]·CI20[−14.9,25.4] · P(S4>L) .48/.59 → 묶음 증분 0. (CL − L P .66/.78)
     단 연도별로는 L 2025 2.08·2026 .26, 7일 C 2025 .04·2026 1.49 — 해마다 다른 원소가 버팀(2023-24 C 1.84 는 C 의 선택 표본 안).
  R6 선택 밖(현물): ETH 2016-17 L .45(보유 2.49·T 2.20, 위약 분위 .49) · BTC 2014-17 L .53(보유 1.01·T 1.74, 분위 .71) → 지지 안 함.
  판정: «S4 통과» → 미통과로 강등(Holm). «힘은 4-13주 규칙» = 2025~ 표본 안에선 유지(펀딩·위약 뒤에도), 선택 밖 2018 이전엔 재현 안 됨.

    nice -n 15 python scripts/research_eth_multihorizon_reverify_20261019.py [--selfcheck]
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT))
from dashboard.dir_signals import FRED_LAG, asof, cb_premium_z, liq_score  # noqa: E402

RAW = MAIN / ".claude/worktrees/medium-long-term-forecast-model-669354/tmp"
ORIG = MAIN / ".claude/worktrees/4hour-forecast-accuracy-5c6776/tmp/multihorizon_20261019/report.json"
OUT = ROOT / "tmp/multihorizon_reverify_20261019"
COST, EXCL = 4.0, 60
MAIN_P = ("2025-01-01", "2026-09-15")
PERIODS = (("2025~", MAIN_P), ("2025", ("2025-01-01", "2026-01-01")), ("2026", ("2026-01-01", MAIN_P[1])), ("2023-24", ("2023-02-01", "2025-01-01")))
st = lambda v: np.where(v >= 1, 1, np.where(v <= -1, -1, 0))


def vote_of(C: pd.Series) -> pd.Series:
    return sum(np.sign(C - C.shift(L)) for L in (7, 14, 28, 56, 90)) / 5


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(np.mean(x) / np.std(x) * np.sqrt(365)) if np.std(x) > 0 else 0.0


def blocks(n: int, blk: int, nb: int, seed: int) -> np.ndarray:
    st_ = np.random.default_rng(seed).integers(0, n - blk + 1, size=(nb, int(np.ceil(n / blk))))
    return (st_[:, :, None] + np.arange(blk)).reshape(nb, -1)[:, :n]


def boot_p(a, b, blk=5, n=4000, seed=11) -> float:   # 원 연구와 같은 블록 부트스트랩 P(샤프 a > 샤프 b)
    ii = blocks(len(a), blk, n, seed)
    return float((a[ii].mean(1) / a[ii].std(1) > b[ii].mean(1) / b[ii].std(1)).mean())


def boot_ci(x, blk=5, n=4000, seed=7):
    m = x[blocks(len(x), blk, n, seed)].mean(1)
    return [round(float(np.quantile(m, .025)), 2), round(float(np.quantile(m, .975)), 2)]


def holm(p: dict, alpha: float) -> dict:
    out, k = {}, len(p)
    for i, (nm, v) in enumerate(sorted(p.items(), key=lambda t: t[1])):
        out[nm] = round(min(1.0, v * (k - i)), 4)
    run = 0.0                                  # 단조 보정(앞 단계 adj 이상)
    for nm in sorted(out, key=lambda n: p[n]):
        run = max(run, out[nm]); out[nm] = run
    return {nm: {"p": round(p[nm], 4), "p_holm": out[nm], "reject": out[nm] <= alpha} for nm in p}


def funding_daily(zips: list[Path]) -> pd.Series:
    """정산 시각 t 의 펀딩을 t 직전에 들고 있던 하루 포지션 날짜에 귀속(0시 정산 = 전날). bp."""
    f = pd.concat([pd.read_csv(zipfile.ZipFile(z).open(zipfile.ZipFile(z).namelist()[0])) for z in zips])
    t = pd.to_datetime(f.calc_time, unit="ms")
    return (pd.Series(f.last_funding_rate.astype(float).to_numpy() * 1e4, index=(t - pd.Timedelta("1min")).dt.normalize()).groupby(level=0).sum())


def swing(pos: pd.Series, r: pd.Series, fund: pd.Series | None) -> pd.Series:
    """d 일 손익 = pos_d · r_d − 회전 비용 − pos_d · 펀딩_d (롱이 양수 펀딩 냄)."""
    x = pos * r - COST * pos.diff().abs().fillna(0)
    if fund is not None:
        x = x - pos * fund.reindex(pos.index).fillna(0)
    return x.dropna()


def macro_Z() -> pd.Series:
    f = lambda s: pd.read_csv(RAW / f"trend_mid_long_20261015/raw/fred_{s}.csv", parse_dates=["observation_date"]).set_index("observation_date").iloc[:, 0]
    idx = pd.date_range("2005-01-01", "2026-10-15")
    return liq_score(*(asof(f(s), idx, FRED_LAG[s]) for s in ("DTWEXBGS", "WALCL", "RRPONTSYD", "WTREGEN")))


def placebo(lstate: pd.Series, days: pd.DatetimeIndex, build, actual: float) -> dict:
    """lstate(전 구간 일 시계열, 결정일 기준 값)를 원형으로 밀어 build(l_on_days) → 샤프."""
    v, N = lstate.to_numpy(), len(lstate)
    pos = lstate.index.get_indexer(days)
    assert (pos >= 0).all()
    ks = [k for k in range(1, N) if min(k, N - k) > EXCL]
    sh = np.array([build(pd.Series(np.roll(v, k)[pos], index=days)) for k in ks])
    return {"actual": round(actual, 3), "n_shifts": len(ks), "pct": round(float((sh < actual).mean()), 3),
            "median": round(float(np.median(sh)), 3), "p90": round(float(np.quantile(sh, .9)), 3), "p95": round(float(np.quantile(sh, .95)), 3)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(MAIN / "data/binance_vision/panel/ETHUSDT.parquet", columns=["timestamp", "close"])
    Cd = v.set_index("timestamp").close.ffill().groupby(lambda t: t.normalize()).last()      # 선물 UTC 일봉(원 연구 그대로)
    days = Cd.index
    cb = pd.read_csv(RAW / "short_fresh_20261015/raw/cb_BTC.csv", parse_dates=["date"]).set_index("date").close
    bn = pd.read_csv(RAW / "trend_mid_long_20261015/raw/bn_BTC.csv", parse_dates=["date"]).set_index("date").close
    Z = macro_Z()
    lstate = pd.Series(st(Z.shift(1).to_numpy()), index=Z.index)[Z.shift(1).notna()]          # 결정일 d 의 상태(d-1 값)
    D = pd.DataFrame(index=days)
    D["t"] = np.where(vote_of(Cd).shift(1).abs() >= 0.6, np.sign(vote_of(Cd).shift(1)), 0)
    D["c"] = st(cb_premium_z(cb, bn).shift(1).reindex(days).to_numpy())
    D["l"] = lstate.reindex(days).to_numpy()
    D = D.fillna(0)
    r = np.log(Cd / Cd.shift(1)) * 1e4
    fund = funding_daily(sorted((OUT / "raw_funding").glob("ETHUSDT-fundingRate-*.zip")))
    # 교차 확인: API 일합(fut_ETH.csv, 정산 날짜 기준)과 월간 아카이브 합이 같은 크기
    api = pd.read_csv(RAW / "trend_mid_long_20261015/raw/fut_ETH.csv", parse_dates=["date"]).set_index("date").funding * 1e4
    j = api.loc["2025-01-01":"2026-09-14"]
    fund_api_check = round(float(j.sum()), 1), round(float(fund.loc["2024-12-31":"2026-09-13"].sum()), 1)
    pos_of = {"S4": np.sign(D.t + D.c + D.l), "T": D.t, "L": D.l, "C": D.c, "TL": np.sign(D.t + D.l), "CL": np.sign(D.c + D.l)}
    dly = {f"{k}{'_f' if fd is not None else ''}": swing(p, r, fd) for k, p in pos_of.items() for fd in (None, fund)}
    win = lambda s, a, b: s[(s.index >= a) & (s.index < b)]
    rep: dict = {"funding_sum_bp_check(api, archive)": fund_api_check, "series": {}}
    for nm, s in dly.items():
        rep["series"][nm] = {lab: {"sharpe": round(sharpe(win(s, a, b)), 2), "mean_bp": round(float(win(s, a, b).mean()), 2),
                                   "long_share": round(float((pos_of[nm.split('_')[0]].pipe(lambda p: win(p, a, b)) > 0).mean()), 3)}
                             for lab, (a, b) in PERIODS if len(win(s, a, b)) > 30}
    se = rep["series"]
    rep["R0_reproduce"] = {"S4": se["S4"]["2025~"]["sharpe"], "T": se["T"]["2025~"]["sharpe"], "L": se["L"]["2025~"]["sharpe"],
                           "ok": abs(se["S4"]["2025~"]["sharpe"] - 1.52) <= .01 and abs(se["T"]["2025~"]["sharpe"] - .69) <= .01 and abs(se["L"]["2025~"]["sharpe"] - 1.53) <= .01}
    m = lambda s: win(s, *MAIN_P).to_numpy()
    rep["funding_mean_bp_per_day_2025~"] = {k: round(float(m(dly[k + "_f"]).mean() - m(dly[k]).mean()), 2) for k in pos_of}
    # R2 원 판정(펀딩 포함·미포함)
    r2 = {}
    for tag in ("", "_f"):
        a, b = dly["S4" + tag], dly["T" + tag]
        r2["S4 vs T" + tag] = {lab: [round(sharpe(win(a, p0, p1)), 2), round(sharpe(win(b, p0, p1)), 2)] for lab, (p0, p1) in PERIODS[:3]}
        r2["S4 vs T" + tag]["P_2025~"] = boot_p(m(a), m(b))
        x = r2["S4 vs T" + tag]
        x["pass"] = x["2025"][0] > x["2025"][1] and x["2026"][0] > x["2026"][1] and x["P_2025~"] >= .90
    rep["R2_original_rule"] = r2
    # R3 Holm(원 5쌍, S4 vs T 는 원 보고 P 와 펀딩 포함 P 둘 다)
    orig = json.loads(ORIG.read_text())["pairs"]
    p = {k: 1 - v["2025~"]["P_better"] for k, v in orig.items()}
    rep["R3_holm_original"] = holm(p, .10)
    rep["R3_holm_with_funded_S4"] = holm({**p, "S4 vs T": 1 - r2["S4 vs T_f"]["P_2025~"]}, .10)
    # R5 증분 S4 vs L, 그리고 조합들 vs L (펀딩 포함)
    inc = {}
    for k in ("S4", "TL", "CL"):
        d = m(dly[k + "_f"]) - m(dly["L_f"])
        inc[f"{k} - L"] = {"mean_bp": round(float(d.mean()), 2), "ci5": boot_ci(d, 5), "ci20": boot_ci(d, 20),
                           "P_sharpe_gt_L_blk5": boot_p(m(dly[k + "_f"]), m(dly["L_f"]), 5), "P_sharpe_gt_L_blk20": boot_p(m(dly[k + "_f"]), m(dly["L_f"]), 20)}
    rep["R5_increment_vs_L_funded"] = inc
    # R4 위약(펀딩 포함, 2025~)
    sd = days[(days >= MAIN_P[0]) & (days < MAIN_P[1])]
    Dm, rm = D.reindex(sd), r.reindex(sd)
    bS4 = lambda l: sharpe(swing(np.sign(Dm.t + Dm.c + l), rm, fund))
    bL = lambda l: sharpe(swing(l, rm, fund))
    lm = Dm.l   # 실제 값도 같은 함수로(창 첫날 회전 비용 0 -- 위약과 같은 조건)
    rep["R4_placebo_2025~_funded"] = {"S4": placebo(lstate, sd, bS4, bS4(lm)), "L": placebo(lstate, sd, bL, bL(lm))}
    # 보조(판정 밖): 밀기 원천을 2018~ · 2022~(검토자 방식) 상태열로 좁힘
    for a in ("2018-01-01", "2022-01-01"):
        ls_ = lstate.loc[a:MAIN_P[1]]
        rep[f"R4b_placebo_src_{a[:4]}~"] = {"S4": placebo(ls_, sd, bS4, bS4(lm)), "L": placebo(ls_, sd, bL, bL(lm))}
    # R6 선택 밖(2018 이전, 현물 CoinMetrics, 펀딩 없음)
    r6 = {}
    for coin, (a, b) in (("eth", ("2016-01-01", "2018-01-01")), ("btc", ("2014-01-01", "2018-01-01"))):
        C = pd.read_csv(RAW / f"trend_mid_long_20261015/raw/cm_{coin}.csv", parse_dates=["date"]).set_index("date").PriceUSD.dropna()
        rc = np.log(C / C.shift(1)) * 1e4
        tv = vote_of(C).shift(1)
        dd = C.index[(C.index >= a) & (C.index < b)]
        tpos = pd.Series(np.where(tv.abs() >= .6, np.sign(tv), 0), index=C.index).reindex(dd)
        lpos = lstate.reindex(dd)
        rr = rc.reindex(dd)
        res = {"days": len(dd), "L": round(sharpe(swing(lpos, rr, None)), 2), "T": round(sharpe(swing(tpos, rr, None)), 2),
               "TL": round(sharpe(swing(np.sign(tpos + lpos), rr, None)), 2), "hold": round(sharpe(rr.dropna()), 2),
               "L_active_share": round(float((lpos != 0).mean()), 3), "L_long_share": round(float((lpos > 0).mean()), 3)}
        res["L_placebo"] = placebo(lstate, dd, lambda l: sharpe(swing(l, rr, None)), sharpe(swing(lpos, rr, None)))
        res["support"] = res["L"] > 0 and res["L_placebo"]["pct"] >= .90
        r6[coin] = res
    rep["R6_out_of_selection"] = r6
    # 판정
    s4 = se["S4_f"]["2025~"]["sharpe"]; l_ = se["L_f"]["2025~"]["sharpe"]
    hold = r2["S4 vs T_f"]["pass"] and rep["R3_holm_with_funded_S4"]["S4 vs T"]["reject"] and rep["R4_placebo_2025~_funded"]["S4"]["pct"] >= .95
    rep["verdict_S4"] = "invalid" if not rep["R0_reproduce"]["ok"] else ("holds" if hold else "downgrade_to_not_passed")
    rep["verdict_power_is_L"] = {"L_ge_0.9_S4": l_ >= .9 * s4, "P_S4_gt_L_lt_.90": inc["S4 - L"]["P_sharpe_gt_L_blk5"] < .90,
                                 "L_placebo_pct_ge_.95": rep["R4_placebo_2025~_funded"]["L"]["pct"] >= .95}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)))
    pd.DataFrame(dly).to_parquet(OUT / "daily.parquet")
    print(json.dumps({k: v for k, v in rep.items() if k != "series"}, ensure_ascii=False, indent=1, default=float))
    print({k: (v.get("2025~", {}).get("sharpe"), v.get("2025", {}).get("sharpe"), v.get("2026", {}).get("sharpe"), v.get("2023-24", {}).get("sharpe")) for k, v in se.items()})


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # 펀딩 귀속: 1-2 00:00 정산 → 1-1 포지션 몫 · 롱이 양수 펀딩을 낸다
        d0 = pd.Timestamp("2025-01-01")
        t = pd.Series([d0 + pd.Timedelta(h, "h") for h in (0, 8, 16, 24)])
        assert list((t - pd.Timedelta("1min")).dt.normalize()) == [d0 - pd.Timedelta(days=1), d0, d0, d0]
        idx = pd.date_range(d0, periods=3)
        pos, r0, fd = pd.Series([1.0, 1.0, -1.0], idx), pd.Series(0.0, idx), pd.Series([1.0, 1.0, 1.0], idx)
        x = swing(pos, r0, fd)
        assert x.iloc[1] == -1.0 and x.iloc[2] == -2 * COST + 1.0       # 롱 −1 · 숏 +1 (전환 비용 8bp)
        # 지연: 결정일 d 는 d-1 까지의 추세표(오늘 종가 미사용)
        C = pd.Series(np.r_[np.linspace(100, 200, 100), [1.0]], index=pd.date_range("2024-01-01", periods=101))
        assert vote_of(C).shift(1).iloc[100] == 1
        # Holm: 가장 작은 p × k, 단조
        h = holm({"a": .01, "b": .04, "c": .03}, .10)
        assert h["a"]["p_holm"] == .03 and h["c"]["p_holm"] == .06 and h["b"]["p_holm"] == .06
        assert len([k for k in range(1, 200) if min(k, 200 - k) > EXCL]) == 200 - 1 - 2 * EXCL
        print("selfcheck ok")
    else:
        main()
