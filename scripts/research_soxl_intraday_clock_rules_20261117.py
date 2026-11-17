#!/usr/bin/env python3
"""SOXL 무기한 장중 시계 규칙 2개 확인 검정 (2026-11-17). 사전등록 docs/prereg/soxl_intraday_clock_rules_20261117.md (계산 전 커밋 4d162280).

  selftest   합성 1분봉: 심은 효과 복원 · 미래 봉을 잘라도 같은 결정 · 서머타임 전후 시각
  build      SOXLUSDT 1분봉 Vision 일 파일 중 아직 없는 날(11-11~)을 받는다
  run        1회 채점 → tmp/soxl_intraday_clock_rules_20261117/result.json

R1 마감 30분 되돌림: s = sign(px(15:30) ÷ 전 거래일 px(16:00) − 1) → 15:30 에 −s, 16:00 청산.
R2 첫 1시간 이어짐: s = sign(px(10:30) ÷ px(09:30) − 1) → 10:30 에 +s, 15:30 청산.
px(T) = 시작이 T−1분인 1분봉 종가(동부 시각). 결정은 px(T) 까지만 본다.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_soxl_session_vol_20261112 as S  # noqa: E402 -- 1분봉 읽기(load)·HOLIDAYS

OUT = ROOT / "tmp/soxl_intraday_clock_rules_20261117"
ETF = ROOT / "tmp/soxl_intraday_explore_20261117/SOXL_60m.parquet"   # 보조(ETF 대조)만
NY, SEED, NB = "America/New_York", 20261117, 10000
HALF = "2026-08-15"                                                  # 앞·뒤 절반 경계(동부 날짜)
COST = {"R1": 12.0, "R2": 13.5}                                      # 왕복 bp: 테이커 5×2 + 미끄러짐 1×2 (+R2 펀딩 1.5)
COST_MK = {"R1": 8.0, "R2": 9.5}                                     # 메이커 진입 2 + 테이커 청산 5 + 미끄러짐 1 (+R2 펀딩 1.5)
CLOCKS = [f"{h:02d}:{m:02d}" for h in range(9, 17) for m in (0, 30) if "09:30" <= f"{h:02d}:{m:02d}" <= "16:00"]


def px_table(k: pd.DataFrame) -> pd.DataFrame:
    """거래일 × 시각 표. px(T) = 시작이 T−1분인 봉의 종가 · 열 «T+1» = 시작이 T 인 봉의 종가(1분 늦춘 체결) · 없으면 NaN."""
    c = pd.Series(k.c.to_numpy(), index=pd.to_datetime(k.t, unit="ms", utc=True))
    days = pd.date_range(c.index[0].tz_convert(NY).normalize().tz_localize(None), c.index[-1].tz_convert(NY).normalize().tz_localize(None), freq="D")
    days = days[(days.dayofweek < 5) & ~np.isin(days.strftime("%Y-%m-%d"), list(S.HOLIDAYS))]
    out = {}
    for T in CLOCKS:
        at = (days + pd.Timedelta(T + ":00")).tz_localize(NY).tz_convert("UTC")
        out[T] = c.reindex(at - pd.Timedelta(minutes=1)).to_numpy()
        out[T + "+1"] = c.reindex(at).to_numpy()
    return pd.DataFrame(out, index=days.strftime("%Y-%m-%d"))


def rules(P: pd.DataFrame, entry_lag: bool = False) -> dict:
    """규칙별 (pos = 든 방향, r = 수익 창의 롱 수익, sig = 신호 크기). pos = 0 또는 NaN 인 날은 거래 없음."""
    e = "+1" if entry_lag else ""
    x1 = P["15:30"] / P["16:00"].shift(1) - 1
    x2 = P["10:30"] / P["09:30"] - 1
    return {"R1": pd.DataFrame({"pos": -np.sign(x1), "r": P["16:00"] / P["15:30" + e] - 1, "sig": x1}),
            "R2": pd.DataFrame({"pos": np.sign(x2), "r": P["15:30"] / P["10:30" + e] - 1, "sig": x2})}


def maker(k: pd.DataFrame, P: pd.DataFrame, R: dict) -> dict:
    """메이커 진입판: T 에 px(T) 지정가, [T, T+5분) 봉이 지정가를 넘어 지나가면 그 값에 체결. 아니면 그날 거래 없음."""
    idx = pd.to_datetime(k.t, unit="ms", utc=True)
    lo, hi = pd.Series(k.l.to_numpy(), index=idx), pd.Series(k.h.to_numpy(), index=idx)
    out = {}
    for name, T in (("R1", "15:30"), ("R2", "10:30")):
        d = R[name].copy()
        at = (pd.to_datetime(P.index) + pd.Timedelta(T + ":00")).tz_localize(NY).tz_convert("UTC")
        mn = np.nanmin([lo.reindex(at + pd.Timedelta(minutes=i)).to_numpy() for i in range(5)], axis=0)
        mx = np.nanmax([hi.reindex(at + pd.Timedelta(minutes=i)).to_numpy() for i in range(5)], axis=0)
        lim = P[T].to_numpy()
        fill = np.where(d.pos.to_numpy() > 0, mn < lim, np.where(d.pos.to_numpy() < 0, mx > lim, False))
        d["pos"] = np.where(fill, d.pos, np.nan)
        out[name] = d
    return out


def boot(x: np.ndarray, rng, groups: np.ndarray | None = None) -> dict:
    """평균의 부트스트랩(거래일 또는 묶음 단위): 평균 · 90/95% CI · 양측 p."""
    if groups is None:
        m = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    else:
        g = pd.DataFrame({"g": groups, "x": x}).groupby("g").x.agg(["sum", "size"]).to_numpy()
        b = g[rng.integers(0, len(g), (NB, len(g)))]
        m = b[:, :, 0].sum(1) / b[:, :, 1].sum(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"mean": float(x.mean()), "ci90": [float(np.percentile(m, 5)), float(np.percentile(m, 95))],
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "p2": float(min(1.0, p))}


def score(d: pd.DataFrame, cost_bp: float, rng) -> dict:
    """한 규칙 채점. 값은 전부 bp. 분모 = 신호와 수익 창이 다 있는 거래일."""
    d = d.dropna(subset=["pos", "r"]); n_all = len(d); d = d[d.pos != 0]
    pos, r = d.pos.to_numpy(), d.r.to_numpy() * 1e4
    g = pos * r; net = g - cost_bp
    perm = np.array([np.mean(pos * rng.permutation(r)) for _ in range(NB)])
    late = np.asarray(d.index >= HALF)
    wk = pd.to_datetime(d.index).strftime("%G-%V").to_numpy()
    cum = np.cumsum(net)
    side = {nm: {"n": int(m.sum()), "mean_g": float(g[m].mean()) if m.any() else None, "wins": f"{int((g[m] > 0).sum())}/{int(m.sum())}"}
            for nm, m in (("롱", pos > 0), ("숏", pos < 0))}
    return {"거래일": n_all, "거래": len(d), "이긴 날(비용 전)": f"{int((g > 0).sum())}/{len(d)}", "이긴 날(비용 뒤)": f"{int((net > 0).sum())}/{len(d)}",
            "gross": boot(g, rng), "net": boot(net, rng), "net_주블록": boot(net, rng, wk), "중앙_g": float(np.median(g)), "표준편차_g": float(g.std(ddof=1)),
            "위약_분위": float(100 * (perm < g.mean()).mean()), "위약_95분위": float(np.percentile(perm, 95)),
            "초과_늘롱": boot(g - r, rng), "초과_늘숏": boot(g + r, rng), "늘롱_평균": float(r.mean()),
            "앞_절반": {"n": int((~late).sum()), "mean_g": float(g[~late].mean()) if (~late).any() else None},
            "뒤_절반": {"n": int(late.sum()), "mean_g": float(g[late].mean()) if late.any() else None},
            "다리": side, "누적_net": float(cum[-1]), "최대낙폭_net": float((cum - np.maximum.accumulate(cum)).min()), "비용": cost_bp}


def verdict(res: dict, mk: dict) -> dict:
    """사전등록 주 판정 — Holm(두 규칙) · 위약 · 양쪽 다리 · 두 절반."""
    def holm(key):
        ps = sorted(((res[n][key]["p2"], n) for n in res), key=lambda t: t[0]); ok = {}
        ok[ps[0][1]] = ps[0][0] < 0.025 and res[ps[0][1]][key]["mean"] > 0
        ok[ps[1][1]] = ok[ps[0][1]] and ps[1][0] < 0.05 and res[ps[1][1]][key]["mean"] > 0
        return ok
    hn, hg, out = holm("net"), holm("gross"), {}
    for n, s in res.items():
        legs = s["초과_늘롱"]["mean"] > 0 and s["초과_늘숏"]["mean"] > 0
        halves = (s["앞_절반"]["mean_g"] or -1) > 0 and (s["뒤_절반"]["mean_g"] or -1) > 0
        out[n] = {"1_순수익_Holm": bool(hn[n]), "2_위약_97.5": s["위약_분위"] >= 97.5, "3_양쪽_다리": bool(legs), "4_두_절반": bool(halves),
                  "엄격_통과": bool(hn[n] and s["위약_분위"] >= 97.5 and legs and halves),
                  "존재_확인": bool(hg[n] and s["위약_분위"] >= 97.5 and halves),
                  "통과_후보": bool(mk[n]["net"]["mean"] > 0 and s["위약_분위"] >= 50 and legs and halves)}
    return out


def load() -> pd.DataFrame:
    k = S.load("SOXLUSDT")
    ps = sorted((OUT / "klines").glob("SOXLUSDT-1m-*.parquet"))
    return pd.concat([k, *[pd.read_parquet(p) for p in ps]]).drop_duplicates("t").sort_values("t").reset_index(drop=True) if ps else k


def build() -> None:
    (OUT / "klines").mkdir(parents=True, exist_ok=True)
    last = pd.to_datetime(S.load("SOXLUSDT").t.iloc[-1], unit="ms").normalize()
    for day in pd.date_range(last + pd.Timedelta(days=1), pd.Timestamp.now("UTC").tz_localize(None).normalize()).strftime("%Y-%m-%d"):
        p = OUT / "klines" / f"SOXLUSDT-1m-{day}.parquet"
        if p.exists():
            continue
        try:                                                        # Vision 정적 파일(REST 아님)
            raw = urllib.request.urlopen(f"https://data.binance.vision/data/futures/um/daily/klines/SOXLUSDT/1m/SOXLUSDT-1m-{day}.zip", timeout=60).read()
        except urllib.error.HTTPError:
            print(day, "아직 없음"); continue
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit():
            d = d.iloc[1:]
        d = d.astype(float)
        pd.DataFrame({"t": d[0].astype("int64"), "h": d[2], "l": d[3], "c": d[4], "v": d[5], "tb": d[9]}).to_parquet(p); print(day, len(d))
    k = load(); print("1분봉", len(k), pd.to_datetime(k.t.iloc[[0, -1]], unit="ms").astype(str).tolist())


def etf_rules() -> dict:
    """보조: 같은 기간 ETF 60분봉(야후) 같은 식 — 무기한이 ETF 를 따라가는지."""
    d = pd.read_parquet(ETF); d["et"] = pd.to_datetime(d.ts, unit="s", utc=True).dt.tz_convert(NY)
    d = d[d.et >= pd.Timestamp("2026-05-15", tz=NY)]; d["day"] = d.et.dt.strftime("%Y-%m-%d"); d["hm"] = d.et.dt.strftime("%H:%M")
    ok = d.groupby("day").hm.apply(lambda s: list(s) == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]); d = d[d.day.map(ok)]
    C, O = d.pivot(index="day", columns="hm", values="c"), d.pivot(index="day", columns="hm", values="o")
    x1, x2 = C["14:30"] / C["15:30"].shift(1) - 1, C["09:30"] / O["09:30"] - 1
    return {"R1": pd.DataFrame({"pos": -np.sign(x1), "r": C["15:30"] / C["14:30"] - 1, "sig": x1}),
            "R2": pd.DataFrame({"pos": np.sign(x2), "r": C["14:30"] / C["09:30"] - 1, "sig": x2})}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k = load(); P = px_table(k); R = rules(P); M = maker(k, P, R); rng = np.random.default_rng(SEED)
    res = {n: score(R[n], COST[n], rng) for n in R}
    mk = {n: score(M[n], COST_MK[n], rng) for n in M}
    out = {"data_last": str(pd.to_datetime(k.t.iloc[-1], unit="ms")), "days_table": [P.index[0], P.index[-1], len(P)],
           "missing_px": {T: int(P[T].isna().sum()) for T in ("09:30", "10:30", "15:30", "16:00")},
           "primary": res, "maker": mk, "verdict": verdict(res, mk)}
    # ── 보조(판정 아님)
    aux, d1 = {}, R["R1"].dropna(subset=["pos", "r"])
    g1 = d1.pos * d1.r * 1e4
    aux["R1_내린날"] = {"n": int((d1.sig < 0).sum()), "mean_g": float(g1[d1.sig < 0].mean())}
    aux["R1_오른날"] = {"n": int((d1.sig > 0).sum()), "mean_g": float(g1[d1.sig > 0].mean())}
    q = pd.qcut(d1.sig.abs(), 3, labels=["작음", "중간", "큼"])
    aux["R1_움직임_3분위"] = {str(c): {"n": int((q == c).sum()), "mean_g": float(g1[q == c].mean())} for c in q.cat.categories}
    pc = P["16:00"].shift(1); tw = {}
    for a, b in zip(CLOCKS[1:-2], CLOCKS[2:-1]):                       # 10:00 … 15:00 시작(끝 15:30 까지)
        v = (-np.sign(P[a] / pc - 1) * (P[b] / P[a] - 1) * 1e4).dropna()
        tw[a] = {"n": len(v), "mean_g": float(v.mean()), "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))}
    aux["시각_위약(R1 식을 다른 30분 창에)"] = tw
    aux["1분_늦춘_체결"] = {n: score(d, COST[n], rng) for n, d in rules(P, entry_lag=True).items()}
    nxt = P["09:30"].shift(-1)
    aux["R1_다음날_개장까지"] = score(pd.DataFrame({"pos": R["R1"].pos, "r": nxt / P["15:30"] - 1, "sig": R["R1"].sig}), COST["R1"] + 3.0, rng)
    if ETF.exists():
        aux["ETF_같은기간"] = {n: score(d, 0.0, rng) for n, d in etf_rules().items()}
    out["aux"] = aux
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def selftest() -> None:
    ts = lambda s: int(pd.Timestamp(s, tz=NY).tz_convert("UTC").value // 10**6)
    rng = np.random.default_rng(1)
    t0, n = ts("2026-10-12 00:00"), 40 * 1440                          # 서머타임 종료(11-01)를 낀 40일
    t = t0 + np.arange(n) * 60000
    et = pd.to_datetime(t, unit="ms", utc=True).tz_convert(NY); hm = et.hour * 60 + et.minute
    r = rng.normal(0, 4e-4, n)
    c = 50 * np.exp(np.cumsum(r)); k = pd.DataFrame({"t": t, "h": c * 1.0003, "l": c * 0.9997, "c": c, "v": 1.0, "tb": 0.5})
    P = px_table(k)
    # 시각: 서머타임 전(10-20)은 15:30 ET = 19:30 UTC, 뒤(11-10)는 20:30 UTC → px 는 그 1분 전 봉 종가
    for day, hh in (("2026-10-20", 19), ("2026-11-10", 20)):
        want = k.c[k.t == int(pd.Timestamp(f"{day} {hh}:29", tz="UTC").value // 10**6)].iloc[0]
        assert P.at[day, "15:30"] == want and P.at[day, "15:30+1"] == k.c[k.t == int(pd.Timestamp(f"{day} {hh}:30", tz="UTC").value // 10**6)].iloc[0]
    assert "2026-10-17" not in P.index and "2026-11-26" not in P.index   # 주말·휴장 없음
    # 효과 심기: 마지막 30분 = 그날 방향의 반대로 30bp · 10:30~15:30 = 첫 1시간 방향으로 50bp(매분 균등)
    R0 = rules(P); lr = np.diff(np.log(c), prepend=np.log(c[0])); day_s = et.strftime("%Y-%m-%d")
    for d in P.index[1:]:
        m = day_s == d
        s2 = R0["R2"].pos.get(d, 0.0)
        if s2 and not np.isnan(s2):
            lr[m & (hm >= 630) & (hm < 930)] += s2 * 50e-4 / 300
    c2 = c[0] * np.exp(np.cumsum(lr) - lr[0]); k2 = k.assign(c=c2, h=c2 * 1.0003, l=c2 * 0.9997); P2 = px_table(k2)
    lr2 = np.diff(np.log(c2), prepend=np.log(c2[0])); R1s = rules(P2)["R1"]
    for d in P2.index[1:]:
        s1 = -R1s.pos.get(d, 0.0)                                       # 그날 방향
        if s1 and not np.isnan(s1):
            lr2[(day_s == d) & (hm >= 930) & (hm < 960)] += -s1 * 30e-4 / 30
    c3 = c2[0] * np.exp(np.cumsum(lr2) - lr2[0]); k3 = k.assign(c=c3, h=c3 * 1.0003, l=c3 * 0.9997); P3 = px_table(k3); R3 = rules(P3)
    sc = {nm: score(R3[nm], COST[nm], np.random.default_rng(0)) for nm in R3}
    base = {nm: score(rules(P)[nm], COST[nm], np.random.default_rng(0)) for nm in R3}
    assert abs(sc["R1"]["gross"]["mean"] - base["R1"]["gross"]["mean"] - 30) < 12, (sc["R1"]["gross"]["mean"], base["R1"]["gross"]["mean"])
    assert abs(sc["R2"]["gross"]["mean"] - base["R2"]["gross"]["mean"] - 50) < 12, (sc["R2"]["gross"]["mean"], base["R2"]["gross"]["mean"])
    assert abs(sc["R1"]["net"]["mean"] - (sc["R1"]["gross"]["mean"] - 12)) < 1e-9 and sc["R1"]["위약_분위"] > 90
    # 미래 봉을 잘라도 같은 결정: 11-05 15:30 까지만 든 자료의 그날 신호 = 전체 자료의 신호
    cut = k3[k3.t < ts("2026-11-05 15:30")]; Rc = rules(px_table(cut))
    assert Rc["R1"].pos["2026-11-05"] == R3["R1"].pos["2026-11-05"] and Rc["R2"].pos["2026-11-05"] == R3["R2"].pos["2026-11-05"]
    assert np.isnan(Rc["R1"].r["2026-11-05"])                           # 수익 창은 아직 없다
    # 메이커 판: 체결 조건 = 지정가를 넘어 지나감 · 안 채워진 날은 거래 없음
    M = maker(k3, P3, R3); f = M["R1"].pos.notna().sum(); assert 0 < f <= R3["R1"].pos.notna().sum()
    # Holm: 작은 p 가 떨어지면 둘 다 탈락
    fake = {"R1": {"net": {"p2": .03, "mean": 5}, "gross": {"p2": .001, "mean": 5}, "위약_분위": 99, "초과_늘롱": {"mean": 1}, "초과_늘숏": {"mean": 1}, "앞_절반": {"mean_g": 1}, "뒤_절반": {"mean_g": 1}},
            "R2": {"net": {"p2": .04, "mean": 5}, "gross": {"p2": .04, "mean": 5}, "위약_분위": 99, "초과_늘롱": {"mean": 1}, "초과_늘숏": {"mean": -1}, "앞_절반": {"mean_g": 1}, "뒤_절반": {"mean_g": 1}}}
    v = verdict(fake, {"R1": {"net": {"mean": 1}}, "R2": {"net": {"mean": 1}}})
    assert not v["R1"]["엄격_통과"] and not v["R2"]["엄격_통과"] and v["R1"]["존재_확인"] and v["R2"]["존재_확인"] and v["R1"]["통과_후보"] and not v["R2"]["통과_후보"]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "build", "run"])
    {"selftest": selftest, "build": build, "run": run}[ap.parse_args().cmd]()
