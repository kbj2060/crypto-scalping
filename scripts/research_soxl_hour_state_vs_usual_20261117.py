#!/usr/bin/env python3
"""SOXL «앞 1시간이 이 시각대 평소보다 조용한가·격한가» 판정 (2026-11-17). 사전등록 docs/prereg/soxl_hour_state_vs_usual_20261117.md (계산 전 커밋 4aa3b549 + 채점 전 수정).

  selftest   합성 1분봉: 평소 U 가 미래·같은 날을 안 봄 · 변동성 두 배 날에 격함
  run        시험 10-01~ 1회 채점 → tmp/soxl_hour_state_20261117/result.json

모델 = 배포 계수 beta60(동결)의 중앙값형 예측 m̂ = exp(har_x·beta60). 평소 U = 같은 날 유형·동부 시각 ±15분·앞선 30일 RV60 중앙값(40개 이상).
ρ = m̂ ÷ U → 조용 ≤ 0.7 · 격함 ≥ 1.5. 실제 = RV60 > U.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_soxl_session_vol_20261112 as S  # noqa: E402 -- frame·har_x(서빙과 같은 식)
import research_soxl_intraday_clock_rules_20261117 as C  # noqa: E402 -- load(): 1분봉 05-15~11-15

OUT = ROOT / "tmp/soxl_hour_state_20261117"
PARAMS = ROOT / "dashboard/risk_radar_params_soxl_20261117.json"
NY, SEED, NB = "America/New_York", 20261117, 2000
TEST0, FRESH0 = "2026-10-01", "2026-11-11"
QUIET, LOUD, LOOK_D, HALF_SLOTS, MIN_N = 0.7, 1.5, 30, 3, 40


def usual(d: pd.DataFrame) -> np.ndarray:
    """U(τ): 같은 날 유형(평일 세션 / 주말·휴장 세션) · 같은 동부 날짜 줄 안의 ±15분 칸 · 앞선 30일 날짜들의 RV60 중앙값. 40개 미만이면 NaN."""
    et = pd.to_datetime(d.tau.to_numpy(), unit="s", utc=True).tz_convert(NY)
    date = et.normalize().tz_localize(None); typ = (d.sess.to_numpy() >= 4).astype(int); slot = (d.tod.to_numpy() // 5).astype(int)
    U = np.full(len(d), np.nan)
    for t in (0, 1):
        m = typ == t
        pv = pd.DataFrame({"date": date[m], "slot": slot[m], "rv": d.rv.to_numpy()[m]}).pivot_table(index="date", columns="slot", values="rv", aggfunc="first")
        pv = pv.reindex(index=pd.DatetimeIndex(sorted(set(date[m]))), columns=range(288))   # 값이 전부 빈 날도 줄을 남긴다(빠지면 그날 τ 가 다른 날의 U 를 집는다 -- selftest ③)
        A, dates = pv.to_numpy(), pv.index
        W = np.full((len(A), 288, 2 * HALF_SLOTS + 1), np.nan)                  # 줄(날짜) × 칸 × 이웃 칸(자정을 넘겨 감지 않는다)
        for j, o in enumerate(range(-HALF_SLOTS, HALF_SLOTS + 1)):
            lo, hi = max(0, -o), min(288, 288 - o)
            W[:, lo:hi, j] = A[:, lo + o:hi + o]
        out = np.full(A.shape, np.nan)
        for i, D in enumerate(dates):
            prior = (dates < D) & (dates >= D - pd.Timedelta(days=LOOK_D))        # 앞선 날만 → 창은 τ 전에 끝났다
            if prior.sum() == 0:
                continue
            V = W[prior].transpose(1, 0, 2).reshape(288, -1)
            n = np.isfinite(V).sum(1)
            with np.errstate(all="ignore"):
                out[i] = np.where(n >= MIN_N, np.nanmedian(V, axis=1), np.nan)
        U[m] = out[dates.get_indexer(date[m]), slot[m]]
    return U


def table(d: pd.DataFrame, beta60: np.ndarray) -> pd.DataFrame:
    """채점 표: 중앙값형 예측 m̂ · 평소 U · ρ(모델)·ρ_p(지속성) · 실제 큼."""
    X = S.har_x(d)
    return d.assign(mhat=np.exp(X @ beta60), U=usual(d)).assign(
        rho=lambda x: x.mhat / x.U, rho_p=lambda x: np.exp(x.lrv60) * np.sqrt(60) / x.U, big=lambda x: x.rv > x.U)


def prec(flag: np.ndarray, hit: np.ndarray, day: np.ndarray, rng) -> dict:
    """정확도 = 맞은 수 ÷ 낸 수, 달력일 블록 부트스트랩 단측 90% 하한."""
    g = pd.DataFrame({"d": day, "f": flag.astype(float), "h": (flag & hit).astype(float)}).groupby("d")[["f", "h"]].sum().to_numpy()
    b = g[rng.integers(0, len(g), (NB, len(g)))].sum(1)
    with np.errstate(all="ignore"):
        ps = b[:, 1] / b[:, 0]
    n, h = int(flag.sum()), int((flag & hit).sum())
    return {"건수": f"{h}/{n}", "정확도": h / n if n else None, "하한90": float(np.nanpercentile(ps, 10)) if n else None, "발동": n / len(flag)}


def score(T: pd.DataFrame, rng) -> dict:
    T = T[(T.sess != 5) & T.rv.notna() & T.U.notna() & T.rho.notna() & T.rho_p.notna()]
    day, big = (T.tau // 86400).to_numpy(), T.big.to_numpy()
    out = {"채점_tau": len(T), "날": int(len(np.unique(day))), "실제_큼_비율": float(big.mean()),
           "rho_분위(10·50·90)": [float(x) for x in np.nanpercentile(T.rho, [10, 50, 90])]}
    sc = {}
    for nm, r in (("모델", T.rho.to_numpy()), ("지속성", T.rho_p.to_numpy())):
        loud, quiet = r >= LOUD, r <= QUIET
        out[nm] = {"격함": prec(loud, big, day, rng), "조용": prec(quiet, ~big, day, rng)}
        sc[nm] = np.where(loud, np.where(big, 1.0, -1.0), 0.0) + np.where(quiet, np.where(~big, 1.0, -1.0), 0.0)
    dsc = sc["모델"] - sc["지속성"]
    g = pd.DataFrame({"d": day, "x": dsc}).groupby("d").x.agg(["sum", "size"]).to_numpy()
    b = g[rng.integers(0, len(g), (NB, len(g)))]
    out["맞힘점수"] = {"모델": float(sc["모델"].mean()), "지속성": float(sc["지속성"].mean()), "차": float(dsc.mean()),
                   "차_하한90": float(np.percentile(b[:, :, 0].sum(1) / b[:, :, 1].sum(1), 10))}
    out["세션별"] = {}
    for s in range(5):
        m = (T.sess == s).to_numpy()
        if m.sum():
            r = T.rho.to_numpy()[m]; lo, qu = r >= LOUD, r <= QUIET
            out["세션별"][S.SESS[s]] = {"n": int(m.sum()), "격함": f"{int((lo & big[m]).sum())}/{int(lo.sum())}", "조용": f"{int((qu & ~big[m]).sum())}/{int(qu.sum())}"}
    return out


def verdict(r: dict) -> dict:
    lo, qu = r["모델"]["격함"], r["모델"]["조용"]
    c3 = lo["발동"] >= 0.05 and qu["발동"] >= 0.05
    return {"1_격함_하한≥75": bool(lo["하한90"] is not None and lo["하한90"] >= 0.75), "2_조용_하한≥75": bool(qu["하한90"] is not None and qu["하한90"] >= 0.75),
            "3_발동≥5%": bool(c3), "통과": bool(c3 and (lo["하한90"] or 0) >= 0.75 and (qu["하한90"] or 0) >= 0.75),
            "통과_후보": bool(c3 and (lo["정확도"] or 0) >= 0.75 and (qu["정확도"] or 0) >= 0.75), "모델_몫": bool(r["맞힘점수"]["차_하한90"] > 0),
            "누수_의심": bool(max(lo["정확도"] or 0, qu["정확도"] or 0) > 0.99 or (lo["정확도"] or 0) - (r["지속성"]["격함"]["정확도"] or 0) > 0.20)}


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    beta60 = np.array(json.loads(PARAMS.read_text())["coins"]["SOXL"]["beta60"])
    k = C.load(); T = table(S.frame(k, 60), beta60); rng = np.random.default_rng(SEED)
    t0, f0 = (pd.Timestamp(x, tz="UTC").timestamp() for x in (TEST0, FRESH0))
    allrows = T[T.tau >= t0]
    res = {"data_last": str(pd.to_datetime(k.t.iloc[-1], unit="ms")), "시험_tau_전체": len(allrows), "U_없음": int(allrows.U.isna().sum()), "휴장_제외": int((allrows.sess == 5).sum()),
           "시험": score(allrows, rng)}
    res["판정"] = verdict(res["시험"])
    res["새_구간(11-11~)"] = score(T[T.tau >= f0], rng)
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False, indent=1))


def selftest() -> None:
    ts = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())
    rng = np.random.default_rng(3); n = 60 * 1440
    t = ts("2026-06-01 00:00") * 1000 + np.arange(n) * 60000
    dayi = np.arange(n) // 1440
    lv = np.zeros(60)                                                          # 날마다 변동성이 느리게 바뀐다(AR1) -- 계수가 지속성을 배우게
    for i in range(1, 60):
        lv[i] = 0.8 * lv[i - 1] + rng.normal(0, 0.25)
    lv[50:] = np.log(3.0)                                                      # 마지막 10일은 평소의 세 배
    vol = 4e-4 * np.exp(lv)[dayi]
    c = 60 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol)); k = pd.DataFrame({"t": t, "h": c * 1.0002, "l": c * 0.9998, "c": c, "v": 1.0})
    d = S.frame(k, 60); U = usual(d)
    assert np.isfinite(U).mean() > 0.4
    # ① 미래를 안 본다: 45일째 정오 뒤 봉을 쓰레기로 바꿔도 그 전 τ 의 U 는 같다
    cut = ts("2026-07-16 12:00"); k2 = k.copy(); m2 = k2.t >= cut * 1000
    k2.loc[m2, "c"] = k2.c[m2] * np.exp(np.cumsum(rng.normal(0, 5e-3, int(m2.sum())))); k2["h"], k2["l"] = k2.c * 1.0002, k2.c * 0.9998
    d2 = S.frame(k2, 60); U2 = usual(d2); before = (d.tau <= cut).to_numpy()
    assert np.allclose(U[before], U2[before], equal_nan=True), "평소 U 가 미래를 본다"
    # ③ 같은 날 제외: 그날의 RV 를 지워도 그날 U 는 같다
    d3 = d.copy(); day35 = (d3.tau >= ts("2026-07-06 04:00")) & (d3.tau < ts("2026-07-07 04:00")); d3.loc[day35, "rv"] = np.nan
    U3 = usual(d3); same = ((d.tau >= ts("2026-07-06 04:05")) & (d.tau < ts("2026-07-06 23:00"))).to_numpy()
    assert np.allclose(U[same], U3[same], equal_nan=True), "같은 날 값이 평소에 들어갔다"
    # ② 변동성 두 배 구간: 지속성·모델 모두 격함을 내고 실제도 평소보다 크다(평소가 아직 옛 수준인 첫 며칠)
    fit = d[(d.tau < ts("2026-07-15")) & d.rv.notna()].dropna(subset=S.HAR_COLS)
    beta = np.linalg.lstsq(S.har_x(fit), np.log(np.maximum(fit.rv, S.RV_FLOOR)), rcond=None)[0]
    T = table(d, beta); hi = T[(T.tau >= ts("2026-07-22 00:00")) & (T.tau < ts("2026-07-25 00:00")) & T.U.notna() & T.rho.notna() & T.rv.notna()]
    assert len(hi) > 200 and (hi.rho >= LOUD).mean() > 0.6 and hi.big[hi.rho >= LOUD].mean() > 0.9, ((hi.rho >= LOUD).mean(), hi.big[hi.rho >= LOUD].mean())
    lo = T[(T.tau >= ts("2026-07-01")) & (T.tau < ts("2026-07-15")) & T.U.notna() & T.rho.notna() & T.rv.notna()]
    assert (lo.rho >= LOUD).mean() < (hi.rho >= LOUD).mean() / 2 and abs(lo.big.mean() - 0.5) < 0.2   # 평소 구간: 격함이 훨씬 드묾 · 기준 비율 약 절반
    r = score(T[T.tau >= ts("2026-07-01")], np.random.default_rng(0)); v = verdict(r)
    assert r["모델"]["격함"]["정확도"] > 0.9 and isinstance(v["통과"], bool)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "run"])
    {"selftest": selftest, "run": run}[ap.parse_args().cmd]()
