"""ETH 4h 방향 HGB «조기 종료 무작위 10% 검증» 문제 사전등록 검정 (2026-10-19, 로컬 CPU 만 -- 서버·거래소·GPU 0).

배경(10-19 리뷰): 라이브 HGB 는 early_stopping='auto' → 1만 행 초과면 sklearn 이 «무작위» 10% 를 검증으로 뗀다.
  그 행은 학습 행과 5분 이웃·4h 라벨 겹침이라 검증 손실이 계속 내려가 n_iter_ 가 늘 300(멈춘 적 없음) · 시드는 어떤 10% 를
  버리느냐만 바꿔 같은 달 시드 간 예측 상관 .84~.89. 시간순 홀드아웃 최소는 4h(MSE) 기준 반복 ~2.

PRE-REGISTRATION (copy verbatim):
  Period: monthly walk-forward 2023-01..2026-09 (live train(): window prior 365d, cut = month − 4h, rows [::2]); main = 2025-01..2026-09, secondary = 2023-24.
  Arms:
    A  = current live: HGB as-is (early_stopping auto), seeds 0,1,2 averaged.
    B2 = early_stopping=False, max_iter=300, one model (deterministic; seeds irrelevant) — "same capacity, all training rows, no random holdout".
    B1 = early_stopping=False, n_iter chosen per month by CHRONOLOGICAL validation: inside the training window, val = last 30 days, train-for-selection = rows ending 4h before val start; grid n_iter ∈ {25, 50, 100, 200, 300}; criterion = Pearson IC on val; then refit on the full window with the chosen n_iter.
  Signals: live rule = past-OOS 20/80 quantiles fixed within month (month_fixed_q) of EACH arm's own OOS predictions (each arm's own history from 2023-01; first ~month without thresholds = no trades, same for all arms).
  Execution: taker 4bp, 8 phases, 5-day block bootstrap.
  Decision rule:
    Adopt B1 iff 2025~ daily PnL B1 − A CI lower > 0 AND 2023-24 mean diff > 0 AND 2025~ IC(B1) ≥ IC(A).
    Else adopt B2 (explicit early_stopping=False, deterministic) iff 2025~ B2 − A mean ≥ −1bp/day AND CI includes or exceeds 0 AND 2023-24 mean diff ≥ −1bp (i.e. not worse; benefit = determinism/transparency).
    Else keep A unchanged.
  Report also: IC per year, signal rates, chosen n_iter distribution for B1, seed sensitivity of A (pairwise corr of seeds 0/1/2 per month, mean), runtime per month per arm.

구현 정의(결과 전 고정):
  A  = research_eth_dir4h_reverify_20261019 의 refit() 캐시(pred_s 시드 0·1·2 = 라이브 train() 적합 그대로) 평균. 시간 = 3달 표본 재적합.
  달 m: cut = m − 4h · 학습 창 tr = [cut − 365일, cut) · 학습 행 X[tr][::2] (라이브와 같음).
  B1 선택: vs = cut − 30일 · 검증 val = [vs, cut)(모든 행) · 선택 학습 sel = [cut − 365일, vs − 4h)[::2]
     → 선택 학습 행 라벨(봉 t 종가→t+48 종가, 끝 = ts+4h+5분) 이 vs 이전에 끝나고, val 라벨은 m(테스트 첫 봉) 전에 끝난다.
     선택은 학습 창 안 행만 본다(cut 이후 행 0). 재적합 = 전체 tr[::2] 에 n_iter = 선택값.
     early_stopping=False 면 무작위 요소가 없다(5.2만 행 < 분할 표본 20만) → «max_iter=n 적합» = «max_iter=300 적합의 staged_predict n번째»
     (selfcheck 에서 같음 확인) → B1 예측 = B2 모델의 n 번째 단계 예측(재적합 비용 절약, 결과 동일).
  IC = 피어슨(모든 봉, 세 판 모두 유한한 행) · 연도별 · IC 차 CI = 일 충분통계 5일 블록.

결과(10-19 실행, 4bp · 위상 8 · 5일 블록 CI · 일 손익 bp):
  A 재현: 표본 3달 재적합 = 캐시 상관 1.000000 · n_iter_ 9/9 = 300(멈춘 적 없음 확인).
              2025~                     2023-24              IC 2025~  IC 23-24  신호 2025~ 롱/숏
  A   +18.7[−0.1,+39.9] 샤프 1.60   −5.5[−17.6,+6.3]       .0633     .0120     22.8% / 23.7%
  B2  +22.9[+4.6,+43.4] 샤프 1.99   −4.1[−16.0,+7.8]       .0667     .0162     23.4% / 23.4%
  B1  +17.2[+0.5,+34.9] 샤프 1.62   −8.5[−21.1,+3.6]       .0632     .0028     17.8% / 21.2%
  짝 차: B1−A 2025~ −1.57[−12.1,+7.9] · 2023-24 −3.02 → B1 불채택(세 조건 모두 미달).
         B2−A 2025~ +4.12[−1.66,+9.64] · 2023-24 +1.46[−3.6,+6.2] · IC 차 +.0035[−.005,+.012] → B2 조건 충족(이득 구분 불가 = 나쁘지 않음).
         B1−B2 2025~ −5.69[−14.3,+2.4].
  연도별 IC A/B2/B1: 2023 −.005/−.002/−.005 · 2024 .018/.020/.000 · 2025 .084/.088/.078 · 2026 .028/.030/.030.
  B1 선택 n_iter 45달: 25×10 · 50×6 · 100×6 · 200×5 · 300×18 (달마다 널뜀 = 30일 val IC 잡음) · val IC 평균 n별 .021/.028/.033/.037/.035.
  A 시드 쌍 상관(달별, 0·1·2) 평균 .922 [.860~.966].  시간/달: A 3시드 4.8~6.5초 · B2 1.9초 · B1 선택 1.9초(+재적합 ≤1.9).
  ⇒ 판정 = B2 채택(early_stopping=False 명시·결정적·시드 1개). 손익 이득 주장은 아님(CI 0 포함).

    nice -n 15 python scripts/research_eth_dir4h_early_stop_20261019.py              # 캐시 tmp/dir4h_early_stop_20261019/fits.npz
    nice -n 15 python scripts/research_eth_dir4h_early_stop_20261019.py --selfcheck
플래그: fresh_forward_bar_by_bar=true(월별 재학습 표본 밖 예측) · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_dir4h_20261015 import FEATS, HGB, H  # noqa: E402
HGB = {**HGB, "early_stopping": "auto"}   # 10-19 이전 라이브(A) 설정 고정 -- 라이브는 B2(early_stopping=False)로 바뀌었다
from research_eth_dir4h_reverify_20261019 import (BD, PERIODS, block_ci, diff, ic_diff_ci, ic_stats, month_fixed_q,  # noqa: E402
                                                  pearson, phase_daily, rates, summ)

OUT = ROOT / "tmp/dir4h_early_stop_20261019"
REFIT = ROOT / "tmp/dir4h_reverify_20261019/refit.npz"
GRID = (25, 50, 100, 200, 300)
HGB_NOES = {**HGB, "early_stopping": False}
PURGE = np.timedelta64(5 * H, "m")


def split(ts: np.ndarray, m: pd.Timestamp):
    """(tr, sel, val, te) 불리언. tr = 라이브 학습 창 · sel/val = B1 시간순 선택 · te = 그 달."""
    cut = np.datetime64(m) - PURGE
    w0, vs = cut - np.timedelta64(365, "D"), cut - np.timedelta64(30, "D")
    tr = (ts >= w0) & (ts < cut)
    sel = (ts >= w0) & (ts < vs - PURGE)
    val = (ts >= vs) & (ts < cut)
    te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
    return tr, sel, val, te


def staged(mdl, X) -> dict[int, np.ndarray]:
    return {i + 1: p for i, p in enumerate(mdl.staged_predict(X)) if i + 1 in GRID}


def pick_n(Xs, ys, Xv, yv) -> tuple[int, dict]:
    """선택 학습 → val 피어슨 IC 최대 n_iter. 인자로 받은 행만 본다."""
    mdl = HistGradientBoostingRegressor(**HGB_NOES).fit(Xs, ys)
    ics = {n: float(np.corrcoef(p, yv)[0, 1]) for n, p in staged(mdl, Xv).items()}
    return max(ics, key=ics.get), ics


def fits(P: pd.DataFrame, months) -> dict:
    f = OUT / "fits.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    ts, y, X = P.ts.to_numpy(), P.y_4h.to_numpy(), P[FEATS].to_numpy(dtype=np.float32)
    ok = np.isfinite(y)
    pB2, pB1 = np.full(len(ts), np.nan), np.full(len(ts), np.nan)
    nsel, ics, secs = [], [], []
    for m in months:
        tr, sel, val, te = split(ts, m)
        tr, sel, val = tr & ok, sel & ok, val & ok
        t0 = time.perf_counter()
        mdl = HistGradientBoostingRegressor(**HGB_NOES).fit(X[tr][::2], y[tr][::2])
        st = staged(mdl, X[te]); t1 = time.perf_counter()
        n, ic = pick_n(X[sel][::2], y[sel][::2], X[val], y[val]); t2 = time.perf_counter()
        pB2[te], pB1[te] = st[300], st[n]
        nsel.append(n); ics.append([ic[k] for k in GRID]); secs.append([t1 - t0, t2 - t1])
        print(f"  {m:%Y-%m} n={n:3d} val IC", {k: round(v, 3) for k, v in ic.items()}, f"{t1 - t0:.1f}s/{t2 - t1:.1f}s", flush=True)
    out = {"pB2": pB2, "pB1": pB1, "nsel": np.array(nsel), "val_ic": np.array(ics), "secs": np.array(secs)}
    np.savez(f, **out)
    return out


def time_a(P, months) -> dict:
    """A 적합 시간·n_iter_ 표본 3달(캐시 예측과 상관 확인)."""
    ts, y, X = P.ts.to_numpy(), P.y_4h.to_numpy(), P[FEATS].to_numpy(dtype=np.float32)
    R = np.load(REFIT)["pred_s"]
    out = []
    for m in (months[0], months[len(months) // 2], months[-1]):
        tr, _, _, te = split(ts, m); tr &= np.isfinite(y)
        t0 = time.perf_counter()
        mdls = [HistGradientBoostingRegressor(random_state=s, **HGB).fit(X[tr][::2], y[tr][::2]) for s in (0, 1, 2)]
        sec = time.perf_counter() - t0
        c = min(float(np.corrcoef(md.predict(X[te]), R[s, te])[0, 1]) for s, md in enumerate(mdls))
        out.append({"month": f"{m:%Y-%m}", "sec_3seeds": round(sec, 1), "n_iter_": [int(md.n_iter_) for md in mdls], "corr_vs_cache_min": round(c, 6)})
        print("  A", out[-1], flush=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(BD / "panel.parquet", columns=["ts", "y_4h"] + FEATS)
    for c in FEATS:
        P[c] = P[c].astype(np.float32)
    ts, y = P.ts.to_numpy(), P.y_4h.to_numpy()
    months = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    R = np.load(REFIT)["pred_s"]
    Fz = fits(P, months)
    ta = time_a(P, months)
    arms = {"A": R.mean(0), "B2": Fz["pB2"], "B1": Fz["pB1"]}
    rep = {"pnl": {}, "rates": {}, "diff": {}, "ic": {}, "ic_diff": {}}
    D = {}
    for nm, p in arms.items():
        lo, hi = month_fixed_q(p, ts, months)
        rep["pnl"][nm], rep["rates"][nm] = {}, {}
        for lab, a, b in PERIODS:
            D[nm, lab] = phase_daily(p, lo, hi, ts, y, a, b); rep["pnl"][nm][lab] = summ(D[nm, lab])
            rep["rates"][nm][lab] = {k: v for k, v in rates(p, lo, hi, ts, a, b).items() if k in ("long", "short")}
    for a_ in ("B1", "B2"):
        rep["diff"][f"{a_} − A"] = {lab: diff(D[a_, lab], D["A", lab]) for lab, _, _ in PERIODS}
    rep["diff"]["B1 − B2"] = {lab: diff(D["B1", lab], D["B2", lab]) for lab, _, _ in PERIODS}
    fin = np.isfinite(y) & np.all([np.isfinite(p) for p in arms.values()], 0)
    yr = pd.DatetimeIndex(ts).year
    spans = [(str(Y), (yr == Y)) for Y in (2023, 2024, 2025, 2026)] + [(lab, (ts >= np.datetime64(a)) & (ts < np.datetime64(b))) for lab, a, b in PERIODS]
    for lab, w in spans:
        m = fin & w
        day = pd.to_datetime(ts[m]).floor("D")
        S = {nm: ic_stats(p[m], y[m], day) for nm, p in arms.items()}
        rep["ic"][lab] = {nm: round(float(pearson(s.sum(0))), 4) for nm, s in S.items()}
        if lab in ("2025~", "2023-24"):
            rep["ic_diff"][lab] = {f"{k} − A": (lambda v: [round(v[0], 4), v[1]])(ic_diff_ci(S[k], S["A"])) for k in ("B1", "B2")}
    # A 시드 민감도: 달별 시드 쌍 상관
    pc = []
    for m in months:
        te = split(ts, m)[3]
        c = np.corrcoef(R[:, te]); pc.append([c[0, 1], c[0, 2], c[1, 2]])
    pc = np.array(pc)
    rep["seed_corr_A"] = {"mean": round(float(pc.mean()), 3), "min": round(float(pc.min()), 3), "max": round(float(pc.max()), 3),
                          "by_year_mean": {str(Y): round(float(pc[months.year == Y].mean()), 3) for Y in (2023, 2024, 2025, 2026)}}
    ns = Fz["nsel"]
    rep["nsel"] = {"counts": {int(k): int((ns == k).sum()) for k in GRID}, "by_month": {f"{m:%Y-%m}": int(n) for m, n in zip(months, ns)}}
    rep["val_ic_mean_by_n"] = {int(k): round(float(v), 4) for k, v in zip(GRID, Fz["val_ic"].mean(0))}
    rep["runtime_sec_per_month"] = {"A_3seeds_sample": ta, "B2_fit": round(float(Fz["secs"][:, 0].mean()), 1),
                                    "B1_select_fit": round(float(Fz["secs"][:, 1].mean()), 1),
                                    "B1_total_upper(select + full refit ≤ B2)": round(float(Fz["secs"].sum(1).mean()), 1)}
    d1, d2 = rep["diff"]["B1 − A"], rep["diff"]["B2 − A"]
    c1 = d1["2025~"]["ci"][0] > 0 and d1["2023-24"]["mean"] > 0 and rep["ic"]["2025~"]["B1"] >= rep["ic"]["2025~"]["A"]
    c2 = d2["2025~"]["mean"] >= -1 and d2["2025~"]["ci"][1] >= 0 and d2["2023-24"]["mean"] >= -1
    rep["decision"] = "B1" if c1 else ("B2" if c2 else "A")
    rep["flags"] = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False,
                    "future_rows_used_for_entry": False}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for k in ("pnl", "diff", "ic", "ic_diff", "rates", "seed_corr_A", "nsel", "val_ic_mean_by_n", "runtime_sec_per_month", "decision"):
        print(k, json.dumps(rep[k], ensure_ascii=False), flush=True)
    print("저장", OUT / "report.json")


def selfcheck() -> None:
    ts = pd.date_range("2024-01-01", "2025-03-01", freq="5min").to_numpy()
    m = pd.Timestamp("2025-02-01")
    tr, sel, val, te = split(ts, m)
    cut, vs = np.datetime64(m) - PURGE, np.datetime64(m) - PURGE - np.timedelta64(30, "D")
    lab_end = lambda t: t + PURGE + np.timedelta64(5, "m")          # 봉 t 시작 → 라벨 = t+48 봉 종가
    assert lab_end(ts[sel].max()) <= vs and ts[val].min() >= vs, "선택 학습 라벨이 val 시작 전에 끝나야"
    assert lab_end(ts[val].max()) <= np.datetime64(m) and lab_end(ts[tr].max()) <= np.datetime64(m) <= ts[te].min()
    assert not (sel & val).any() and ((sel | val) <= tr).all() and ts[tr].min() >= cut - np.timedelta64(365, "D")
    assert ts[val].max() < cut and ts[sel].max() < vs - PURGE + np.timedelta64(1, "s")
    # staged n = max_iter n 적합 · 시드 무관(early_stopping=False)
    r = np.random.default_rng(0); X = r.normal(size=(3000, 5)).astype(np.float32); yy = X[:, 0] + r.normal(size=3000)
    kw = {**HGB_NOES, "min_samples_leaf": 50}
    full = HistGradientBoostingRegressor(random_state=0, **kw).fit(X, yy)
    st = {i + 1: p for i, p in enumerate(full.staged_predict(X[:200]))}
    for n in (25, 100):
        assert np.allclose(st[n], HistGradientBoostingRegressor(random_state=0, **{**kw, "max_iter": n}).fit(X, yy).predict(X[:200]))
    assert np.array_equal(full.predict(X), HistGradientBoostingRegressor(random_state=7, **kw).fit(X, yy).predict(X)), "시드 영향 없어야"
    # pick_n: 인자 행만 사용 → 같은 입력이면 같은 선택(결정적)
    X3 = r.normal(size=(12000, 5)).astype(np.float32); y3 = X3[:, 0] + r.normal(size=12000)
    n1, ic1 = pick_n(X3[:9000], y3[:9000], X3[9000:], y3[9000:])
    assert n1 == pick_n(X3[:9000], y3[:9000], X3[9000:], y3[9000:])[0] and ic1[n1] == max(ic1.values()) and np.isfinite(list(ic1.values())).all()
    # month_fixed_q 인과: 그 달·미래 예측을 바꿔도 그 달 문턱 불변
    t2 = pd.date_range("2024-01-01", "2024-04-01", freq="5min").to_numpy(); p = r.normal(size=len(t2))
    mo = pd.date_range("2024-03-01", "2024-03-01", freq="MS")
    lo0, hi0 = month_fixed_q(p, t2, mo); p2 = p.copy(); p2[t2 >= np.datetime64("2024-03-01")] = 1e6
    lo1, hi1 = month_fixed_q(p2, t2, mo)
    assert np.array_equal(lo0, lo1, equal_nan=True) and np.array_equal(hi0, hi1, equal_nan=True) and np.isfinite(lo0[(t2 >= np.datetime64("2024-03-01")) & (t2 < np.datetime64("2024-04-01"))]).all()
    x = r.normal(0, 10, 600); assert block_ci(x + 3)[0] > 0
    print("selfcheck ok")


if __name__ == "__main__":
    selfcheck()
    if "--selfcheck" not in sys.argv:
        main()
