"""실원장 «행동 상태» 위험 규칙 검정 (2026-10-08, 사용자 «리스크 관리를 극한으로 — 수익과 위험을 동시에»).

이전에 닫힌 축(시장 상태 필터·물타기 게이트·손절·켈리·명목 상한)과 겹치지 않는 축: 시장이 아니라 «나의 상태».
연패·당일 손실·손실 직후 재진입·과매매·계정 낙폭·큰 손익 직후·심야 — 문헌(Coval–Shumway 손실 후 위험 증가,
일일 손실 한도, 낙폭 스로틀 Grossman–Zhou)이 권하는 규칙들이 이 원장에서 수익·위험을 바꾸는가.

사전 고정 (결과 보기 전, 2026-10-08)
  상태는 «실제» 원장 경로에서, 진입 시각 t0 이전에 닫힌 왕복만으로 계산한다(양 심볼·양 측면 합).
  날짜 = KST 자정 기준. 왕복 수익률 = 왕복 손익 / 진입 직전 순자산.
  규칙(왕복 크기 배수 s, 0 = 건너뛰기):
    R1a 직전 연패 ≥2 → 0.5      R1b 연패 ≥3 → 0
    R2a 당일 닫힌 손익 ≤ −5% → 0  R2b ≤ −10% → 0
    R3  손실 왕복 청산 30분 안 재진입 → 0.5
    R4  당일 4번째 이상 진입 → 0.5
    R5a 계정 NAV 낙폭 ≥20% → 0.5  R5b ≥30% → 0.5
    R6  직전 손실 ≥5% 이고 12시간 안 → 0
    R7  직전 이익 ≥5% 이고 12시간 안 → 0.5
    R8  KST 01~07시 진입 → 0.5
  정보 검정(주): 상태 안 − 상태 밖 평균 왕복 수익률, 진입일 클러스터 부트스트랩 95% CI.
  규칙 판정: «수익도» = Δ손익 일 클러스터 CI 하한 > 0 · «위험만» = Δ손익 CI 하한 > −10%·|실제| 이고
             NAV MDD 가 같은 배수 벡터를 무작위 왕복에 준 위약 50개의 95분위보다 좋음.
  위약 = 같은 개수·같은 배수를 무작위 왕복에 배정(시드 50).
  한계: 반사실 경로에서 상태가 달라지는 효과(손실이 줄면 연패·낙폭도 달라짐)는 무시 — «실제 흐름에 규칙이 켜졌을 곳».

실행: python scripts/research_ledger_behavioral_risk_rules_20261008.py [--selftest]
산출: tmp/behavioral_risk_20261008/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")          # 원장 체결·income 은 메인 체크아웃 tmp/ 에 있다
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_avgdown_gate_notional_cap_20261004 as A  # noqa: E402

A.ROOT, A.OUT = MAIN, MAIN / "tmp/avgdown_gate_cap_20261004"
OUT = ROOT / "tmp/behavioral_risk_20261008"
KST = 9 * 3_600_000
H = 3_600_000
SEED, N_BOOT, N_PLAC = 20261008, 4000, 50

RULES = {   # 이름: (상태 컬럼 조건, 배수)
    "R1a": (lambda d: d.streak >= 2, 0.5), "R1b": (lambda d: d.streak >= 3, 0.0),
    "R2a": (lambda d: d.day_ret <= -0.05, 0.0), "R2b": (lambda d: d.day_ret <= -0.10, 0.0),
    "R3": (lambda d: d.since_loss <= 0.5 * H, 0.5),
    "R4": (lambda d: d.n_today >= 3, 0.5),
    "R5a": (lambda d: d.dd <= -0.20, 0.5), "R5b": (lambda d: d.dd <= -0.30, 0.5),
    "R6": (lambda d: (d.prev_ret <= -0.05) & (d.since_prev <= 12 * H), 0.0),
    "R7": (lambda d: (d.prev_ret >= 0.05) & (d.since_prev <= 12 * H), 0.5),
    "R8": (lambda d: d.hour_kst.between(1, 6), 0.5),
}


def states(tr: pd.DataFrame, nav_t: np.ndarray, nav: np.ndarray, E: np.ndarray) -> pd.DataFrame:
    """tr: trip, t0, t1, closed, pnl, ret. 진입 시각 이전에 닫힌 왕복만 본다."""
    tr = tr.sort_values("t0").reset_index(drop=True).copy()
    peak = np.maximum.accumulate(nav)
    cols = {k: [] for k in ("streak", "day_ret", "since_loss", "n_today", "dd", "prev_ret", "since_prev")}
    for _, r in tr.iterrows():
        done = tr[tr.closed & (tr.t1 < r.t0)].sort_values("t1")
        s = 0
        for v in done.pnl.to_numpy()[::-1]:
            if v < 0: s += 1
            else: break
        day0 = (r.t0 + KST) // 86_400_000 * 86_400_000 - KST
        i0 = max(np.searchsorted(nav_t, day0, "right") - 1, 0)
        today = done[done.t1 >= day0]
        losses = done[done.pnl < 0]
        j = max(np.searchsorted(nav_t, r.t0, "right") - 1, 0)
        cols["streak"].append(s)
        cols["day_ret"].append(today.pnl.sum() / E[i0] if len(today) else 0.0)
        cols["since_loss"].append(r.t0 - losses.t1.iloc[-1] if len(losses) else np.inf)
        cols["n_today"].append(int(((tr.t0 >= day0) & (tr.t0 < r.t0)).sum()))
        cols["dd"].append(nav[j] / peak[j] - 1)
        cols["prev_ret"].append(done.ret.iloc[-1] if len(done) else 0.0)
        cols["since_prev"].append(r.t0 - done.t1.iloc[-1] if len(done) else np.inf)
    out = tr.assign(**cols)
    out["hour_kst"] = ((out.t0 + KST) // H) % 24
    return out


def boot_diff(ret, day, mask, rng):
    """상태 안 − 밖 평균 수익률의 진입일 클러스터 CI."""
    days = np.unique(day)
    idx = rng.integers(0, len(days), (N_BOOT, len(days)))
    by = {d: np.where(day == d)[0] for d in days}
    out = []
    for row in idx:
        ii = np.concatenate([by[days[k]] for k in row])
        m = mask[ii]
        if m.any() and (~m).any():
            out.append(ret[ii][m].mean() - ret[ii][~m].mean())
    return float(ret[mask].mean() - ret[~mask].mean()), np.quantile(out, [0.025, 0.975]).tolist()


def main() -> int:
    if "--selftest" in sys.argv:
        selftest(); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    f = A.tag_events(A.load_fills())
    inc = pd.DataFrame(map(json.loads, open(A.OUT / "income_full.jsonl")))
    wt, wc, flow = A.flows_wallet(inc)
    now = json.load(open(A.OUT / "account_now.json"))["fetched_ms"]
    t0 = int(f.time.min()) // 60_000 * 60_000
    k1 = A.load_k1m(pd.Timestamp(t0 - 5 * A.DAY, unit="ms").strftime("%Y-%m-%d"),
                    pd.Timestamp(now, unit="ms").strftime("%Y-%m-%d"))
    k1 = k1[k1.t + 60_000 <= now]
    pa = (k1, wt, wc, flow, t0, int(k1.t.max()))
    trips = f.groupby("trip").agg(t0=("time", "min"), t1=("time", "max"), sym=("symbol", "first"),
                                  ps=("positionSide", "first")).reset_index()
    trips["j"] = [A.STREAMS.index((a, b)) for a, b in zip(trips.sym, trips.ps)]
    q = f.assign(d=np.where(f.inc, f.qty, -f.qty)).groupby("trip").d.sum()
    trips["closed"] = trips.trip.map(q.abs() < 1e-9).to_numpy()
    trips.loc[~trips.closed, "t1"] = int(k1.t.max())
    base, _, mp = A.evaluate(f, trips, pa, set(), None)
    E, T = mp["a"]["E"], mp["t"]
    r, mdd0 = A.nav_mdd(E, mp["flow"]); nav = np.cumprod(1 + r)
    trips["pnl"] = trips.trip.map(base["a"]["trip"]).fillna(0.0)
    trips["E0"] = E[np.maximum(np.searchsorted(T, trips.t0, "right") - 1, 0)]
    trips["ret"] = trips.pnl / trips.E0
    trips["day"] = (trips.t0 + KST) // 86_400_000
    S = states(trips, T, nav, E)
    rng = np.random.default_rng(SEED)
    ret, day = S.ret.to_numpy(), S.day.to_numpy()
    days = np.unique(day)
    bidx = rng.integers(0, len(days), (N_BOOT, len(days)))
    by = {d: np.where(day == d)[0] for d in days}
    rep = {"data": {"trips": len(S), "closed": int(S.closed.sum()), "days": len(days),
                    "first": str(pd.Timestamp(S.t0.min(), unit="ms", tz="UTC")),
                    "last": str(pd.Timestamp(S.t0.max(), unit="ms", tz="UTC")),
                    "pnl": float(S.pnl.sum()), "nav_mdd": mdd0, "nav_ret": float(nav[-1] - 1),
                    "win_rate": float((S.pnl > 0).mean())}, "rules": {}}
    print(json.dumps(rep["data"], indent=1), flush=True)

    def run(scale):
        res, _, mpc = A.evaluate(f, trips, pa, set(), None, scale=scale)
        rc, mddc = A.nav_mdd(mpc["c"]["E"], mpc["flow"])
        d = np.array([res["c"]["trip"].get(t, 0.0) - res["a"]["trip"].get(t, 0.0) for t in S.trip])
        return res, mddc, float(np.prod(1 + rc) - 1), d

    for name, (cond, s) in RULES.items():
        m = cond(S).to_numpy(bool)
        if m.sum() == 0:
            rep["rules"][name] = {"n": 0}; print(name, "n=0"); continue
        diff, ci = boot_diff(ret, day, m, rng) if (~m).any() else (np.nan, [np.nan, np.nan])
        scale = {t: s for t in S.trip[m]}
        res, mddc, navc, d = run(scale)
        dsum = np.array([d[by[x]].sum() for x in days])
        dci = np.quantile(dsum[bidx].sum(1), [0.025, 0.975]).tolist()
        plac = []
        for sd in range(N_PLAC):
            pick = np.random.default_rng(SEED + 1 + sd).choice(S.trip.to_numpy(), int(m.sum()), replace=False)
            _, mp_, nv_, dp = run({t: s for t in pick})
            plac.append([dp.sum(), mp_, nv_])
        plac = np.array(plac)
        base_pnl = float(S.pnl.sum())
        row = {"n": int(m.sum()), "scale": s, "ret_in": float(ret[m].mean()), "ret_out": float(ret[~m].mean()),
               "win_in": float((ret[m] > 0).mean()), "diff": diff, "diff_ci": ci,
               "dpnl": float(d.sum()), "dpnl_ci": dci, "mdd": mddc, "nav": navc,
               "plac_dpnl_pct": float((plac[:, 0] < d.sum()).mean()),
               "plac_mdd_q95": float(np.quantile(plac[:, 1], 0.95)),
               "plac_mdd_pct": float((plac[:, 1] < mddc).mean()),
               "worst_trip": float(min(res["c"]["trip"].values())), "worst_mae": float(min(res["c"]["mae"].values()))}
        row["pass_profit"] = bool(dci[0] > 0)
        row["pass_risk_only"] = bool(dci[0] > -0.10 * abs(base_pnl) and mddc > row["plac_mdd_q95"])
        rep["rules"][name] = row
        print(f"{name:4s} n={row['n']:3d} 안 {row['ret_in']:+.4f} 밖 {row['ret_out']:+.4f} 차 {diff:+.4f} "
              f"CI[{ci[0]:+.4f},{ci[1]:+.4f}] · Δ$ {row['dpnl']:+7.0f} CI[{dci[0]:+.0f},{dci[1]:+.0f}] 위약 {row['plac_dpnl_pct']:.2f} · "
              f"MDD {mdd0:+.3f}→{mddc:+.3f} 위약분위 {row['plac_mdd_pct']:.2f} · NAV {navc:+.2f} · "
              f"수익 {row['pass_profit']} 위험 {row['pass_risk_only']}", flush=True)
    rep["base_worst_trip"] = float(min(base["a"]["trip"].values()))
    rep["base_worst_mae"] = float(min(base["a"]["mae"].values()))
    S.to_csv(OUT / "trip_states.csv", index=False)
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1, default=A._json)
    return 0


def selftest() -> None:
    # 왕복 4개: 손실 2연속 뒤 3번째 진입 → streak 2 · 당일 손익 −3/100 · 손실 청산 10분 뒤 → since_loss 600s
    tr = pd.DataFrame({"trip": [0, 1, 2, 3], "t0": [0, 2 * H, 3 * H + 600_000, 4 * H + 60_000],
                       "t1": [H, 3 * H, 5 * H, 6 * H], "closed": True,
                       "pnl": [-1.0, -2.0, 5.0, 1.0], "ret": [-0.01, -0.02, 0.05, 0.01]})
    T = np.arange(0, 7 * H, 60_000); E = np.full(len(T), 100.0)
    nav = np.ones(len(T)); nav[T >= 3 * H] = 0.7
    S = states(tr, T, nav, E)
    r2 = S[S.trip == 2].iloc[0]
    assert r2.streak == 2 and abs(r2.day_ret + 0.03) < 1e-12 and r2.since_loss == 600_000, r2
    assert r2.n_today == 2 and abs(r2.dd + 0.3) < 1e-12 and r2.prev_ret == -0.02
    r3 = S[S.trip == 3].iloc[0]
    assert r3.streak == 2 and r3.n_today == 3           # 2번 왕복은 아직 안 닫혔다 → 연패 유지
    assert RULES["R1a"][0](S).sum() == 2 and RULES["R5b"][0](S).sum() == 2
    print("selftest OK -- 연패·당일 손익·손실 후 경과·당일 진입 수·낙폭")


if __name__ == "__main__":
    raise SystemExit(main())
