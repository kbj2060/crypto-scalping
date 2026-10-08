"""리스크 규칙 2회차 (2026-10-08, /loop «리스크 관리를 극한으로»): 1회차 권고의 실원장 재생 + 온라인 α 사이징.

A. 실원장 재생 — 1회차 권고(사다리 총 명목 = 순자산 × L 고정 + 첫 진입가 −3% 전량 손절)를 실제 153왕복에.
   왕복 단위 근사: 실제 체결로 왕복 손익 r(최대 명목당)과 MAE(최대 명목당)를 다시 계산(손절판은 1분 고저가
   선을 넘은 분 시작 이전 체결만 남기고 남은 물량을 선 가격(이미 넘었으면 직전 종가)·테이커 4bp 로 청산).
   반사실 순자산: 진입 때 크기 = L × 그때 반사실 순자산, 손익은 청산 시각에 반영(입출금 무시 = NAV).
   일치 점검: 실제 L_i 로 같은 근사를 돌려 실제 NAV 수익률과 비교.
B. 온라인 α 사이징(사전 고정, 결과 보기 전) — 지난 왕복의 용량당 수익 x 로 μ̂·σ̂·SE 추정,
   L = clip(0.5 × (μ̂ − SE)/σ̂², 0, 2.25), 처음 50왕복은 L 0.5. 손절 S1. 1회차 프런티어 경로(trades_*.parquet)에
   α 0/5/10/15/20(손절 왕복 α 없음)을 넣어 K1.5 와 비교. 1년 경로마다 추정을 새로 시작(보수).
   판정(보고): «모르는 α 에 강건» = α0·α5 에서 p10 이 K1.5 보다 높고, α≥10 에서 중앙이 K1.5 의 80% 이상.
   A 에도 같은 규칙(원장 왕복 r 로 추정)을 건다.

실행: python scripts/research_risk_rule_ledger_replay_online_alpha_20261008.py [--selftest]
산출: tmp/risk_rule_replay_20261008/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_risk_frontier_ledger_templates_20261008 as F  # noqa: E402

A = F.A
OUT = ROOT / "tmp/risk_rule_replay_20261008"
STOP, TAKER = 0.03, 4e-4
KELLY_FRAC, L_MAX, WARM_N, WARM_L = 0.5, 2.25, 50, 0.5


def trip_stats(g: pd.DataFrame, k1: pd.DataFrame, stop: bool, t_end: int | None = None,
               stop_pct: float = STOP) -> tuple[float, float, float]:
    """한 왕복 체결 g(시간순) → (손익$, 최대 명목$, MAE$). stop 이면 첫 진입가 −3% 에서 남은 물량 청산.
    t_end: 열린 왕복의 평가 끝 분(그 분 종가로 남은 물량 평가)."""
    s = float(g.sgn.iloc[0]); p0 = float(g.price.iloc[0]); lvl = p0 * (1 - s * stop_pct)
    a, b = int(g.time.min()) // 60_000 * 60_000 + 60_000, int(t_end if t_end is not None else g.time.max())
    k = k1[(k1.t >= a) & (k1.t <= max(b, a))]
    adv = (k.l if s > 0 else k.h).to_numpy()
    ts = None
    if stop:
        hit = (adv <= lvl) if s > 0 else (adv >= lvl)
        if hit.any() and int(k.t.to_numpy()[np.argmax(hit)]) < b:
            j = int(np.argmax(hit)); ts = int(k.t.to_numpy()[j])
            prev = float(k.c.to_numpy()[j - 1]) if j > 0 else p0
            ex = min(lvl, prev) if s > 0 else max(lvl, prev)
    gg = g[g.time < ts] if ts is not None else g
    q = avg = pnl = notl = 0.0
    qs = []
    for r in gg.itertuples():
        if r.inc:
            avg = (avg * q + r.price * r.qty) / (q + r.qty); q += r.qty
        else:
            pnl += s * (r.price - avg) * r.qty; q -= r.qty
        pnl -= r.commission
        notl = max(notl, q * r.price); qs.append((r.time, q, avg))
    if ts is not None and q > 1e-9:
        pnl += s * (ex - avg) * q - TAKER * ex * q
    elif q > 1e-9 and len(k):                                  # 열린 왕복: 마지막 분 종가로 평가
        pnl += s * (float(k.c.iloc[-1]) - avg) * q
    # MAE: 분 마감 시점의 포지션으로 그 분 역행 극값 평가
    T = np.array([x[0] for x in qs]); Q = np.array([x[1] for x in qs]); AV = np.array([x[2] for x in qs])
    kt = k.t.to_numpy() + 60_000
    if ts is not None:
        keep = k.t.to_numpy() <= ts; kt, adv = kt[keep], adv[keep].copy()
        if len(adv): adv[-1] = ex
    i = np.searchsorted(T, kt, "right") - 1
    ok = i >= 0
    u = np.where(ok, Q[np.maximum(i, 0)] * s * (adv - AV[np.maximum(i, 0)]), 0.0) if len(kt) else np.zeros(1)
    return pnl, notl, float(min(u.min(), 0.0))


def online_L(x_hist: np.ndarray) -> float:
    n = len(x_hist)
    if n < WARM_N:
        return WARM_L
    mu, sd = x_hist.mean(), x_hist.std(ddof=1)
    return float(np.clip(KELLY_FRAC * (mu - sd / np.sqrt(n)) / max(sd * sd, 1e-12), 0.0, L_MAX))


def compound(trips: pd.DataFrame, lev_fn) -> dict:
    """trips: t0, t1, r(최대 명목당), mae(최대 명목당). 진입 때 크기 고정, 청산 때 반영."""
    ev = sorted([(t, 1, i) for i, t in enumerate(trips.t0)] + [(t, 0, i) for i, t in enumerate(trips.t1)])
    E, peak, mdd, size, done = 1.0, 1.0, 0.0, {}, []
    for t, kind, i in ev:
        if kind == 1:
            L = lev_fn(np.array([trips.r.iloc[j] for j in done]))
            size[i] = L * E
            low = E + size[i] * trips.mae.iloc[i]
            mdd = min(mdd, low / peak - 1)
        else:
            E += size.pop(i) * trips.r.iloc[i]; done.append(i)
            peak = max(peak, E); mdd = min(mdd, E / peak - 1)
            if E <= 0:
                return {"mult": 0.0, "mdd": -1.0}
    return {"mult": E, "mdd": mdd}


def part_a() -> dict:
    f = A.tag_events(A.load_fills())
    st = pd.read_csv(ROOT / "tmp/behavioral_risk_20261008/trip_states.csv").set_index("trip")
    k1 = A.load_k1m("2026-07-25", "2026-10-07")
    rows = []
    for tr, g in f.groupby("trip"):
        g = g.sort_values(["time", "id"])
        te = None if st.loc[tr, "closed"] else int(st.loc[tr, "t1"])
        p0, n0, m0 = trip_stats(g, k1, False, te)
        p1, _, m1 = trip_stats(g, k1, True, te)
        rows.append({"trip": tr, "t0": st.loc[tr, "t0"], "t1": st.loc[tr, "t1"], "E0": st.loc[tr, "E0"],
                     "pnl_eng": st.loc[tr, "pnl"], "pnl0": p0, "pnl1": p1, "notl": n0, "mae0": m0, "mae1": m1})
    T = pd.DataFrame(rows).sort_values("t0").reset_index(drop=True)
    T["L_act"] = T.notl / T.E0
    closed = st.closed.reindex(T.trip).to_numpy()
    out = {"check": {"sum_pnl_engine": float(T.pnl_eng.sum()), "sum_pnl_recalc": float(T.pnl0.sum()),
                     "max_abs_diff": float((T.pnl_eng - T.pnl0).abs().max()), "closed_trips_max_abs_diff": float((T.pnl_eng - T.pnl0)[closed].abs().max()),
                     "stopped_trips": int((T.pnl1 != T.pnl0).sum()),
                     "stop_delta_usd": float((T.pnl1 - T.pnl0).sum())}}
    res = {}
    for S, pc, mc in (("S0", "pnl0", "mae0"), ("S1", "pnl1", "mae1")):
        X = T.assign(r=T[pc] / T.notl, mae=T[mc] / T.notl)
        Lq = iter(T.L_act.tolist())
        res[f"actual_L_{S}"] = compound(X, lambda h, it=Lq: next(it))
        for L in (1.0, 1.5, 2.0, 4.0):
            res[f"K{L:g}_{S}"] = compound(X, lambda h, L=L: L)
        res[f"online_{S}"] = compound(X, online_L)
    out["res"] = res
    out["stopped"] = T[T.pnl1 != T.pnl0][["trip", "pnl0", "pnl1", "L_act"]].round(2).to_dict("records")
    print(json.dumps(out["check"], indent=1))
    for k, v in res.items():
        print(f"  {k:14s} 배수 {v['mult']:.3f}  MDD {v['mdd']:+.3f}")
    print("손절된 왕복", out["stopped"])
    return out


def ledger_stop_scan() -> dict:
    """3회차: 원장 왕복에 손절 폭 3·5·8·12% — 고정 L 1.5/2/4 복리와 걸린 왕복."""
    f = A.tag_events(A.load_fills())
    st = pd.read_csv(ROOT / "tmp/behavioral_risk_20261008/trip_states.csv").set_index("trip")
    k1 = A.load_k1m("2026-07-25", "2026-10-07")
    out = {}
    for sp in (None, 0.03, 0.05, 0.08, 0.12):
        rows = []
        for tr, g in f.groupby("trip"):
            g = g.sort_values(["time", "id"])
            te = None if st.loc[tr, "closed"] else int(st.loc[tr, "t1"])
            p, _, m = trip_stats(g, k1, sp is not None, te, sp or STOP)
            p0, n0, _ = trip_stats(g, k1, False, te)          # 용량 = 원래 왕복 최대 명목(손절로 덜 찬 사다리도 같은 용량)
            rows.append({"t0": st.loc[tr, "t0"], "t1": st.loc[tr, "t1"], "r": p / n0, "mae": m / n0, "pnl": p, "pnl0": p0})
        X = pd.DataFrame(rows).sort_values("t0").reset_index(drop=True)
        hit = X.pnl != X.pnl0
        key = "none" if sp is None else f"{sp:.0%}"
        out[key] = {"stopped": int(hit.sum()), "recovered_after": int((X.pnl0[hit] > X.pnl[hit]).sum()),
                    "delta_usd_actual_size": float((X.pnl - X.pnl0).sum()),
                    **{f"K{L:g}": compound(X, lambda h, L=L: L) for L in (1.5, 2.0, 4.0)}}
        o = out[key]
        print(f"손절 {key:5s} 걸림 {o['stopped']} (그 뒤 더 나았던 {o['recovered_after']}) 실제크기 Δ$ {o['delta_usd_actual_size']:+.0f} | "
              + " | ".join(f"K{L:g} {o[f'K{L:g}']['mult']:.3f}/{o[f'K{L:g}']['mdd']:+.3f}" for L in (1.5, 2.0, 4.0)), flush=True)
    json.dump(out, open(OUT / "ledger_stop_scan.json", "w"), indent=1)
    return out


def ledger_committed() -> dict:
    """사용자 «내 원장 기준 맞아?» — 실제 왕복(진입 시각·방향·물타기·청산 시각) 그대로, 크기·손절만 바꾼 직접 재생.
    커밋 규칙(5cbcc38b build_rule_ladder): L = min(12.4×권장배수, 청산안전, 10) · 손절 첫 진입가 ∓3σ24 (진입 직전 마감 5분봉 288개).
    권고: L 1.5/2 · 손절 5%. 용량 = 원래 왕복 최대 명목(사다리가 덜 차도 같은 용량)."""
    f = A.tag_events(A.load_fills())
    st = pd.read_csv(ROOT / "tmp/behavioral_risk_20261008/trip_states.csv").set_index("trip")
    k1 = A.load_k1m("2026-07-25", "2026-10-07")
    b = k1.assign(b=k1.t // 300_000).groupby("b").c.last()
    sg = F.sigma_day(b.to_numpy()); bt = b.index.to_numpy()
    arms = {"실제": None, "커밋": "c", "커밋+상한6": "c6", "권고1.5·5%": (1.5, 0.05), "권고2·5%": (2.0, 0.05), "권고1.5·손절없음": (1.5, None)}
    rows = []
    for tr, g in f.groupby("trip"):
        g = g.sort_values(["time", "id"])
        te = None if st.loc[tr, "closed"] else int(st.loc[tr, "t1"])
        i = np.searchsorted(bt, int(g.time.min()) // 300_000, "left") - 1          # 진입 직전 마감 5분봉
        sig = float(sg[i]); vm = float(np.clip(sig / np.sqrt(6) / F.VOLM_MED, 0.25, 3.0) / F.VOLM_REF)
        p0, n0, m0 = trip_stats(g, k1, False, te)
        side = pd.DataFrame({"sig": [sig], "vm": [vm], "side": [float(g.sgn.iloc[0])]})
        pc, _, mc = trip_stats(g, k1, True, te, 3 * sig / 1e4)
        p5, _, m5 = trip_stats(g, k1, True, te, 0.05)
        rows.append({"t0": st.loc[tr, "t0"], "t1": st.loc[tr, "t1"], "E0": st.loc[tr, "E0"], "n0": n0, "sig": sig,
                     "Lact": n0 / st.loc[tr, "E0"], "Lc": F.committed_L(side)[0], "Lc6": F.committed_L(side, 6.0)[0],
                     "p0": p0, "m0": m0, "pc": pc, "mc": mc, "p5": p5, "m5": m5})
    T = pd.DataFrame(rows).sort_values("t0").reset_index(drop=True)
    out = {}
    for name, arm in arms.items():
        if arm is None:
            pcol, mcol, Ls = "p0", "m0", T.Lact.to_numpy()
        elif arm in ("c", "c6"):
            pcol, mcol, Ls = "pc", "mc", T[{"c": "Lc", "c6": "Lc6"}[arm]].to_numpy()
        else:
            pcol, mcol = ("p5", "m5") if arm[1] else ("p0", "m0")
            Ls = np.full(len(T), arm[0])
        X = T.assign(r=T[pcol] / T.n0, mae=T[mcol] / T.n0)
        it = iter(Ls.tolist())
        res = compound(X, lambda h, it=it: next(it))
        hit = (T[pcol] != T.p0).to_numpy()
        out[name] = {**res, "stops": int(hit.sum()), "L_med": float(np.median(Ls)), "L_max": float(np.max(Ls)),
                     "worst_trip_pct": float(np.min(Ls * X.mae.to_numpy())),
                     "stop_losses_pct": [round(float(x) * 100, 1) for x in (Ls * X.r.to_numpy())[hit]]}
        o_ = out[name]
        print(f"{name:14s} 67일 배수 {o_['mult']:.3f}  MDD {o_['mdd']:+.3f}  L 중앙 {o_['L_med']:.1f} 최대 {o_['L_max']:.1f}  "
              f"손절 {o_['stops']}회 {o_['stop_losses_pct']}  왕복 중 최악 평가손 {o_['worst_trip_pct']:+.1%}", flush=True)
    print("σ24 중앙", round(float(T.sig.median())), "bp · 커밋 3σ 손절 폭 중앙", round(float(3 * T.sig.median() / 100), 1), "%")
    json.dump(out, open(OUT / "ledger_committed.json", "w"), indent=1, ensure_ascii=False)
    return out


def part_b() -> dict:
    out = {}
    for per in F.PERIODS:
        D = pd.read_parquet(F.OUT / f"trades_{per}.parquet")
        stopped = (D.f_S1 != D.f_S0).to_numpy()
        for a in (0, 5, 10, 15, 20):
            x = D.f_S1.to_numpy() + a / 1e4 * np.where(stopped, 0.0, D.fill_S1.to_numpy())
            lev = np.empty(len(D))
            for _, ix in D.groupby("path").indices.items():          # 경로마다 추정 새로 시작(누적 합으로 O(n))
                xs = x[ix]; n = np.arange(len(xs))
                c1 = np.r_[0, np.cumsum(xs)[:-1]]; c2 = np.r_[0, np.cumsum(xs * xs)[:-1]]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mu = c1 / n; var = (c2 - n * mu * mu) / (n - 1)
                    L = np.clip(KELLY_FRAC * (mu - np.sqrt(var / n)) / np.maximum(var, 1e-12), 0.0, L_MAX)
                lev[ix] = np.where(n < WARM_N, WARM_L, L)
            E = D.assign(f_S1=x)
            on = F.evaluate(E, lev, "S1", 0.0)
            k15 = F.evaluate(E, np.full(len(D), 1.5), "S1", 0.0)
            out[f"{per}_a{a}"] = {"online": on, "K1.5": k15}
            print(f"{per} α{a:2d} 온라인 중앙 {on['med_mult']:.2f} p10 {on['p10_mult']:.2f} DD50 {on['p_dd50']:.2f} "
                  f"L중앙 {on['med_lev']:.2f} p90 {on['p90_lev']:.2f} | K1.5 중앙 {k15['med_mult']:.2f} "
                  f"p10 {k15['p10_mult']:.2f} DD50 {k15['p_dd50']:.2f}", flush=True)
    return out


def selftest() -> None:
    # 롱 1 @100 → 2분 뒤 추가 1 @98 → 청산 2 @101. 1분봉 저가 97.5. 손절 −3% = 97 안 걸림.
    g = pd.DataFrame({"time": [0, 120_000, 300_000], "id": [1, 2, 3], "price": [100.0, 98.0, 101.0], "qty": [1.0, 1.0, 2.0],
                      "inc": [True, True, False], "sgn": [1, 1, 1], "commission": [0.0, 0.0, 0.0]})
    k1 = pd.DataFrame({"t": np.arange(0, 360_000, 60_000), "h": [100.5] * 6, "l": [99.5, 98.5, 97.5, 99, 100, 100.5],
                       "c": [100, 99, 98, 99.5, 100.5, 101.0]})
    p, n, m = trip_stats(g, k1, True)
    assert abs(p - 4.0) < 1e-12 and abs(n - 196.0) < 1e-12, (p, n)
    assert abs(m - 2 * (97.5 - 99)) < 1e-12, m                       # 분 2(마감 180s) 포지션 2 @ 평단 99
    # 저가 96.9 면 손절: 분 시작 120s 전 체결(1 @100)만, 97 에서 청산, 테이커 4bp
    k2 = k1.assign(l=[99.5, 98.5, 96.9, 99, 100, 100.5])
    g2 = g.assign(time=[0, 150_000, 300_000])
    p2, _, m2 = trip_stats(g2, k2, True)
    assert abs(p2 - (97 - 100 - TAKER * 97)) < 1e-12 and abs(m2 + 3.0) < 1e-12, (p2, m2)
    # 복리: 두 왕복 겹침 없이 r +10%, −10%, L 1 → 0.99 · MDD −10%
    tr = pd.DataFrame({"t0": [0, 10], "t1": [5, 15], "r": [0.1, -0.1], "mae": [0.0, -0.1]})
    c = compound(tr, lambda h: 1.0)
    assert abs(c["mult"] - 0.99) < 1e-12 and abs(c["mdd"] + 0.1) < 1e-12, c
    assert online_L(np.zeros(10)) == WARM_L and online_L(np.r_[np.full(60, 0.01), np.full(60, -0.01)]) == 0.0
    print("selftest OK -- 왕복 재계산·손절 청산·MAE·복리·온라인 L")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest(); return 0
    if "--stops" in sys.argv:
        ledger_stop_scan(); return 0
    if "--committed" in sys.argv:
        ledger_committed(); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"A_ledger": part_a()}
    if "--a-only" not in sys.argv:
        rep["B_online_sim"] = part_b()
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1, default=A._json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
