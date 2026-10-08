"""위험–수익 프런티어: 내 원장 왕복 «모양» × 4.7년 ETH 1분봉 × 재량 우위 α (2026-10-08).

질문(사용자 «리스크 관리를 극한으로 — 수익률과 위험을 동시에»): 앞선 검정들이 «고르기(필터)·게이트·손절로는
수익이 안 오르고, 꼬리는 크기에서 온다»로 수렴했다. 남은 손잡이는 «크기 수준과 크기 규칙» 하나다.
원장 67일 한 경로로는 꼬리를 못 본다(재표집은 관측 최대를 못 넘는다) — 그래서 원장의 왕복 모양(물타기 간격·
비중·보유시간)을 4.7년 실제 가격 위에 무작위 방향으로 순차 실행해 시장 꼬리를 입히고, 재량 우위는 α 로 따로 둔다.

사전 고정 (결과 보기 전, 2026-10-08)
  틀(template) = 원장 153왕복 각각의 (이벤트별 역행 거리 bp, 이벤트 비중, 보유 분). 역행 거리 < −5bp 인 추가는
    가격 도달 시(지정가, 이미 넘어 있으면 직전 종가) 체결, 나머지는 원장 시간 오프셋에 체결. 추가는 순서대로만.
  청산 = 원장 보유 시간 끝 종가(시간 청산). 비용 왕복 3bp/체결 명목. 방향 50/50(시장 중립 — 우위는 α 로만).
  재량 우위 α = 체결 명목당 bp 를 청산 때 더함. 시나리오 α ∈ {0, 5, 10, 20}. 주 = α 5
    (원장 명목가중 +4.7bp ≈ 보수적, 동일가중 +20bp 는 낙관).
  경로 = 1년 순차 걷기(무작위 시작, 다음 왕복 = 이전 청산 + 지수 간격, 연 ~834회 = 원장 빈도 2.28/일),
    경로 400개/기간. 기간 주 2025-01~2026-09 · 보조 2022-01~2024-12.
  손절 S0 없음 · S1 첫 진입가 −3% · S2 첫 진입가 −3σ24(일간 σ = √Σ 288개 5분 로그수익²).
  크기 규칙 (L = 모든 이벤트가 찼을 때 명목 ÷ 순자산):
    K(L) 고정 L · V(L) L × 대시보드 권장 배수(크기∝σ) · B(b) 손절 시 손실 = 순자산 b (S1/S2, L ≤ 30) ·
    U 원장 실제 L 그대로(현 습관 기준선).
  파산 = 분 고저 역행 평가손으로 순자산 ≤ MMR(0.5%) × 체결 명목 → 순자산 0, 경로 끝.
  MDD = 왕복 안 최저점 포함.
  선택 규칙: 주(2025~, α5)에서 P(파산) ≤ 1% 이고 P(MDD ≥ 50%) ≤ 10% 인 것 중 중앙 1년 로그성장 최대.
  강건성: 같은 설정을 보조 기간·α 0/10/20 에 그대로 보고(재선택 금지).

실행: python scripts/research_risk_frontier_ledger_templates_20261008.py [--selftest | --kelly(사후 보조, 보수 α 레버 탐색) | --stops(3회차 손절 폭)]
산출: tmp/risk_frontier_20261008/report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_avgdown_gate_notional_cap_20261004 as A  # noqa: E402

A.ROOT, A.OUT = MAIN, MAIN / "tmp/avgdown_gate_cap_20261004"
OUT = ROOT / "tmp/risk_frontier_20261008"
SEED, P_PATHS, YEAR_MIN = 20261008, 400, 365 * 1440
COST, MMR, TRADES_YR = 3.0, 0.005, 834
PERIODS = {"main": ("2025-01-01", "2026-09-30"), "aux": ("2022-01-01", "2024-12-31")}
ALPHAS = (0.0, 5.0, 10.0, 20.0)
STOPS = ("S0", "S1", "S2", "S5", "S8", "S12", "F2", "F4")   # F2/F4 = −2/−4% (3회차 사후 경계 확인)   # S5/S8/S12 = 첫 진입가 −5/−8/−12% (3회차 추가, 같은 시드라 S0~S2 불변)
VOLM_MED, VOLM_REF = 157.3941696083474, 0.7657638410664205


def templates() -> list[dict]:
    """원장 왕복 → (dist bp, 비중, 이벤트 시간 오프셋 분, 보유 분, 실제 L)."""
    f = A.tag_events(A.load_fills())
    st = pd.read_csv(ROOT / "tmp/behavioral_risk_20261008/trip_states.csv").set_index("trip")
    out = []
    for tr, g in f.groupby("trip"):
        g = g.sort_values("time"); s = g.sgn.iloc[0]
        inc = g[g.inc].assign(pq=lambda d: d.price * d.qty)
        ev = inc.groupby("ev").agg(t=("time", "first"), qty=("qty", "sum"), pq=("pq", "sum"))
        vw = (ev.pq / ev.qty).to_numpy()
        q = np.cumsum(np.where(g.inc, g.qty, -g.qty))
        out.append({"dist": s * (vw / vw[0] - 1) * 1e4, "w": (ev.qty / ev.qty.sum()).to_numpy(),
                    "dt": (ev.t.to_numpy() - ev.t.iloc[0]) / 60_000,
                    "hold": max(1, int(round((g.time.max() - g.time.min()) / 60_000))),
                    "L": float((q * g.price).max() / st.loc[tr, "E0"])})
    return out


def run_trade(c, h, l, i0, s, tp, sig):
    """시작 분 i0(그 분 종가 진입), 방향 s, 틀 tp → 손절별 (f, m, filled, lossstop) — 모두 «용량 1» 단위."""
    H = tp["hold"]
    px = c[i0:i0 + H + 1]; adv = (l if s > 0 else h)[i0 + 1:i0 + H + 1]
    p0 = px[0]
    res = {}
    for S in STOPS:
        stop_bp = {"S0": np.inf, "S1": 300.0, "S2": 3 * sig, "S5": 500.0, "S8": 800.0, "S12": 1200.0, "F2": 200.0, "F4": 400.0}[S]
        lots, j_prev = [(p0, tp["w"][0], 0)], 0
        stop_lvl = p0 * (1 - s * stop_bp / 1e4) if np.isfinite(stop_bp) else None
        if stop_lvl is not None:                               # 손절 분(첫 진입 기준 고정선)
            hit = (adv <= stop_lvl) if s > 0 else (adv >= stop_lvl)
            js = int(np.argmax(hit)) if hit.any() else H
        else:
            js = H
        for k in range(1, len(tp["w"])):
            d = tp["dist"][k]
            if d < -5:
                lvl = p0 * (1 + s * d / 1e4)
                seg = adv[j_prev:js]
                hit = (seg <= lvl) if s > 0 else (seg >= lvl)
                if not hit.any():
                    break
                j = j_prev + int(np.argmax(hit))
                prev = px[j]                                   # 그 분 시작 직전 종가
                fill = min(lvl, prev) if s > 0 else max(lvl, prev)
            else:
                j = max(j_prev, int(round(tp["dt"][k])) - 1)
                if j >= js:
                    break
                fill = px[j + 1]
            lots.append((fill, tp["w"][k], j)); j_prev = j
        P = np.array([x[0] for x in lots]); W = np.array([x[1] for x in lots]); J = np.array([x[2] for x in lots])
        if stop_lvl is not None and js < H:
            prev = px[js]
            ex = min(stop_lvl, prev) if s > 0 else max(stop_lvl, prev)
            a = adv[:js + 1].copy(); a[-1] = ex                # 손절 분은 손절가까지만 노출
        else:
            ex, a = px[H], adv[:H]
        held = np.arange(len(a))[None, :] >= J[:, None]         # (lots, 분) — 체결 분부터 그 분 역행에 노출
        upath = (held * W[:, None] * (s * (a[None, :] / P[:, None] - 1))).sum(0)
        res[S] = (float((W * (s * (ex / P - 1))).sum() - COST / 1e4 * W.sum()),
                  float(min(upath.min(), 0.0)), float(W.sum()),
                  float((tp["w"] * np.maximum(stop_bp - np.maximum(-tp["dist"], 0), 0) / 1e4).sum()) if np.isfinite(stop_bp) else np.nan)
    return res


def sigma_day(c5: np.ndarray) -> np.ndarray:
    r = np.diff(np.log(c5), prepend=np.nan) * 1e4
    return np.sqrt(pd.Series(r * r).rolling(288).sum().to_numpy())


def walks(k1, tps, period, rng):
    t, c, h, l = k1.t.to_numpy(), k1.c.to_numpy(), k1.h.to_numpy(), k1.l.to_numpy()
    b = k1.assign(b=k1.t // 300_000).groupby("b").c.last()
    sg5 = sigma_day(b.to_numpy())
    bi = np.searchsorted(b.index.to_numpy(), t // 300_000, "left") - 1   # 마감된 마지막 5분봉
    sig_m = np.where(bi >= 0, sg5[np.maximum(bi, 0)], np.nan)
    a0, a1 = (int(np.searchsorted(t, pd.Timestamp(x, tz="UTC").value // 10**6)) for x in period)
    mean_hold = np.mean([x["hold"] for x in tps])
    gap = max(1.0, YEAR_MIN / TRADES_YR - mean_hold)
    rows = []
    for p in range(P_PATHS):
        i = int(rng.integers(a0 + 300, a1 - YEAR_MIN))
        stop_at = i + YEAR_MIN
        while True:
            tp = tps[int(rng.integers(len(tps)))]
            if i + tp["hold"] + 1 >= min(stop_at, len(c) - 1):
                break
            s = 1.0 if rng.random() < 0.5 else -1.0
            sig = sig_m[i]
            if np.isfinite(sig):
                r = run_trade(c, h, l, i, s, tp, sig)
                vm = float(np.clip(sig / np.sqrt(6) / VOLM_MED, 0.25, 3.0) / VOLM_REF)
                rows.append([p, tp["L"], vm, s, sig] + [v for S in STOPS for v in r[S]])
            i += tp["hold"] + 1 + int(rng.exponential(gap))
    cols = ["path", "Lu", "vm", "side", "sig"] + [f"{k}_{S}" for S in STOPS for k in ("f", "m", "fill", "ls")]
    return pd.DataFrame(rows, columns=cols)


def evaluate(D: pd.DataFrame, lev: np.ndarray, S: str, alpha: float) -> dict:
    """lev: 왕복별 L. 경로별 복리 + 파산 + 왕복 안 최저점 포함 MDD."""
    f, m, fill = D[f"f_{S}"].to_numpy(), D[f"m_{S}"].to_numpy(), D[f"fill_{S}"].to_numpy()
    R = lev * (f + alpha / 1e4 * fill)
    trough = lev * m
    ruin = 1 + trough <= MMR * lev * fill
    gl, md, ru = [], [], []
    for _, ix in D.groupby("path").indices.items():
        r_, tr_, rn = R[ix], trough[ix], ruin[ix]
        if rn.any():
            gl.append(np.log(1e-12)); md.append(-1.0); ru.append(True); continue
        W = np.cumprod(np.r_[1.0, 1 + r_])
        low = np.minimum(W[:-1] * (1 + tr_), W[1:])            # 왕복 안 최저점과 청산 후 값
        peak = np.maximum.accumulate(W)[:-1]
        md.append(float(np.min(low / peak - 1))); ru.append(False); gl.append(np.log(W[-1]))
    gl, md, ru = np.array(gl), np.array(md), np.array(ru)
    return {"med_mult": float(np.exp(np.median(gl))), "p10_mult": float(np.exp(np.quantile(gl, 0.10))),
            "p90_mult": float(np.exp(np.quantile(gl, 0.90))), "p_ruin": float(ru.mean()),
            "p_dd50": float((md <= -0.5).mean()), "med_mdd": float(np.median(md)),
            "med_lev": float(np.median(lev)), "p90_lev": float(np.quantile(lev, 0.9))}


def configs(D: pd.DataFrame):
    for L in (1, 2, 3, 4, 6, 8, 10, 15, 20, 30):
        for S in STOPS:
            yield f"K{L}_{S}", S, np.full(len(D), float(L))
            yield f"V{L}_{S}", S, L * D.vm.to_numpy()
    for b in (0.025, 0.05, 0.10, 0.15, 0.20, 0.30):
        for S in ("S1", "S2"):
            yield f"B{int(b * 1000)}_{S}", S, np.minimum(b / D[f"ls_{S}"].to_numpy(), 30.0)   # ponytail: L 30 상한(파산 판정은 그대로)
    for S in STOPS:
        yield f"U_{S}", S, D.Lu.to_numpy()


def main() -> int:
    if "--selftest" in sys.argv:
        selftest(); return 0
    if "--kelly" in sys.argv:
        kelly_scan(); return 0
    if "--stops" in sys.argv:
        stop_scan(); return 0
    if "--compare" in sys.argv:
        compare(); return 0
    OUT.mkdir(parents=True, exist_ok=True)
    tps = templates()
    k1 = A.load_k1m("2021-12-25", "2026-10-07")
    rep = {"templates": len(tps), "periods": {}}
    for per, rng_p in PERIODS.items():
        D = walks(k1, tps, rng_p, np.random.default_rng(SEED + (0 if per == "main" else 1)))
        D.to_parquet(OUT / f"trades_{per}.parquet")
        mkt = {S: float((D[f"f_{S}"] / D[f"fill_{S}"]).mean() * 1e4) for S in STOPS}
        print(per, "왕복", len(D), "경로당", round(len(D) / P_PATHS), "무작위 bp/체결명목", mkt, flush=True)
        res = {}
        for name, S, lev in configs(D):
            res[name] = {f"a{int(a)}": evaluate(D, lev, S, a) for a in ALPHAS}
        rep["periods"][per] = {"n_trades": len(D), "mkt_bp": mkt, "res": res}
    main_ = rep["periods"]["main"]["res"]
    ok = {k: v["a5"] for k, v in main_.items() if v["a5"]["p_ruin"] <= 0.01 and v["a5"]["p_dd50"] <= 0.10}
    best = max(ok, key=lambda k: ok[k]["med_mult"]) if ok else None
    rep["selected"] = best
    if best:
        rep["selected_all"] = {per: rep["periods"][per]["res"][best] for per in PERIODS}
    json.dump(rep, open(OUT / "report.json", "w"), ensure_ascii=False, indent=1)
    show = [k for k in main_ if k.split("_")[0] in ("K2", "K4", "K6", "K10", "K20", "V2", "V4", "V6", "B50", "B100", "B150", "B200", "U")]
    for per in PERIODS:
        for a in ("a0", "a5", "a20"):
            print(f"── {per} {a}")
            for k in show:
                r = rep["periods"][per]["res"][k][a]
                print(f"  {k:9s} 중앙 {r['med_mult']:6.2f}배 p10 {r['p10_mult']:5.2f} 파산 {r['p_ruin']:.3f} "
                      f"DD50 {r['p_dd50']:.3f} 중앙MDD {r['med_mdd']:+.2f} L중앙 {r['med_lev']:.1f}")
    print("선택", best, json.dumps(rep.get("selected_all"), indent=1))
    return 0


def kelly_scan() -> dict:
    """사후(결과 본 뒤) 보조: S1 고정 L 0.25 단위 탐색, 손절된 왕복엔 α 없음(보수). 저장된 trades_*.parquet 재사용."""
    Ls = np.round(np.arange(0.25, 8.01, 0.25), 2)
    out = {}
    for per in PERIODS:
        D = pd.read_parquet(OUT / f"trades_{per}.parquet")
        stopped = (D.f_S1 != D.f_S0).to_numpy()
        for a in (0, 5, 10, 15, 20):
            E = D.assign(f_S1=D.f_S1 + a / 1e4 * np.where(stopped, 0.0, D.fill_S1))
            r = {float(L): evaluate(E, np.full(len(E), L), "S1", 0.0) for L in Ls}
            ok = [L for L in r if r[L]["p_dd50"] <= 0.10]
            out[f"{per}_a{a}"] = {"growth_opt": max(r, key=lambda L: r[L]["med_mult"]),
                                  "dd50_le10_opt": max(ok, key=lambda L: r[L]["med_mult"]) if ok else None, "grid": r}
            print(per, a, {k: v for k, v in out[f"{per}_a{a}"].items() if k != "grid"}, flush=True)
    json.dump(out, open(OUT / "kelly_scan.json", "w"), indent=1)
    return out


def stop_scan() -> dict:
    """3회차(사전 고정): 손절 폭 S0·S1(3%)·S5·S8·S12·S2(3σ) × 고정 L × α 5/10/15(손절 왕복 α 없음).
    판정 = 각 (기간, α) 칸에서 DD50≤10% 최적 L 의 중앙 1년 배수가 가장 큰 손절 — 칸 6개 중 다수."""
    Ls = np.round(np.arange(0.25, 4.01, 0.25), 2)
    out, wins = {}, {}
    for per in PERIODS:
        D = pd.read_parquet(OUT / f"trades_{per}.parquet")
        for a in (5, 10, 15):
            cell = {}
            for S in ("S0", "F2", "S1", "F4", "S5", "S8", "S12", "S2"):
                stopped = (D[f"f_{S}"] != D.f_S0).to_numpy()
                E = D.assign(**{f"f_{S}": D[f"f_{S}"] + a / 1e4 * np.where(stopped, 0.0, D[f"fill_{S}"])})
                r = {float(L): evaluate(E, np.full(len(E), L), S, 0.0) for L in Ls}
                ok = [L for L in r if r[L]["p_dd50"] <= 0.10]
                Lc = max(ok, key=lambda L: r[L]["med_mult"]) if ok else None
                cell[S] = {"stopped_frac": float(stopped.mean()), "L_dd10": Lc,
                           **({k: r[Lc][k] for k in ("med_mult", "p10_mult", "p_dd50", "med_mdd")} if Lc else {}),
                           "at_L1.5": {k: r[1.5][k] for k in ("med_mult", "p10_mult", "p_dd50", "med_mdd")}}
                print(f"{per} α{a:2d} {S:4s} 손절률 {stopped.mean():.3f} | DD50≤10% 최적 L {Lc} 중앙 {cell[S].get('med_mult', 0):.2f} "
                      f"p10 {cell[S].get('p10_mult', 0):.2f} | L1.5 중앙 {r[1.5]['med_mult']:.2f} p10 {r[1.5]['p10_mult']:.2f} "
                      f"DD50 {r[1.5]['p_dd50']:.2f}", flush=True)
            best = max(cell, key=lambda S: cell[S].get("med_mult", 0)); wins[best] = wins.get(best, 0) + 1
            out[f"{per}_a{a}"] = {"cells": cell, "best": best}
    out["wins"] = wins
    print("칸별 최고 손절", wins)
    json.dump(out, open(OUT / "stop_scan.json", "w"), indent=1)
    return out


def committed_L(D: pd.DataFrame, cap: float = 10.0) -> np.ndarray:
    """커밋된 대시보드 규칙(5cbcc38b live_manual_peg_entry.build_rule_ladder)의 4분할 합계 배수:
    min(4 × 0.303 × 10.26 × 권장 배수, 청산가가 3σ 손절선보다 0.5% 바깥인 최대, 증거금 50% × 20배 = 10)."""
    sd, off, buf, mmr = 3 * D.sig.to_numpy() / 1e4, 0.0066, 0.005, 0.02
    lo = 1 - (1 - mmr) * (1 - sd) * (1 - buf) / (1 - off)
    sh = (1 + mmr) * (1 + sd) * (1 + buf) / (1 + off) - 1
    den = np.where(D.side.to_numpy() > 0, lo, sh)
    liq = np.where(den > 0, 1 / np.maximum(den, 1e-12), np.inf)
    return np.minimum(np.minimum(4 * 0.303 * 10.26 * D.vm.to_numpy(), liq), cap)


def compare() -> dict:
    """사후 비교(사용자 «지금 커밋된 진입크기·분할매수와 비교»): 같은 경로에서 커밋 규칙 vs 권고.
    모양 두 가지 — 원장 틀(내 실제 물타기) · 4분할 틀(0/44/88/132bp 같은 크기, 보유시간만 원장)."""
    tps = templates()
    k1 = None
    out = {}
    for geo in ("ledger", "ladder4"):
        for per, rng_p in PERIODS.items():
            fp = OUT / f"trades_{geo}_{per}.parquet"
            if geo == "ledger":
                D = pd.read_parquet(OUT / f"trades_{per}.parquet")
            elif fp.exists():
                D = pd.read_parquet(fp)
            else:
                k1 = k1 if k1 is not None else A.load_k1m("2021-12-25", "2026-10-07")
                lad = [{**t, "dist": np.array([0.0, -44, -88, -132]), "w": np.full(4, 0.25), "dt": np.zeros(4)} for t in tps]
                D = walks(k1, lad, rng_p, np.random.default_rng(SEED + 7 + (0 if per == "main" else 1)))
                D.to_parquet(fp)
            arms = {"커밋(3σ·노출맞춤·상한10)": ("S2", committed_L(D)), "커밋+상한6": ("S2", committed_L(D, 6.0)),
                    "권고 1.5배·5%": ("S5", np.full(len(D), 1.5)), "권고 2배·5%": ("S5", np.full(len(D), 2.0)),
                    "권고 1.5배·3σ": ("S2", np.full(len(D), 1.5))}
            for a in (0, 5, 10, 15):
                for name, (S, lev) in arms.items():
                    stopped = (D[f"f_{S}"] != D.f_S0).to_numpy()
                    E = D.assign(**{f"f_{S}": D[f"f_{S}"] + a / 1e4 * np.where(stopped, 0.0, D[f"fill_{S}"])})
                    r = evaluate(E, lev, S, 0.0)
                    ls = lev * D[f"ls_{S}"].to_numpy() if S != "S0" else np.full(len(D), np.nan)
                    r["med_stop_loss"] = float(np.nanmedian(ls)); r["p90_stop_loss"] = float(np.nanquantile(ls, 0.9))
                    out[f"{geo}|{per}|a{a}|{name}"] = r
                    print(f"{geo:7s} {per} α{a:2d} {name:20s} 중앙 {r['med_mult']:6.2f} p10 {r['p10_mult']:5.2f} 파산 {r['p_ruin']:.3f} "
                          f"DD50 {r['p_dd50']:.2f} 중앙MDD {r['med_mdd']:+.2f} L중앙 {r['med_lev']:4.1f} p90 {r['p90_lev']:4.1f} "
                          f"손절손실 중앙 {r['med_stop_loss']:.0%} p90 {r['p90_stop_loss']:.0%}", flush=True)
    json.dump(out, open(OUT / "compare_committed.json", "w"), indent=1, ensure_ascii=False)
    return out


def selftest() -> None:
    # 롱 틀: 0bp 50% + −100bp 50%, 보유 4분. 가격 100 → 98.6(저가 98.5) → 99 → 101 → 101
    tp = {"dist": np.array([0.0, -100.0]), "w": np.array([0.5, 0.5]), "dt": np.array([0, 1]), "hold": 4, "L": 1}
    c = np.array([100, 98.6, 99, 101, 101.0]); l = np.array([100, 98.5, 98.9, 100.5, 100.9]); h = c + 0.1
    r = run_trade(c, h, l, 0, 1.0, tp, sig=1000.0)
    f, m, fill, _ = r["S0"]
    want_f = 0.5 * (101 / 100 - 1) + 0.5 * (101 / 99 - 1) - COST / 1e4
    assert abs(f - want_f) < 1e-12 and fill == 1.0, (f, want_f)
    want_m = 0.5 * (98.5 / 100 - 1) + 0.5 * (98.5 / 99 - 1)     # 추가는 그 분 지정가 99 → 같은 분 저가 98.5 노출
    assert abs(m - want_m) < 1e-12, (m, want_m)
    assert abs(r["S1"][0] - f) < 1e-12                          # S1(−3%)은 안 걸림 → S0 과 같다
    # 숏 틀, S2 = 3σ, σ=50bp → 150bp 손절: 가격 100 → 고가 101.6 에서 손절 101.5
    tp2 = {"dist": np.array([0.0]), "w": np.array([1.0]), "dt": np.array([0]), "hold": 4, "L": 1}
    c2 = np.array([100, 101, 101.2, 100, 99.0]); h2 = np.array([100, 101.1, 101.6, 100.2, 99.2]); l2 = c2 - 0.1
    r2 = run_trade(c2, h2, l2, 0, -1.0, tp2, sig=50.0)
    assert abs(r2["S2"][0] - (-(101.5 / 100 - 1) - COST / 1e4)) < 1e-12, r2["S2"]
    assert abs(r2["S2"][1] + 0.015) < 1e-12                     # 최저점 = 손절가(분 고가 101.6 이 아님)
    assert abs(r2["S0"][0] - (-(99 / 100 - 1) - COST / 1e4)) < 1e-12
    assert abs(r2["S2"][3] - 0.015) < 1e-12                    # 손절 시 손실(용량 단위) = 150bp
    # 파산: L=60, m=−0.02 → 1−1.2 < 0 → 파산
    D = pd.DataFrame({"path": [0, 0], "f_S0": [0.01, 0.01], "m_S0": [-0.02, 0.0], "fill_S0": [1.0, 1.0]})
    e = evaluate(D, np.array([60.0, 60.0]), "S0", 0.0)
    assert e["p_ruin"] == 1.0
    e = evaluate(D, np.array([1.0, 1.0]), "S0", 0.0)
    assert abs(e["med_mult"] - 1.01 ** 2) < 1e-12 and abs(e["med_mdd"] + 0.02) < 1e-12, e
    print("selftest OK -- 추가 지정가 체결·역행 경로·손절 체결·손절 손실·파산·MDD")


if __name__ == "__main__":
    raise SystemExit(main())

