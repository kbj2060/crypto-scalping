"""실원장 × 먼 OTM 옵션 보험 재생 (2026-10-12) — «노출이 커질 때만 산 OTM 옵션이 손절·상한보다 싸게 꼬리를 막나».

사전등록(결과 보기 전 고정, 전 팔 보고 · 골라 보고 금지):
  실제 진입·물타기·청산은 원장 그대로(재생 오차 $0). 보험은 그 위에 «얹기»만 한다.
  트리거: 총 명목(수량 × 분 종가) ≥ N × 순자산, N ∈ {3, 5, 8}. 순자산 = 왕복 시작 시점 원장 E0 + (이 팔 누적 − 실제 누적).
  보호 수량: ① 초과분 = (명목 − N·E)/가격 ETH · ② 전체 = 보유 수량 전부.
     목표 − 보유 ≥ 0.1 ETH 면 0.1 단위로 내림해 추가 매수(물타기마다 새 행사가). 노출이 줄어도 왕복 끝까지 안 판다.
  행사가: 매수 시 가격에서 5·10·15% OTM(롱=풋, 숏=콜). 만기 7·14일. 왕복이 만기를 넘기면 만기에 내재가치 정산 후
     트리거가 아직 켜져 있으면 그 분에 다시 산다(= 만기 시점 모델가 롤).
  가격: Black-Scholes r=0. IV = 직전 마감 시간 DVOL + 스큐 가산 × (|K/S−1| / 10%). 스큐 가산 ∈ {+0, +5, +10}vol pt(주 판정 +5).
     비용: 매수 IV+1pt · 매도 IV−1pt(반스프레드) · 수수료 min(0.03% × 기초 명목, 12.5% × 프리미엄) 매수·매도 각각,
     만기 정산 수수료 min(0.015% × 기초 명목, 12.5% × 내재가치). 최소 0.1 ETH.
  왕복 종료(포지션 0) → 마지막 원장 체결가로 남은 옵션 전부 모델가 매도.
  주 지표(목적): 최악 왕복 손실 % 순자산 · MDD %(순차, 왕복 안 최저 평가 = 선물 + 옵션 청산가치) · 순자산 20% 넘게 잃은 왕복 수 ·
     파산(순차 최저 순자산 ≤ 0) 여부. 보조: 총손익 · 평상시(10-06 큰 롱 제외) 손익 유지율 · 프리미엄 총지불.
  비교 기준(10-12 재생 엔진 그대로): A 실제 · S 전체 시스템 · S 고정3% · 상한 1.5배만 · 후속 «실제 크기 + 모델 손절만» ·
     «실제 크기 + 고정 3% 손절» · «실제 크기 + 첫 진입 −5% 손절» · «명목 6배 상한만».
  강건성: 10-06 큰 롱을 빼고(왕복 목록에서 제거 후 재생) 다시 순위 · 스큐 셋 각각.
한계(코드로 못 고침): 과거 실호가 없음 → 모델가. DVOL(30일) 을 7·14일 ATM 으로 씀. 옵션 손익은 Deribit 계좌 몫이라
  바이낸스 선물 증거금을 보태지 않는다(청산 방지 아님, 계좌 합산 손익만). 표본 = 원장 하나 · 대재앙 1건.
ponytail: 트리거·평가는 분 종가 기준(분 안 경로 무시), 최저 평가는 분 역행 극값에서 선물·옵션 동시 — 정밀 체결 시뮬은 실호가 생기면.

실행: python scripts/research_ledger_option_insurance_replay_20261012.py [--selftest]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp/ledger_option_insurance_20261012"
DVOL_CSV = MAIN / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv"
DVOL_API = OUT / "eth_dvol_hourly_deribit_api_20260804_20261012.csv"   # public/get_volatility_index_data 로 받은 뒷부분

GRID = {"N": (3, 5, 8), "size": ("초과분", "전체"), "otm": (0.05, 0.10, 0.15), "tenor_d": (7, 14)}
SKEWS = (0.0, 5.0, 10.0)
MAIN_SKEW = 5.0
HALF_SPREAD, FEE, FEE_CAP, SETTLE_FEE, MIN_Q = 1.0, 3e-4, 0.125, 1.5e-4, 0.1
YEAR_MS = 365 * 86_400_000


def bs(S, K, T, iv, cp):
    """r=0 Black-Scholes. cp=+1 콜·−1 풋. T 년, iv 비율. 배열 가능."""
    S, K, T, iv = (np.asarray(x, float) for x in (S, K, T, iv))
    intr = np.maximum(cp * (S - K), 0.0)
    v = iv * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(S / K) / v + v / 2
        p = cp * (S * ndtr(cp * d1) - K * ndtr(cp * (d1 - v)))
    return np.where(v > 1e-12, p, intr)


def fee(S, prem, rate=FEE):
    return np.minimum(rate * S, FEE_CAP * prem)


def iv_of(dv, skew, K, S, adj):
    return np.maximum((dv + skew * np.abs(K / S - 1) / 0.10 + adj) / 100, 0.01)


def load_dvol() -> tuple[np.ndarray, np.ndarray]:
    a = pd.read_csv(DVOL_CSV, parse_dates=["timestamp"])
    b = pd.read_csv(DVOL_API, parse_dates=["timestamp"])
    d = pd.concat([a[a.timestamp < b.timestamp.min()], b]).sort_values("timestamp")
    return (d.timestamp.astype("int64") // 10**6).to_numpy() + 3_600_000, d.close.to_numpy(float)   # 마감 시각 · 종가


def perp_path(tr, k) -> dict:
    """원장 체결 그대로 분 단위 경로(scale 1, 손절·상한 없음 = RS.sim 의 A 팔과 같은 회계)."""
    s, ev = tr["s"], tr["ev"]
    i0 = k.i(ev[0][0]); i1 = min(k.i(ev[-1][0]), len(k.t) - 1)
    q = avg = R = 0.0; ptr = 0; lastp = ev[0][2]
    Q, RR, UN, AD = [], [], [], []
    for m in range(i0, i1 + 1):
        while ptr < len(ev) and ev[ptr][0] < k.t[m] + 60_000:
            t, inc, p, qq, f, _, _ = ev[ptr]; ptr += 1; lastp = p
            if inc:
                avg = (avg * q + p * qq) / (q + qq); q += qq; R -= f * p * qq
            else:
                x = min(qq, q); R += s * (p - avg) * x - f * p * x; q -= x
        adv = k.l[m] if s > 0 else k.h[m]
        Q.append(q); RR.append(R); UN.append(s * q * (adv - avg)); AD.append(adv)
    if q > 1e-9:
        R += s * (k.c[i1] - avg) * q - 5e-4 * k.c[i1] * q; RR[-1] = R            # RS.sim 과 같은 잔량 처리
    return {"m0": i0, "q": np.array(Q), "R": np.array(RR), "un": np.array(UN), "adv": np.array(AD),
            "c": k.c[i0:i1 + 1].copy(), "tc": k.t[i0:i1 + 1] + 60_000, "pend": lastp, "pnl": R, "s": s}


def overlay(P, E, dvt, dvv, N, size, otm, tenor_d, skew) -> dict:
    """한 왕복 경로 P 위에 옵션 보험. 반환 = 옵션 순손익·최저 평가(선물+옵션)·지불 프리미엄."""
    s, cp, n = P["s"], -P["s"], len(P["q"])
    dvi = np.clip(np.searchsorted(dvt, P["tc"], "right") - 1, 0, len(dvt) - 1)
    dv = dvv[dvi]
    live, done = [], []          # [K, 만기 ms, 수량, 매수 분 idx]
    cash = prem = 0.0; wanted = False
    for j in range(n):
        c, t = P["c"][j], P["tc"][j]
        for o in [o for o in live if o[1] <= t]:                                       # 만기 정산(내재가치)
            intr = max(cp * (c - o[0]), 0.0)
            cash += o[2] * (intr - float(fee(c, intr, SETTLE_FEE))); live.remove(o); done.append(o + [j])
        q = P["q"][j]
        if q <= 1e-9:
            continue
        notional = q * c
        tgt = 0.0
        if E > 0 and notional >= N * E:                                              # 순자산 ≤ 0 = 이미 파산, 프리미엄 낼 돈 없음
            tgt = (notional - N * E) / c if size == "초과분" else q
        wanted |= tgt > 0
        buy = math.floor((tgt - sum(o[2] for o in live)) / MIN_Q + 1e-9) * MIN_Q
        if buy >= MIN_Q - 1e-12:
            K = c * (1 - s * otm)
            pr = float(bs(c, K, tenor_d / 365, iv_of(dv[j], skew, K, c, HALF_SPREAD), cp))
            cost = buy * (pr + float(fee(c, pr)))
            cash -= cost; prem += cost
            live.append([K, t + tenor_d * 86_400_000, buy, j])
    # 왕복 종료: 마지막 체결가로 남은 옵션 모델가 매도
    pe, te = P["pend"], P["tc"][-1]
    for o in live:
        bid = float(bs(pe, o[0], (o[1] - te) / YEAR_MS, iv_of(dv[-1], skew, o[0], pe, -HALF_SPREAD), cp))
        cash += o[2] * (bid - float(fee(pe, bid)))
        done.append(o + [n])
    # 최저 평가: 매 분 역행 극값에서 선물 평가 + 옵션 청산가치(매도 IV·수수료 차감) + 그때까지 옵션 현금
    val = np.zeros(n); flow = np.zeros(n)
    for K, exp, qq, jb, je in done:
        sl = slice(jb, min(je, n))
        S_ = P["adv"][sl]
        b = bs(S_, K, (exp - P["tc"][sl]) / YEAR_MS, iv_of(dv[sl], skew, K, S_, -HALF_SPREAD), cp)
        val[sl] += qq * (b - fee(S_, b))
        pr = float(bs(P["c"][jb], K, tenor_d / 365, iv_of(dv[jb], skew, K, P["c"][jb], HALF_SPREAD), cp))
        flow[jb] -= qq * (pr + float(fee(P["c"][jb], pr)))
        if je < n:                                                                     # 만기 정산 현금
            intr = max(cp * (P["c"][je] - K), 0.0)
            flow[je] += qq * (intr - float(fee(P["c"][je], intr, SETTLE_FEE)))
    path = P["R"] + P["un"] + val + np.cumsum(flow)
    total = P["pnl"] + cash
    return {"opt": cash, "trough": float(min(path.min(), total)), "prem": prem, "buys": len(done),
            "impossible": bool(wanted and not done)}


def run_overlay(trips, paths, E_of, arm) -> pd.DataFrame:
    rows, hist = [], []                                                                # hist: (청산 시각, 옵션 손익)
    for i in sorted(range(len(trips)), key=lambda i: trips[i]["t0"]):
        tr = trips[i]
        E = tr["E0"] + sum(o for t1, o in hist if t1 <= tr["t0"])
        r = overlay(paths[i], E, *E_of, *arm)
        hist.append((tr["t1"], r["opt"]))
        rows.append({"i": i, "t0": tr["t0"], "E": E, "s": tr["s"], "big": tr["big"], "real": tr["pnl"],
                     "pnl": tr["pnl"] + r["opt"], "how": "원장", "lev": tr["N"] / E, **r})
    return pd.DataFrame(rows)


def purpose(df: pd.DataFrame, E_start: float, base_nb: float) -> dict:
    eq = E_start + df.pnl.cumsum().to_numpy()
    prev = np.r_[E_start, eq[:-1]]
    low = np.minimum(prev + df.trough.to_numpy(), eq)
    peak = np.maximum.accumulate(np.r_[E_start, eq])[:-1]
    ok = df.E.to_numpy() > 1.0                                                        # 순자산 ≤ 0 인 왕복은 % 가 정의 안 됨 → 제외
    loss_pct = 100 * df.pnl.to_numpy()[ok] / df.E.to_numpy()[ok]
    nb = df[~df.big].pnl.sum()
    j = int(np.argmax(low <= 0))
    return {"최악 왕복 %E": round(float(loss_pct.min()), 1), "MDD %": round(float(100 * np.min((low - peak) / peak)), 1),
            "−20%E 넘는 왕복": int((loss_pct < -20).sum()), "파산": bool(low.min() <= 0),
            "첫 파산 왕복": pd.to_datetime(df.t0.iloc[j], unit="ms").strftime("%m-%d %H:%M") if low.min() <= 0 else "",
            "순자산≤0 왕복": int((~ok).sum()), "10-06 $": round(float(df[df.big].pnl.sum()), 1),
            "총손익 $": round(float(df.pnl.sum()), 1), "평상시 손익 $": round(float(nb), 1),
            "평상시 유지 %": round(float(100 * nb / base_nb), 1),
            "프리미엄 $": round(float(df.get("prem", pd.Series([0.0])).sum()), 1)}


def baselines(RS, trips, k, safe, bidx) -> dict:
    """10-12 재생 엔진 그대로의 비교 팔. E 는 RS.run_arm 과 같은 규칙."""
    out = {}
    for a in ["A 실제", "S 전체 시스템", "S 고정3% 손절(모델 대신)", "상한 1.5배만"]:
        out[a] = RS.run_arm(trips, k, a, safe, bidx)
    follow = {"실제 크기 + 모델 손절만": lambda tr, E: RS.sim(tr, k, E, d=RS.d_at(safe, bidx, tr["t0"], tr["s"])),
              "실제 크기 + 고정 3% 손절": lambda tr, E: RS.sim(tr, k, E, d=0.03),
              "실제 크기 + 첫 진입 −5% 손절": lambda tr, E: RS.sim(tr, k, E, d=0.05),
              "명목 6배 상한만": lambda tr, E: RS.sim(tr, k, E, cap=6.0)}
    for a, f in follow.items():
        rows, done = [], []
        for i in sorted(range(len(trips)), key=lambda i: trips[i]["t0"]):
            tr = trips[i]
            E = max(tr["E0"] + sum(b - x for t1, x, b in done if t1 <= tr["t0"]), 1.0)
            r = f(tr, E); done.append((tr["t1"], tr["pnl"], r["pnl"]))
            rows.append({"i": i, "t0": tr["t0"], "E": E, "s": tr["s"], "big": tr["big"], "real": tr["pnl"], **r})
        out[a] = pd.DataFrame(rows)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MAIN / ".claude/worktrees/order-flow-five-strategies-735495/scripts"))
    import research_ledger_risk_system_replay_20261012 as RS  # noqa: E402  10-12 재생 엔진(G·W·sim·run_arm·risk_model)
    d1 = RS.G.load_1m(); k = RS.W.K(d1)
    trips = RS.W.load_trips(k)
    E_start = min(trips, key=lambda t: t["t0"])["E0"]
    big = [t for t in trips if t["big"]]
    print(f"왕복 {len(trips)} · 실제 합 ${sum(t['pnl'] for t in trips):,.2f} · 시작 E ${E_start:.1f} · 큰 롱 {len(big)}", flush=True)
    dvt, dvv = load_dvol()
    t_min = min(t["t0"] for t in trips); t_max = max(t["t1"] for t in trips)
    assert dvt[0] < t_min and dvt[-1] >= t_max, ("DVOL 범위 부족", dvt[0], dvt[-1], t_min, t_max)
    paths = [perp_path(t, k) for t in trips]
    err = max(abs(p["pnl"] - t["pnl"]) for p, t in zip(paths, trips))
    print("선물 경로 재생 오차 $", round(err, 6), flush=True)

    safe, bidx, mchk = RS.risk_model(RS.bars5(d1))
    base = baselines(RS, trips, k, safe, bidx)
    base_nb = float(base["A 실제"][~base["A 실제"].big].pnl.sum())
    rep = {"시작 순자산 $": round(E_start, 1), "왕복 수": len(trips), "선물 경로 재생 오차 $": round(err, 6),
           "A 재생 오차 $(RS)": round(float((base["A 실제"].pnl - base["A 실제"].real).abs().max()), 4),
           "리스크 모델": mchk, "기준": {}, "보험": [], "10-06 제외": {"기준": {}, "보험": []}}
    for a, df in base.items():
        rep["기준"][a] = {**purpose(df, E_start, base_nb), **{x: RS.summarize(df, E_start)[x] for x in ("MDD %(순차)", "총손익 $")}}
        print(a, rep["기준"][a], flush=True)

    keep = [i for i, t in enumerate(trips) if not t["big"]]
    trips_nb = [trips[i] for i in keep]; paths_nb = [paths[i] for i in keep]
    base_nb_df = baselines(RS, trips_nb, k, safe, bidx)
    for a, df in base_nb_df.items():
        rep["10-06 제외"]["기준"][a] = purpose(df, E_start, base_nb)
    arms = list(itertools.product(GRID["N"], GRID["size"], GRID["otm"], GRID["tenor_d"]))
    for skew in SKEWS:
        for N, size, otm, ten in arms:
            arm = (N, size, otm, ten, skew)
            df = run_overlay(trips, paths, (dvt, dvv), arm)
            name = {"N": N, "크기": size, "OTM %": int(otm * 100), "만기 d": ten, "스큐 +pt": skew}
            row = {**name, **purpose(df, E_start, base_nb), "옵션 순손익 $": round(float(df.opt.sum()), 1),
                   "10-06 옵션 손익 $": round(float(df[df.big].opt.sum()), 1), "매수 왕복": int((df.buys > 0).sum()),
                   "0.1ETH 미만 불가 왕복": int(df.impossible.sum())}
            rep["보험"].append(row)
            dfn = run_overlay(trips_nb, paths_nb, (dvt, dvv), arm)
            rep["10-06 제외"]["보험"].append({**name, **purpose(dfn, E_start, base_nb)})
        print(f"스큐 +{skew} 완료", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    T = pd.DataFrame(rep["보험"]); T.to_csv(OUT / "arms.csv", index=False)
    pd.DataFrame(rep["10-06 제외"]["보험"]).to_csv(OUT / "arms_drop1006.csv", index=False)
    plot(rep)
    return 0


EN = {"A 실제": "A actual", "S 전체 시스템": "S full system", "S 고정3% 손절(모델 대신)": "S fixed 3% stop",
      "상한 1.5배만": "cap 1.5x", "실제 크기 + 모델 손절만": "actual size + model stop",
      "실제 크기 + 고정 3% 손절": "actual size + 3% stop", "실제 크기 + 첫 진입 −5% 손절": "actual size + 5% stop",
      "명목 6배 상한만": "cap 6x"}


def plot(rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = pd.DataFrame(rep["보험"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a_i, (ycol, lab) in enumerate((("최악 왕복 %E", "worst trip loss, % equity"), ("MDD %", "max drawdown %"))):
        for sk, mk in zip(SKEWS, ("o", "s", "^")):
            t = T[T["스큐 +pt"] == sk]
            ax[a_i].scatter(t["평상시 유지 %"], t[ycol], marker=mk, s=18, alpha=.6, label=f"option insurance, skew +{sk:g}")
        for nme, r in rep["기준"].items():
            ax[a_i].scatter(r["평상시 유지 %"], r[ycol], marker="X", s=70, color="k")
            ax[a_i].annotate(EN.get(nme, nme), (r["평상시 유지 %"], r[ycol]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[a_i].set_xlabel("normal-time profit kept vs actual (%, excl. 10-06)"); ax[a_i].set_ylabel(lab)
        ax[a_i].axhline(-20 if a_i == 0 else -100, lw=.6, ls=":", color="r"); ax[a_i].grid(alpha=.3)
        ax[a_i].set_ylim(-420, 5)                                                       # 파산 후 % 가 터지는 팔은 잘림(표 참고)
    ax[0].legend(fontsize=7)
    fig.suptitle("ETH ledger 154 trips: OTM option insurance (36 arms x 3 skews) vs stop/cap baselines (X)")
    fig.tight_layout(); fig.savefig(OUT / "frontier.png", dpi=130)


def selftest():
    # ① BS 알려진 값: S=K=100, T=1, σ=20% → 콜 7.9656 · ② 풋-콜 패리티(r=0) C − P = S − K
    c = float(bs(100, 100, 1, 0.2, 1)); assert abs(c - 7.96557) < 1e-4, c
    for K in (80, 100, 125):
        assert abs(float(bs(100, K, 0.3, 0.6, 1) - bs(100, K, 0.3, 0.6, -1)) - (100 - K)) < 1e-9
    # ③ 수수료 상한: 싼 옵션은 12.5% × 프리미엄, 비싼 옵션은 0.03% × 기초
    assert abs(float(fee(2500, 0.4)) - 0.05) < 1e-12 and abs(float(fee(2500, 100)) - 0.75) < 1e-12
    # ④ 롤 회계: 롱 10 ETH, 20일 왕복, 만기 7일 → 7·14일 만기에 내재가치 정산 후 재매수, 마지막에 모델가 매도
    day = 1440
    n = 20 * day + 1
    c_ = np.r_[np.full(6 * day, 100.0), np.full(n - 6 * day, 80.0)]
    P = {"s": 1, "q": np.full(n, 10.0), "R": np.zeros(n), "un": 10 * (c_ - 100), "adv": c_, "c": c_,
         "tc": np.arange(n) * 60_000.0 + 60_000, "pend": 80.0, "pnl": 10 * (80 - 100.0)}
    dvt, dvv = np.array([0.0]), np.array([60.0])
    r = overlay(P, 100.0, dvt, dvv, 3, "전체", 0.05, 7, 0.0)
    # 1번: K=95 풋, 만기(7일) 정산 내재 15 − 수수료 min(0.015%·80, 12.5%·15)=0.012
    p1 = float(bs(100, 95, 7 / 365, iv_of(60, 0, 95, 100, 1), -1)); c1 = 10 * (p1 + float(fee(100, p1)))
    s1 = 10 * (15 - 0.012)
    # 2번: 7일 만기 분에 S=80 에서 K=76 풋 재매수 → 14일 정산 내재 0 · 3번: 14일에 재매수 → 20일에 모델가 매도
    p2 = float(bs(80, 76, 7 / 365, iv_of(60, 0, 76, 80, 1), -1)); c2 = 10 * (p2 + float(fee(80, p2)))
    T3 = (14 * day + 7 * day - (n - 1)) * 60_000 / YEAR_MS
    b3 = float(bs(80, 76, T3, iv_of(60, 0, 76, 80, -1), -1)); s3 = 10 * (b3 - float(fee(80, b3)))
    want = -c1 + s1 - c2 - c2 + s3
    assert r["buys"] == 3 and abs(r["opt"] - want) < 1e-6, (r, want)
    assert r["trough"] > -200 + 100, r                                                   # 보험이 −200 선물 손실의 절반 이상 막음
    # ⑤ 0.1 ETH 미만 목표 → 매수 불가 표시
    P2 = {**P, "q": np.full(n, 3.05)}
    r2 = overlay(P2, 100.0, dvt, dvv, 3, "초과분", 0.05, 7, 0.0)                           # 초과 명목 5 → 0.05 ETH
    assert r2["buys"] == 0 and r2["impossible"], r2
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else main())
