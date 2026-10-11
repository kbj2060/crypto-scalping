"""실원장 진입에 «리스크 관리 시스템»을 적용했다면 (2026-10-12, 사용자 «실원장의 진입에서 이 리스크 관리 시스템을 적용했다면»).

시스템(10-12 설명 그대로, 결과 보기 전 고정):
  ① 손절 거리 d = 리스크 모델의 «4h 최대 역행폭 99%»(GBM q0.9 × 보정 배수, 방향별) — 진입 직전 마감 5분봉 기준
  ② 크기: 손절에 걸리면 잃는 돈 ≤ 순자산 5% (수수료 포함) → 첫 진입 = 예산 ÷ d
  ③ 상한: 총 명목 ≤ 순자산 1.5배
  ④ 손절은 거래소 스톱(첫 진입가 기준 고정, 물타기로 안 옮김) — 닿는 분에 그 가격(갭이면 시가)·미끄러짐 3bp·테이커
  ⑤ 보유시간 상한 4h: 첫 체결 240분 뒤 시가에 남은 물량 정리(테이커)
  물타기: 원장의 추가 주문을 같은 배율로 따르되 ②·③ 을 넘는 만큼은 잘라낸다. 청산: 원장 청산 비율을 그대로 따른다.
  순자산: 왕복 시작 시점 = 원장 E0 + (이 팔 실현 누적 − 실제 실현 누적).
팔: A 실제 · S 전체 시스템 · S−시간상한 · S 손절을 고정 3% 로(모델 몫 분리) · 상한 1.5배만.
리스크 모델: live_eth_sizing_vol_model 피쳐(체결수 nt_z 제외 21열 — 2026-10 1분봉에 체결수 없음) + 방향, 4h MAE, HistGBR q0.9.
  학습 2022-01~2025-12 · 보정(1% 목표 배수) 2026-01~06 · 원장 08-01~10-08 은 표본 밖.
ponytail: MDD 는 왕복 순차(실현 + 왕복 안 최저 평가) — 겹친 왕복의 동시 평가손은 안 더한다.

실행: python scripts/research_ledger_risk_system_replay_20261012.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(MAIN / ".claude/worktrees/agent-a1dc713d584c49ce4/scripts"))
sys.path.insert(0, str(MAIN / "scripts"))
import research_ledger_sr_first_entry_grid_20261011 as G  # noqa: E402  load_1m · W(K·load_trips·SLIP)
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

W = G.W
OUT = ROOT / "tmp/ledger_risk_system_20261012"
TK, SLIP = 5e-4, W.SLIP
FEATS = [f for f in svm.FEATURES if f != "nt_z"]
H4 = 48


# ── 리스크 모델 ─────────────────────────────────────────────────────────────
def bars5(d1: pd.DataFrame) -> pd.DataFrame:
    old = pd.concat([pd.read_parquet(MAIN / f"data/binance_vision/klines1m/ETHUSDT-1m-{m}.parquet", columns=list("thlcv"))
                     for m in pd.period_range("2022-01", "2026-05", freq="M")])
    d = pd.concat([old, d1[list("thlcv")]]).drop_duplicates("t", keep="last").sort_values("t")
    d["b"] = d.t // 300_000 * 300_000
    g = d.groupby("b")
    o = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "v": g.v.sum()})
    return o.reindex(np.arange(o.index[0], o.index[-1] + 1, 300_000)).ffill()


def risk_model(o: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingRegressor
    ts = pd.to_datetime(o.index, unit="ms")
    c, hi, lo = o.c.to_numpy(), o.h.to_numpy(), o.l.to_numpy()
    F = svm.build_features(pd.Series(ts), c, (o.v * o.c).to_numpy(), np.ones(len(o)), hi, lo)[FEATS].to_numpy()
    fmin = pd.Series(lo).iloc[::-1].rolling(H4).min().iloc[::-1].shift(-1).to_numpy()
    fmax = pd.Series(hi).iloc[::-1].rolling(H4).max().iloc[::-1].shift(-1).to_numpy()
    Y = np.c_[(c - fmin) / c, (fmax - c) / c]                                  # 롱·숏 MAE(비율)
    okF = np.isfinite(F).all(1) & (np.arange(len(o)) >= 600)
    ok = okF & np.isfinite(Y).all(1)
    tr = ok & (ts < "2026-01-01"); ca = ok & (ts >= "2026-01-01") & (ts < "2026-07-01")
    X = lambda m, s: np.c_[F[m], np.full(m.sum(), s)]
    msub = np.zeros(len(o), bool); msub[np.flatnonzero(tr)[::3]] = True
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=300, learning_rate=0.06, max_depth=6,
                                      min_samples_leaf=200, random_state=20260913)
    m.fit(np.r_[X(msub, 1), X(msub, -1)], np.r_[Y[msub, 0], Y[msub, 1]])
    pred = {}
    for s in (1, -1):
        pred[s] = np.full(len(o), np.nan); pred[s][okF] = m.predict(X(okF, s))
    mult = {s: float(np.quantile(Y[ca, 0 if s > 0 else 1] / pred[s][ca], 0.99)) for s in (1, -1)}
    te = ok & (ts >= "2026-07-01")
    chk = {s: round(float(np.mean(Y[te, 0 if s > 0 else 1] > pred[s][te] * mult[s])), 4) for s in (1, -1)}
    return {s: pred[s] * mult[s] for s in (1, -1)}, o.index.to_numpy(), {"배수": mult, "2026-07~ 초과율(목표 1%)": chk}


def d_at(safe, bidx, t_ms: int, s: int) -> float:
    j = np.searchsorted(bidx, t_ms, side="right") - 2                         # 진입 시각 직전에 «마감된» 5분봉
    return float(safe[s][j])


# ── 왕복 재생 ───────────────────────────────────────────────────────────────
def sim(tr, k, E, d=None, budget=None, cap=None, tcap=None, scale=1.0, first=1.0, exits=None) -> dict:
    """d=None → 손절 없음. budget·cap 은 E 배수. scale=None → 첫 진입을 예산 ÷ d 로 정하고 원장 수량에 같은 배율."""
    s, ev = tr["s"], tr["ev"]
    beyond = lambda p, x: (p <= x) if s > 0 else (p >= x)
    line = tr["p0"] * (1 - s * d) if d else None
    if scale is None:
        scale = first * (budget * E / d) / (tr["ordq"][0] * tr["ordp"][0])   # first<1 → 남은 예산 = 물타기 여유
    st = {"q": 0.0, "avg": 0.0, "R": 0.0}
    qa = 0.0
    i0 = k.i(ev[0][0]); iend = k.i(ev[-1][0]) + 1
    if tcap:
        iend = min(iend, i0 + tcap)
    ptr, trough, how, peakN = 0, 0.0, "원장 청산", 0.0

    def close(px, fee):
        st["R"] += s * (px - st["avg"]) * st["q"] - fee * px * st["q"]; st["q"] = 0.0

    for m in range(i0, min(iend, len(k.t) - 1) + 1):
        if tcap and m == i0 + tcap and st["q"] > 0:
            close(k.o[m] * (1 - s * SLIP), TK); how = "시간 상한"; break
        had = False
        while ptr < len(ev) and ev[ptr][0] < k.t[m] + 60_000:
            t, inc, p, qq, fee, o, _ = ev[ptr]; ptr += 1; had = True
            q, avg, R = st["q"], st["avg"], st["R"]
            if q > 0 and line is not None and beyond(p, line):
                close(line * (1 - s * SLIP), TK); how = "손절"; break
            if inc:
                qa += qq
                x = qq * scale
                if cap:
                    x = min(x, max(cap * E / p - q, 0.0))
                if budget and line is not None:                                  # 손절 시 총손실(수수료 포함) ≤ 예산
                    L0 = R + s * (line - avg) * q - TK * line * q
                    coef = -fee * p + s * (line - p) - TK * line
                    if coef < 0:
                        x = min(x, max((L0 + budget * E) / -coef, 0.0))
                if x > 0:
                    st["avg"] = (avg * q + p * x) / (q + x); st["q"] = q + x; st["R"] = R - fee * p * x
                    peakN = max(peakN, st["q"] * p)
            else:
                frac = qq / qa if qa > 0 else 1.0
                qa -= qq
                xo = q if qa <= 1e-9 else q * frac
                st["R"] = R + s * (p - avg) * xo - fee * p * xo; st["q"] = q - xo
        if how == "손절":
            break
        adv = k.l[m] if s > 0 else k.h[m]
        if st["q"] > 0 and line is not None and beyond(adv, line):
            px = k.o[m] if (not had and beyond(k.o[m], line)) else line
            close(px * (1 - s * SLIP), TK); how = "손절"; break
        trough = min(trough, st["R"] + s * st["q"] * (adv - st["avg"]))
        # 익절 훅(exits[s] = 조건 충족 «5분봉 마감 분» 집합): 평가 이익 > 수수료면 그 분 종가에 전량 정리
        if exits is not None and st["q"] > 0 and m in exits[s] and s * (k.c[m] - st["avg"]) > 2 * TK * k.c[m]:
            close(k.c[m] * (1 - s * SLIP), TK); how = "익절 훅"; break
        if st["q"] <= 1e-12 and qa <= 1e-9 and ptr >= len(ev):
            break
    if st["q"] > 0:                                                              # 원장 끝났는데 남음(반올림) → 마지막 종가
        close(k.c[min(iend, len(k.c) - 1)], TK)
    return {"pnl": st["R"], "trough": min(trough, st["R"]), "how": how, "lev": peakN / E}


def run_arm(trips, k, arm, safe, bidx):
    rows, done = [], []                                                          # done: (청산 시각, 실제 pnl, 이 팔 pnl)
    for i in sorted(range(len(trips)), key=lambda i: trips[i]["t0"]):
        tr = trips[i]
        real = [(a, b) for t1, a, b in done if t1 <= tr["t0"]]
        E = max(tr["E0"] + sum(b - a for a, b in real), 1.0)
        if arm == "A 실제":
            r = sim(tr, k, E)
        elif arm == "상한 1.5배만":
            r = sim(tr, k, E, cap=1.5)
        else:
            d = 0.03 if arm.startswith("S 고정3%") else d_at(safe, bidx, tr["t0"], tr["s"])
            r = sim(tr, k, E, d=d, budget=0.05, cap=1.5, tcap=None if "시간상한 없음" in arm else 240, scale=None,
                    first=1 / 3 if "첫진입 1/3" in arm else 1.0)
            r["d"] = d
        done.append((tr["t1"], tr["pnl"], r["pnl"]))
        rows.append({"i": i, "t0": tr["t0"], "E": E, "s": tr["s"], "big": tr["big"], "real": tr["pnl"],
                     "N_real_x": tr["N"] / tr["E0"], **r})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, E_start: float) -> dict:
    eq = E_start + df.pnl.cumsum().to_numpy()
    prev = np.r_[E_start, eq[:-1]]
    low = np.minimum(prev + df.trough.to_numpy(), eq)
    peak = np.maximum.accumulate(np.r_[E_start, eq])[:-1]
    nb = df[~df.big]
    out = {"총손익 $": round(float(df.pnl.sum()), 1), "10-06 큰 롱 $": round(float(df[df.big].pnl.sum()), 1),
           "그 1건 뺀 손익 $": round(float(nb.pnl.sum()), 1), "최악 왕복 $": round(float(df.pnl.min()), 1),
           "왕복 중 최악 평가손 $": round(float(df.trough.min()), 1), "MDD $(순차)": round(float(np.min(low - peak)), 1),
           "MDD %(순차)": round(float(100 * np.min((low - peak) / peak)), 1), "승률": round(float((df.pnl > 0).mean()), 3),
           "청산 방식": df.how.value_counts().to_dict(),
           "최대 명목/순자산 중앙·최대": [round(float(df.lev.median()), 2), round(float(df.lev.max()), 2)]}
    if "d" in df:
        out["손절 거리 d 중앙·p10·p90 %"] = [round(100 * float(np.nanpercentile(df.d, q)), 2) for q in (50, 10, 90)]
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d1 = G.load_1m(); k = W.K(d1)
    trips = W.load_trips(k)
    print(f"왕복 {len(trips)} · 실제 합 ${sum(t['pnl'] for t in trips):,.2f}", flush=True)
    safe, bidx, mchk = risk_model(bars5(d1))
    print("리스크 모델", mchk, flush=True)
    E_start = min(trips, key=lambda t: t["t0"])["E0"]
    rep = {"리스크 모델": mchk, "시작 순자산 $": round(E_start, 1), "왕복 수": len(trips)}
    for j, a in enumerate(["A 실제", "S 전체 시스템", "S 시간상한 없음", "S 고정3% 손절(모델 대신)", "상한 1.5배만",
                            "S 첫진입 1/3", "S 첫진입 1/3 시간상한 없음"]):
        df = run_arm(trips, k, a, safe, bidx)
        if j == 0:
            rep["재생 오차 $(왕복별 최대)"] = round(float((df.pnl - df.real).abs().max()), 4)
            print("재생 오차", rep["재생 오차 $(왕복별 최대)"], flush=True)
        df.to_parquet(OUT / f"trips_arm{j}.parquet")
        rep[a] = summarize(df, E_start)
        print(a, json.dumps(rep[a], ensure_ascii=False), flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


def selftest():
    class KK:   # 1분 격자: 100 → 97 로 밀림
        t = np.arange(6) * 60_000.0; o = np.array([100, 100, 99, 98, 97.0, 97.0]); h = o + 0.1; l = o - 1.5; c = o.copy()
        def i(self, ms):
            return int(ms // 60_000)
    tr = {"s": 1, "p0": 100.0, "ev": [(0, True, 100.0, 1.0, 0.0, 0, True), (4 * 60_000, False, 97.0, 1.0, 0.0, -2, False)],
          "ordq": np.array([1.0]), "ordp": np.array([100.0])}
    r0 = sim(tr, KK(), 1000.0)
    assert abs(r0["pnl"] + 3.0) < 1e-9, r0                       # 실제 = −3
    r = sim(tr, KK(), 1000.0, d=0.02, budget=0.05, cap=1.5, scale=None)
    assert r["how"] == "손절" and -32.0 < r["pnl"] < -30.0, r      # 상한 1.5배 → 명목 $1,500 → 2% 손절(+미끄러짐·수수료) ≈ −$31
    r = sim(tr, KK(), 1000.0, d=0.02, budget=0.01, cap=1.5, scale=None)
    assert r["how"] == "손절" and -10.3 < r["pnl"] < -9.5, r       # 예산 1%=$10 → 명목 ≈ $500 → ≈ −$10
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    sys.exit(selftest() if ap.parse_args().selftest else main())
