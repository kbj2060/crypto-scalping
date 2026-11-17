#!/usr/bin/env python3
"""자동 매매 결정 모델(롱·숏·건너뛰기 + 익절·손절·시한) — ETH · SOXL (2026-11-17). 사전등록 docs/prereg/auto_trade_decision_model_20261117.md (계산 전 커밋 110b0d63).

  selftest   배리어 판정 직접 대조 · 메이커 체결 · 순차 모의 겹침 없음 · 무작위 보행 ≈ −비용 · 심은 우위를 벌어들임
  cv         설계 구간 6겹 정화 교차 예측 → 설정 8 × 경로 2 성과표 · PBO · DSR → 관문 1 → docs/experiments/auto_trade_decision_selection_20261117.json
  test       관문 1 을 통과한 (종목·경로)만 잠금 구간 1회 채점 → tmp/auto_trade_decision_20261117/result.json

결정 τ = 5분 경계. 진입가 P0 = τ 직전 1분봉 종가. 배리어 ±B(B = k × σ15), 시한 H 분, τ 부터 1분봉 고가·저가로 판정(같은 분 둘 다 = 손절).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_time_class_direction_20261117 as M  # noqa: E402 -- 5분봉·지표 피쳐·자료 읽기

OUT = ROOT / "tmp/auto_trade_decision_20261117"
SEL = ROOT / "docs/experiments/auto_trade_decision_selection_20261117.json"
SEED, NB, BAR, MIN = 20261117, 2000, 300_000, 60_000
DESIGN = {"ETH": ("2024-01-08", "2026-01-01"), "SOXL": ("2026-05-22", "2026-10-01")}      # [시작, 끝)
LOCK = {"ETH": ("2026-01-01", "2026-11-16"), "SOXL": ("2026-10-01", "2026-11-16")}
HALF = {"ETH": "2026-06-09", "SOXL": "2026-10-23"}
GRID = [(k, H, m) for k in (1.0, 2.0) for H in (15, 60) for m in (1.0, 2.0)]
# 비용(bp): 엄격 = 테이커×2 + 미끄러짐 1×2 / 메이커 경로 = 진입 지정가 · 익절 지정가 · 손절·시한 테이커+1
COST = {"ETH": dict(rt=10.0, mk_in=0.0, mk_tp=0.0, mk_sl=5.0, mk_ref=2.5), "SOXL": dict(rt=12.0, mk_in=2.0, mk_tp=2.0, mk_sl=6.0, mk_ref=6.0)}
HGB = dict(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
FOLDS = 6


def prep(sym: str) -> dict:
    """1분 격자 배열 + 5분 결정 표(피쳐·P0·σ15·τ 의 분 색인)."""
    import research_soxl_intraday_clock_rules_20261117 as C
    eb, bb = M.bars5(M.load_crypto("ETHUSDT")), M.bars5(M.load_crypto("BTCUSDT"))
    btc = M.features(bb)
    if sym == "ETH":
        k1, b, cross = M.load_crypto("ETHUSDT"), eb, {"btc": btc}
    else:
        k1 = C.load(); b = M.bars5(k1); cross = {"eth": M.features(eb), "btc": btc}
    return make(k1, b, M.features(b, cross))


def make(k1: pd.DataFrame, b: pd.DataFrame, F: pd.DataFrame) -> dict:
    m0 = int(k1.t.iloc[0]) // MIN; n = int(k1.t.iloc[-1]) // MIN - m0 + 1
    hi, lo, cl = (np.full(n + 200, np.nan) for _ in range(3)); ix = (k1.t.to_numpy() // MIN - m0).astype(int)
    hi[ix], lo[ix], cl[ix] = k1.h.to_numpy(), k1.l.to_numpy(), k1.c.to_numpy()
    sig15 = np.log(b.c).diff().rolling(288, min_periods=200).std() * np.sqrt(3)
    F = F.assign(lsig=np.log(sig15), P0=b.c, sig15=sig15, mt=((F.index + BAR) // MIN - m0).astype(int))
    return dict(F=F, hi=hi, lo=lo, cl=cl, m0=m0, cols=[c for c in M.feat_cols(F) if c not in ("P0", "sig15", "mt")])


def barrier(D: dict, mt, P0, B, H: int, side: int, start=None, tp_delay: int = 0):
    """배리어 결과: (그쪽 기준 총수익, τ 로부터 청산 분, 종류 0 익절·1 손절·2 시한, 유효). start = τ 로부터 시작 분(메이커 체결 분), tp_delay = 익절은 그만큼 뒤부터."""
    n = len(mt); start = np.zeros(n, int) if start is None else start
    g, xo, kind, ok = np.full(n, np.nan), np.zeros(n, int), np.full(n, 2), np.zeros(n, bool)
    for a in range(0, n, 40000):
        s = slice(a, a + 40000); idx = (mt[s] + start[s])[:, None] + np.arange(H)[None, :]
        inb = (idx[:, -1] < len(D["cl"])) & (idx[:, 0] >= 1)
        idx = np.clip(idx, 0, len(D["cl"]) - 1); p = P0[s][:, None]
        hh, ll = D["hi"][idx] / p - 1, D["lo"][idx] / p - 1; b_ = B[s][:, None]
        tp, sl = (hh >= b_, ll <= -b_) if side > 0 else (ll <= -b_, hh >= b_)
        if tp_delay:
            tp[:, :tp_delay] = False
        t_tp = np.where(tp.any(1), tp.argmax(1), H); t_sl = np.where(sl.any(1), sl.argmax(1), H)
        kd = np.where((t_sl <= t_tp) & (t_sl < H), 1, np.where(t_tp < H, 0, 2))                    # 같은 분 둘 다 = 손절
        last = side * (D["cl"][idx[:, -1]] / P0[s] - 1)
        g[s] = np.where(kd == 1, -B[s], np.where(kd == 0, B[s], last)); kind[s] = kd
        xo[s] = start[s] + np.where(kd == 2, H - 1, np.minimum(t_tp, t_sl))
        ok[s] = inb & np.isfinite(hh).all(1) & np.isfinite(ll).all(1) & np.isfinite(P0[s]) & np.isfinite(B[s]) & (B[s] > 0)
    return g, xo, kind, ok


def fills(D: dict, mt, P0, side: int):
    """메이커 진입: [τ, τ+5분) 중 가격이 지정가 P0 를 넘어 지나간 첫 분(없으면 −1)."""
    idx = np.clip(mt[:, None] + np.arange(5)[None, :], 0, len(D["cl"]) - 1)
    thru = (D["lo"][idx] < P0[:, None]) if side > 0 else (D["hi"][idx] > P0[:, None])
    return np.where(thru.any(1), thru.argmax(1), -1)


def outcomes(D: dict, k: float, H: int) -> dict:
    """설정 (k, H) 의 결정별 결과표 — 시장가 경로·메이커 경로 × 롱·숏, 학습 표적 y."""
    F = D["F"]; mt, P0, B = F.mt.to_numpy(), F.P0.to_numpy(), (k * F.sig15).to_numpy()
    O = {"B": B}
    for nm, side in (("L", 1), ("S", -1)):
        O[f"t{nm}"] = barrier(D, mt, P0, B, H, side)
        f = fills(D, mt, P0, side); O[f"f{nm}"] = f
        O[f"m{nm}"] = barrier(D, mt, P0, B, H, side, start=np.maximum(f, 0), tp_delay=1)
    O["y"] = np.where(O["tL"][3] & O["tS"][3], (O["tL"][0] - O["tS"][0]) / (2 * B), np.nan)
    return O


def decide(yhat, B, m: float, c_ref: float):
    e = np.abs(yhat) * B * 1e4
    return np.where(np.isfinite(e) & (e >= m * c_ref), np.sign(yhat), 0).astype(int)


def simulate(F: pd.DataFrame, O: dict, act, rows, path: str, C: dict) -> pd.DataFrame:
    """봉 단위 순차 진행(한 번에 한 포지션). rows = 채점할 결정의 위치(시간순). 거래 표를 돌려준다."""
    mt, ok_feat = F.mt.to_numpy(), F.ok.to_numpy(); free, rec = -1, []
    for i in rows:
        a = act[i]
        if a == 0 or mt[i] < free or not ok_feat[i]:
            continue
        nm = "L" if a > 0 else "S"
        if path == "taker":
            g, xo, kd, ok = (x[i] for x in O[f"t{nm}"])
            if not ok:
                continue
            net = g * 1e4 - C["rt"]
        else:
            f = O[f"f{nm}"][i]
            if f < 0:
                free = mt[i] + 5; continue                              # 미체결: 취소하고 그 5분은 쉰다
            g, xo, kd, ok = (x[i] for x in O[f"m{nm}"])
            if not ok:
                continue
            net = g * 1e4 - C["mk_in"] - (C["mk_tp"] if kd == 0 else C["mk_sl"])
        free = mt[i] + xo + 1
        rec.append((i, a, net, g * 1e4, kd))
    return pd.DataFrame(rec, columns=["i", "side", "net", "gross", "kind"])


def daily(F: pd.DataFrame, tr: pd.DataFrame, rows) -> pd.Series:
    """하루 순손익(bp, 1배 명목 · 거래 없는 날 = 0). 날 = 채점 구간의 모든 달력일."""
    days = (F.index.to_numpy()[rows] + BAR) // 86_400_000; alld = np.unique(days)
    s = pd.Series(0.0, index=alld)
    if len(tr):
        s = s.add(tr.groupby((F.index.to_numpy()[tr.i.to_numpy()] + BAR) // 86_400_000).net.sum(), fill_value=0.0)
    return s


def fit_predict(F, cols, y, train_rows, pred_rows):
    from sklearn.ensemble import HistGradientBoostingRegressor
    tr = train_rows[np.isfinite(y[train_rows]) & F.ok.to_numpy()[train_rows]]
    mdl = HistGradientBoostingRegressor(**HGB).fit(F[cols].to_numpy()[tr], y[tr])
    out = np.full(len(F), np.nan); out[pred_rows] = mdl.predict(F[cols].to_numpy()[pred_rows]); return out


def rng_rows(F, a: str, b: str):
    t = F.index.to_numpy() + BAR; t0, t1 = (pd.Timestamp(x, tz="UTC").value // 10**6 for x in (a, b))
    return np.flatnonzero((t >= t0) & (t < t1))


def pbo(perf: np.ndarray) -> float:
    """CSCV: 토막 절반으로 고른 최고 설정이 나머지 절반에서 중앙 이하인 비율. perf = 토막 × 설정."""
    n = perf.shape[0]; bad = 0; tot = 0
    for S in itertools.combinations(range(n), n // 2):
        rest = [j for j in range(n) if j not in S]; best = int(np.argmax(perf[list(S)].mean(0)))
        oos = perf[rest].mean(0); rank = (oos < oos[best]).sum() + 0.5 * (oos == oos[best]).sum()
        bad += rank / len(oos) <= 0.5; tot += 1
    return bad / tot


def dsr(x: np.ndarray, sr_trials: np.ndarray) -> float:
    """Deflated Sharpe: 시도 수만큼 부풀 수 있는 최대 샤프를 빼고 본 확률."""
    from scipy.stats import norm, skew, kurtosis
    if x.std() == 0:
        return 0.0
    sr = x.mean() / x.std(ddof=1); T = len(x); N = len(sr_trials); v = np.var(sr_trials, ddof=1) if N > 1 else 0.0
    g = 0.5772156649; sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    den = np.sqrt(max(1e-12, 1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))


def cv(syms=("ETH", "SOXL")) -> None:
    OUT.mkdir(parents=True, exist_ok=True); sel = {"prereg": "110b0d63", "grid": GRID, "result": {}}
    for sym in syms:
        D = prep(sym); F, cols = D["F"], D["cols"]; rows = rng_rows(F, *DESIGN[sym]); C = COST[sym]
        edges = np.linspace(0, len(rows), FOLDS + 1).astype(int); folds = [rows[edges[j]:edges[j + 1]] for j in range(FOLDS)]
        perf = {p: np.zeros((FOLDS, len(GRID))) for p in ("taker", "maker")}; ser = {p: {} for p in perf}; ntr = {p: np.zeros(len(GRID)) for p in perf}
        for k, H in sorted({(g[0], g[1]) for g in GRID}):
            O = outcomes(D, k, H); yhat = np.full(len(F), np.nan)
            for j, te in enumerate(folds):
                tr = np.setdiff1d(rows, rows[max(0, edges[j] - 288):min(len(rows), edges[j + 1] + 288)])       # 토막 앞뒤 1일 purge
                yhat[te] = fit_predict(F, cols, O["y"], tr, te)[te]
            np.save(OUT / f"{sym}_yhat_k{k}_H{H}.npy", yhat)
            for gi, (kk, HH, m) in enumerate(GRID):
                if (kk, HH) != (k, H):
                    continue
                for p, cref in (("taker", C["rt"]), ("maker", C["mk_ref"])):
                    act = decide(yhat, O["B"], m, cref); alld = []
                    for j, te in enumerate(folds):
                        t = simulate(F, O, act, te, p, C); d = daily(F, t, te); perf[p][j, gi] = d.mean(); alld.append(d); ntr[p][gi] += len(t)
                    ser[p][gi] = pd.concat(alld)
        res = {}
        for p in perf:
            mean = perf[p].mean(0); best = int(np.argmax(mean)); x = ser[p][best].to_numpy()
            srs = np.array([s.mean() / s.std(ddof=1) if s.std() > 0 else 0.0 for s in (ser[p][g].to_numpy() for g in range(len(GRID)))])
            r = {"표(설정: 하루 순손익 bp 평균 · 양수 토막 · 거래 수)": {str(GRID[g]): [round(float(mean[g]), 3), int((perf[p][:, g] > 0).sum()), int(ntr[p][g])] for g in range(len(GRID))},
                 "고른_설정": list(GRID[best]), "평균_하루bp": float(mean[best]), "토막별": [round(float(v), 3) for v in perf[p][:, best]], "양수_토막": int((perf[p][:, best] > 0).sum()),
                 "PBO": float(pbo(perf[p])), "DSR": dsr(x, srs), "거래수": int(ntr[p][best]), "날": int(len(x))}
            r["관문1_통과"] = bool(r["평균_하루bp"] > 0 and r["양수_토막"] >= 4 and r["PBO"] < 0.2 and r["DSR"] > 0.95)
            res[p] = r
        sel["result"][sym] = res; print(sym, json.dumps(res, ensure_ascii=False, indent=1))
    SEL.write_text(json.dumps(sel, ensure_ascii=False, indent=1))


def judge(F, O, act, rows, path, C, sym, rng) -> dict:
    t = simulate(F, O, act, rows, path, C); d = daily(F, t, rows); x = d.to_numpy()
    b = x[rng.integers(0, len(x), (NB, len(x)))].mean(1); p2 = float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
    cum = np.cumsum(x); sub = F["sub"].to_numpy()
    r = {"날": int(len(x)), "거래": int(len(t)), "하루거래": float(len(t) / max(len(x), 1)), "하루_순손익_bp": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "p2": p2,
         "누적_bp": float(cum[-1]) if len(x) else 0.0, "최대낙폭_bp": float((cum - np.maximum.accumulate(cum)).min()) if len(x) else 0.0}
    if not len(t):
        r.update({"무작위_분위": None, "초과_늘롱": None, "초과_늘숏": None, "앞절반": 0.0, "뒤절반": 0.0}); return r
    ii, sd = t.i.to_numpy(), t.side.to_numpy()
    def net_side(side):                                             # 같은 진입 시각·같은 배리어에서 그 방향으로 했을 때(뒤 진입 시각은 그대로 둔다)
        out = np.zeros(len(ii))
        for s_, nm in ((1, "L"), (-1, "S")):
            m_ = side == s_
            if path == "taker":
                g, _, kd, ok = (v[ii] for v in O[f"t{nm}"]); val = np.where(ok, g * 1e4 - C["rt"], 0.0)
            else:
                g, _, kd, ok = (v[ii] for v in O[f"m{nm}"]); f = O[f"f{nm}"][ii]
                val = np.where(ok & (f >= 0), g * 1e4 - C["mk_in"] - np.where(kd == 0, C["mk_tp"], C["mk_sl"]), 0.0)
            out[m_] = val[m_]
        return out
    nd = len(x); tot = t.net.sum() / nd
    rnd = np.array([net_side(rng.choice([-1, 1], len(ii))).sum() / nd for _ in range(1000)])
    al, as_ = net_side(np.ones(len(ii), int)).sum() / nd, net_side(-np.ones(len(ii), int)).sum() / nd
    hb = pd.Timestamp(HALF[sym], tz="UTC").value // 86_400_000_000_000
    r.update({"이긴거래": f"{int((t.net > 0).sum())}/{len(t)}", "건당_평균bp": float(t.net.mean()), "건당_중앙bp": float(t.net.median()), "건당_비용전bp": float(t.gross.mean()),
              "청산(익절·손절·시한)": [int((t.kind == k_).sum()) for k_ in (0, 1, 2)], "롱": [int((sd > 0).sum()), float(t.net[sd > 0].mean()) if (sd > 0).any() else None],
              "숏": [int((sd < 0).sum()), float(t.net[sd < 0].mean()) if (sd < 0).any() else None],
              "시간대(거래·합계bp)": {nm: [int(m_.sum()), float(t.net[m_].sum())] for nm, m_ in (("활발", sub[ii] <= 1), ("나머지", sub[ii] >= 2))},
              "무작위_분위": float(100 * (rnd < tot).mean()), "무작위_평균": float(rnd.mean()), "늘롱_하루bp": float(al), "늘숏_하루bp": float(as_), "초과_늘롱": float(tot - al), "초과_늘숏": float(tot - as_),
              "앞절반": float(d[d.index < hb].mean()), "뒤절반": float(d[d.index >= hb].mean())})
    return r


def test() -> None:
    sel = json.loads(SEL.read_text()); out = {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False, "result": {}}
    rng = np.random.default_rng(SEED)
    for sym, R in sel["result"].items():
        opened = [p for p in R if R[p]["관문1_통과"]]
        if not opened:
            out["result"][sym] = {"잠금": "열지 않음(관문 1 불통과)"}; continue
        D = prep(sym); F, cols = D["F"], D["cols"]; dr, lr = rng_rows(F, *DESIGN[sym]), rng_rows(F, *LOCK[sym]); C = COST[sym]; out["result"][sym] = {}
        for p in opened:
            k, H, m = R[p]["고른_설정"]; O = outcomes(D, k, H); tr = dr[:-288]                          # 설계 끝 1일 purge
            yhat = fit_predict(F, cols, O["y"], tr, lr); act = decide(yhat, O["B"], m, C["rt"] if p == "taker" else C["mk_ref"])
            r = judge(F, O, act, lr, p, C, sym, rng)
            days = np.unique((F.index.to_numpy()[tr] + BAR) // 86_400_000); rs = np.random.default_rng(1); spread = []
            for _ in range(10):                                         # 강건성: 학습 날짜 재추출
                pick = rs.choice(days, len(days)); dd = (F.index.to_numpy()[tr] + BAR) // 86_400_000
                tr2 = np.concatenate([tr[dd == d_] for d_ in pick]); yh = fit_predict(F, cols, O["y"], tr2, lr)
                t2 = simulate(F, O, decide(yh, O["B"], m, C["rt"] if p == "taker" else C["mk_ref"]), lr, p, C); spread.append(round(float(daily(F, t2, lr).mean()), 3))
            r["학습_날짜_재추출_하루bp"] = spread
            r["1_절대"] = bool(r["하루_순손익_bp"] > 0 and r["p2"] < 0.05); r["2_무작위_97.5"] = bool((r["무작위_분위"] or 0) >= 97.5)
            r["3_양쪽다리"] = bool((r["초과_늘롱"] or -1) > 0 and (r["초과_늘숏"] or -1) > 0); r["4_낙폭"] = bool(r["최대낙폭_bp"] >= -2000); r["5_두절반"] = bool(r["앞절반"] > 0 and r["뒤절반"] > 0)
            out["result"][sym][p] = r
    ps = sorted((v["taker"]["p2"], s) for s, v in out["result"].items() if isinstance(v, dict) and "taker" in v)
    for rank, (p_, s) in enumerate(ps):                               # Holm(엄격 경로, 연 종목만 · 종목 수 2 기준)
        out["result"][s]["taker"]["1_절대_Holm"] = bool(out["result"][s]["taker"]["1_절대"] and p_ < 0.05 / (2 - rank))
    for s, v in out["result"].items():
        if not isinstance(v, dict):
            continue
        if "taker" in v:
            t = v["taker"]; t["엄격_통과"] = bool(t.get("1_절대_Holm") and t["2_무작위_97.5"] and t["3_양쪽다리"] and t["4_낙폭"] and t["5_두절반"]); t["상대_통과"] = bool(t["2_무작위_97.5"] and t["3_양쪽다리"] and t["4_낙폭"])
        if "maker" in v:
            mk = v["maker"]; mk["통과_후보"] = bool(mk["하루_순손익_bp"] > 0 and (mk["무작위_분위"] or 0) >= 50 and mk["3_양쪽다리"] and mk["4_낙폭"]); mk["상대_통과"] = bool(mk["2_무작위_97.5"] and mk["3_양쪽다리"] and mk["4_낙폭"])
    (OUT / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=1))


def selftest() -> None:
    # ① 배리어: 손으로 만든 1분 경로
    P = 100.0; hi = np.full(300, np.nan); lo = hi.copy(); cl = hi.copy()
    def path(start, hs, ls, cs):
        hi[start:start + len(hs)], lo[start:start + len(hs)], cl[start:start + len(hs)] = hs, ls, cs
    cl[9] = P
    path(10, [100.1, 100.6, 100.2], [99.9, 100.0, 99.95], [100.0, 100.5, 100.1])            # 2번째 분에 +0.6%
    D = dict(hi=hi, lo=lo, cl=cl); mt, P0 = np.array([10]), np.array([P])
    g, xo, kd, ok = barrier(D, mt, P0, np.array([0.005]), 3, 1); assert ok[0] and kd[0] == 0 and abs(g[0] - 0.005) < 1e-12 and xo[0] == 1          # 롱 익절
    g, xo, kd, ok = barrier(D, mt, P0, np.array([0.005]), 3, -1); assert kd[0] == 1 and abs(g[0] + 0.005) < 1e-12 and xo[0] == 1                    # 숏 손절
    g, xo, kd, ok = barrier(D, mt, P0, np.array([0.02]), 3, 1); assert kd[0] == 2 and abs(g[0] - 0.001) < 1e-9 and xo[0] == 2                       # 시한 = 마지막 분 종가
    hi[11], lo[11] = 100.6, 99.4
    g, xo, kd, ok = barrier(D, mt, P0, np.array([0.005]), 3, 1); assert kd[0] == 1                                                                   # 같은 분 둘 다 = 손절
    g, xo, kd, ok = barrier(D, mt, P0, np.array([0.005]), 5, 1); assert not ok[0]                                                                    # 창에 빈 분 = 무효
    # ② 메이커 체결: 지정가를 넘어 지나가야 체결 · 익절은 체결 다음 분부터
    lo[10] = 100.0; assert fills(D, mt, P0, 1)[0] == 1                 # 첫 분 저가 = 지정가(안 지나감) → 둘째 분 99.4
    assert fills(D, np.array([12]), np.array([99.0]), 1)[0] == -1
    hi[11], lo[11] = 100.6, 99.99; g, xo, kd, ok = barrier(D, mt, P0, np.array([0.005]), 2, 1, start=np.array([1]), tp_delay=1)
    assert kd[0] == 2 and xo[0] == 2                                   # 체결 분(11)의 고가 100.6 은 익절로 안 친다
    # ③ 무작위 보행: 무작위 방향 진입의 건당 순손익 ≈ −비용 · 포지션 겹침 없음
    rng = np.random.default_rng(2); n = 30 * 1440; t = pd.Timestamp("2026-10-12", tz="UTC").value // 10**6 + np.arange(n) * MIN
    c = 100 * np.exp(np.cumsum(rng.normal(0, 4e-4, n))); v = rng.uniform(1, 3, n); w = np.abs(rng.normal(0, 2e-4, n))
    k1 = pd.DataFrame({"t": t, "h": c * (1 + w), "l": c * (1 - w), "c": c, "v": v, "tb": v * rng.uniform(.3, .7, n)})
    b = M.bars5(k1); Dd = make(k1, b, M.features(b)); F = Dd["F"]; O = outcomes(Dd, 1.0, 15); rows = np.flatnonzero(F.ok.to_numpy() & np.isfinite(O["y"]))
    act = rng.choice([-1, 1], len(F)); C = COST["ETH"]
    tk = simulate(F, O, act, rows, "taker", C); assert len(tk) > 300 and abs(tk.net.mean() + C["rt"]) < 4, tk.net.mean()
    ent = F.mt.to_numpy()[tk.i.to_numpy()]; ex = ent + np.array([O["tL" if s > 0 else "tS"][1][i] for i, s in zip(tk.i, tk.side)]); assert (ent[1:] > ex[:-1]).all(), "포지션이 겹친다"
    mk = simulate(F, O, act, rows, "maker", C); assert len(mk) > 200 and mk.net.mean() < 1.0                                                          # 관통 체결의 역선택: 공짜 수익 없음
    # ④ 미래 불변: τ 뒤 1분봉을 바꿔도 그 결정(피쳐·B)은 같다 / 결과는 바뀐다
    i0 = rows[len(rows) // 2]; k2 = k1.copy(); fut = k2.t >= F.index[i0] + BAR; k2.loc[fut, ["h", "l", "c"]] = k2.loc[fut, ["h", "l", "c"]] * 1.05
    b2 = M.bars5(k2); D2 = make(k2, b2, M.features(b2)); F2 = D2["F"]
    assert np.allclose(F[Dd["cols"]].iloc[i0].to_numpy(float), F2[D2["cols"]].iloc[i0].to_numpy(float), equal_nan=True) and F.sig15.iloc[i0] == F2.sig15.iloc[i0] and F.P0.iloc[i0] == F2.P0.iloc[i0]
    # ⑤ 심은 우위: 예측이 실제 결과와 같으면(오라클) 순손익 > 0, 반대면 < −비용 · 결정 문턱이 거래를 거른다
    y = O["y"]; orc = simulate(F, O, decide(y, O["B"], 0.0, 1.0), rows, "taker", C); anti = simulate(F, O, decide(-y, O["B"], 0.0, 1.0), rows, "taker", C)
    assert orc.net.mean() > 0 and anti.net.mean() < -C["rt"], (orc.net.mean(), anti.net.mean())
    assert (decide(np.array([0.01, -0.5, np.nan]), np.array([0.002, 0.002, 0.002]), 1.0, 10.0) == np.array([0, -1, 0])).all()                          # 0.2bp < 10 → 건너뜀 · 10bp ≥ 10 → 숏
    # ⑥ PBO: 늘 같은 설정이 최고면 0, 완전 무작위면 중간
    good = np.tile(np.arange(8.0), (6, 1)); assert pbo(good) == 0.0 and 0.2 < pbo(np.random.default_rng(0).normal(size=(6, 8))) < 0.9
    d = daily(F, tk, rows); assert len(d) >= 28 and abs(d.sum() - tk.net.sum()) < 1e-6
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "cv", "test"]); ap.add_argument("--sym", nargs="*", default=["ETH", "SOXL"])
    a = ap.parse_args()
    {"selftest": selftest, "cv": lambda: cv(tuple(a.sym)), "test": test}[a.cmd]()
