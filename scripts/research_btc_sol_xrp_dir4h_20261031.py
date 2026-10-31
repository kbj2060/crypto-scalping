#!/usr/bin/env python3
"""BTC·SOL·XRP 4시간 방향 예측 — 배포 ETH B2 레시피를 코인별로 (사전등록 docs/prereg/btc_sol_xrp_dir4h_20261031.md).

레시피 = scripts/live_eth_dir4h_20261015.py 그대로: 65피쳐 build_features · y_4h(봉 τ 종가 → τ+48 종가 로그수익 bp) ·
매달 1일 재학습 · 학습 창 [m−4h−365일, m−4h) · 행 [::2] · HGB(early_stopping=False, 결정적) · 문턱 = 그 팔 자신의 과거 표본 밖 예측 20/80(달 안 고정).
대상 코인 바꾸기 = frames["ETH"] 자리에 대상 코인, frames[대상] 자리에 ETH(교차 19코인 묶음 유지 · BTC 대상이면 X_btc_*·U_btc_* 가 ETH).
팔: S = 그 코인 단독 학습 · E = ETH 학습 모형을 그 코인 피쳐에 · J = ETH·BTC·SOL·XRP 합동(코인별 행 [::4], y/그 코인 학습 창 표준편차).

    OMP_NUM_THREADS=3 python scripts/research_btc_sol_xrp_dir4h_20261031.py --selftest
    OMP_NUM_THREADS=3 python scripts/research_btc_sol_xrp_dir4h_20261031.py          # 피쳐 + 46달 적합 → tmp/btc_sol_xrp_dir4h_20261031/preds.npz
    OMP_NUM_THREADS=3 python scripts/research_btc_sol_xrp_dir4h_20261031.py --score  # 양성 대조 → 채점 → report.json
플래그: fresh_forward_bar_by_bar=true(월별 재학습 표본 밖 예측) · trade_ledgers_used_as_input=false · saved_parent_exit_timestamps_used=false · future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_dir4h_20261015 as d4  # noqa: E402
from research_eth_dir4h_reverify_20261019 import month_fixed_q, phase_daily, side, summ  # noqa: E402

# 같은 vision 5분 패널(20코인 2022-01-01~2026-10-27) = XRP 15분·1시간 연구(xrp_dir_short_20261031)가 쓴 파일
PANEL = Path(os.environ.get("DIR4H_PANEL", "/home/kbj20/crypto-scalping/.claude/worktrees/ema5-20-vwap-trading-rules-d6cb37/data/live/dirshort_panel"))
OOS_SEED = Path("/home/kbj20/crypto-scalping/data/live/eth_dir4h_oos_seed.parquet")   # 배포 B2 표본 밖 예측(2023-01~2026-09-14)
OUT = ROOT / "tmp/btc_sol_xrp_dir4h_20261031"
COINS4 = ["ETH", "BTC", "SOL", "XRP"]
ARMS = ["S", "E", "J"]
MONTHS = pd.date_range("2023-01-01", "2026-10-01", freq="MS")
VAL = (np.datetime64("2026-01-01"), np.datetime64("2026-06-01"))
TEST = (np.datetime64("2026-06-01"), np.datetime64("2026-10-28"))
H, PURGE = d4.H, np.timedelta64(5 * d4.H, "m")
NB, NP, BLK = 2000, 200, 5


def frames_for(allf: dict, target: str) -> dict:
    f = dict(allf)
    if target != "ETH":
        f["ETH"], f[target] = allf[target], allf["ETH"]
    return f


def features(allf: dict, target: str):
    F = d4.build_features(frames_for(allf, target))
    return F.ts.to_numpy(), F.y_4h.to_numpy(), F[d4.FEATS].to_numpy(np.float32)


def windows(ts, m):
    cut = np.datetime64(m) - PURGE
    tr = (ts >= cut - np.timedelta64(365, "D")) & (ts < cut)
    te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
    return tr, te


def hgb():
    return HistGradientBoostingRegressor(random_state=0, **d4.HGB)


def fit_month(D: dict, ts, m, out: dict, coins=COINS4):
    """그 달 te 행 예측을 out[f"{코인}_{팔}"] 에 쓴다. D[c] = (y, X)."""
    tr, te = windows(ts, m)
    trc = {c: tr & np.isfinite(D[c][0]) for c in COINS4}
    S = {c: hgb().fit(D[c][1][trc[c]][::2], D[c][0][trc[c]][::2]) for c in COINS4}
    Xs = np.vstack([D[c][1][trc[c]][::4] for c in COINS4])
    ys = np.concatenate([D[c][0][trc[c]][::4] / np.nanstd(D[c][0][trc[c]]) for c in COINS4])
    J = hgb().fit(Xs, ys)
    for c in coins:
        Xt = D[c][1][te]
        out[f"{c}_S"][te] = S[c].predict(Xt)
        out[f"{c}_E"][te] = S["ETH"].predict(Xt)
        out[f"{c}_J"][te] = J.predict(Xt)
    print(f"  {m:%Y-%m} 학습 {', '.join(f'{c} {int(trc[c].sum()):,}' for c in COINS4)} → 예측 {int(te.sum()):,}", flush=True)


def build_D():
    allf = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS}
    D, ts0 = {}, None
    for c in COINS4:
        ts, y, X = features(allf, c)
        assert ts0 is None or np.array_equal(ts, ts0), "코인별 5분 격자가 다르다"
        ts0 = ts; D[c] = (y, X)
        print(f"피쳐 {c} {len(ts):,}행 ~{pd.Timestamp(ts[-1])}", flush=True)
    return ts0, D


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    ts, D = build_D()
    np.savez_compressed(OUT / "feats.npz", ts=ts, **{f"y_{c}": D[c][0] for c in COINS4}, **{f"X_{c}": D[c][1] for c in COINS4})
    P = {f"{c}_{a}": np.full(len(ts), np.nan) for c in COINS4 for a in ARMS}
    for m in MONTHS:
        fit_month(D, ts, m, P)
    np.savez_compressed(OUT / "preds.npz", ts=ts, **{f"y_{c}": D[c][0] for c in COINS4}, **P)
    print("저장", OUT / "preds.npz")


# ── 채점 ──────────────────────────────────────────────────────────────────
def spear(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def block_rows(day):
    ud = np.unique(day)
    return [np.flatnonzero(np.isin(day, ud[s:s + BLK])) for s in range(0, len(ud), BLK)]


def hit_excess(p, y, lo, hi):
    """문턱 밖 행 부호 적중 − 같은 롱·숏 구성의 기저(같은 구간 상승 비율) 적중. 반환 (적중 수, 신호 수, 적중률, 기저, 초과)."""
    s = side(p, lo, hi); m = s != 0; up = y > 0
    hit = np.where(s[m] > 0, up[m], ~up[m]); pu = up.mean()
    base = np.where(s[m] > 0, pu, 1 - pu).mean() if m.any() else np.nan
    return int(hit.sum()), int(m.sum()), float(hit.mean()) if m.any() else np.nan, float(base), float(hit.mean() - base) if m.any() else np.nan


def boot(p, y, lo, hi, day, rng):
    rows = block_rows(day); ic, hx = np.empty(NB), np.empty(NB)
    for b in range(NB):
        i = np.concatenate([rows[k] for k in rng.integers(0, len(rows), len(rows))])
        ic[b] = spear(p[i], y[i]); hx[b] = hit_excess(p[i], y[i], lo[i], hi[i])[4]
    return ic, hx


def positive_control(ts, P, y):
    """배포 B2 표본 밖 예측(eth_dir4h_oos_seed) 재현: 학습 창이 2024-03-04(metrics 규약 전환) 이후인 2025-04~2026-09-14."""
    s = pd.read_parquet(OOS_SEED)
    seed = pd.Series(s.pred.to_numpy(float), index=s.ts).reindex(pd.DatetimeIndex(ts)).to_numpy()
    out = {}
    for lab, a, b in (("2025-04~2026-09-14", "2025-04-01", "2026-09-15"), ("2023-01~2025-03", "2023-01-01", "2025-04-01")):
        m = (ts >= np.datetime64(a)) & (ts < np.datetime64(b)) & np.isfinite(seed) & np.isfinite(P) & np.isfinite(y)
        out[lab] = {"n": int(m.sum()), "corr": round(float(np.corrcoef(P[m], seed[m])[0, 1]), 5),
                    "ic_mine": round(spear(P[m], y[m]), 4), "ic_seed": round(spear(seed[m], y[m]), 4),
                    "max_abs_diff_bp": round(float(np.abs(P[m] - seed[m]).max()), 3)}
    r = out["2025-04~2026-09-14"]
    out["pass"] = bool(r["corr"] >= .99 and abs(r["ic_mine"] - r["ic_seed"]) < .003)
    return out


def score_arm(ts, p, y, lo, hi):
    mv = (ts >= VAL[0]) & (ts < VAL[1]) & np.isfinite(y)
    mt = (ts >= TEST[0]) & (ts < TEST[1]) & np.isfinite(y)
    yr = pd.DatetimeIndex(ts).year
    r = {"val_ic": spear(p[mv], y[mv]), "test_ic": spear(p[mt], y[mt]),
         "test_pearson": float(np.corrcoef(p[mt], y[mt])[0, 1]),
         "year_ic": {str(Y): round(spear(p[(yr == Y) & np.isfinite(y) & np.isfinite(p)], y[(yr == Y) & np.isfinite(y) & np.isfinite(p)]), 4) for Y in (2023, 2024, 2025)},
         "month_ic": {f"{m:%Y-%m}": round(spear(p[w], y[w]), 4) for m in MONTHS[MONTHS >= "2026-01-01"]
                      for w in [(ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1))) & np.isfinite(y)] if w.sum() > 1000}}
    h = hit_excess(p[mt], y[mt], lo[mt], hi[mt])
    r["hit"] = {"hits": f"{h[0]}/{h[1]}", "rate": h[2], "base": h[3], "excess": h[4], "signal_share": h[1] / int(mt.sum())}
    hv = hit_excess(p[mv], y[mv], lo[mv], hi[mv]); r["val_hit"] = {"hits": f"{hv[0]}/{hv[1]}", "excess": hv[4]}
    s = side(p[mt], lo[mt], hi[mt]); g = s != 0
    r["legs_bp"] = {"model": float((s[g] * y[mt][g]).mean()), "all_long": float(y[mt][g].mean()), "all_short": float(-y[mt][g].mean()),
                    "long_share": float((s[g] > 0).mean())}
    r["pnl_4bp_phase8"] = summ(phase_daily(p, lo, hi, ts, y, str(TEST[0]), str(TEST[1])))
    return r


def score():
    z = np.load(OUT / "preds.npz"); ts = z["ts"]; rng = np.random.default_rng(20261031)
    rep = {"flags": {"fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
                     "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}}
    rep["positive_control"] = pc = positive_control(ts, z["ETH_S"], z["y_ETH"])
    print("양성 대조", json.dumps(pc, ensure_ascii=False), flush=True)
    assert pc["pass"], "ETH 배포 B2 표본 밖 예측 재현 실패 -- 새 코인 채점 전에 멈춘다"
    for c in COINS4:
        y = z[f"y_{c}"]; arms = ARMS if c != "ETH" else ["S"]
        R = {}
        for a in arms:
            p = z[f"{c}_{a}"]; lo, hi = month_fixed_q(p, ts, MONTHS)
            R[a] = score_arm(ts, p, y, lo, hi); R[a]["thr_test_first_month"] = [float(lo[ts >= TEST[0]][0]), float(hi[ts >= TEST[0]][0])]
        pick = max(arms, key=lambda a: R[a]["val_ic"])
        p = z[f"{c}_{pick}"]; lo, hi = month_fixed_q(p, ts, MONTHS)
        mt = (ts >= TEST[0]) & (ts < TEST[1]) & np.isfinite(y); day = pd.DatetimeIndex(ts[mt]).normalize().asi8
        bic, bhx = boot(p[mt], y[mt], lo[mt], hi[mt], day, rng)
        idx = np.flatnonzero((ts >= TEST[0]) & (ts < TEST[1])); ok = np.isfinite(y[idx]); nday = len(idx) // 288
        ks = rng.choice(np.arange(1, nday), size=min(NP, nday - 1), replace=False)
        circ = np.array([spear(np.roll(p[idx], int(k) * 288)[ok], y[idx][ok]) for k in ks])
        rnd = np.array([spear(rng.standard_normal(ok.sum()), y[idx][ok]) for _ in range(NP)])
        ic, hx = R[pick]["test_ic"], R[pick]["hit"]["excess"]
        v = {"pick": pick, "val_ic": {a: round(R[a]["val_ic"], 4) for a in arms}, "test_ic": ic,
             "ic_p10": float(np.quantile(bic, .10)), "ic_p05_p95": [float(np.quantile(bic, .05)), float(np.quantile(bic, .95))],
             "hit_excess": hx, "hit_p10": float(np.quantile(bhx, .10)),
             "circ_q90": float(np.quantile(circ, .90)), "circ_q95": float(np.quantile(circ, .95)), "circ_pct": float((circ < ic).mean()),
             "rand_q95": float(np.quantile(rnd, .95)), "n_circ": int(len(ks)), "n_test": int(mt.sum()), "test_days": int(len(np.unique(day))),
             "n_val": int(((ts >= VAL[0]) & (ts < VAL[1]) & np.isfinite(y)).sum())}
        strict = v["ic_p10"] > 0 and ic > v["circ_q95"] and ic > v["rand_q95"] and v["hit_p10"] > 0
        cand = (not strict) and ic > 0 and ic > v["circ_q90"] and hx > 0
        v["verdict"] = "엄격 통과" if strict else ("통과 후보" if cand else "불통과")
        if c == "ETH":
            v["verdict"] = "참고(배포 중 · 판정 대상 아님) -- " + v["verdict"]
        v["leak_tripwire_needed"] = bool(ic > .08)
        rep[c] = {"verdict": v, "arms": R}
        print(c, json.dumps({k: v[k] for k in ("pick", "val_ic", "verdict")}, ensure_ascii=False),
              f"시험 IC {ic:.4f} p10 {v['ic_p10']:.4f} · 적중 초과 {hx:+.4f} p10 {v['hit_p10']:+.4f} · 원형 95 {v['circ_q95']:.4f}", flush=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    print("저장", OUT / "report.json")


def selftest():
    allf = {s: pd.read_parquet(PANEL / f"{s}USDT.parquet") for s in ["ETH"] + d4.COINS}
    sub = {k: v[v.timestamp >= "2026-08-01"].reset_index(drop=True) for k, v in allf.items()}
    ts, y, X = features(sub, "BTC")
    btc = sub["BTC"].set_index("timestamp").close; eth = sub["ETH"].set_index("timestamp").close
    i = 9000; t = pd.Timestamp(ts[i]); prev = t - pd.Timedelta("5min")
    want = (np.log(btc[t]) - np.log(btc[prev])) * 1e4
    assert abs(X[i, d4.FEATS.index("P_r5m")] - want) < 1e-3                     # 대상 = BTC 자신
    assert abs(X[i, d4.FEATS.index("X_btc_r5m")] - (np.log(eth[t]) - np.log(eth[prev])) * 1e4) < 1e-3   # 교차 기준 = ETH
    assert abs(y[i] - (np.log(btc[t + pd.Timedelta(minutes=5 * H)]) - np.log(btc[t])) * 1e4) < 1e-3     # 표적 = 봉 τ 종가 → τ+48
    # 학습 행 라벨 끝(봉 시작 + 49봉) ≤ 예측 달 첫 봉 시작
    tsg = pd.date_range("2025-01-01", "2025-04-01", freq="5min").to_numpy()
    tr, te = windows(tsg, pd.Timestamp("2025-03-01"))
    assert tsg[tr].max() + PURGE + np.timedelta64(5, "m") <= tsg[te].min()
    # 적중 초과: 완벽 예측 > 0 · 무작위 ≈ 0
    rng = np.random.default_rng(0); yy = rng.standard_normal(20000) + .05
    lo, hi = np.full(20000, -.84), np.full(20000, .84)
    assert hit_excess(yy, yy, lo, hi)[4] > .4 and abs(hit_excess(rng.standard_normal(20000), yy, lo, hi)[4]) < .02
    k = slice(0, 3000); day = np.repeat(np.arange(30), 100)
    bi, bh = boot(yy[k], yy[k], lo[k], hi[k], day, np.random.default_rng(1))
    assert abs(np.median(bi) - 1) < 1e-9 and np.median(bh) > .4
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--score" in sys.argv:
        score()
    else:
        run()
