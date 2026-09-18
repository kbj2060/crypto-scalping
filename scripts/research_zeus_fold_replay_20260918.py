#!/usr/bin/env python3
"""Zeus 폴드 재생 엔진 (2026-09-18, 학습 0).

`research_zeus_n0x2_sltp_grid_20260917.py::seq` 와 «같은 숫자»를 내되, 집계 대신
**체결 원장**을 돌려준다. v4 동결 수치를 비트 단위로 재현하는 것이 계약이다
(F5 2.35건/일 +4.40bp · CAND 2.03 +11.87 — `--selfcheck` 가 확인한다).

왜 따로 있나: 격자 스크립트는 팔·배리어 비교용 집계만 내고, 차트/진단에 필요한
«언제 어느 방향으로 들어가 언제 나왔는가»를 남기지 않는다. 여기에만 그게 있다.

포함:
  panel()        라벨 조인된 5분봉 프레임 (폴드 슬라이스의 기준)
  first_touch()  더블배리어 -- K._first_touch_open 규약 그대로(진입=신호봉 종가,
                 스캔=다음 봉부터, 동시터치 SL 우선)
  trades()       게이트(롤링분위) -> 후보 -> 1슬롯 순차 -> 원장
  causal_state() 지그재그 상태기계의 «스트리밍» 판. 봉 i 까지만 보고 age/rtr/trend/conf.
                 🔴피벗 확정 지연이 중앙 11봉이라 «극점 직후 차단»은 인과적으로 불가능하다
                 (docs/experiments/zeus_label_and_gate_axis_20260918.md §4).
"""
from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
BASE = ROOT / "tmp/omega461_longwindow_20260917"
STAGE = BASE / "stageE"
SEEDS = [613042, 27851, 904377, 155690, 488213, 178618]
TP, SL, MAXBARS, ROLLQ, ROLLQQ = 0.015, 0.007, 2016, 1000, 0.85

# 폴드 TEST 창 -- research_omega461_side_skill_decomposition_20260917.FOLDS 와 같아야 한다.
FOLDS = {"F1": ("2023-07-01", "2023-12-31"), "F2": ("2024-01-01", "2024-06-30"),
         "F3": ("2024-07-01", "2024-12-31"), "F4": ("2025-01-01", "2025-06-30"),
         "F5": ("2025-07-01", "2025-12-31"), "CAND": ("2026-03-01", "2026-06-30"),
         "SHADOW": ("2026-07-01", "2026-08-19")}
# 라벨 변형별 확률 캐시와 게이트 점수. `--dirlabel=<name>` 학습이 만든 이름 규약을 따른다.
SPEC = {
    "v3": (STAGE / "stageP_probs_noreg_purge30.npz", "q"),
    "v4": (STAGE / "stageP_probs_noreg_purge30_ftall.npz", "edge"),
    "v5": (STAGE / "stageP_probs_noreg_purge30_dlzigzag_rev016_ftall.npz", "edge"),
    "v6": (STAGE / "stageP_probs_noreg_purge30_dlzigzag_softargmax_ftall.npz", "edge"),
    "v7": (STAGE / "stageP_probs_noreg_purge30_dlzigzag_b0soft_ftall.npz", "edge"),
}


def panel() -> pd.DataFrame:
    """피쳐 프레임 ∩ zigzag 라벨. 폴드 행수가 캐시 배열과 맞는 유일한 조합이다."""
    df = pd.read_parquet(BASE / "features_with_regime_2022_2026_realfunding.parquet",
                         columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df.timestamp).dt.tz_localize(None)
    lab = pd.concat([pd.read_csv(p, usecols=["timestamp"])
                     for p in sorted(glob.glob(str(BASE / "zigzag_labels_full/*.csv")))])
    lab["timestamp"] = pd.to_datetime(lab.timestamp).dt.tz_localize(None)
    return (df.merge(lab.drop_duplicates("timestamp"), on="timestamp")
            .sort_values("timestamp").reset_index(drop=True))


def first_touch(ei, side, hi, lo, cl, tp, sl, maxbars=MAXBARS, block=288):
    """시간 배리어 없는 더블배리어. 반환 (ret, hold봉, reason 0=SL/1=TP/2=미해소)."""
    n, m = len(cl), len(ei)
    e, sd = cl[ei], side[ei]
    ret = np.full(m, np.nan); hold = np.full(m, maxbars, np.int64)
    reason = np.full(m, 2, np.int8); alive = np.ones(m, bool)
    for st in range(0, maxbars, block):
        if not alive.any():
            break
        a = np.where(alive)[0]
        j = ei[a][:, None] + np.arange(st + 1, min(st + block, maxbars) + 1)[None, :]
        okj = j < n; j = np.clip(j, 0, n - 1)
        up = (hi[j] - e[a][:, None]) / e[a][:, None]
        dn = (lo[j] - e[a][:, None]) / e[a][:, None]
        gh = np.where(sd[a][:, None] > 0, up, -dn); gl = np.where(sd[a][:, None] > 0, dn, -up)
        gh = np.where(okj, gh, -np.inf); gl = np.where(okj, gl, np.inf)
        big = gh.shape[1] + 1
        t_sl = np.where((gl <= -sl).any(1), (gl <= -sl).argmax(1), big)
        t_tp = np.where((gh >= tp).any(1), (gh >= tp).argmax(1), big)
        hs = (t_sl < big) & (t_sl <= t_tp)              # 동시터치는 SL 우선(보수적)
        done = hs | ((t_tp < big) & ~hs)
        if done.any():
            k = a[done]
            ret[k] = np.where(hs[done], -sl, tp)
            hold[k] = st + 1 + np.where(hs[done], t_sl[done], t_tp[done])
            reason[k] = np.where(hs[done], 0, 1); alive[k] = False
        alive[a[~done & ~okj[:, -1]]] = False
    un = np.isnan(ret)
    if un.any():                                        # 미해소는 데이터 끝 종가로 청산
        k = np.where(un)[0]; je = np.minimum(ei[k] + maxbars, n - 1)
        ret[k] = sd[k] * (cl[je] - e[k]) / e[k]; hold[k] = je - ei[k]
    return ret, hold, reason


def thr_seq(qf_c, q=ROLLQQ, win=ROLLQ):
    """후보 계열의 인과적 롤링 분위. shift(1) 로 자기 자신을 빼고, 워밍업은 확장창."""
    s = pd.Series(qf_c).shift(1)
    t = s.rolling(win, min_periods=200).quantile(q)
    return t.fillna(s.expanding(min_periods=50).quantile(q)).to_numpy()


def gate_score(D, Q, kind):
    da = D.argmax(1); ar = np.arange(len(D))
    if kind == "edge":
        return da, D[ar, da] - D[:, 0]
    if kind == "dq":
        return da, D[ar, da] * np.where(da > 0, Q[ar, da], Q[:, 0])
    return da, np.where(da > 0, Q[ar, da], Q[:, 0])


def candidates(df, fold, ver, q=ROLLQQ, score=None):
    """게이트만 통과시킨 후보 (배리어 무관). 반환 (te, side 배열)."""
    cache, sc = SPEC[ver]
    sc = score or sc
    z = np.load(cache, allow_pickle=True)
    v0, v1 = FOLDS[fold]
    te = df[(df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")].reset_index(drop=True)
    P = [dict(z[f"{fold}|N1s{s}"].item()) for s in SEEDS]      # 6시드 확률 평균
    D = np.mean([p["D"] for p in P], 0); Q = np.mean([p["Q"] for p in P], 0)
    assert len(D) == len(te), f"{fold} 정렬 불일치 {len(D)} vs {len(te)}"
    da, qf = gate_score(D, Q, sc)
    m = da != 0
    t = thr_seq(qf[m], q)
    ok = np.zeros(len(da), bool); ok[np.where(m)[0]] = np.isfinite(t) & (qf[m] >= t)
    return te, np.where((da == 1) & ok, 1.0, np.where((da == 2) & ok, -1.0, 0.0))


def sequential(te, side, tp=TP, sl=SL):
    """슬롯 1개 순차 집행. 배리어 결과는 이전 거래와 무관하므로 일괄 계산 후 선택만 한다."""
    hi, lo, cl = (pd.to_numeric(te[c]).to_numpy(float) for c in ("high", "low", "close"))
    idx = np.where(side != 0)[0]
    r, h, rsn = first_touch(idx, side, hi, lo, cl, tp, sl)
    take, cur = [], -1
    for j in range(len(idx)):
        if idx[j] <= cur:
            continue
        take.append(j); cur = idx[j] + int(h[j])          # 청산 봉까지 슬롯 점유
    t = np.array(take, int)
    return pd.DataFrame({"i": idx[t], "ts": te.timestamp.to_numpy()[idx[t]],
                         "side": side[idx[t]], "entry": cl[idx[t]], "hold": h[t],
                         "bp": r[t] * 1e4, "exit_i": idx[t] + h[t],
                         "why": np.where(rsn[t] == 0, "SL", np.where(rsn[t] == 1, "TP", "TO"))}), len(idx)


def trades(df, fold, ver, q=ROLLQQ, score=None, tp=TP, sl=SL):
    te, side = candidates(df, fold, ver, q, score)
    tr, nsig = sequential(te, side, tp, sl)
    return te, tr, nsig


def causal_state(frame, min_reversal_pct=0.010, atr_window=14, atr_multiplier=0.0):
    """지그재그 상태기계의 스트리밍 판 -- 봉 i 는 i 까지만 본다.

    age=러닝 극단 갱신 후 봉수 · rtr=극단에서 되돌린 폭 · trend=상태 · conf=확정 후 봉수 ·
    lag=그 확정이 가리킨 피벗까지의 지연(확정 봉에만, 진단용) · thr=그 봉의 반전 문턱.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(np.float64)
    prev = np.roll(close, 1); prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = pd.Series(tr).ewm(span=atr_window, adjust=False, min_periods=1).mean().to_numpy()
    atr_pct = atr / np.maximum(close, 1e-12)
    n = len(close)
    age = np.zeros(n, np.int32); rtr = np.zeros(n); trend = np.zeros(n, np.int8)
    conf = np.full(n, 10 ** 6, np.int32); lag = np.full(n, -1, np.int32)
    t = 0; li = hi_ = 0; lp = hp = float(close[0]); last = -10 ** 6
    for i in range(1, n):
        p = float(close[i])
        if not np.isfinite(p):
            age[i], rtr[i], trend[i], conf[i] = age[i - 1], rtr[i - 1], trend[i - 1], conf[i - 1]
            continue
        thr = max(min_reversal_pct, float(atr_pct[i]) * atr_multiplier)
        if t == 0:
            if p < lp: li, lp = i, p
            if p > hp: hi_, hp = i, p
            if hp / max(lp, 1e-12) - 1.0 >= thr:
                if li < hi_: lag[i] = i - li; t, hi_, hp, last = 1, i, p, i
                else: lag[i] = i - hi_; t, li, lp, last = -1, i, p, i
        elif t == 1:
            if p > hp: hi_, hp = i, p
            if hp / max(p, 1e-12) - 1.0 >= thr:
                lag[i] = i - hi_; t, li, lp, last = -1, i, p, i
        else:
            if p < lp: li, lp = i, p
            if p / max(lp, 1e-12) - 1.0 >= thr:
                lag[i] = i - li; t, hi_, hp, last = 1, i, p, i
        ext_i, ext_p = (hi_, hp) if t == 1 else (li, lp)
        age[i] = i - ext_i
        rtr[i] = (ext_p / max(p, 1e-12) - 1.0) if t == 1 else (p / max(ext_p, 1e-12) - 1.0)
        trend[i] = t; conf[i] = i - last
    return pd.DataFrame({"age": age, "rtr": rtr, "trend": trend, "conf": conf, "lag": lag,
                         "thr": np.maximum(min_reversal_pct, atr_pct * atr_multiplier)})


def report(df, vers, folds, q=ROLLQQ, score=None, tp=TP, sl=SL):
    print(f"{'구성':>8}{'폴드':>8}{'신호':>8}{'체결':>7}{'체결/일':>9}{'건당bp':>9}{'TP율':>7}")
    out = {}
    for ver in vers:
        B, tot = [], 0
        for f in folds:
            te, tr, ns = trades(df, f, ver, q, score, tp, sl)
            d = (pd.Timestamp(FOLDS[f][1]) - pd.Timestamp(FOLDS[f][0])).days + 1
            print(f"{ver:>8}{f:>8}{ns:>8,}{len(tr):>7,}{len(tr)/d:>9.2f}"
                  f"{tr.bp.mean():>+9.2f}{(tr.why=='TP').mean()*100:>6.1f}%")
            B.append(tr.bp.to_numpy()); tot += d
        b = np.concatenate(B)
        print(f"{ver:>8}{'합계':>8}{'':>8}{len(b):>7,}{len(b)/tot:>9.2f}{b.mean():>+9.2f}\n")
        out[ver] = (len(b) / tot, b.mean())
    return out


SELFCHECK = {("v4", "F5"): (2.35, 4.40), ("v4", "CAND"): (2.03, 11.87)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vers", default="v4")
    ap.add_argument("--folds", default="F1,F2,F3,CAND,F4,F5")
    ap.add_argument("--score", default=None, choices=[None, "edge", "dq", "q"])
    ap.add_argument("--q", type=float, default=ROLLQQ)
    ap.add_argument("--tp", type=float, default=TP)
    ap.add_argument("--sl", type=float, default=SL)
    ap.add_argument("--selfcheck", action="store_true",
                    help="v4 동결 수치를 재현하는지 확인한다(계약)")
    a = ap.parse_args()
    df = panel()
    if a.selfcheck:
        bad = 0
        for (ver, f), (d_exp, bp_exp) in SELFCHECK.items():
            _, tr, _ = trades(df, f, ver)
            d = (pd.Timestamp(FOLDS[f][1]) - pd.Timestamp(FOLDS[f][0])).days + 1
            got = (len(tr) / d, tr.bp.mean())
            ok = abs(got[0] - d_exp) < 0.01 and abs(got[1] - bp_exp) < 0.01
            bad += not ok
            print(f"{'OK ' if ok else 'FAIL'} {ver} {f}: {got[0]:.2f}건/일 {got[1]:+.2f}bp "
                  f"(기대 {d_exp} / {bp_exp:+})")
        print("자체점검", "통과" if not bad else f"실패 {bad}건")
        return 1 if bad else 0
    report(df, a.vers.split(","), a.folds.split(","), a.q, a.score, a.tp, a.sl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
