#!/usr/bin/env python3
"""**틱 단위 스프레드 폭발 경보 — 트레이딩에 도움이 되는가** (2026-09-30)

사용자 질문 «트레이딩에 도움이 되는가». 모델 학습이 아니라 **경보 조건을 과거 bookTicker 에
적용해 그 뒤 가격을 세는** 검정이다. 스프레드는 **틱 수**로 본다(bp 스프레드는 1/가격 인공물).

원천: 서버 `data/live/orderflow/bookticker/ETHUSDT/*.bt(.gz)` (32B 고정폭, `live_book_ticker_collector_20260914`).
     ts = 거래소 T(ms). ts<=0 · bid<=0 · ask<=bid 행은 버리고 개수를 보고한다.

── 결과를 보기 전에 본 데이터 사실(정의를 고른 근거) ─────────────────────────────────────
  · 스프레드 ≥2틱은 **업데이트의 2.5~3.2%** 지만 **시간가중 0.03~0.07%**. 폭발 연속구간의 길이는
    중앙 0ms · p90 1~3ms · 100ms 이상 0건(표본 3시간) — 즉 폭발은 **밀리초 과도상태**(최우선 한 층이
    먹히고 같은 ms 안에 다시 채워짐)다. «n초 지속» 변형은 표본이 없어 정의하지 않는다.
  · 5초 묶음 후 사건 수가 k=2·5·10 에서 거의 같다(시간당 150~250) — 폭발은 대개 한 번에 여러 틱이다.

── 사전 고정 정의 ────────────────────────────────────────────────────────────────────────────
경보 변형 4개(트리거 = bookTicker 업데이트 한 행):
  k2 / k3 / k5 : 스프레드 ≥ k틱
  q999        : 스프레드 > Q 이고 ≥2틱. Q = **직전 완결 60분**(현재 분 제외) 업데이트 가중 스프레드
                분포의 99.9분위(누적비율 ≥0.999 가 되는 가장 작은 틱값). 그 60분 중 데이터 있는 분 <50 이면 무효.
사건 묶기: 트리거 시각 정렬 후 직전 트리거와 **5,000ms 초과** 간격이면 새 사건. 사건 시각 t_a = 첫 트리거 ts.
시간 격자: 초 s 의 가격 m(s) = 그 초 끝까지의 **마지막 1틱 호가의 중간가**(폭발 과도 mid 배제), 앞채움.
         업데이트 0건인 초 = 결측(수집 공백). 분석 초 s 는 [s−300, s+900] 에 결측 0 이어야 한다.
기준가·라벨(경보 **이후**): 경보 초 s_a 의 기준가 m(s_a) 는 반드시 **트리거 뒤** 같은 초 안에 나온
         1틱 호가여야 한다(아니면 사건 제외·개수 보고). 전방 라벨 = m(s_a+h), h ∈ {60, 300, 900}초 —
         **시각 기준**(행 이동 아님). 대조군도 같은 규약(초 끝 기준가)이라 대칭이다.
대조군: 같은 변형의 트리거가 [s−5초, s초 끝] 에 하나도 없는 초(= 그 순간 경보가 조용한 시각).
주장 a  «조기 경보»: |r_h| = |log m(s+h) − log m(s)|(bp)와 전방 실현변동성 RV_h(1초 수익 제곱합의 √).
         매칭 셀 = 같은 **달력 시각(UTC 날짜+시)** × 직전 5분 RV(1초 수익, [s−299, s−1]) 전체 10분위.
         효과 = 사건값 − 같은 셀 대조군 평균(대조 ≥10 셀만).
주장 a2 «이미 움직인 것 통제»: 셀에 경보 초 자체의 |Δm| 틱 구간(0 | 0.5–1 | 1.5–2 | 2.5–4 | 4.5–8 | 8.5–16 | >16)
         을 더한다. 「스프레드가 가격이 이미 움직였다는 것 이상을 말하는가」.
주장 b  «선행 vs 동행»: 1초 |수익| 프로파일 j = −60..+60 (j=0 은 경보가 든 초). 매칭 셀 = 날짜+시 × RV[s−299, s−61]
         10분위(프로파일 창을 매칭에 쓰지 않는다). 초과 = 사건 − 셀 대조 평균.
         판정(사전): 사후 초과합(+1..+60) CI 가 0 초과 **이고** 사전 초과합(−60..−1)의 2배 이상 **이고**
         경보초(j=0) 초과보다 크면 «선행». 아니면 «동행/후행»(= 경보의 값은 «이미 움직였다»뿐).
주장 c  방향: d_pre = sign(m(s) − m(s−60)) (경보초 포함, 기준가 시점에 알려짐), d_side = 폭발 쪽
         (+1: 직전 1틱 호가 대비 ask 가 더 벌어짐 = 위로 쓸림, −1: bid). 지속 = d×r_h(bp).
         d_pre 초과 = 사건 d·r − 셀 대조 평균 d·r · d_side 초과 = d_side·(r − 셀 대조 평균 r).
CI: **1시간 블록**(UTC 날짜+시) 부트스트랩 2,000회, 보조로 1일 블록. 독립 일수 보고.
앞/뒤 절반: 데이터 첫·끝 시각의 중점으로 사건을 나눠 효과 부호 일치를 본다.

⚠️라벨 창이 겹친다(사건 간격 ~18초 ≪ 900초). 그래서 사건 수가 아니라 블록 수가 검정력이다.

실행: python scripts/research_eth_spread_blowout_alert_20260930.py [--rebuild] | --selftest
출력: tmp/spread_blowout_alert_20260930/ (grid.npz 캐시 · report.json · report.txt)
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_eth_microprice_second_horizon_20260914 import read_bt  # noqa: E402  (.bt 리더 재사용)

BT_DIR = ROOT / "data/live/orderflow/bookticker/ETHUSDT"
OUT = ROOT / "tmp/spread_blowout_alert_20260930"
TICK = 0.01
NB = 128                      # 스프레드 히스토그램 칸(≥127틱은 마지막 칸)
GAP_MS = 5000
HS = (60, 300, 900)
PRE, POST = 300, 900
VARIANTS = ("k2", "k3", "k5", "q999")
R0_EDGES = np.array([0.25, 1.25, 2.25, 4.25, 8.25, 16.25])   # |Δm| 틱 구간 경계(중간가는 0.5틱 단위)
RUN_EDGES = np.array([1, 2, 5, 10, 50, 100, 250, 1000])       # 폭발 지속 ms 구간 경계


# ── 1. 파일 하나 → 압축 요약 ────────────────────────────────────────────────────────────────
def scan_arrays(ts, bid, ask):
    """정렬·청소된 틱에서 초 격자·분 히스토그램·후보 트리거(≥2틱)를 뽑는다."""
    bad_ts = int((ts <= 0).sum())
    ok = (ts > 0) & (bid > 0) & (ask > bid)
    crossed = int(((ts > 0) & ~ok).sum())
    ts, bid, ask = ts[ok], bid[ok], ask[ok]
    o = np.argsort(ts, kind="stable")
    ts, bid, ask = ts[o], bid[o], ask[o]
    sp = np.rint((ask - bid) / TICK).astype(np.int64)
    n = len(ts)
    sec = ts // 1000
    normal = sp == 1
    # 초별 업데이트 수 · 마지막 1틱 중간가
    usec, cnt = np.unique(sec, return_counts=True)
    ni = np.flatnonzero(normal)
    nsec = sec[ni]
    last = ni[np.r_[nsec[1:] != nsec[:-1], True]] if len(ni) else ni
    # 분 히스토그램(업데이트 가중) · 시간가중 분포
    mnt = ts // 60000
    umin, minv = np.unique(mnt, return_inverse=True)
    spc = np.minimum(sp, NB - 1)
    mhist = np.bincount(minv * NB + spc, minlength=len(umin) * NB).reshape(len(umin), NB)
    dur = np.diff(ts, append=ts[-1]) if n else ts
    tw = np.bincount(spc, weights=dur, minlength=NB)
    # 후보(≥2틱): 직전 1틱 호가 대비 폭발 쪽 · 같은 초 안에 뒤따르는 1틱 호가가 있는가
    idx = np.arange(n)
    prev_n = np.maximum.accumulate(np.where(normal, idx, -1)) if n else idx
    nxt = np.where(normal, idx, n)[::-1]
    next_n = np.minimum.accumulate(nxt)[::-1] if n else idx
    ci = np.flatnonzero(sp >= 2)
    pj, nj = prev_n[ci], next_n[ci]
    side = np.zeros(len(ci), np.int8)
    h = pj >= 0
    au = (ask[ci[h]] - ask[pj[h]]) / TICK
    bd = (bid[pj[h]] - bid[ci[h]]) / TICK
    side[h] = np.sign(au - bd).astype(np.int8)
    after = (nj < n) & (sec[np.minimum(nj, n - 1)] == sec[ci])
    # 폭발 연속구간 길이(k=2, k=5): 시작 → 다음 1틱 호가
    runs = {}
    for k in (2, 5):
        w = sp >= k
        st = np.flatnonzero(w & ~np.r_[False, w[:-1]])
        e = next_n[st]
        d = np.where(e < n, ts[np.minimum(e, n - 1)] - ts[st], -1)   # 파일 끝까지 안 풀림 = −1
        runs[k] = np.bincount(np.searchsorted(RUN_EDGES, d[d >= 0], side="right"),
                              minlength=len(RUN_EDGES) + 1)
    return dict(usec=usec, cnt=cnt, nsec=sec[last], nmid=(bid[last] + ask[last]) / 2,
                umin=umin, mhist=mhist, tw=tw, n=n, bad_ts=bad_ts, crossed=crossed,
                c_ts=ts[ci], c_sp=sp[ci].astype(np.int32), c_side=side, c_after=after,
                run2=runs[2], run5=runs[5])


def scan_file(p: Path):
    x = read_bt(p)
    r = scan_arrays(x["ts"], x["bid_px"], x["ask_px"])
    r["name"] = p.name
    return r


# ── 2. 파일 요약들 → 전체 격자 ───────────────────────────────────────────────────────────────
def assemble(res):
    s0 = min(int(r["usec"][0]) for r in res if len(r["usec"]))
    s1 = max(int(r["usec"][-1]) for r in res if len(r["usec"]))
    N = s1 - s0 + 1
    nupd = np.zeros(N, np.int64)
    m = np.full(N, np.nan)
    for r in res:
        nupd[r["usec"] - s0] += r["cnt"]
        m[r["nsec"] - s0] = r["nmid"]
    i = np.maximum.accumulate(np.where(np.isfinite(m), np.arange(N), 0))
    m = m[i]                                     # 앞채움(마지막 1틱 중간가, 초 끝 기준)
    m0 = min(int(r["umin"][0]) for r in res if len(r["umin"]))
    m1 = max(int(r["umin"][-1]) for r in res if len(r["umin"]))
    mh = np.zeros((m1 - m0 + 1, NB), np.int64)
    for r in res:
        mh[r["umin"] - m0] += r["mhist"]
    cat = lambda k: np.concatenate([r[k] for r in res])
    c = dict(ts=cat("c_ts"), sp=cat("c_sp"), side=cat("c_side"), after=cat("c_after"))
    o = np.argsort(c["ts"], kind="stable")
    c = {k: v[o] for k, v in c.items()}
    stats = dict(files=len(res), rows=int(sum(r["n"] for r in res)),
                 bad_ts=int(sum(r["bad_ts"] for r in res)), crossed=int(sum(r["crossed"] for r in res)),
                 upd_hist=mh.sum(0), tw_hist=np.sum([r["tw"] for r in res], 0),
                 run2=np.sum([r["run2"] for r in res], 0), run5=np.sum([r["run5"] for r in res], 0))
    return dict(s0=s0, nupd=nupd, m=m, m0=m0, mhist=mh, c=c, stats=stats)


def group_events(t_ms, gap_ms=GAP_MS):
    """정렬된 트리거 시각 → 사건 시작 위치(bool). 직전 트리거와 gap_ms **초과** 간격이면 새 사건."""
    return np.r_[True, np.diff(t_ms) > gap_ms] if len(t_ms) else np.zeros(0, bool)


def q999_threshold(mhist, need=50):
    """분별 Q = 직전 완결 60분 업데이트 가중 분포의 99.9분위(틱). 무효는 큰 값(트리거 불가)."""
    C = np.vstack([np.zeros((1, NB), np.int64), np.cumsum(mhist, 0)])
    M = len(mhist)
    Q = np.full(M, 10 ** 9)
    have = np.r_[0, np.cumsum(mhist.sum(1) > 0)]
    for mi in range(60, M):
        if have[mi] - have[mi - 60] < need:
            continue
        w = C[mi] - C[mi - 60]
        Q[mi] = int(np.argmax(np.cumsum(w) * 1000 >= 999 * w.sum()))   # 정수 비교(0.999 부동소수 경계 회피)
    return Q


def triggers(G, v):
    c = G["c"]
    if v == "q999":
        Q = q999_threshold(G["mhist"])
        mi = c["ts"] // 60000 - G["m0"]
        return c["sp"] > Q[mi]
    return c["sp"] >= int(v[1:])


# ── 3. 통계 도구 ───────────────────────────────────────────────────────────────────────────
def block_ci(vals, blocks, B=2000, seed=7):
    u, inv = np.unique(blocks, return_inverse=True)
    S = np.bincount(inv, vals)
    Nn = np.bincount(inv).astype(float)
    W = np.random.default_rng(seed).multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)
    boot = (W @ S) / (W @ Nn)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def cell_mean(y, cell, ctrl, ncell):
    s = np.bincount(cell[ctrl], y[ctrl], ncell)
    n = np.bincount(cell[ctrl], minlength=ncell)
    return s / np.maximum(n, 1), n


def summarize(e, ev_s, s0, t_mid_s):
    """효과 벡터 e(사건별) → 평균·1h CI·1일 CI·앞/뒤 절반."""
    hr = (ev_s + s0) // 3600
    dy = (ev_s + s0) // 86400
    mean, lo, hi = block_ci(e, hr)
    _, dlo, dhi = block_ci(e, dy)
    first = (ev_s + s0) < t_mid_s
    a, b = float(e[first].mean()), float(e[~first].mean())
    return dict(n=int(len(e)), mean=mean, ci=[lo, hi], ci_day=[dlo, dhi],
                first=a, second=b, sign_agree=bool(np.sign(a) == np.sign(b)))


# ── 4. 분석 ─────────────────────────────────────────────────────────────────────────────────
def analyze(G):
    s0, m, nupd = G["s0"], G["m"], G["nupd"]
    N = len(m)
    lm = np.log(m) * 1e4                                    # bp 단위 로그가격
    r1 = np.r_[np.nan, np.diff(lm)]
    gap = (nupd == 0) | ~np.isfinite(m)
    cg = np.r_[0, np.cumsum(gap)]
    s = np.arange(N)
    ok = (s >= PRE + 1) & (s + POST < N)
    ok[ok] = (cg[s[ok] + POST + 1] - cg[s[ok] - PRE]) == 0
    r1z = np.where(np.isfinite(r1), r1, 0.0)
    c2 = np.r_[0, np.cumsum(r1z ** 2)]                      # c2[i] = Σ r1[0..i-1]²
    sq = lambda a, b: np.sqrt(np.maximum(c2[np.clip(b + 1, 0, N)] - c2[np.clip(a, 0, N)], 0))  # Σ r1[a..b]²
    rv_pre = sq(s - 299, s - 1)
    rv_preb = sq(s - 299, s - 61)
    sh = lambda k: np.clip(s + k, 0, N - 1)
    y = {}
    for h in HS:
        y[f"absr{h}"] = np.abs(lm[sh(h)] - lm)
        y[f"rv{h}"] = sq(s + 1, s + h)
        y[f"r{h}"] = lm[sh(h)] - lm
    d_pre = np.sign(lm - lm[sh(-60)])
    r0t = np.abs(m - m[sh(-1)]) / TICK
    hr = (s + s0) // 3600 - (s0 // 3600)
    edges = np.quantile(rv_pre[ok], np.linspace(0, 1, 11)[1:-1])
    edges_b = np.quantile(rv_preb[ok], np.linspace(0, 1, 11)[1:-1])
    rvb = np.searchsorted(edges, rv_pre)
    rvbb = np.searchsorted(edges_b, rv_preb)
    r0b = np.searchsorted(R0_EDGES, r0t)
    nr0 = len(R0_EDGES) + 1
    cellA = hr * 10 + rvb
    cellA2 = cellA * nr0 + r0b
    cellB = hr * 10 + rvbb
    nA, nA2, nB = cellA.max() + 1, cellA2.max() + 1, cellB.max() + 1
    t_mid_s = (s0 + s0 + N - 1) / 2

    c = G["c"]
    csec = c["ts"] // 1000 - s0
    out = {}
    for v in VARIANTS:
        trig = triggers(G, v)
        tt, tsec = c["ts"][trig], csec[trig]
        st = group_events(tt)
        ev_t, ev_sec = tt[st], tsec[st]
        ev_after, ev_side = c["after"][trig][st], c["side"][trig][st]
        # 대조군: [s−5, s] 에 트리거 0
        tc = np.bincount(tsec, minlength=N)[:N]
        ct = np.r_[0, np.cumsum(tc)]
        quiet = (ct[s + 1] - ct[np.maximum(s - 5, 0)]) == 0
        ctrl = ok & quiet
        keep0 = ok[ev_sec]
        keep = keep0 & ev_after
        es = ev_sec[keep]
        esd = ev_side[keep]
        res = dict(n_events_raw=int(len(ev_t)), n_trig=int(trig.sum()),
                   n_dropped_gap=int((~keep0).sum()), n_dropped_no_after=int((keep0 & ~ev_after).sum()),
                   events_per_hour=float(len(ev_t) / (N / 3600)), n_ctrl_sec=int(ctrl.sum()),
                   days=int(len(np.unique((es + s0) // 86400))), hours=int(len(np.unique((es + s0) // 3600))),
                   side_up_share=float((esd > 0).mean()) if len(esd) else None,
                   ev_r0_ticks_med=float(np.median(r0t[es])) if len(es) else None,
                   ctrl_r0_ticks_med=float(np.median(r0t[ctrl])))
        if len(es) < 50:
            out[v] = res
            continue

        def matched(yy, cell, ncell, idx=es):
            cm, cn = cell_mean(yy, cell, ctrl, ncell)
            k = cn[cell[idx]] >= 10
            return k, yy[idx][k] - cm[cell[idx]][k], float(yy[idx][k].mean()), float(cm[cell[idx]][k].mean())

        A = {}
        for name, cell, ncell in (("a", cellA, nA), ("a2", cellA2, nA2)):
            for h in HS:
                for kind in ("absr", "rv"):
                    k, e, ev_m, ct_m = matched(y[f"{kind}{h}"], cell, ncell)
                    r = summarize(e, es[k], s0, t_mid_s)
                    r.update(ev_mean=ev_m, ctrl_mean=ct_m, ratio=ev_m / ct_m if ct_m else None,
                             n_drop_cell=int((~k).sum()))
                    A[f"{name}_{kind}{h}"] = r
            # 균형 점검: 매칭 셀 안 직전 RV 차
            k, e, ev_m, ct_m = matched(rv_pre, cell, ncell)
            A[f"{name}_balance_rvpre"] = dict(ev=ev_m, ctrl=ct_m)
        res["claim_a"] = A

        # 주장 b: 프로파일
        kB = np.bincount(cellB[ctrl], minlength=nB)[cellB[es]] >= 10
        eb = es[kB]
        prof, exc = {}, np.zeros((len(eb), 121))
        for j in range(-60, 61):
            yj = np.abs(r1[sh(j)])
            yj = np.where(np.isfinite(yj), yj, 0.0)
            cm, _ = cell_mean(yj, cellB, ctrl, nB)
            exc[:, j + 60] = yj[eb] - cm[cellB[eb]]
            prof[j] = dict(ev=float(yj[eb].mean()), ctrl=float(cm[cellB[eb]].mean()))
        B = dict(profile={j: prof[j] for j in (-60, -30, -20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20, 30, 60)},
                 pre=summarize(exc[:, :60].sum(1), eb, s0, t_mid_s),
                 j0=summarize(exc[:, 60], eb, s0, t_mid_s),
                 post=summarize(exc[:, 61:].sum(1), eb, s0, t_mid_s),
                 post_1_5=summarize(exc[:, 61:66].sum(1), eb, s0, t_mid_s),
                 pre_5_1=summarize(exc[:, 55:60].sum(1), eb, s0, t_mid_s))
        pre_m, post_m, j0_m = B["pre"]["mean"], B["post"]["mean"], B["j0"]["mean"]
        B["verdict"] = ("선행" if (B["post"]["ci"][0] > 0 and post_m >= 2 * max(pre_m, 0) and post_m > j0_m)
                        else "동행/후행")
        # 동행 진단(사후 추가, 판정엔 안 씀): 큰 1초 움직임이 난 초 중 트리거가 든 비율
        B["trig_share_given_move"] = {f">={t}틱": float((tc[ok & (r0t >= t)] > 0).mean())
                                      for t in (0.5, 4.5, 8.5, 16.5)}
        B["trig_share_all_sec"] = float((tc[ok] > 0).mean())
        res["claim_b"] = B

        # 주장 c: 방향
        Cc = {}
        for h in HS:
            yr = y[f"r{h}"]
            k, e, ev_m, ct_m = matched(d_pre * yr, cellA, nA)
            r = summarize(e, es[k], s0, t_mid_s)
            r.update(ev_mean=ev_m, ctrl_mean=ct_m)
            Cc[f"dpre_r{h}"] = r
            cm, cn = cell_mean(yr, cellA, ctrl, nA)
            k = (cn[cellA[es]] >= 10) & (esd != 0)
            e = esd[k] * (yr[es][k] - cm[cellA[es]][k])
            Cc[f"side_r{h}"] = summarize(e, es[k], s0, t_mid_s)
            Cc[f"side_r{h}"]["hit"] = float((np.sign(yr[es][k]) == esd[k]).mean())
        res["claim_c"] = Cc
        out[v] = res
    return out


# ── 5. 보고 ─────────────────────────────────────────────────────────────────────────────────
def data_table(G):
    st = G["stats"]
    uh, th = st["upd_hist"].astype(float), st["tw_hist"].astype(float)
    frac = lambda h: {"1": h[1] / h.sum(), "2": h[2] / h.sum(), "3+": h[3:].sum() / h.sum(),
                      "5+": h[5:].sum() / h.sum(), "20+": h[20:].sum() / h.sum()}
    N = len(G["m"])
    gap = int((G["nupd"] == 0).sum())
    run_lbl = ["0", "1", "2-4", "5-9", "10-49", "50-99", "100-249", "250-999", "1000+"]
    return dict(source="server data/live/orderflow/bookticker/ETHUSDT", files=st["files"],
                start_utc=str(np.datetime64(G["s0"], "s")), end_utc=str(np.datetime64(G["s0"] + N - 1, "s")),
                hours=N / 3600, rows=st["rows"], bad_ts=st["bad_ts"], crossed_or_locked=st["crossed"],
                gap_seconds=gap, gap_share=gap / N,
                spread_update_weighted=frac(uh), spread_time_weighted=frac(th),
                run_ms_k2=dict(zip(run_lbl, st["run2"].tolist())), run_ms_k5=dict(zip(run_lbl, st["run5"].tolist())))


def fmt(r, scale=1.0):
    return (f"{r['mean']*scale:+.3f} [{r['ci'][0]*scale:+.3f},{r['ci'][1]*scale:+.3f}] "
            f"일블록[{r['ci_day'][0]*scale:+.3f},{r['ci_day'][1]*scale:+.3f}] "
            f"앞{r['first']*scale:+.3f}/뒤{r['second']*scale:+.3f}{'' if r['sign_agree'] else ' ✗'}")


def render(D, R):
    L = ["== 데이터 ==", json.dumps(D, ensure_ascii=False, indent=1)]
    for v, r in R.items():
        L.append(f"\n== 변형 {v}: 트리거 {r['n_trig']:,} · 사건 {r['n_events_raw']:,} ({r['events_per_hour']:.1f}/시간) · "
                 f"공백제외 {r['n_dropped_gap']} · 뒤따르는 1틱호가 없음 {r['n_dropped_no_after']} · "
                 f"{r['days']}일 {r['hours']}시간 · 위쪽폭발 {r['side_up_share']} · 경보초 |Δm| 중앙 {r['ev_r0_ticks_med']}틱 (대조 {r['ctrl_r0_ticks_med']})")
        if "claim_a" not in r:
            continue
        for name in ("a", "a2"):
            for h in HS:
                for kind in ("absr", "rv"):
                    x = r["claim_a"][f"{name}_{kind}{h}"]
                    L.append(f"  {name} {kind}{h:>4}  n{x['n']:>6} 사건 {x['ev_mean']:.2f} 대조 {x['ctrl_mean']:.2f}bp "
                             f"×{x['ratio']:.3f}  차 {fmt(x)}  (셀없음 {x['n_drop_cell']})")
            b = r["claim_a"][f"{name}_balance_rvpre"]
            L.append(f"  {name} 균형: 직전5분RV 사건 {b['ev']:.2f} 대조 {b['ctrl']:.2f}")
        B = r["claim_b"]
        L.append("  b 프로파일 |r1s| bp (사건/대조): " + " ".join(
            f"{j}:{p['ev']:.2f}/{p['ctrl']:.2f}" for j, p in B["profile"].items()))
        for key in ("pre", "pre_5_1", "j0", "post_1_5", "post"):
            L.append(f"  b {key:>9} 초과합 {fmt(B[key])}")
        L.append(f"  b 판정: {B['verdict']}")
        L.append(f"  b 1초 |Δm| 조건부 트리거 동반율 {B['trig_share_given_move']} (전체 초 {B['trig_share_all_sec']:.3f})")
        for h in HS:
            L.append(f"  c d_pre·r{h:<4} 초과 {fmt(r['claim_c'][f'dpre_r{h}'])}")
            x = r["claim_c"][f"side_r{h}"]
            L.append(f"  c side·r{h:<4} 초과 {fmt(x)} 적중 {x['hit']:.3f}")
    return "\n".join(L)


def build(rebuild: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "grid.npz"
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return z["G"].item()
    files = sorted(p for p in BT_DIR.glob("*.bt*") if not p.name.startswith("."))   # rsync 임시파일 제외
    assert files, f"bookTicker 파일 없음: {BT_DIR}"
    with Pool(6) as pool:
        res = []
        for i, r in enumerate(pool.imap(scan_file, files, chunksize=2)):
            res.append(r)
            if i % 50 == 0:
                print(f"  스캔 {i+1}/{len(files)} {r['name']} {r['n']:,}행", flush=True)
    G = assemble(res)
    np.savez(cache, G=np.array(G, dtype=object))
    return G


# ── 6. 자체점검 ─────────────────────────────────────────────────────────────────────────────
def selftest():
    # 사건 묶기: 5,000ms 초과만 새 사건
    t = np.array([0, 1000, 5999, 10999, 16000, 16001])
    assert group_events(t).tolist() == [True, False, False, False, True, False]
    # 합성 스트림: 30초, 100ms 마다 1틱 호가(mid 100.005). 폭발 두 번.
    ts = np.arange(0, 30000, 100, dtype=np.int64) + 1_700_000_000_000
    bid = np.full(len(ts), 100.00)
    ask = np.full(len(ts), 100.01)
    base = int(ts[0])
    up = ts > base + 5300                    # 폭발 뒤 가격이 2틱 올라간 세계 — 기준가가 «경보 뒤»인지 가려낸다
    bid[up] += 0.02
    ask[up] += 0.02
    # ① 5.300초: ask 가 5틱 벌어짐 → 5.301초 1틱 복귀(같은 초 안에 뒤따르는 1틱 호가 있음)
    ins = [(base + 5300, 100.00, 100.05), (base + 5301, 100.02, 100.03),
           # ② 25.999초: bid 가 7틱 벌어짐 → 다음 1틱 호가는 26.000초(다음 초) ⇒ 기준가가 경보 뒤가 아님 → 제외
           (base + 25999, 99.96, 100.03),
           # ③ 깨진 행
           (0, 100.0, 100.01), (base + 7000, 100.01, 100.01)]
    ts = np.r_[ts, [x[0] for x in ins]]
    bid = np.r_[bid, [x[1] for x in ins]]
    ask = np.r_[ask, [x[2] for x in ins]]
    r = scan_arrays(ts, bid, ask)
    assert r["bad_ts"] == 1 and r["crossed"] == 1
    assert r["c_ts"].tolist() == [base + 5300, base + 25999]
    assert r["c_after"].tolist() == [True, False], r["c_after"]
    assert r["c_side"].tolist() == [1, -1]
    G = assemble([r])
    s_a = (base + 5300) // 1000 - G["s0"]
    # 경보 초의 기준가는 경보 **뒤** 1틱 호가(100.025), 직전 초는 경보 전 가격(100.005)
    assert abs(G["m"][s_a] - 100.025) < 1e-9, G["m"][s_a]
    assert abs(G["m"][s_a - 1] - 100.005) < 1e-9
    # 폭발 과도 호가(스프레드≥2, mid 99.995)는 격자 mid 에 들어가지 않는다
    s_b = (base + 25999) // 1000 - G["s0"]
    assert abs(G["m"][s_b] - 100.025) < 1e-9, G["m"][s_b]
    # q999: 직전 60분이 모자라면 무효
    assert (q999_threshold(np.ones((70, NB), np.int64)) [:60] == 10 ** 9).all()
    Q = q999_threshold(np.tile(np.r_[0, 998, 1, 1, np.zeros(NB - 4)].astype(np.int64), (70, 1)))
    assert Q[60] == 2 and Q[69] == 2, Q[60:]
    # 블록 CI: 상수면 폭 0
    mn, lo, hi = block_ci(np.full(10, 3.0), np.arange(10) // 2)
    assert mn == lo == hi == 3.0
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    G = build(a.rebuild)
    D = data_table(G)
    R = analyze(G)
    txt = render(D, R)
    print(txt)
    (OUT / "report.txt").write_text(txt)
    (OUT / "report.json").write_text(json.dumps(dict(data=D, results=R), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
