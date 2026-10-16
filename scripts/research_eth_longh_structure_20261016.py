#!/usr/bin/env python3
"""과거 불합격·단기 전용 «구조/플레이북/확장» 4가족을 긴 지평(4h·12h·1d·3d)으로 다시 잰다 (2026-10-16).

가족(정의는 원 메모·원 스크립트 그대로, 봉 크기만 키움 -- 봉 수 규약은 유지):
  F1 플레이북  trader_playbooks_yush_carmine_nill_rejected_20261004
     YushA  4h 봉 · 전일 VA(1분 tp 프로파일 0.05% 칸 70%) 끝 거부 → POC 쪽. 일·측면당 첫 건.
     YushB  일봉 · 전주(월 00:00 UTC 시작) VA 끝 거부 → POC 쪽. 주·측면당 첫 건.
     Carm   4h 봉 · 직전 7일 프로파일 LVN(market_ctx.profile_nodes) 을 1.5×ATR(48봉) 봉이 관통 → 12봉 안 재시험 방어 → 임펄스 쪽.
     N1/N2  Nill PBD 를 4h 봉에(ATR 96봉 · 8봉 임펄스 4ATR · 8봉 범위 · 96봉 안 첫 거부). N1 역추세 · N2 추세 쪽.
     N2w    N2 + 주간 VA 정렬(새 정의): 롱은 진입가 > 전주 POC, 숏은 < 전주 POC.
     보고 전용: 원판 손절/목표(최대 보유 3d, 1분 고·저, 같은 분 = 손절).
  F2 볼륨 프로파일  volume_profile_rules_5_rejected_20261004 -- 1d·3d, 세션 = 일(d) / 주(w)
     R80  시가가 전 세션 VA 밖 → 30분 연속 VA 안 첫 분 → 반대편 VA 끝 쪽.
     nPOC 아직 안 닿은 전 세션 POC(나이 ≤10 세션) 0.5% 안 첫 진입 분 → POC 쪽.
     VAmig VA 위/아래 이동(겹침 < 0.5) → 세션 끝에 그 방향.   Pb  (POC−저)/(고−저) > 2/3 롱 · < 1/3 숏.
     비방향(보고 표 따로): H3b VA 겹침 → 다음 H 변동성 ↓ (이동일 대비, RV 5분위 층화) · H5 손절 HVN 너머 vs LVN 안 터치율.
  F3 가격×OI 사분면  price_oi_quadrant_works_at_1h_as_exit_20260921 -- 창 W = H. 하락 = Δ가격 < 직전 365일 25분위,
     상승 = > 75분위(🔴원판은 전 표본 분위 -- 미래 분위라 직전 365일로 바꿈). ΔOI 부호로 가름. 진입 = 다음 5분봉 종가(원판).
     DnOIdn 롱(소진 되돌림) · DnOIup 숏(신규 숏 지속) · UpOIup 롱 · UpOIdn 숏.
     🔴OI 스탬프: 2024-03-04 부터 행 t = t+5분 스냅샷 → 봉 τ 에 쓰는 OI = 봉 τ 시작 스냅샷(그 전 행 τ, 그 뒤 행 τ−1).
  F4 소진형 대량 델타 지속  exhaustion_delta_4h_reversal_prereg_rejected_20260930 정의 그대로(5분봉, 288봉 기준선) · 방향 = 델타 쪽.
     🔴ETH 전진 사전등록(2026-10-01 이후 사건) 보호: ETH 데이터는 2026-09-30 23:59 에서 자른다(사건·청산 모두 그 전).
     ETH 2020–21 은 이전 세션이 봤다 -- 여기 데이터는 2022~ 라 안 쓴다. ETH 4h 2022~26 도 이전 CONFIRM 에서 봤다(참고 칸).
     BTC = 독립 복제(패널 5분봉 2022-01~2026-09-14).

공통 규약: 사건은 봉 τ 마감까지의 데이터만, 수익은 τ 뒤(진입 = τ 종가, F3·F4 는 τ+1 종가 -- 원판). 청산 = H 시간 청산.
비용 4bp 왕복. MAIN = 2025-01-01~ · AUX = 그 전. 블록 부트스트랩(일, 3d 는 주) 포아송 B=1000.
위약 = MAIN 안 같은 봉 격자 무작위 시각 같은 건수·같은 방향 구성(방향 순열) P=200회 · 차 CI = 부트스트랩 b − 위약 b.
PASS = MAIN 순 평균 > 0 & CI 하한 > 0 & 위약 차 CI 하한 > 0 & AUX 순 평균 > 0.
추세 상관 = 일별 시가평가 손익 vs «28일 수익 부호를 하루 보유» 일 손익(MAIN 일).

실행: nice -n 10 env OMP_NUM_THREADS=3 python scripts/research_eth_longh_structure_20261016.py [--selftest]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/home/kbj20/crypto-scalping")          # 원천 데이터는 메인 체크아웃에만 있다
sys.path.insert(0, str(ROOT / "dashboard"))
import market_ctx as MC                              # noqa: E402  value_area · profile_nodes · PROFILE_BIN_FRAC

OUT = ROOT / "tmp/longh_structure_20261016"
K1M = REPO / "data/binance_vision/klines1m"
ZIPS = ROOT / "tmp/pingpong_hedge_20261016/vision_1m"
PANEL = REPO / "data/binance_vision/panel"
T0 = pd.Timestamp("2022-01-01")
MAIN0 = np.datetime64("2025-01-01", "ms")
OI_SWITCH = np.datetime64("2024-03-04", "ms")
ETH_CUT = np.datetime64("2026-10-01", "ms")
COST = 4.0
HS = {"4h": 240, "12h": 720, "1d": 1440, "3d": 4320}       # 분
D1, W1, WOFF = 1440, 10080, 2880                           # 2022-01-03(월) 00:00 = 분 2880
BUF = 5e-4
B, P = 1000, 200
swv = np.lib.stride_tricks.sliding_window_view


# ───────────────────────────── 데이터 ─────────────────────────────
def load_eth_1m() -> dict:
    fs = sorted(glob.glob(str(K1M / "ETHUSDT-1m-*.parquet")))
    parts = [pd.read_parquet(f, columns=["t", "h", "l", "c", "v", "tb"]) for f in fs]
    for z in sorted(ZIPS.glob("ETHUSDT-1m-*.zip")):
        with zipfile.ZipFile(z) as f:
            x = pd.read_csv(f.open(f.namelist()[0]), usecols=["open_time", "high", "low", "close", "volume", "taker_buy_volume"])
        parts.append(x[["open_time", "high", "low", "close", "volume", "taker_buy_volume"]].set_axis(["t", "h", "l", "c", "v", "tb"], axis=1))
    m = pd.concat(parts).drop_duplicates("t").sort_values("t")
    t0 = int(T0.value // 10**6)
    m = m[m.t >= t0].set_index("t")
    t1 = (int(m.index.max()) // 86_400_000 + 1) * 86_400_000
    m = m.reindex(np.arange(t0, t1, 60_000))
    c = m.c.ffill()
    return dict(t=m.index.to_numpy().astype("datetime64[ms]"), c=c.to_numpy(float), h=m.h.fillna(c).to_numpy(float),
                l=m.l.fillna(c).to_numpy(float), v=m.v.fillna(0).to_numpy(float), tb=m.tb.fillna(0).to_numpy(float),
                ok=m.c.notna().to_numpy())


def agg(D: dict, k: int) -> dict:
    """k분봉. 행 i = 분 [k·i, k·i+k), 마감 분 = k·i+k−1. o = 직전 봉 종가."""
    n = len(D["c"]) // k * k
    r = lambda a: D[a][:n].reshape(-1, k)          # noqa: E731
    c = r("c")[:, -1]
    return dict(h=r("h").max(1), l=r("l").min(1), c=c, o=np.r_[np.nan, c[:-1]], v=r("v").sum(1),
                delta=(2 * r("tb") - r("v")).sum(1), t=D["t"][:n:k])


# ───────────────────────────── 평가 엔진 ─────────────────────────────
def boot(v, blk, seed=1):
    u, inv = np.unique(blk, return_inverse=True)
    s, k = np.bincount(inv, v), np.bincount(inv).astype(float)
    W = np.random.default_rng(seed).poisson(1.0, (B, len(u))).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s.sum() / k.sum(), (W @ s) / (W @ k)


def greedy(e, x):
    """시간순으로 겹치지 않게 순차 선택(직전 청산 뒤 첫 진입만)."""
    keep, last = np.zeros(len(e), bool), -1
    for i in np.argsort(e, kind="stable"):
        if e[i] >= last:
            keep[i] = True; last = x[i]
    return keep


_DAYC: dict = {}


def day_tools(t):
    key = (id(t), len(t))
    if key not in _DAYC:
        day = t.astype("datetime64[D]")
        du, first = np.unique(day, return_index=True)
        _DAYC[key] = (du, np.searchsorted(du, day), np.r_[first[1:] - 1, len(t) - 1])
    return _DAYC[key]


def daily_mtm(t, c, e, x, side):
    """일별 시가평가 손익(bp, 거래 단위 합) · 28일 추세 기준선 일 손익."""
    du, did, last = day_tools(t)
    lc = np.log(c); pnl = np.zeros(len(du))
    span = int((did[x] - did[e]).max()) if len(e) else 0
    for k in range(span + 1):
        d = did[e] + k; m = d <= did[x]
        dm = d[m]
        s = np.maximum(e[m], np.where(dm > 0, last[np.maximum(dm - 1, 0)], 0))
        f = np.minimum(x[m], last[dm])
        np.add.at(pnl, dm, np.where(f > s, side[m] * (lc[f] - lc[s]) * 1e4, 0.0))
    bl = np.full(len(du), np.nan)
    L = last
    bl[29:] = np.sign(lc[L[28:-1]] - lc[L[:-29]]) * (lc[L[29:]] - lc[L[28:-1]]) * 1e4
    return du, pnl, bl


def evaluate(fam, rule, H, t, c, idx, side, hb, off=0, grid=1, cut=None, gx=None, seed=7, note=""):
    """idx = 결정 봉(그 봉 마감까지 앎), 진입 = idx+off 종가, 청산 = +hb 봉 종가. gx = (총 bp, 청산 인덱스) 덮어쓰기(손절/목표판)."""
    idx = np.asarray(idx, np.int64); side = np.asarray(side, float)
    e = idx + off; x = e + hb if gx is None else np.asarray(gx[1], np.int64)
    lim = len(c) if cut is None else int(np.searchsorted(t, cut))
    ok = (e >= 0) & (x < lim)
    if gx is not None:
        g_all = np.asarray(gx[0], float)
    e, x, side = e[ok], x[ok], side[ok]
    g = side * (c[x] / c[e] - 1) * 1e4 if gx is None else g_all[ok]
    net = g - COST; te = t[e]
    blk = te.astype("datetime64[D]").astype(np.int64)
    if H == "3d":
        blk = (blk + 3) // 7                         # 주 블록(월요일 시작)
    mn, ax = te >= MAIN0, te < MAIN0
    row = dict(fam=fam, rule=rule, H=H, n=int(len(e)), n_main=int(mn.sum()), n_aux=int(ax.sum()), note=note)
    if mn.sum() < 10:
        row.update(verdict="FAIL(n<10)"); return row
    mu, dist = boot(net[mn], blk[mn], seed)
    lo, hi = np.nanpercentile(dist, [2.5, 97.5])
    # 위약: MAIN 안 같은 격자·유효 청산 무작위 시각, 같은 방향 구성
    rng = np.random.default_rng(seed + 1)
    res = int(np.bincount(e[mn] % grid).argmax())
    hb_p = hb if gx is None else max(1, int(np.median(x[mn] - e[mn])))   # 손절/목표판 위약 = 같은 중앙 보유의 시간 청산(보고 전용)
    ar = np.arange(max(0, lim - hb_p))
    cand = ar[(t[ar] >= MAIN0) & (ar % grid == res)]
    sm, nm = side[mn], int(mn.sum())
    pm = np.empty(P)
    for p in range(P):
        pk = rng.choice(cand, nm, replace=True)
        pm[p] = np.mean(rng.permutation(sm) * (c[pk + hb_p] / c[pk] - 1) * 1e4) - COST
    dd = dist - pm[np.arange(B) % P]
    dlo, dhi = np.nanpercentile(dd, [2.5, 97.5])
    aux_mu = float(net[ax].mean()) if ax.sum() >= 10 else np.nan
    keep = greedy(e, x)
    k_m = keep & mn
    nmu, ndist = boot(net[k_m], blk[k_m], seed + 2) if k_m.sum() >= 5 else (np.nan, np.array([np.nan]))
    yrs = te.astype("datetime64[Y]").astype(int) + 1970
    per_year = {int(y): round(float(net[yrs == y].mean()), 1) for y in np.unique(yrs)}
    du, pnl, bl = daily_mtm(t, c, e, x, side)
    dm = (du >= MAIN0.astype("datetime64[D]")) & np.isfinite(bl)
    tc = float(np.corrcoef(pnl[dm], bl[dm])[0, 1]) if pnl[dm].std() > 0 else np.nan
    passed = bool(mu > 0 and lo > 0 and dlo > 0 and np.isfinite(aux_mu) and aux_mu > 0)
    row.update(gross_main=round(float(g[mn].mean()), 2), net_main=round(float(mu), 2), ci=[round(float(lo), 2), round(float(hi), 2)],
               plac_net=round(float(pm.mean()), 2), diff=round(float(mu - pm.mean()), 2), diff_ci=[round(float(dlo), 2), round(float(dhi), 2)],
               net_aux=None if not np.isfinite(aux_mu) else round(aux_mu, 2), gross_aux=None if ax.sum() < 10 else round(float(g[ax].mean()), 2),
               n_seq=int(keep.sum()), n_seq_main=int(k_m.sum()),
               seq_net_main=None if not np.isfinite(nmu) else round(float(nmu), 2),
               seq_ci=[round(float(v), 2) for v in np.nanpercentile(ndist, [2.5, 97.5])] if k_m.sum() >= 5 else None,
               per_year=per_year, trend_corr=None if not np.isfinite(tc) else round(tc, 3),
               verdict="PASS" if passed else "FAIL")
    return row


def simulate(h, l, c, j, side, stop, tgt, tmax):
    """c[j] 진입, j+1~j+tmax 1분 고·저로 손절/목표 첫 닿음(같은 분 = 손절), 아니면 c[j+tmax]. 반환 (총 bp, 청산 분)."""
    e = c[j]; end = min(j + tmax, len(c) - 1)
    hh, ll = h[j + 1:end + 1], l[j + 1:end + 1]
    hs, ht = (ll <= stop, hh >= tgt) if side > 0 else (hh >= stop, ll <= tgt)
    i_s = int(np.argmax(hs)) if hs.any() else 10**9
    i_t = int(np.argmax(ht)) if ht.any() else 10**9
    if min(i_s, i_t) == 10**9:
        return side * (c[end] / e - 1) * 1e4, end
    x, k = (stop, i_s) if i_s <= i_t else (tgt, i_t)
    return side * (x / e - 1) * 1e4, j + 1 + k


# ───────────────────────────── 프로파일 ─────────────────────────────
def sessions(D: dict, start0: int, L: int) -> dict:
    """start0 분부터 길이 L 세션들. 행 s 는 세션 s **자신**의 VA -- 쓸 때는 s+1 에서."""
    c, h, l, v = D["c"], D["h"], D["l"], D["v"]
    tp = (h + l + c) / 3
    ns = (len(c) - start0) // L
    S = {k: np.full(ns, np.nan) for k in ("val", "poc", "vah", "hi", "lo", "last", "rate")}
    S["start"] = start0 + np.arange(ns) * L; S["L"] = L
    r2 = np.r_[0.0, np.diff(np.log(c))] ** 2
    for s in range(ns):
        a = start0 + s * L; b = a + L; vv = v[a:b]; m = vv > 0
        if m.sum() < L // 2:
            continue
        bw = c[b - 1] * MC.PROFILE_BIN_FRAC
        p = MC.value_area(np.floor(tp[a:b][m] / bw).astype(np.int64), vv[m], bw)
        if p:
            S["val"][s], S["poc"][s], S["vah"][s] = p["val"], p["poc"], p["vah"]
        S["hi"][s], S["lo"][s], S["last"][s] = h[a:b].max(), l[a:b].min(), c[b - 1]
        S["rate"][s] = np.sqrt(r2[a + 1:b].mean())
    return S


def nodes_4h(D: dict, win: int) -> tuple[list, list]:
    """4h 경계 b 마다 [240b−win, 240b) 1분 tp 프로파일의 HVN·LVN 가격(market_ctx.profile_nodes)."""
    c, h, l, v = D["c"], D["h"], D["l"], D["v"]
    tp = (h + l + c) / 3; nb = len(c) // 240
    HV, LV = [np.empty(0)] * nb, [np.empty(0)] * nb
    for b in range(win // 240, nb):
        a, z = 240 * b - win, 240 * b
        vv = v[a:z]; m = vv > 0
        if m.sum() < win // 2:
            continue
        bw = c[z - 1] * MC.PROFILE_BIN_FRAC
        nd = MC.profile_nodes(np.floor(tp[a:z][m] / bw).astype(np.int64), vv[m], bw)
        HV[b] = np.array([q["px"] for q in nd["hvn"]]); LV[b] = np.array([q["px"] for q in nd["lvn"]])
    return HV, LV


# ───────────────────────────── F1 플레이북 ─────────────────────────────
def ev_yush(D, k, S):
    """k분봉 i 의 세션 s = 봉 시작 분이 속한 세션. 직전 봉 종가가 전 세션 VA 안 → 고가 ≥ VAH·종가 < VAH = 숏(거울 롱)."""
    f = agg(D, k); Hh, Ll, C = f["h"], f["l"], f["c"]; out, seen = [], set()
    st0, L = int(S["start"][0]), S["L"]
    for i in range(1, len(C)):
        s = (k * i - st0) // L
        if s < 1 or s >= len(S["val"]) or not np.isfinite(S["vah"][s - 1]):
            continue
        val, poc, vah = S["val"][s - 1], S["poc"][s - 1], S["vah"][s - 1]
        if not val < C[i - 1] < vah:
            continue
        for side, hit, back in ((-1, Hh[i] >= vah, C[i] < vah), (1, Ll[i] <= val, C[i] > val)):
            if not (hit and back) or (s, side) in seen:
                continue
            seen.add((s, side))
            if side * (poc - C[i]) <= 0:
                continue
            stop = Hh[i] * (1 + BUF) if side < 0 else Ll[i] * (1 - BUF)
            out.append((k * i + k - 1, side, stop, poc))
    return out


def ev_carmine(D, LV):
    f = agg(D, 240); Hh, Ll, C, O = f["h"], f["l"], f["c"], f["o"]
    atr = pd.Series(np.fmax(Hh - Ll, np.abs(Hh - O))).rolling(48).mean().shift(1).to_numpy()
    h1, l1 = D["h"], D["l"]; out, armed = [], []
    for i in range(1, len(C)):
        j = 240 * i + 239
        armed = [a for a in armed if a[3] >= i]
        for lv, side, j0, _ in armed:                 # 방어(진입) 먼저 -- 이 봉의 새 임펄스로는 진입 안 함
            if not ((Ll[i] <= lv < C[i]) if side > 0 else (Hh[i] >= lv > C[i])):
                continue
            e = C[i]
            stop = min(Ll[i], lv) * (1 - BUF) if side > 0 else max(Hh[i], lv) * (1 + BUF)
            ext = h1[j0:j + 1].max() if side > 0 else l1[j0:j + 1].min()
            if side * (ext - e) < abs(e - stop):
                continue
            out.append((j, side, stop, ext)); armed = []
            break
        if i < len(LV) and np.isfinite(atr[i]) and Hh[i] - Ll[i] >= 1.5 * atr[i]:
            for lv in LV[i]:
                if C[i - 1] < lv < C[i]:
                    armed.append((lv, 1, 240 * i, i + 12))
                elif C[i - 1] > lv > C[i]:
                    armed.append((lv, -1, 240 * i, i + 12))
    return out


def ev_nill(D, mode, k=240):
    f = agg(D, k); n = len(f["c"])
    c, h, l, o = f["c"], f["h"], f["l"], f["o"]
    atr = pd.Series(np.fmax(h - l, np.abs(h - o))).rolling(96).mean().shift(1).to_numpy()
    out = []; t = 104
    while t < n - 9:
        imp = c[t] - c[t - 8]
        if not (np.isfinite(atr[t]) and abs(imp) >= 4 * atr[t]):
            t += 1; continue
        s = np.sign(imp); org = c[t - 8]
        RH, RL = h[t + 1:t + 9].max(), l[t + 1:t + 9].min()
        retr = (c[t] - RL) / imp if s > 0 else (RH - c[t]) / -imp
        if RH - RL > 0.5 * abs(imp) or retr >= 0.5:
            t += 1; continue
        w = RH - RL; k0 = t + 9; nxt = k0 + 96
        for q in range(k0, min(k0 + 96, n)):
            if mode == "N1":
                ok = (h[q] >= RH > c[q]) if s > 0 else (l[q] <= RL < c[q])
                side, stop, tgt = -s, (RH + 0.25 * w if s > 0 else RL - 0.25 * w), org
            else:
                ok = (l[q] <= RL < c[q]) if s > 0 else (h[q] >= RH > c[q])
                side, stop, tgt = s, (RL - 0.25 * w if s > 0 else RH + 0.25 * w), (RH if s > 0 else RL)
            if ok:
                if side * (tgt - c[q]) > 0 and side * (c[q] - stop) > 0:
                    out.append((k * q + k - 1, side, stop, tgt))
                nxt = q + 1
                break
        t = nxt
    return out


def weekly_align(tr, D, SW):
    out = []
    for j, side, stop, tgt in tr:
        w = (j - WOFF) // W1
        if w >= 1 and np.isfinite(SW["poc"][w - 1]) and side * (D["c"][j] - SW["poc"][w - 1]) > 0:
            out.append((j, side, stop, tgt))
    return out


# ───────────────────────────── F2 볼륨 프로파일 ─────────────────────────────
def ev_r80(D, S, run=30):
    c, L, out = D["c"], S["L"], []
    for s in range(1, len(S["val"])):
        val, vah, opn = S["val"][s - 1], S["vah"][s - 1], S["last"][s - 1]
        if not (np.isfinite(val) and np.isfinite(opn)):
            continue
        side = -1 if opn > vah else (1 if opn < val else 0)
        if side == 0:
            continue
        a = int(S["start"][s]); cc = c[a:a + L]
        cs = np.r_[0, np.cumsum(~((cc >= val) & (cc <= vah)))]
        m = np.flatnonzero(cs[run:] - cs[:-run] == 0)          # m+run−1 = 30 연속 안 마지막 분
        m = m[m + run - 1 < L - 30]
        if len(m):
            out.append((a + int(m[0]) + run - 1, side))
    return out


def ev_npoc(D, S, near=0.005, age=10):
    h, l, c, L, n = D["h"], D["l"], D["c"], S["L"], len(D["c"])
    out = []
    for d in range(len(S["poc"]) - 1):
        lv = S["poc"][d]
        if not np.isfinite(lv):
            continue
        a = int(S["start"][d + 1]); b = min(n, a + age * L)
        hh, ll, cc = h[a:b], l[a:b], c[a:b]
        tch = (ll <= lv) & (hh >= lv); dist = np.abs(cc / lv - 1)
        tau = int(np.argmax(tch)) if tch.any() else b - a
        cp = np.r_[abs(c[a - 1] / lv - 1), dist[:-1]]
        cand = np.flatnonzero((dist <= near) & (cp > near)); cand = cand[cand < tau]
        if len(cand):
            j = int(cand[0]); out.append((a + j, 1 if lv > cc[j] else -1))
    return out


def va_class(S):
    val, poc, vah = S["val"], S["poc"], S["vah"]
    pv, ph = np.r_[np.nan, val[:-1]], np.r_[np.nan, vah[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        ov = np.clip(np.minimum(vah, ph) - np.maximum(val, pv), 0, None) / (np.maximum(vah, ph) - np.minimum(val, pv))
        up = (val >= pv) & (vah > ph) & (ov < 0.5); dn = (val <= pv) & (vah < ph) & (ov < 0.5); bal = ov >= 0.5
        x = (poc - S["lo"]) / (S["hi"] - S["lo"])
    return up, dn, bal, x


def ev_vamig(S):
    up, dn, _, _ = va_class(S); end = S["start"] + S["L"] - 1
    return [(int(end[s]), 1 if up[s] else -1) for s in range(len(up)) if up[s] or dn[s]]


def ev_pb(S):
    _, _, _, x = va_class(S); end = S["start"] + S["L"] - 1
    return [(int(end[s]), 1 if x[s] > 2 / 3 else -1) for s in range(len(x)) if np.isfinite(x[s]) and (x[s] > 2 / 3 or x[s] < 1 / 3)]


def strat_diff(y, g, blk, z, nq=5, seed=7):
    """g=1 − g=0 평균 차, z(사건 기준 분위) 층마다 차 → 사건 비중 평균. 블록 포아송 부트스트랩."""
    y, g, z = np.asarray(y, float), np.asarray(g, int), np.asarray(z, float)
    b = np.searchsorted(np.nanquantile(z[g == 1], np.linspace(0, 1, nq + 1)[1:-1]), z) if nq > 1 else np.zeros(len(y), int)
    _, inv = np.unique(blk, return_inverse=True); Dn = inv.max() + 1
    ix = inv * nq + b
    A = lambda w: np.bincount(ix, weights=w, minlength=Dn * nq).reshape(Dn, nq)   # noqa: E731
    S1, N1, S0, N0 = A(y * (g == 1)), A((g == 1) * 1.0), A(y * (g == 0)), A((g == 0) * 1.0)

    def stat(W):
        s1, n1, s0, n0 = W @ S1, W @ N1, W @ S0, W @ N0
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s1 / n1 - s0 / n0
        okk = (n1 > 0) & (n0 > 0); p = np.where(okk, n1, 0); p = p / p.sum(-1, keepdims=True)
        return np.nansum(np.where(okk, p * d, 0), -1)
    W = np.random.default_rng(seed).poisson(1.0, (B, Dn)).astype(float)
    lo, hi = np.nanpercentile(stat(W), [2.5, 97.5])
    return float(stat(np.ones((1, Dn)))[0]), float(lo), float(hi)


def nd_row(fam, rule, H, est, lo, hi, n1, n0, aux, sign, note=""):
    ok = (lo > 0) if sign > 0 else (hi < 0)
    auxok = aux is not None and np.isfinite(aux[0]) and np.sign(aux[0]) == sign
    return dict(fam=fam, rule=rule, H=H, n_main=int(n1), n_ctl=int(n0), est=round(est, 4), ci=[round(lo, 4), round(hi, 4)],
                aux=None if aux is None else [round(v, 4) for v in aux], sign=sign,
                verdict="PASS" if ok and auxok else "FAIL", note=note)


def h3b(D, S, hm):
    """VA 겹침 세션 → 다음 hm 분 RV(분당) 비 log 가 이동 세션보다 작나. 층 = RV / 직전 30세션 중앙 5분위."""
    up, dn, bal, _ = va_class(S)
    r2 = np.r_[0.0, np.diff(np.log(D["c"]))] ** 2; G = np.r_[0.0, np.cumsum(r2)]
    end = S["start"] + S["L"]; okn = end + hm < len(G)
    nxt = np.full(len(end), np.nan)
    nxt[okn] = np.sqrt((G[end[okn] + hm] - G[end[okn]]) / hm)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.log(nxt / S["rate"])
        lvl = S["rate"] / pd.Series(S["rate"]).rolling(30, min_periods=20).median().shift(1).to_numpy()
    te = D["t"][np.minimum(end - 1, len(D["t"]) - 1)]
    m = np.isfinite(y) & np.isfinite(lvl) & (bal | up | dn)
    res = {}
    for nm, pm in (("main", te >= MAIN0), ("aux", te < MAIN0)):
        k = m & pm
        res[nm] = (strat_diff(y[k], bal[k].astype(int), np.arange(len(y))[k], lvl[k]), int((bal & k).sum()), int((~bal & k).sum()))
    return res


def h5(D, HV, LV, win_tag, hm, dist, d_h=0.0015, d_l=0.0010):
    """4h 경계 직전 분 종가 진입(롱·숏) · 손절 거리 dist · hm 분 안 터치율: HVN 너머 − LVN 안(방향×RV24h 5분위 층)."""
    c, h, l = D["c"], D["h"], D["l"]; nb = len(c) // 240
    lc = np.log(c); r2 = np.r_[0.0, np.diff(lc)] ** 2; G = np.r_[0.0, np.cumsum(r2)]
    rows = []
    for b in range(42, nb):
        j = 240 * b - 1
        if j + hm >= len(c) or (not len(HV[b]) and not len(LV[b])):
            continue
        fmin, fmax = l[j + 1:j + hm + 1].min(), h[j + 1:j + hm + 1].max()
        rv = np.sqrt(G[j + 1] - G[j + 1 - 1440])
        for side in (1, -1):
            sp = c[j] * (1 - side * dist)
            hv, lv = HV[b], LV[b]
            beyond = len(hv) and np.any((side * (hv - sp) > 0) & (side * (c[j] - hv) > 0) & (np.abs(hv / sp - 1) <= d_h))
            inside = len(lv) and np.any(np.abs(lv / sp - 1) <= d_l)
            cls = 1 if beyond else (0 if inside else -1)
            if cls < 0:
                continue
            hit = (fmin <= sp) if side > 0 else (fmax >= sp)
            rows.append((j, side, cls, float(hit), rv))
    a = np.array(rows)
    out = {}
    te = D["t"][a[:, 0].astype(np.int64)]
    for nm, pm in (("main", te >= MAIN0), ("aux", te < MAIN0)):
        x = a[pm]
        q = np.searchsorted(np.nanquantile(x[:, 4], [0.2, 0.4, 0.6, 0.8]), x[:, 4])
        z = (x[:, 1] > 0) * 5 + q                         # 방향×RV 5분위 = 10층 (strat_diff 에는 층 번호를 z 로)
        blk = te[pm].astype("datetime64[D]").astype(np.int64)
        out[nm] = (strat_diff_cat(x[:, 3], x[:, 2].astype(int), blk, z.astype(int)), int((x[:, 2] == 1).sum()), int((x[:, 2] == 0).sum()))
    return out


def strat_diff_cat(y, g, blk, cat, seed=7):
    _, inv = np.unique(blk, return_inverse=True); Dn = inv.max() + 1; nc = int(cat.max()) + 1
    ix = inv * nc + cat
    A = lambda w: np.bincount(ix, weights=w, minlength=Dn * nc).reshape(Dn, nc)   # noqa: E731
    S1, N1, S0, N0 = A(y * (g == 1)), A((g == 1) * 1.0), A(y * (g == 0)), A((g == 0) * 1.0)

    def stat(W):
        s1, n1, s0, n0 = W @ S1, W @ N1, W @ S0, W @ N0
        with np.errstate(invalid="ignore", divide="ignore"):
            d = s1 / n1 - s0 / n0
        okk = (n1 > 0) & (n0 > 0); p = np.where(okk, n1, 0); p = p / p.sum(-1, keepdims=True)
        return np.nansum(np.where(okk, p * d, 0), -1)
    W = np.random.default_rng(seed).poisson(1.0, (B, Dn)).astype(float)
    lo, hi = np.nanpercentile(stat(W), [2.5, 97.5])
    return float(stat(np.ones((1, Dn)))[0]), float(lo), float(hi)


# ───────────────────────────── F3 사분면 ─────────────────────────────
def oi_known(t, oi_raw):
    """봉 τ 에 쓰는 OI = 봉 τ 시작 스냅샷. 2024-03-04 전 행 τ = τ 스냅샷 · 그 뒤 행 τ = τ+5분 → 행 τ−1."""
    return np.where(t >= OI_SWITCH, np.r_[np.nan, oi_raw[:-1]], oi_raw)


def ev_quadrant(t, c, oik, W, qwin=365 * 288, qmin=90 * 288):
    lc = np.log(c)
    dp = np.r_[np.full(W, np.nan), lc[W:] - lc[:-W]] * 1e4
    with np.errstate(invalid="ignore", divide="ignore"):
        do = np.r_[np.full(W, np.nan), oik[W:] / oik[:-W] - 1]
    s = pd.Series(dp)
    q25 = s.rolling(qwin, min_periods=qmin).quantile(0.25).shift(1).to_numpy()
    q75 = s.rolling(qwin, min_periods=qmin).quantile(0.75).shift(1).to_numpy()
    ok = np.isfinite(dp) & np.isfinite(do) & np.isfinite(q25)
    dn, up = ok & (dp < q25), ok & (dp > q75)
    return {"DnOIdn": (np.flatnonzero(dn & (do < 0)), 1), "DnOIup": (np.flatnonzero(dn & (do > 0)), -1),
            "UpOIup": (np.flatnonzero(up & (do > 0)), 1), "UpOIdn": (np.flatnonzero(up & (do < 0)), -1)}


# ───────────────────────────── F4 소진형 ─────────────────────────────
N_EX, MULT, RQ, DEDUP = 288, 20.0, 0.90, 48


def ev_exhaust(hi, lo, c, delta):
    """원판 events_a 그대로(직전 288봉·t 제외 기준선) + 48봉 중복 제거. 반환 (k, 방향=델타 부호)."""
    d = pd.DataFrame({"high": hi, "low": lo, "close": c, "delta": delta})
    ad = d.delta.abs(); med = ad.shift(1).rolling(N_EX, min_periods=N_EX).median()
    rng = (d.high - d.low) / d.close; rq = rng.shift(1).rolling(N_EX, min_periods=N_EX).quantile(RQ)
    hi24 = d.high.shift(1).rolling(N_EX, min_periods=N_EX).max(); lo24 = d.low.shift(1).rolling(N_EX, min_periods=N_EX).min()
    s = np.sign(d.delta); pre = np.sign(d.close.shift(1) - d.close.shift(13))
    ev = (ad >= MULT * med) & (rng >= rq) & (((s > 0) & (d.high >= hi24)) | ((s < 0) & (d.low <= lo24))) & (pre == s) & (s != 0)
    ks, last = [], -10**9
    for k in np.flatnonzero(ev.fillna(False).to_numpy(bool)):
        if k - last > DEDUP:
            ks.append(k); last = k
    ks = np.array(ks, np.int64)
    return ks, np.sign(delta[ks]) if len(ks) else np.empty(0)


def load_btc_5m():
    d = pd.read_parquet(PANEL / "BTCUSDT.parquet", columns=["timestamp", "high", "low", "close", "volume", "taker_buy_base"])
    d = d.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="5min"))
    return dict(t=d.index.to_numpy().astype("datetime64[ms]"), h=d.high.to_numpy(float), l=d.low.to_numpy(float),
                c=d.close.to_numpy(float), delta=(2 * d.taker_buy_base - d.volume).to_numpy(float))


# ───────────────────────────── 실행 ─────────────────────────────
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nd_rows, rep = [], [], {}
    say = lambda *a: print(*a, flush=True)   # noqa: E731
    D = load_eth_1m(); t, c = D["t"], D["c"]
    say(f"ETH 1분 {len(c):,}행 {t[0]} ~ {t[-1]} · 빈 분 {int((~D['ok']).sum()):,}")

    # F4 먼저(ETH 자르기 전 원 프레임에서 5분 집계 → 2026-10-01 전에서 자름)
    f5 = agg(D, 5); m = f5["t"] < ETH_CUT
    e5 = {k: f5[k][m] for k in ("t", "h", "l", "c", "delta")}
    ks, sd = ev_exhaust(e5["h"], e5["l"], e5["c"], e5["delta"])
    old = (e5["t"][ks] >= np.datetime64("2022-01-01")) & (e5["t"][ks] <= np.datetime64("2026-09-15T23:55"))
    say(f"F4 ETH 사건 {len(ks)} (원판 재구성 점검: 2022-01-01~2026-09-15 {int(old.sum())}건 vs 원판 1,102)")
    rep["F4_eth_recon"] = int(old.sum())
    bt = load_btc_5m(); kb, sb = ev_exhaust(bt["h"], bt["l"], bt["c"], bt["delta"])
    say(f"F4 BTC 사건 {len(kb)} · {bt['t'][0]} ~ {bt['t'][-1]}")
    for H, hm in HS.items():
        rows.append(evaluate("F4", "ExhCont_ETH", H, e5["t"], e5["c"], ks, sd, hm // 5, off=1, cut=ETH_CUT,
                             note="4h=이전 CONFIRM 에서 이미 봄" if H == "4h" else ""))
        rows.append(evaluate("F4", "ExhCont_BTC", H, bt["t"], bt["c"], kb, sb, hm // 5, off=1))
    say("F4 끝")

    # F3 사분면 (패널 5분)
    pn = pd.read_parquet(PANEL / "ETHUSDT.parquet", columns=["timestamp", "close", "sum_open_interest"])
    pn = pn.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    pn = pn.reindex(pd.date_range(pn.index.min(), pn.index.max(), freq="5min"))
    pt = pn.index.to_numpy().astype("datetime64[ms]"); pc = pn.close.ffill().to_numpy(float)
    oik = oi_known(pt, pn.sum_open_interest.to_numpy(float))
    rep["F3_diff_down"] = {}
    for H, hm in HS.items():
        Q = ev_quadrant(pt, pc, oik, hm // 5)
        for nm, (ix, sg) in Q.items():
            rows.append(evaluate("F3", nm, H, pt, pc, ix, np.full(len(ix), sg), hm // 5, off=1))
        # 원판 지표(보고 전용): 하락 안 OI↓ − OI↑ 앞 수익(총, 롱 기준)
        hb = hm // 5; fw = lambda ix: (pc[ix + 1 + hb] / pc[ix + 1] - 1) * 1e4   # noqa: E731
        for per, f in (("main", lambda ix: ix[(pt[ix] >= MAIN0) & (ix + 1 + hb < len(pc))]), ("aux", lambda ix: ix[pt[ix] < MAIN0])):
            a, b = f(Q["DnOIdn"][0]), f(Q["DnOIup"][0])
            rep["F3_diff_down"][f"{H}_{per}"] = round(float(fw(a).mean() - fw(b).mean()), 2)
        say(f"F3 {H} 끝")
    del pn, oik

    # F1 플레이북
    SD, SW = sessions(D, 0, D1), sessions(D, WOFF, W1)
    say("세션 프로파일 끝")
    HVd, LVd = nodes_4h(D, D1); HVw, LVw = nodes_4h(D, 7 * D1)
    say("4h 노드 끝")
    F1 = {"YushA": ev_yush(D, 240, SD), "YushB": ev_yush(D, D1, SW), "Carm": ev_carmine(D, LVw),
          "N1": ev_nill(D, "N1"), "N2": ev_nill(D, "N2")}
    F1["N2w"] = weekly_align(F1["N2"], D, SW)
    for nm, tr in F1.items():
        if not tr:
            say(f"F1 {nm}: 0건"); continue
        j = np.array([x[0] for x in tr], np.int64); sg = np.array([x[1] for x in tr], float)
        grid = 240 if nm != "YushB" else D1
        for H, hm in HS.items():
            rows.append(evaluate("F1", nm, H, t, c, j, sg, hm, grid=grid))
        sim = [simulate(D["h"], D["l"], c, int(x[0]), x[1], x[2], x[3], HS["3d"]) for x in tr]
        rows.append(evaluate("F1", nm + "_stoptgt", "≤3d", t, c, j, sg, HS["3d"], grid=grid,
                             gx=(np.array([s[0] for s in sim]), np.array([s[1] for s in sim])), note="보고 전용(원판 손절/목표, 최대 3d)"))
    say("F1 끝")

    # F2 볼륨 프로파일 (방향)
    for tag, S in (("d", SD), ("w", SW)):
        evs = {"R80": ev_r80(D, S), "nPOC": ev_npoc(D, S), "VAmig": ev_vamig(S), "Pb": ev_pb(S)}
        for nm, tr in evs.items():
            j = np.array([x[0] for x in tr], np.int64); sg = np.array([x[1] for x in tr], float)
            grid = 1 if nm in ("R80", "nPOC") else (D1 if tag == "d" else W1)
            for H in ("1d", "3d"):
                rows.append(evaluate("F2", f"{nm}_{tag}", H, t, c, j, sg, HS[H], grid=grid))
        for H in ("1d", "3d"):
            r = h3b(D, S, HS[H])
            nd_rows.append(nd_row("F2", f"H3b_{tag}", H, *r["main"][0], r["main"][1], r["main"][2], r["aux"][0], -1,
                                  note="log(RV 다음 H / RV 세션) 겹침 − 이동"))
    for tag, (HV, LV) in (("d24h", (HVd, LVd)), ("w7d", (HVw, LVw))):
        for H in ("1d", "3d"):
            sc = np.sqrt(HS[H] / 240)
            for d0 in (0.005, 0.010):
                r = h5(D, HV, LV, tag, HS[H], d0 * sc)
                nd_rows.append(nd_row("F2", f"H5_{tag}_D{d0 * sc * 100:.2f}%", H, *r["main"][0], r["main"][1], r["main"][2], r["aux"][0], -1,
                                      note="터치율 HVN 너머 − LVN 안"))
    say("F2 끝")
    return rows, nd_rows, rep


def fmt(rows, nd_rows, rep):
    L = ["fam rule H | n(all/seq) main/aux | MAIN 총 | MAIN 순 [CI] | 위약차 [CI] | AUX 순 | seq MAIN 순 [CI] | 연도 순 | 추세ρ | 판정"]
    for r in rows:
        if "net_main" not in r:
            L.append(f"{r['fam']} {r['rule']} {r['H']} | n {r['n']} main {r['n_main']} | {r['verdict']}"); continue
        yr = " ".join(f"{str(k)[2:]}:{v:+.0f}" for k, v in r["per_year"].items())
        L.append(f"{r['fam']} {r['rule']} {r['H']} | {r['n']}/{r['n_seq']} {r['n_main']}/{r['n_aux']} | {r['gross_main']:+.1f} | "
                 f"{r['net_main']:+.1f} [{r['ci'][0]:+.1f},{r['ci'][1]:+.1f}] | {r['diff']:+.1f} [{r['diff_ci'][0]:+.1f},{r['diff_ci'][1]:+.1f}] | "
                 f"{r['net_aux'] if r['net_aux'] is None else format(r['net_aux'], '+.1f')} | "
                 f"{r['n_seq_main']}: {r['seq_net_main']} {r['seq_ci']} | {yr} | {r['trend_corr']} | {r['verdict']} {r['note']}")
    L.append("\n비방향(H3b·H5): est [CI] main · aux")
    for r in nd_rows:
        L.append(f"{r['rule']} {r['H']} | n {r['n_main']}/{r['n_ctl']} | {r['est']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] | aux {r['aux']} | {r['verdict']}")
    cnt = [r for r in rows if "≤3d" != r["H"]]
    L.append(f"\n방향 칸 {len(cnt)} · PASS {sum(r['verdict'] == 'PASS' for r in cnt)} · 비방향 {len(nd_rows)} PASS {sum(r['verdict'] == 'PASS' for r in nd_rows)}"
             f" · 보고 전용(손절/목표) {len(rows) - len(cnt)} PASS {sum(r['verdict'] == 'PASS' for r in rows if r['H'] == '≤3d')}")
    L.append(f"보고: {json.dumps(rep, ensure_ascii=False)}")
    return "\n".join(L)


# ───────────────────────────── 자체 점검 ─────────────────────────────
def synth(days=70, seed=0):
    rng = np.random.default_rng(seed); n = days * D1
    c = 2000 * np.exp(np.cumsum(rng.normal(0, 8e-4, n)))
    v = rng.gamma(2.0, 50.0, n)
    return dict(t=(np.datetime64("2022-01-01T00:00", "ms") + np.arange(n) * np.timedelta64(60_000, "ms")),
                c=c, h=c * (1 + np.abs(rng.normal(0, 4e-4, n))), l=c * (1 - np.abs(rng.normal(0, 4e-4, n))),
                v=v, tb=v * rng.uniform(0.3, 0.7, n), ok=np.ones(n, bool))


def perturb(D, cut, seed=99):
    """분 cut 뒤 전부를 다른 경로로 바꾼다(cut 까지는 그대로)."""
    E = synth(len(D["c"]) // D1, seed)
    out = {k: v.copy() for k, v in D.items()}
    sc = D["c"][cut] / E["c"][cut]
    for k in ("c", "h", "l"):
        out[k][cut + 1:] = E[k][cut + 1:] * sc
    out["v"][cut + 1:] = E["v"][cut + 1:] * 3; out["tb"][cut + 1:] = E["tb"][cut + 1:] * 3
    return out


def selftest():
    # 1) 엔진: 수익 부호·진입 지연·순차 비중첩·일별 시가평가 합 = 거래 수익
    n = 20 * D1; t = np.datetime64("2025-01-01", "ms") + np.arange(n) * np.timedelta64(60_000, "ms")
    c = 100 * np.exp(np.arange(n) * 1e-5)
    e = np.array([10, 100, 3000]); x = e + 1440
    assert greedy(e, x).tolist() == [True, False, True]
    du, pnl, _ = daily_mtm(t, c, e[:1], x[:1], np.array([1.0]))
    assert abs(pnl.sum() - 1440 * 1e-5 * 1e4) < 1e-6, pnl.sum()
    r = evaluate("T", "up", "1d", t, c, np.arange(0, n - 2000, 7), np.ones(len(range(0, n - 2000, 7))), 1440)
    assert abs(r["gross_main"] - (np.exp(1440e-5) - 1) * 1e4) < 0.01 and abs(r["diff"]) < 0.01, r
    # 2) 손절/목표
    hh = np.array([100, 101, 103, 100.5]); ll = np.array([100, 99.5, 100, 100]); cc = np.array([100, 100.5, 102, 100.2])
    assert abs(simulate(hh, ll, cc, 0, 1, 99.0, 102.5, 3)[0] - 250.0) < 1e-9
    assert abs(simulate(np.array([100, 103.]), np.array([100, 98.]), np.array([100, 100.]), 0, 1, 99.0, 102.0, 1)[0] + 100.0) < 1e-9
    # 3) 미래참조 없음: cut 뒤를 바꿔도 cut 까지의 사건은 같아야 한다(가족마다)
    D = synth(); cut = 49 * D1 + 517
    D2 = perturb(D, cut)
    for k in ("c", "h", "l", "v"):
        assert np.array_equal(D[k][:cut + 1], D2[k][:cut + 1])

    oi = np.exp(np.cumsum(np.random.default_rng(3).normal(0, 1e-3, len(D["c"]) // 5)))
    oi2 = oi.copy(); oi2[cut // 5 + 1:] *= 1.5          # OI 도 cut 뒤를 바꾼다(행 = 봉 시작 스냅샷)

    def ev_all(X, oi):
        SD, SW = sessions(X, 0, D1), sessions(X, WOFF, W1)
        _, LVw = nodes_4h(X, 7 * D1)
        out = {"YushA": ev_yush(X, 240, SD), "YushB": ev_yush(X, D1, SW), "Carm": ev_carmine(X, LVw), "N2": ev_nill(X, "N2"),
               "R80d": ev_r80(X, SD), "R80w": ev_r80(X, SW), "nPOCd": ev_npoc(X, SD), "VAmig": ev_vamig(SD), "Pb": ev_pb(SD)}
        out["N2w"] = weekly_align(out["N2"], X, SW)
        f5 = agg(X, 5); ks, sd = ev_exhaust(f5["h"], f5["l"], f5["c"], f5["delta"])
        out["Exh"] = [(5 * int(k) + 4, s) for k, s in zip(ks, sd)]
        Q = ev_quadrant(f5["t"], f5["c"], oi_known(f5["t"], oi), 48, qwin=20 * 288, qmin=5 * 288)
        out["Quad"] = [(5 * int(k) + 4, 1) for k in Q["DnOIdn"][0]]
        return out
    A, Bv = ev_all(D, oi), ev_all(D2, oi2)
    tot = 0
    for k in A:
        a = [x[:2] for x in A[k] if x[0] <= cut]; b = [x[:2] for x in Bv[k] if x[0] <= cut]
        assert a == b, (k, len(a), len(b)); tot += len(a)
    assert len(A["YushA"]) and len(A["R80d"]) and len(A["nPOCd"]) and len(A["VAmig"]) and len(A["Quad"]), {k: len(v) for k, v in A.items()}
    # 4) 소진형: 심은 사건을 잡고(원판 selfcheck 축약) 방향 = 델타 쪽
    rng = np.random.default_rng(0); m = 2000
    cc = 2000 + np.cumsum(rng.normal(0, 0.5, m)); hi, lo = cc + 0.5, cc - 0.5; dl = rng.normal(0, 4, m)
    k = 1500; up = np.linspace(cc[k - 13], cc[k - 13] + 30, 13); cc[k - 13:k] = up; hi[k - 13:k] = up + 0.5; lo[k - 13:k] = up - 0.5
    top = hi[k - N_EX:k].max(); hi[k], lo[k], cc[k] = top + 20, top - 5, top + 10; dl[k] = 4900
    ks, sd = ev_exhaust(hi, lo, cc, dl)
    assert k in ks and sd[list(ks).index(k)] == 1
    # 5) OI 스탬프: 전환 뒤엔 한 행 앞 값
    tt = np.array(["2024-03-03T23:55", "2024-03-04T00:00", "2024-03-04T00:05"], dtype="datetime64[ms]")
    assert np.allclose(oi_known(tt, np.array([1., 2., 3.]))[1:], [1., 2.]) and oi_known(tt, np.array([1., 2., 3.]))[0] == 1.
    print(f"selftest ok (미래 불변 사건 {tot}건 확인)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    rows, nd_rows, rep = run()
    (OUT / "result.json").write_text(json.dumps({"rows": rows, "nondir": nd_rows, "report": rep}, ensure_ascii=False, indent=1, default=str))
    txt = fmt(rows, nd_rows, rep)
    (OUT / "report.txt").write_text(txt)
    print(txt)
