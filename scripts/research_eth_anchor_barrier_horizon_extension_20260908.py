#!/usr/bin/env python3
"""앵커 지속 진입의 **배리어/지평 확장** -- 총수익이 비용을 넘는 규모가 있는가 (2026-09-08).

## 왜 이걸 보는가
방향 모델은 ~120셀 전수 기각(AUC 0.50). 그런데 **모델 없이 전건 지속 진입**의 실현 bp 를
비용 이전(총수익)으로 환산하면 배리어 크기에 따라 다르다:
    H=12/±0.5% → 순 −7.14bp (비용 7.8) ⇒ 총 +0.66bp
    H=48/±1.0% → 순 −1.46bp (비용 7.8) ⇒ 총 **+6.34bp**
비용은 건당 고정(7.8bp)이고 총수익은 배리어에 따라 커진다. 그렇다면 **더 큰 배리어/지평에서
총수익이 7.8bp 를 넘는 지점**이 있을 수 있다. 모델이 필요 없는 순수 산술 검정.

## 함정 방어
- ⚠️**측면 편향(beta)**: VAL/OOS 는 하락장이라 무작위 숏도 +5bp
  ([[feedback_side_asymmetry_needs_same_side_null_20260905]]). ⇒ **같은-측면 무작위 봉 귀무** 필수.
  헤드라인은 관측이 아니라 **관측 − 같은측면귀무**, 일군집 CI 하한 > 0.
- ⚠️**커버리지 100%**: 미터치는 배제가 아니라 H봉 뒤 종가 청산으로 채점
  ([[feedback_outcome_selected_subset_inflates_metrics_20260908]]).
- ⚠️**트레일링 없음**: 배리어+시간청산만. [[eth_trailing_stop_infeasible_fill_bug_20260907]] 회피.
- 동시터치(ambig)는 **보수적으로 손실** 처리.
- 진입 `open[t+1]`, 1분봉 첫터치 -- 기존 라벨 빌더와 동일 규약(L4).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

KL1 = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-1m-api.csv"
SRC = ROOT / "tmp/eth_anchor_label_dataset_20260907/anchors_labels.parquet"
OUT = ROOT / "tmp/eth_anchor_barrier_ext_20260908"
ANCHOR = "any3/Wc3"
H_GRID = (48, 96, 144, 288)          # 4h · 8h · 12h · 24h
P_GRID = (1.0, 1.5, 2.0, 3.0)        # ± %
COST = 7.8
MAX_MIN = max(H_GRID) * 5
CHUNK = 2000
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
NULL_B = 60
BOOT = 2000
SEED = 20260908


def first_touch(hi1, lo1, start, up, dn, max_min=MAX_MIN):
    n = len(start)
    t_up = np.full(n, -1, np.int32); t_dn = np.full(n, -1, np.int32)
    amb = np.zeros(n, bool)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        idx = start[a:b, None] + np.arange(max_min)[None, :]
        hu = hi1[idx] >= up[a:b, None]; hd = lo1[idx] <= dn[a:b, None]
        au = hu.any(1); ad = hd.any(1)
        iu = np.where(au, hu.argmax(1), -1); idd = np.where(ad, hd.argmax(1), -1)
        t_up[a:b] = iu; t_dn[a:b] = idd; amb[a:b] = au & ad & (iu == idd)
    return t_up, t_dn, amb


def pnl(t_up, t_dn, amb, entry, exit_close, sgn, H, P):
    """지속방향 실현 bp. 커버리지 100%: 미터치는 H봉 뒤 종가."""
    lim = H * 5
    up_ok = (t_up >= 0) & (t_up < lim); dn_ok = (t_dn >= 0) & (t_dn < lim)
    tu = np.where(up_ok, t_up, 1 << 30); td = np.where(dn_ok, t_dn, 1 << 30)
    y = (exit_close - entry) / entry * 1e4 * sgn          # 기본 = 시간청산
    hit = up_ok | dn_ok
    win_up = hit & (tu < td)                               # 위 배리어 먼저
    win_dn = hit & (td < tu)
    tie = hit & (tu == td) & (tu < (1 << 30))
    bp = P * 100.0
    # sgn=+1(롱, bottom 지속=상승) 이면 위 터치가 이익
    y = np.where(win_up, np.where(sgn > 0, bp, -bp), y)
    y = np.where(win_dn, np.where(sgn > 0, -bp, bp), y)
    y = np.where(tie | amb & hit, -bp, y)                  # 보수적
    return y


def day_ci(v, day, rng, B=BOOT):
    u = np.unique(day); idx = {x: np.flatnonzero(day == x) for x in u}
    if len(u) < 5: return (np.nan, np.nan)
    o = [v[np.concatenate([idx[x] for x in rng.choice(u, len(u), True)])].mean() for _ in range(B)]
    return tuple(np.percentile(o, [2.5, 97.5]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("[1/3] 로드 ...", flush=True)
    D = pd.read_parquet(SRC)
    D = D[D["anchor"] == ANCHOR].reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL)
    m1 = pd.read_csv(KL1, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    m1 = m1.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts1 = m1["timestamp"].to_numpy(); hi1 = m1["high"].to_numpy(float); lo1 = m1["low"].to_numpy(float)
    ts5 = eth["timestamp"].to_numpy(); O5 = eth["open"].to_numpy(float); C5 = eth["close"].to_numpy(float)
    print(f"      앵커 {len(D):,} · split {dict(D['split'].value_counts())}", flush=True)

    # ---- 평가 대상 = 앵커 + 같은측면 무작위 봉 풀 (귀무) -------------------------------
    bi = D["bar_idx"].to_numpy()
    side = D["side"].to_numpy()
    sgn_a = np.where(side == "bottom", 1.0, -1.0)          # 지속: bottom→롱, top→숏
    split_a = D["split"].to_numpy()
    day_a = pd.to_datetime(D["timestamp"]).dt.floor("D").to_numpy()

    # 각 split 의 봉 범위에서 무작위 봉 풀 구성 (같은 측면 비율 유지 = 앵커의 sgn 을 그대로 씀)
    lo_hi = {}
    for w in WINS:
        m = split_a == w
        if m.sum() == 0: continue
        lo_hi[w] = (int(bi[m].min()), int(bi[m].max()))
    CAP = len(eth) - MAX_MIN // 5 - 2
    lo_hi = {w: (a, min(b, CAP)) for w, (a, b) in lo_hi.items()}
    nA = len(D)
    rnd_bi = np.zeros((NULL_B, nA), np.int64)
    for w, (a, b) in lo_hi.items():
        m = np.flatnonzero(split_a == w)
        rnd_bi[:, m] = rng.integers(a, b + 1, size=(NULL_B, len(m)))

    def eval_set(bidx, sgn):
        """(t_up, t_dn, amb, entry, close_at) 사전계산 -- 배리어별."""
        e_i = np.minimum(bidx + 1, len(O5) - 1)
        entry = O5[e_i]
        start = np.searchsorted(ts1, ts5[e_i])
        ok = (start < len(ts1) - MAX_MIN) & (ts1[np.minimum(start, len(ts1) - 1)] == ts5[e_i])
        s = np.where(ok, start, 0)
        out = {}
        for P in P_GRID:
            up = entry * (1 + P / 100); dn = entry * (1 - P / 100)
            out[P] = first_touch(hi1, lo1, s, up, dn)
        return entry, e_i, ok, out

    print("[2/3] 앵커 배리어 터치 ...", flush=True)
    entry_a, ei_a, ok_a, tt_a = eval_set(bi, sgn_a)

    rows = []
    print("[3/3] 귀무 B=%d ..." % NULL_B, flush=True)
    null_cache = {}
    for b_ in range(NULL_B):
        e, ei, ok, tt = eval_set(rnd_bi[b_], sgn_a)
        null_cache[b_] = (e, ei, ok, tt)
        if (b_ + 1) % 50 == 0: print(f"      귀무 {b_+1}/{NULL_B}", flush=True)

    print()
    hdr = f"{'창':>14} {'H':>4} {'배리어':>7} {'n':>5} {'총bp':>8} {'순bp':>8} {'귀무총':>8} {'초과':>8} {'[일군집 CI95]':>22} {'터치%':>6}"
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for w in WINS:
        m = (split_a == w) & ok_a
        if m.sum() < 30: continue
        for H in H_GRID:
            x_i = np.minimum(ei_a + H, len(C5) - 1)
            for P in P_GRID:
                tu, td, ab = tt_a[P]
                y = pnl(tu, td, ab, entry_a, C5[x_i], sgn_a, H, P)[m]
                nul = np.zeros((NULL_B, m.sum()))
                for b_ in range(NULL_B):
                    e, ei, okn, tt = null_cache[b_]
                    xi = np.minimum(ei + H, len(C5) - 1)
                    tu2, td2, ab2 = tt[P]
                    nul[b_] = pnl(tu2, td2, ab2, e, C5[xi], sgn_a, H, P)[m]
                nmean = nul.mean(0)
                exc = y - nmean
                lo, hi = day_ci(exc, day_a[m], rng)
                touched = float((((tu[m] >= 0) & (tu[m] < H * 5)) | ((td[m] >= 0) & (td[m] < H * 5))).mean())
                rows.append(dict(win=w, H=H, P=P, n=int(m.sum()), gross=float(y.mean()),
                                 net=float(y.mean() - COST), null_gross=float(nmean.mean()),
                                 excess=float(exc.mean()), lo=lo, hi=hi, touch=touched))
                print(f"{w:>14} {H:>4} {'±'+str(P)+'%':>7} {m.sum():>5} {y.mean():>+8.2f} "
                      f"{y.mean()-COST:>+8.2f} {nmean.mean():>+8.2f} {exc.mean():>+8.2f} "
                      f"[{lo:>+7.2f},{hi:>+7.2f}] {touched:>6.1%}", flush=True)
    A = pd.DataFrame(rows); A.to_csv(OUT / "grid.csv", index=False)
    piv = A.pivot_table(index=["H", "P"], columns="win", values="excess")
    both = [(h, p) for (h, p) in piv.index
            if all(A[(A.H == h) & (A.P == p) & (A.win == w)].lo.iloc[0] > 0 for w in ("VAL", "OOS")
                   if len(A[(A.H == h) & (A.P == p) & (A.win == w)]))]
    print()
    print(f"⭐두 창(VAL·OOS) 초과 CI 하한 > 0 인 셀: {len(both)}/{len(piv)}  {both}", flush=True)
    print(f"⭐순bp > 0 인 셀(비용 {COST}): "
          f"{int((A.net > 0).sum())}/{len(A)}", flush=True)
    print(json.dumps({"cells": len(A), "both_windows_pass": len(both)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
