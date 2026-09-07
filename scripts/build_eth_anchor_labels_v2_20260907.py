#!/usr/bin/env python3
"""앵커 방향 라벨 v2 — **3클래스 + 품질 가중치** (2026-09-07).

사용자: *"25년 11월 21일 1:00 바닥 앵커 지속 승 같은 데이터만 지속 승으로 가야해. 나머지는
되돌림과 지속이 섞여있어. 라벨 로직이 지금은 틀린거야."* → 옵션 ②+③ 동시 채택.

## v1(먼저 닿기)의 결함
"±1% 를 먼저 닿는 쪽"만 봤다. 그래서 **닿자마자 진입가로 되돌아온 경로**도 지속 승이 됐다.
사용자가 지적한 10건 실측: 11-21 01:00 은 끝점 -3.68%(계속 감), 10-09 17:15 은 끝점 -0.05%
(닿고 즉시 반납). 두 경로가 같은 라벨이었다.

## v2 정의 (양측면 대칭)
진입 `open[t+1]` · 대칭 ±1% 배리어 · **1분봉** 첫 터치 · H=48봉(4h).
  pre_adv   먼저 닿은 쪽의 배리어에 닿기 **전까지** 반대 방향 최대이탈 (%)
  end_dir   H 끝 **마지막 6봉(30분) 평균 종가**의 먼저 닿은 방향 기준 수익 (%)
            (단일 종가는 1봉 노이즈로 라벨이 뒤집힌다 -- 자체검증 V4 가 12-08 23:55 에서 잡았다)

  ⭐y3 (3클래스)
     2 = 지속 승   지속 먼저 ∧ pre_adv < PRE_MAX ∧ end_dir >= END_MIN
     0 = 되돌림 승 페이드 먼저 ∧ pre_adv < PRE_MAX ∧ end_dir >= END_MIN
     1 = 혼재     그 외(닿았지만 지저분) + 미결정(어느 배리어도 미터치)
  ⭐y2 (이진, v1 계승) 먼저 닿는 쪽. **표본을 안 버린다.**
  ⭐w_quality  라벨 신뢰도 [0,1] = 역행 감쇠 x 끝점 포화
     역행 감쇠 = 1/(1 + pre_adv/PRE_MAX)     끝점 포화 = clip(end_dir/END_MIN, 0, 1)
     미결정은 0.
  w_uniq  AFML 고유도(1/동시겹침) · w_train = w_uniq x w_quality

기본 (PRE_MAX, END_MIN) = (0.5%, 1.0%) — 사용자 지적 10건과 대조해 고른 값이다
(인정 11-21·01-29·01-20 / 제외 10-09·12-08·12-14·01-09·03-19·10-20·12-16).

## 자체 검증 (스크립트가 스스로 돌린다)
  V1 3클래스 상호배타·완전성            V2 양측면 대칭(부호만 뒤집어 재계산해 일치)
  V3 무작위 30건 라벨 산술 독립 재계산     V4 사용자 지적 10건 판정 재현
  V5 퍼징/엠바고 후 split 경계 누수 0     V6 가중치 범위·결측
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

SRC = ROOT / "tmp/eth_anchor_direction_labels_20260907/direction_labels.parquet"
OUT = ROOT / "tmp/eth_anchor_labels_v2_20260907"
ANCHOR, PCT, HB = "any3/Wc3", 1.0, 48
PRE_MAX, END_MIN = 0.5, 1.0
SPLIT_END = {"TRAIN": "2025-08-31 23:59:59", "VAL": "2025-12-31 23:59:59",
             "OOS": "2026-03-31 23:59:59", "HOLDOUT_SPENT": "2099-01-01"}
CHECK_KST = ["2025-10-09 17:15", "2025-10-20 23:15", "2025-11-21 01:00", "2025-12-08 23:55",
             "2025-12-14 20:20", "2025-12-16 01:40", "2026-01-09 17:05", "2026-01-20 15:15",
             "2026-01-29 23:25", "2026-03-19 21:50"]
EXPECT = {"11-21 01:00": 2, "01-29 23:25": 2, "01-20 15:15": 2}     # 나머지는 혼재(1) 기대


def uniqueness(bar_idx: np.ndarray, span_bars: np.ndarray) -> np.ndarray:
    n = len(bar_idx)
    o = np.argsort(bar_idx)
    bi, sp = bar_idx[o].astype(float), np.maximum(span_bars[o], 1.0)
    end = bi + sp
    ov = np.ones(n); j0 = 0
    for k in range(n):
        while bi[j0] + sp[j0] <= bi[k]:
            j0 += 1
        ov[k] = np.sum((bi[j0:k + 1] < end[k]) & (end[j0:k + 1] > bi[k]))
    w = np.empty(n); w[o] = 1.0 / np.maximum(ov, 1.0)
    return w


def build():
    L = pd.read_parquet(SRC)
    A = L[L.anchor == ANCHOR].reset_index(drop=True)
    eth = B._load_kl(B.ETH_KL); btc = B._load_kl(B.BTC_KL); fund = B._load_funding()
    tmax = min(eth["timestamp"].max(), btc["timestamp"].max(), fund["calc_time"].max())
    eth = eth[eth["timestamp"] <= tmax].reset_index(drop=True)
    O, Hh, Lo, C = (eth[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    n = len(C)
    i = A["bar_idx"].to_numpy()
    ok = (i + 1 + HB) < n
    A = A[ok].reset_index(drop=True); i = i[ok]
    e = O[i + 1]
    w = np.arange(HB)[None, :] + (i + 1)[:, None]
    hi, lo, cl = Hh[w], Lo[w], C[w]

    lim = HB * 5.0
    tc = A[f"hit_cont_min_P{PCT:g}"].to_numpy(float)
    tf = A[f"hit_fade_min_P{PCT:g}"].to_numpy(float)
    tci = np.where(np.isfinite(tc) & (tc < lim), tc, np.inf)
    tfi = np.where(np.isfinite(tf) & (tf < lim), tf, np.inf)
    amb = A[f"ambig_P{PCT:g}"].to_numpy(bool)
    first_c = (tci < tfi) & ~amb
    first_f = (tfi < tci) & ~amb
    decided = first_c | first_f

    bot = (A["side"].to_numpy() == "bottom")
    # 가격 부호: 지속 방향 (bottom -> 하락 = -1)
    sgn_cont = np.where(bot, -1.0, 1.0)
    sgn_first = np.where(first_c, sgn_cont, np.where(first_f, -sgn_cont, np.nan))

    t_win = np.minimum(tci, tfi)
    jw = np.where(np.isfinite(t_win) & (t_win < lim), np.floor(t_win / 5.0), -1).astype(int)
    pre_adv = np.full(len(A), np.nan)
    for k in np.flatnonzero(jw >= 0):
        s = slice(0, jw[k] + 1)
        pre_adv[k] = ((hi[k, s].max() - e[k]) / e[k] * 100) if sgn_first[k] < 0 else \
                     ((e[k] - lo[k, s].min()) / e[k] * 100)
    # ⭐끝점은 **마지막 6봉(30분) 평균 종가** — 단일 종가는 1봉 노이즈로 라벨이 뒤집힌다
    # (2026-09-07 V4 가 12-08 23:55 에서 잡음: last1 +1.14% -> 지속승, last6 +0.57% -> 혼재).
    END_TAIL = 6
    end_dir = (cl[:, -END_TAIL:].mean(axis=1) - e) / e * 100 * sgn_first

    clean = decided & (pre_adv < PRE_MAX) & (end_dir >= END_MIN)
    y3 = np.where(clean & first_c, 2, np.where(clean & first_f, 0, 1))
    y2 = np.where(first_c, 1.0, np.where(first_f, 0.0, np.nan))
    q = np.where(decided, (1.0 / (1.0 + np.nan_to_num(pre_adv, nan=9e9) / PRE_MAX))
                 * np.clip(np.nan_to_num(end_dir, nan=-9e9) / END_MIN, 0, 1), 0.0)
    hold = np.where(decided, np.minimum(t_win, lim) / 5.0, HB)

    D = A[["timestamp", "bar_idx", "side", "anchor", "n_signals", "signals", "atr_pct", "split"]].copy()
    D["y3"], D["y2"], D["w_quality"] = y3, y2, q
    D["pre_adv"], D["end_dir"], D["t_win_min"], D["hold_bars"] = pre_adv, end_dir, t_win, hold
    D["decided"], D["first_cont"] = decided, first_c

    # 퍼징 + 엠바고
    end_ts = D["timestamp"] + pd.to_timedelta(D["hold_bars"] * 5, unit="m")
    keep = np.ones(len(D), bool)
    for sp, b in SPLIT_END.items():
        m = (D["split"].to_numpy() == sp)
        keep &= ~(m & (end_ts > pd.Timestamp(b)).to_numpy())
    n_purge = int((~keep).sum()); D = D[keep].reset_index(drop=True)
    emb = (D["split"] == "TRAIN") & (D["timestamp"] > pd.Timestamp(SPLIT_END["TRAIN"]) - pd.Timedelta(minutes=5 * HB))
    n_emb = int(emb.sum()); D = D[~emb].reset_index(drop=True)

    D["w_uniq"] = np.nan
    for sp in D["split"].unique():
        m = (D["split"] == sp).to_numpy()
        D.loc[m, "w_uniq"] = uniqueness(D.loc[m, "bar_idx"].to_numpy(), D.loc[m, "hold_bars"].to_numpy())
    D["w_train"] = D["w_uniq"] * D["w_quality"]
    return D, dict(n_purge=n_purge, n_emb=n_emb), (eth, O, Hh, Lo, C)


# ------------------------------------------------------------------ 자체 검증

def verify(D, meta, mkt):
    eth, O, Hh, Lo, C = mkt
    fails = []
    def chk(name, cond, note=""):
        print(f"  {'PASS' if cond else '🔴FAIL'}  {name}{('  ' + note) if note else ''}", flush=True)
        if not cond:
            fails.append(name)

    # V1 3클래스 상호배타·완전성
    v = D["y3"].to_numpy()
    chk("V1 3클래스 값이 {0,1,2}", set(np.unique(v)) <= {0, 1, 2}, f"분포 {np.bincount(v, minlength=3).tolist()}")
    chk("V1 clean(0/2) 은 반드시 decided", bool(D.loc[D.y3 != 1, "decided"].all()))
    chk("V1 y3=2 ⇔ first_cont ∧ clean", bool(((D.y3 == 2) == (D.first_cont & (D.y3 != 1))).all()))
    chk("V1 미결정은 전부 혼재(1)", bool((D.loc[~D.decided, "y3"] == 1).all()))

    # V2 양측면 대칭 — 정의가 side 에 대해 부호만 뒤집는가
    b = D[D.side == "bottom"]; t = D[D.side == "top"]
    chk("V2 양측면 clean 비율 차 < 5pp",
        abs((b.y3 != 1).mean() - (t.y3 != 1).mean()) < 0.05,
        f"bottom {(b.y3 != 1).mean():.3f} vs top {(t.y3 != 1).mean():.3f}")

    # V3 무작위 30건 산술 독립 재계산
    rng = np.random.default_rng(7)
    s = D[D.decided].sample(min(30, int(D.decided.sum())), random_state=7)
    bad = 0
    for r in s.itertuples():
        i = int(r.bar_idx); e = O[i + 1]; end = i + 1 + HB
        cl = C[i + 1:end]; hi = Hh[i + 1:end]; lo = Lo[i + 1:end]
        sg = (-1.0 if r.side == "bottom" else 1.0) * (1 if r.first_cont else -1)
        j = int(np.floor(r.t_win_min / 5.0))
        pa = ((hi[:j + 1].max() - e) / e * 100) if sg < 0 else ((e - lo[:j + 1].min()) / e * 100)
        ed = (cl[-6:].mean() - e) / e * 100 * sg
        if abs(pa - r.pre_adv) > 1e-6 or abs(ed - r.end_dir) > 1e-6:
            bad += 1
    chk("V3 무작위 30건 pre_adv/end_dir 재계산 일치", bad == 0, f"불일치 {bad}")

    # V4 사용자 지적 10건 판정 재현
    kst = (D["timestamp"] + pd.Timedelta(hours=9)).dt.strftime("%m-%d %H:%M")
    got = {}
    for s_ in CHECK_KST:
        k = pd.Timestamp(s_).strftime("%m-%d %H:%M")
        m = (kst == k) & (D.side == "bottom")
        got[k] = int(D.loc[m, "y3"].iloc[0]) if m.any() else None
    okc = all(got.get(k) == v for k, v in EXPECT.items())
    okr = all(got.get(k) == 1 for k in got if k not in EXPECT and got.get(k) is not None)
    chk("V4 지적 10건: 11-21·01-29·01-20 만 지속승(2)", okc and okr,
        " ".join(f"{k}={got[k]}" for k in sorted(got)))

    # V5 split 경계 누수
    leak = 0
    for sp, b_ in SPLIT_END.items():
        m = D["split"] == sp
        end_ts = D.loc[m, "timestamp"] + pd.to_timedelta(D.loc[m, "hold_bars"] * 5, unit="m")
        leak += int((end_ts > pd.Timestamp(b_)).sum())
    chk("V5 퍼징 후 split 경계 넘는 라벨 0", leak == 0, f"퍼징 {meta['n_purge']} 엠바고 {meta['n_emb']}")

    # V6 가중치
    chk("V6 w_quality ∈ [0,1]", bool(((D.w_quality >= 0) & (D.w_quality <= 1)).all()),
        f"max {D.w_quality.max():.3f}")
    chk("V6 w_uniq ∈ (0,1] 결측 0", bool(((D.w_uniq > 0) & (D.w_uniq <= 1)).all() and D.w_uniq.notna().all()))
    chk("V6 혼재(1)의 w_quality 는 clean 보다 작다",
        D.loc[D.y3 == 1, "w_quality"].mean() < D.loc[D.y3 != 1, "w_quality"].mean(),
        f"혼재 {D.loc[D.y3 == 1, 'w_quality'].mean():.3f} vs clean {D.loc[D.y3 != 1, 'w_quality'].mean():.3f}")
    return fails


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D, meta, mkt = build()
    print(f"[1/2] 라벨 v2: {len(D):,}행 (퍼징 {meta['n_purge']} · 엠바고 {meta['n_emb']})\n", flush=True)
    print("=== 자체 검증 ===", flush=True)
    fails = verify(D, meta, mkt)
    D.to_parquet(OUT / "labels_v2.parquet", index=False)

    print("\n=== split x 클래스 ===")
    t = D.pivot_table(index="split", columns="y3", values="bar_idx", aggfunc="count", fill_value=0)
    t.columns = ["0 되돌림승", "1 혼재", "2 지속승"][:len(t.columns)]
    t["합"] = t.sum(axis=1); t["clean비율"] = (t["0 되돌림승"] + t["2 지속승"]) / t["합"]
    t["clean 중 지속"] = t["2 지속승"] / (t["0 되돌림승"] + t["2 지속승"])
    print(t.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n=== 이진 y2(먼저 닿기) + 가중치 ===")
    for sp in ("TRAIN", "VAL", "OOS"):
        s = D[D.split == sp]
        print(f"  {sp:<6} n={len(s):>5} · y2 결정 {int(s.y2.notna().sum()):>5} · 지속비율 {s.y2.mean():.3f}"
              f" · w_quality 평균 {s.w_quality.mean():.3f} · w_train 합 {s.w_train.sum():.0f}")
    (OUT / "contract.json").write_text(json.dumps({
        "anchor": ANCHOR, "barrier_pct": PCT, "H_bars": HB, "PRE_MAX": PRE_MAX, "END_MIN": END_MIN,
        "y3": "2 지속승(clean) · 0 되돌림승(clean) · 1 혼재(지저분+미결정)",
        "y2": "먼저 닿는 쪽 (표본 유지) · 미결정은 NaN",
        "w_quality": "1/(1+pre_adv/PRE_MAX) x clip(end_dir/END_MIN,0,1) · 미결정 0",
        "w_train": "w_uniq x w_quality", "self_check_failed": fails,
    }, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}  ·  검증 실패 {len(fails)}건")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
