#!/usr/bin/env python3
"""시각대×극단분위 신호의 **자산 축 확장** (2026-09-10). 60종 무기한선물 · 지평 1시간.

사용자: *"그렇게 진행해줘"* — ETH 단일자산에서 연 75건이 구조적 상한이라 표본이 모자랐다.
넓히기(분위 완화·시각 풀링)는 **초과분이 더 빨리 죽어** 비용 아래로 갔고, 딥러닝은 정확도 축이라
이 신호의 수익 경로(이긴 판의 크기)가 아니었다. 남은 레버가 자산 축이다.
전제: `docs/eth_1h_signal_literature_candidates_20260910.md`

## ⭐설계의 핵심 — 확인과 탐색을 가른다
ETH 결과는 **192셀 격자를 뒤져 나온 것**이다. 같은 격자를 60종에서 또 뒤지면 그건 확인이 아니라
더 큰 낚시다. 그래서:
  **1차(확인·탐색 없음)** ETH 가 고른 셀 `L6·h17·모멘텀` 을 **나머지 59종에 그대로** 적용.
                          방향·시각·되돌아보기·분위 전부 ETH 에서 고정. 사후 반전 금지.
  **2차(탐색)**          전 격자를 패널에 얹어 구조가 패널 전반에 있는지. 격자 통과수 귀무로 보정.
1차가 실패하면 2차 결과가 좋아도 그건 새 가설이지 확인이 아니다.

## 🔴표본을 곱했다고 검정력을 산 게 아니다
암호자산은 같이 움직인다. 같은 시각에 60종이 발동하면 그건 **관측 60개가 아니라 군집 1개**다.
[[겹침 트랜치]] 에서 48주→331일로 늘려도 독립 주가 47 그대로였던 것과 **같은 함정**이다.
그래서 이 스크립트는 처음부터:
  · 귀무 = 전 종목에 **동일한 shift** 를 건 순환이동(횡단면 상관 보존)
  · CI  = **시각 군집** 블록부트(한 타임스탬프의 모든 종목을 통째로 재표집)
  · 보고 = 관측 수와 **구별되는 타임스탬프 수(=유효 군집)** 를 나란히
'건수'가 아니라 '군집 수'가 늘어야 검정력을 산 것이다.

## 🔴유동성 함정
09-08 횡단면 스크린에서 **극단이탈 효과는 비유동 종목 전용**이었다. 유동성 계층을 갈라 보지 않으면
비용으로 지워질 효과를 엣지로 오독한다. 상위10/30/60 세 계층 전부 보고하고,
비용은 **10bp(유동)·20bp(비유동)** 두 가정으로 함께 낸다.

출력 tmp/xsec_hod_1h_20260910/report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = Path("/home/kbj20/crypto-scalping/tmp/xsec_perp_screen_20260908/panel.npz")
OUT = ROOT / "tmp/xsec_hod_1h_20260910"

ETH_L, ETH_H, ETH_Q, ETH_DIR = 6, 17, 0.10, +1.0     # ETH 가 고른 셀. **여기서 고정, 사후 변경 금지**
LOOKBACKS = [1, 2, 3, 6]
HOURS = list(range(24))
QS = [0.10, 0.20]
TIERS = {"top10": 10, "top30": 30, "all60": 60}
COSTS = {"liquid_10bp": 10.0, "illiquid_20bp": 20.0}
SPLIT = pd.Timestamp("2025-09-01")
B_NULL = 400
B_BOOT = 2000


def log(m):
    print(f"[xhod {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_hourly():
    d = np.load(PANEL, allow_pickle=True)
    ts = pd.DatetimeIndex(d["ts"]); syms = [str(x) for x in d["syms"]]
    C = d["C"].astype(np.float64); Q = d["Q"].astype(np.float64)
    hh = ts.floor("h")
    codes, uniq = pd.factorize(hh, sort=True)
    n_h, n_s = len(uniq), C.shape[1]
    close = np.full((n_h, n_s), np.nan); dv = np.zeros((n_h, n_s))
    last = np.full(n_s, -1)
    for i in range(len(ts)):                       # 시간별 마지막 유효 종가 + 달러거래량 합
        k = codes[i]
        row = C[i]; m = np.isfinite(row)
        close[k, m] = row[m]
        q = Q[i]; dv[k] += np.where(np.isfinite(q), q, 0.0)
    return pd.DatetimeIndex(uniq), close, dv, syms


def liquidity_rank(dv, win=720):
    """직전 30일(720시간) 중앙 달러거래량 순위. 인과적."""
    n_h, n_s = dv.shape
    med = np.full((n_h, n_s), np.nan)
    for t in range(win, n_h, 24):                  # 하루 1회 갱신(비용 절감), 사이는 직전값 유지
        med[t] = np.nanmedian(dv[t - win:t], axis=0)
    return pd.DataFrame(med).ffill().to_numpy()


def fire_mask(close, hours, L, h, q, tier_ok):
    """(시각 h, 직전 L시간 수익 자기 분위 극단) 발동. **분위는 종목별·인과적**으로 잡는다."""
    n_h, n_s = close.shape
    past = np.full_like(close, np.nan)
    past[L:] = close[L:] / close[:-L] - 1.0
    fwd = np.full_like(close, np.nan)
    fwd[:-1] = close[1:] / close[:-1] - 1.0        # 다음 1시간 수익 = 타깃
    hm = (hours == h)
    hi = np.zeros_like(close, dtype=bool); lo = np.zeros_like(close, dtype=bool)
    rows = np.flatnonzero(hm)
    for j in range(n_s):
        v = past[rows, j]
        ok = np.isfinite(v) & np.isfinite(fwd[rows, j]) & tier_ok[rows, j]
        if ok.sum() < 100:
            continue
        vv = v[ok]
        thi, tlo = np.quantile(vv, 1 - q), np.quantile(vv, q)
        sel = rows[ok]
        hi[sel[v[ok] >= thi], j] = True
        lo[sel[v[ok] <= tlo], j] = True
    return hi, lo, fwd


def stat(fwd, hi, lo, direction=+1.0):
    """양측 평균 bp. 방향 +1 = 모멘텀(상승분위 롱), −1 = 되돌림."""
    a = fwd[hi]; b = fwd[lo]
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 20 or len(b) < 20:
        return np.nan
    return float(direction * (a.mean() - b.mean()) / 2 * 1e4)


def balanced(fwd, hi, lo, direction=+1.0):
    """⭐**시각 내 균형** 규약: 한 타임스탬프에서 롱/숏이 **둘 다 있는 시각만** 쓰고,
    각 팔을 시각 내 평균으로 접은 뒤 시각끼리 평균낸다.
    이유: 종목별 자기분위로 뽑으면 시장이 오른 시각엔 상승분위에 종목이 몰려 양팔이 안 맞고,
    그 불균형이 그대로 **시장 베타**로 새어든다
    ([[feedback_side_mirror_excess_is_residual_beta_not_signal_20260908]])."""
    rows = np.flatnonzero(hi.any(axis=1) & lo.any(axis=1))     # 둘 다 있는 시각만
    per = []
    for t in rows:
        a = fwd[t][hi[t]]; c = fwd[t][lo[t]]
        a = a[np.isfinite(a)]; c = c[np.isfinite(c)]
        if len(a) and len(c):
            per.append(direction * (a.mean() - c.mean()) / 2 * 1e4)
    return (float(np.mean(per)) if per else np.nan), len(per)


def balanced_null(fwd, hi, lo, direction, b=B_NULL, seed=0):
    n = fwd.shape[0]; rng = np.random.default_rng(seed); o = []
    for s in rng.integers(1, n, b):
        v, _ = balanced(fwd, np.roll(hi, s, axis=0), np.roll(lo, s, axis=0), direction)
        if np.isfinite(v):
            o.append(v)
    return np.array(o)


def balanced_boot(fwd, hi, lo, direction, b=B_BOOT, seed=0):
    rows = np.flatnonzero(hi.any(axis=1) & lo.any(axis=1))
    per = []
    for t in rows:
        a = fwd[t][hi[t]]; c = fwd[t][lo[t]]
        a = a[np.isfinite(a)]; c = c[np.isfinite(c)]
        if len(a) and len(c):
            per.append(direction * (a.mean() - c.mean()) / 2 * 1e4)
    if len(per) < 20:
        return [float("nan")] * 2
    per = np.array(per); rng = np.random.default_rng(seed); n = len(per)
    o = [per[rng.integers(0, n, n)].mean() for _ in range(b)]
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def shift_null(fwd, hi, lo, direction, b=B_NULL, seed=0):
    """⭐전 종목에 **같은 shift** — 횡단면 상관·시각 군집을 보존한다."""
    n = fwd.shape[0]; rng = np.random.default_rng(seed); o = []
    for s in rng.integers(1, n, b):
        o.append(stat(fwd, np.roll(hi, s, axis=0), np.roll(lo, s, axis=0), direction))
    o = np.array(o)
    return o[np.isfinite(o)]


def cluster_boot(fwd, hi, lo, direction, b=B_BOOT, seed=0):
    """⭐**시각 군집** 부트: 타임스탬프를 재표집해 그 시각의 전 종목을 통째로 가져온다."""
    rows = np.flatnonzero(hi.any(axis=1) | lo.any(axis=1))
    if len(rows) < 20:
        return [float("nan")] * 2, 0
    per = []
    for t in rows:                                  # 시각별 건당 손익(부호 맞춤) 목록
        a = fwd[t][hi[t]]; c = fwd[t][lo[t]]
        v = np.concatenate([direction * a[np.isfinite(a)], -direction * c[np.isfinite(c)]]) * 1e4
        per.append(v)
    rng = np.random.default_rng(seed); n = len(per); o = []
    for _ in range(b):
        pick = rng.integers(0, n, n)
        cat = np.concatenate([per[i] for i in pick])
        if len(cat):
            o.append(cat.mean())        # ⚠️cat.mean() 은 na≈nb 일 때 이미 (mean(a)−mean(b))/2 다.
                                        # 여기서 또 2로 나누면 CI 중심이 관측치의 절반이 된다(초판 버그).
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))], len(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    hrs_idx, close, dv, syms = load_hourly()
    hours = hrs_idx.hour.to_numpy()
    log(f"시간봉 {close.shape} · {hrs_idx[0]} → {hrs_idx[-1]} · 종목 {len(syms)}")
    lrank = liquidity_rank(dv)
    order = np.argsort(-np.where(np.isfinite(lrank), lrank, -np.inf), axis=1)
    tier_masks = {}
    for name, k in TIERS.items():
        m = np.zeros_like(close, dtype=bool)
        np.put_along_axis(m, order[:, :k], True, axis=1)
        tier_masks[name] = m & np.isfinite(close)
    eth_j = syms.index("ETHUSDT") if "ETHUSDT" in syms else None
    log(f"ETH 패널 인덱스 {eth_j} · 유동성 계층 {list(TIERS)}")

    rep = {"panel": {"n_hours": int(close.shape[0]), "n_syms": len(syms),
                     "span": [str(hrs_idx[0]), str(hrs_idx[-1])]},
           "prereg": {"eth_cell": f"L{ETH_L}|h{ETH_H}|q{int(ETH_Q*100)}",
                      "direction": "모멘텀(ETH 에서 고정, 사후 반전 금지)",
                      "primary": "ETH 제외 59종에 그대로 적용", "costs": COSTS},
           "confirm": {}, "explore": {}}

    # ── 1차 확인: ETH 셀을 나머지 59종에 그대로 ───────────────────────────────
    for tier, tm in tier_masks.items():
        tm2 = tm.copy()
        if eth_j is not None:
            tm2[:, eth_j] = False                    # **ETH 제외** — 그래야 표본외 확인이다
        hi, lo, fwd = fire_mask(close, hours, ETH_L, ETH_H, ETH_Q, tm2)
        obs = stat(fwd, hi, lo, ETH_DIR)
        if not np.isfinite(obs):
            continue
        nl = shift_null(fwd, hi, lo, ETH_DIR, seed=hash(tier) % 9999)
        ci, ncl = cluster_boot(fwd, hi, lo, ETH_DIR, seed=7)
        n_obs = int(hi.sum() + lo.sum())
        is_m = np.asarray(hrs_idx < SPLIT)
        hi_i = hi & is_m[:, None]; lo_i = lo & is_m[:, None]
        hi_o = hi & ~is_m[:, None]; lo_o = lo & ~is_m[:, None]
        rep["confirm"][tier] = {
            "n_obs": n_obs, "n_time_clusters": ncl,
            "obs_per_cluster": round(n_obs / max(ncl, 1), 2),
            "obs_bp": obs, "null_mean_bp": float(nl.mean()),
            "excess_bp": obs - float(nl.mean()),
            "null_ci95": [float(np.percentile(nl, 2.5)), float(np.percentile(nl, 97.5))],
            "beats_null": bool(obs > np.percentile(nl, 97.5)),
            "cluster_ci95": ci, "ci_excludes_zero": bool(ci[0] > 0),
            "net_bp": {k: obs - float(nl.mean()) - c for k, c in COSTS.items()},
            "is_bp": stat(fwd, hi_i, lo_i, ETH_DIR), "oos_bp": stat(fwd, hi_o, lo_o, ETH_DIR)}
        bv, bn = balanced(fwd, hi, lo, ETH_DIR)
        bnl = balanced_null(fwd, hi, lo, ETH_DIR, seed=hash(tier) % 7777)
        bci = balanced_boot(fwd, hi, lo, ETH_DIR, seed=13)
        rep["confirm"][tier]["balanced"] = {
            "n_times_both_sides": bn, "obs_bp": bv,
            "excess_bp": bv - float(bnl.mean()) if len(bnl) else None,
            "beats_null": bool(len(bnl) and bv > np.percentile(bnl, 97.5)),
            "boot_ci95": bci, "ci_excludes_zero": bool(np.isfinite(bci[0]) and bci[0] > 0),
            "is_bp": balanced(fwd, hi_i, lo_i, ETH_DIR)[0],
            "oos_bp": balanced(fwd, hi_o, lo_o, ETH_DIR)[0]}
        c = rep["confirm"][tier]
        log(f"  1차 {tier:6s} n={n_obs:6d} 군집={ncl:5d}({c['obs_per_cluster']}건/군집) "
            f"초과 {c['excess_bp']:+7.2f}bp 귀무{'통과' if c['beats_null'] else '—'} "
            f"군집CI[{ci[0]:+6.2f},{ci[1]:+6.2f}] IS {c['is_bp']:+6.2f} OOS {c['oos_bp']:+6.2f}")
        b = c["balanced"]
        log(f"     └베타제거 시각{b['n_times_both_sides']:5d} 초과 {b['excess_bp']:+7.2f}bp "
            f"{'귀무통과' if b['beats_null'] else '귀무—'} CI[{b['boot_ci95'][0]:+6.2f},{b['boot_ci95'][1]:+6.2f}] "
            f"{'**CI 0제외**' if b['ci_excludes_zero'] else 'CI 0포함'} "
            f"IS {b['is_bp']:+6.2f} OOS {b['oos_bp']:+6.2f}")
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


def xsec_rank_arm(close, hours, tier_ok, L, h, k, direction=+1.0):
    """⭐**횡단면 순위** 팔 — 설계상 양팔이 항상 같은 수다(베타가 구조적으로 제거된다).
    자기분위 방식은 h17 에서 양팔 공존 시각이 528 중 **26** 뿐이었다: 사실상 방향 베팅이었다.
    여기서는 매 시각 종목을 직전 L시간 수익으로 **서로 비교**해 상위 k 롱 / 하위 k 숏."""
    n_h, n_s = close.shape
    past = np.full_like(close, np.nan); past[L:] = close[L:] / close[:-L] - 1.0
    fwd = np.full_like(close, np.nan); fwd[:-1] = close[1:] / close[:-1] - 1.0
    per = []
    for t in np.flatnonzero(hours == h):
        ok = np.flatnonzero(tier_ok[t] & np.isfinite(past[t]) & np.isfinite(fwd[t]))
        if len(ok) < 2 * k + 2:
            continue
        o = ok[np.argsort(past[t][ok])]
        per.append(direction * (fwd[t][o[-k:]].mean() - fwd[t][o[:k]].mean()) / 2 * 1e4)
    return np.array(per)


def rank_null(close, hours, tier_ok, L, h, k, b=200, seed=0):
    """무작위 배정 귀무: 같은 시각·같은 종목 수에서 롱/숏을 **무작위로** 고른다."""
    n_h, n_s = close.shape
    past = np.full_like(close, np.nan); past[L:] = close[L:] / close[:-L] - 1.0
    fwd = np.full_like(close, np.nan); fwd[:-1] = close[1:] / close[:-1] - 1.0
    rows = np.flatnonzero(hours == h); rng = np.random.default_rng(seed); out = []
    for _ in range(b):
        per = []
        for t in rows:
            ok = np.flatnonzero(tier_ok[t] & np.isfinite(past[t]) & np.isfinite(fwd[t]))
            if len(ok) < 2 * k + 2:
                continue
            o = rng.permutation(ok)
            per.append((fwd[t][o[:k]].mean() - fwd[t][o[k:2 * k]].mean()) / 2 * 1e4)
        if per:
            out.append(float(np.mean(per)))
    return np.array(out)


if __name__ == "__main__":
    raise SystemExit(main())
