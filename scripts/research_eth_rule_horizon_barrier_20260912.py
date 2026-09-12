#!/usr/bin/env python3
"""**지평·배리어 축** — b 를 키워 비용 비중을 낮추면 규칙이 사는가 (2026-09-12).

사용자: *"지평 늘려서 배리어 키우는 축으로 돌려줘."*
근거 항등식은 `순익 = (2a−1)·b − 비용`. 앞선 채굴(`research_eth_rule_direction_probability_20260912`)이
±0.8×ATR(중앙 11bp)에서 실패한 건 a 가 낮아서만이 아니라 b 가 비용선과 비슷해서다.
**필요 정확도 a\\* = (1 + 비용/b) / 2** — b 가 11bp면 a\\*≈95%, b 가 100bp면 a\\*≈55%다.

원자·규칙 열거·검증 규약은 앞 스크립트를 그대로 임포트해 쓴다. 여기서 새로 하는 건 셋뿐이다.

## 1. 배리어를 키우고 탐색창을 맞춘다
k ∈ {0.8 … 12}×ATR, 탐색창 maxbars 를 k 에 맞춰 늘린다. **미해결(timeout)을 버리지 않고**
그 시점 종가로 청산한 실현 bp 를 쓴다 — 버리면 결과로 걸러진 부분집합이 된다(09-08 교훈).

## 2. 🔴펀딩비가 비용에 들어온다
5분 스캘핑에선 무시했지만 1일 보유는 8시간 펀딩 3회, 1주는 21회다(실측 평균 회당 0.32bp →
1주 6.66bp). 롱은 내고 숏은 받는다. 실제 펀딩 시계열로 **보유 구간 누적**을 건별로 뺀다.
`data/TOTAL_ETHUSDT_fundingRate_2025_2026.csv` 는 2025-01-01 부터라 그 이전은 0 으로 두고 비율을 병기한다.

## 3. 🔴일블록 부트는 지평이 하루를 넘으면 무효다
블록이 보유기간보다 짧으면 같은 트레이드가 여러 블록에 쪼개져 독립인 척한다.
**블록 길이 = max(1일, 중앙 보유기간)** 으로 잡고 블록 단위로 재표집한다.

산출 tmp/eth_rule_horizon_20260912/{sweep.csv, rules_<라벨>.csv}
자체점검 `--selftest`
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402
import research_eth_rule_direction_probability_20260912 as R  # noqa: E402

OUT = ROOT / "tmp/eth_rule_horizon_20260912"
# (k×ATR, 탐색창 봉) — 배리어가 클수록 닿는 데 오래 걸리므로 창을 같이 늘린다
CELLS = [(0.8, 96), (1.5, 96), (3.0, 288), (5.0, 576), (8.0, 1152), (12.0, 2016)]
FEE = {"테이커": 10.0, "메이커": 7.8}
NSHIFT, B_FAM, MIN_N = 400, 12, 100
RNG = np.random.default_rng(20260912)


def log(m: str) -> None:
    print(f"[hz {time.strftime('%H:%M:%S')}] {m}", flush=True)


def funding_cum(ts: np.ndarray) -> tuple[np.ndarray, float]:
    """봉 i 까지의 누적 펀딩률. 보유 구간 누적 = cf[청산] − cf[진입]."""
    f = pd.read_csv(B.FUNDING, parse_dates=["calc_time"]).sort_values("calc_time")
    idx = pd.DatetimeIndex(ts)
    s = pd.Series(f["last_funding_rate"].to_numpy(), index=pd.DatetimeIndex(f["calc_time"]))
    cum = s.cumsum().reindex(idx.union(s.index)).ffill().reindex(idx).fillna(0.0).to_numpy()
    cover = float((idx >= s.index.min()).mean())
    return cum, cover


def trade_labels(p: pd.DataFrame, k: float, maxbars: int) -> dict[str, np.ndarray]:
    """봉 i 의 규칙에 대해 i+1 시가 진입, ±k×ATR 첫터치 또는 maxbars 종가 청산.

    반환은 전부 **롱 기준**이다(숏은 부호만 뒤집는다).
      hit   1 위 먼저 / 0 아래 먼저 / NaN 미해결   ← 방향 확률 화면용
      ret   실현 bp (미해결은 그 시점 종가)        ← 경제성용, 버리지 않는다
      hold  보유 봉 수                              ← 펀딩·블록 길이용
    """
    op = p["open"].to_numpy(float); cl = p["close"].to_numpy(float)
    hi = p["high"].to_numpy(float); lo = p["low"].to_numpy(float)
    atr = p["atr_pct"].to_numpy(float)
    n = len(p)
    ent = np.roll(op, -1); ent[-1] = np.nan
    up, dn = ent * (1.0 + k * atr), ent * (1.0 - k * atr)
    hit = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    hold = np.full(n, np.nan)
    live = np.isfinite(ent)
    ar = np.arange(n)
    for j in range(1, maxbars + 1):
        idx = np.flatnonzero(live & (ar + j < n))
        if not len(idx):
            break
        t = idx + j
        hu, hd = hi[t] >= up[idx], lo[t] <= dn[idx]
        done = hu | hd
        if done.any():
            d = idx[done]
            u = hu[done] & ~hd[done]                 # 같은 봉 양쪽 터치는 보수적으로 아래로 본다
            hit[d] = u.astype(float)
            ret[d] = np.where(u, k * atr[d], -k * atr[d]) * 1e4
            hold[d] = j
            live[d] = False
    to = np.flatnonzero(live & (ar + maxbars < n))    # 미해결 — 창 끝 종가로 청산
    ret[to] = (cl[to + maxbars] / ent[to] - 1.0) * 1e4
    hold[to] = maxbars
    return {"hit": hit, "ret": ret, "hold": hold}


def net_bp(ret: np.ndarray, hold: np.ndarray, cf: np.ndarray, side: int, idx: np.ndarray,
           fee: float) -> np.ndarray:
    """건별 순익 bp = 방향부호 실현 − 수수료 − 방향부호 펀딩 누적."""
    ex = np.clip(idx + hold[idx].astype(int), 0, len(cf) - 1)
    fund = (cf[ex] - cf[idx]) * 1e4 * side            # 롱은 내고 숏은 받는다
    return side * ret[idx] - fee - fund


def block_boot(v: np.ndarray, t: np.ndarray, block_ns: float, nboot: int = 2000) -> float:
    """블록 부트 95% 하한. 블록 길이 ≥ 보유기간이어야 트레이드가 쪼개지지 않는다."""
    bid = ((t - t.min()) / block_ns).astype(int)
    ub = np.unique(bid)
    by = {b: v[bid == b] for b in ub}
    rng = np.random.default_rng(20260912)
    bs = np.array([np.concatenate([by[b] for b in rng.choice(ub, len(ub))]).mean() for _ in range(nboot)])
    return float(np.percentile(bs, 2.5))


def run_cell(p: pd.DataFrame, D: dict, F: dict, rules: list, k: float, maxbars: int,
             cf: np.ndarray, ts: np.ndarray) -> dict:
    lab = f"bar{k}x{maxbars}"
    T = trade_labels(p, k, maxbars)
    hit, ret, hold = T["hit"], T["ret"], T["hold"]
    n = len(p)
    base_valid = np.zeros(n, bool); base_valid[R.WARM : n - maxbars - 2] = True
    wins = {}
    for nm, (s, e) in R.SPLITS.items():
        m = base_valid.copy()
        if s: m &= (ts >= np.datetime64(s)).astype(bool)
        if e: m &= (ts <= np.datetime64(e + "T23:59:59")).astype(bool)
        wins[nm] = m
    base = {nm: float(np.nanmean(hit[w & np.isfinite(hit)])) for nm, w in wins.items()}
    b_med = float(np.nanmedian(np.abs(ret[wins["OOS"] & np.isfinite(ret)])))
    resolved = float(np.mean(np.isfinite(hit[wins["OOS"]])))
    a_need = {cn: (1.0 + c / b_med) / 2.0 for cn, c in FEE.items()}
    hold_med = float(np.nanmedian(hold[wins["OOS"]]))

    def screen(hv, bs):
        keep = []
        for da, fa, side in rules:
            m = R.rule_mask(da, fa, D, F)
            row = {"rule": R.rule_name(da, fa), "side": "롱" if side > 0 else "숏", "_da": da, "_fa": fa}
            ok = True
            for nm in ("VAL", "OOS"):
                kk, nn, ph = R.rate_at(np.flatnonzero(m & wins[nm]), side, hv)
                row[f"n_{nm}"], row[f"p_{nm}"], row[f"base_{nm}"] = nn, ph, R.sided(bs[nm], side)
                if nn < MIN_N or R.wilson_lo(kk, nn) <= R.sided(bs[nm], side):
                    ok = False; break
            if not ok:
                continue
            kk, nn, ph = R.rate_at(np.flatnonzero(m & wins["TRAIN"]), side, hv)
            row["n_TRAIN"], row["p_TRAIN"] = nn, ph
            keep.append(row)
        return keep

    passed = screen(hit, base)
    fam = []
    for _ in range(B_FAM):
        sh = int(RNG.integers(2000, n - 2 * R.WARM))
        hs = np.roll(hit, sh)
        fam.append(len(screen(hs, {nm: float(np.nanmean(hs[w & np.isfinite(hs)])) for nm, w in wins.items()})))
    fam_mean = float(np.mean(fam))

    # 통과 규칙 경제성 — 미해결 포함 실현 bp, 수수료 + 펀딩, 블록부트
    day_ns = 86400e9
    block = max(day_ns, hold_med * 300e9)
    for r in passed:
        side = 1 if r["side"] == "롱" else -1
        m = R.rule_mask(r["_da"], r["_fa"], D, F)
        idx = np.flatnonzero(m & wins["OOS"] & np.isfinite(ret))
        v = net_bp(ret, hold, cf, side, idx, FEE["테이커"])
        r["net_bp"] = float(v.mean())
        r["boot_lo"] = block_boot(v, ts[idx].astype("datetime64[ns]").astype(np.int64).astype(float), block)
        itr = np.flatnonzero(m & wins["TRAIN"] & np.isfinite(ret))
        r["train_net_bp"] = float(net_bp(ret, hold, cf, side, itr, FEE["테이커"]).mean()) if len(itr) >= 30 else np.nan
        r["n_blocks"] = int(len(np.unique(((ts[idx].astype("datetime64[ns]").astype(np.int64) -
                                            ts[idx].astype("datetime64[ns]").astype(np.int64).min()) / block).astype(int))))
        r.pop("_da"); r.pop("_fa")
    d = pd.DataFrame(passed)
    if len(d):
        d.sort_values("net_bp", ascending=False).to_csv(OUT / f"rules_{lab}.csv", index=False)
    return {"label": lab, "k": k, "maxbars": maxbars, "barrier_bp": b_med, "hold_med_bars": hold_med,
            "resolved_OOS": resolved, "base_OOS": base["OOS"], "a_need_taker": a_need["테이커"],
            "a_need_maker": a_need["메이커"], "passed": len(passed), "fam_null": fam_mean,
            "best_net_bp": float(d.net_bp.max()) if len(d) else np.nan,
            "best_boot_lo": float(d.boot_lo.max()) if len(d) else np.nan,
            "n_boot_pos": int((d.boot_lo > 0).sum()) if len(d) else 0,
            "n_train_pos": int((d.train_net_bp > 0).sum()) if len(d) else 0,
            "n_both": int(((d.boot_lo > 0) & (d.train_net_bp > 0)).sum()) if len(d) else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    p = pd.read_parquet(R.PANEL)
    ts = p["timestamp"].to_numpy()
    cf, cover = funding_cum(ts)
    log(f"펀딩 커버리지 {cover:.1%} (그 이전 봉은 펀딩 0 으로 둔다)")
    D, F = R.atoms(p)
    rules = R.enumerate_rules(D, F)
    log(f"규칙 {len(rules):,}개")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, mb in CELLS:
        t0 = time.time()
        rows.append(run_cell(p, D, F, rules, k, mb, cf, ts))
        log(f"k={k} 창={mb}봉 완료 ({time.time()-t0:.0f}s) — {rows[-1]['passed']}통과 / 귀무 {rows[-1]['fam_null']:.1f}")
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sweep.csv", index=False)
    print("\n" + "=" * 124)
    print("지평·배리어 스윕 — 필요정확도 a* = (1 + 비용/b)/2 · 순익은 미해결 포함 실현 bp − 수수료 − 펀딩")
    print("=" * 124)
    print(f"{'k×ATR':>7}{'창(봉)':>8}{'배리어b':>9}{'중앙보유':>9}{'해결률':>8}{'a*테이커':>9}"
          f"{'통과':>6}{'귀무':>7}{'최고순익':>9}{'부트하한':>9}{'부트>0':>7}{'TRAIN>0':>8}{'둘다':>6}")
    for r in rows:
        print(f"{r['k']:>7.1f}{r['maxbars']:>8,}{r['barrier_bp']:>9.1f}{r['hold_med_bars']:>9.0f}"
              f"{r['resolved_OOS']:>8.2f}{r['a_need_taker']:>9.3f}{r['passed']:>6}{r['fam_null']:>7.1f}"
              f"{r['best_net_bp']:>9.2f}{r['best_boot_lo']:>9.2f}{r['n_boot_pos']:>7}{r['n_train_pos']:>8}{r['n_both']:>6}")
    return 0


def _selftest() -> int:
    p = pd.DataFrame({"open": [10, 10, 10, 10, 10.0], "high": [10, 10, 12, 10, 10.0],
                      "low": [10, 10, 10, 7, 10.0], "close": [10, 10, 11, 8, 10.0],
                      "atr_pct": [0.1] * 5})
    T = trade_labels(p, 1.0, 3)
    assert T["hit"][0] == 1 and T["hold"][0] == 2, (T["hit"][0], T["hold"][0])
    assert abs(T["ret"][0] - 1000.0) < 1e-6, T["ret"][0]     # +10% = 1000bp
    assert T["hit"][2] == 0 and abs(T["ret"][2] + 1000.0) < 1e-6, (T["hit"][2], T["ret"][2])
    T2 = trade_labels(pd.DataFrame({"open": [10.0] * 5, "high": [10.0] * 5, "low": [10.0] * 5,
                                    "close": [10, 10, 10, 10, 10.0], "atr_pct": [0.1] * 5}), 1.0, 3)
    assert np.isnan(T2["hit"][0]) and T2["ret"][0] == 0.0, "미해결은 창 끝 종가로 청산"
    cf = np.array([0.0, 0.0001, 0.0002, 0.0003])
    v = net_bp(np.array([100.0, 0, 0, 0]), np.array([2.0, 0, 0, 0]), cf, 1, np.array([0]), 10.0)
    assert abs(v[0] - (100 - 10 - 2.0)) < 1e-6, v[0]          # 펀딩 0.0002 = 2bp, 롱은 낸다
    v2 = net_bp(np.array([100.0, 0, 0, 0]), np.array([2.0, 0, 0, 0]), cf, -1, np.array([0]), 10.0)
    assert abs(v2[0] - (-100 - 10 + 2.0)) < 1e-6, v2[0]       # 숏은 받는다
    print("selftest OK — 첫터치/미해결 청산/실현bp · 펀딩 롱은 내고 숏은 받는다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
