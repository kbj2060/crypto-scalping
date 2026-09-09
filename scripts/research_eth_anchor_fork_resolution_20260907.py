#!/usr/bin/env python3
"""증거신호 앵커 = **방향 무관 갈림길(fork)** 인가 (2026-09-07).

사용자: *"이 증거신호로 바닥에는 롱, 천장에는 숏을 치는게 말이 안돼. 단지 이 증거신호 8가지는
이 신호가 추세와 되돌림의 갈림길이 될 수 있다는 거야. 이 지점을 잡는게 중요해."*

## 이 재구성이 데이터와 맞는 이유 (이미 측정된 것)
- 학습 라벨(K x ATR 터치)에 **방향 내용이 거의 없다** — 유리/불리 터치율 차이 바닥 8종 중 6종 음수(부록 D).
- 페이드/지속 레짐은 전 지평 **동전**(09-06, 부록 A~C에서 앵커를 바꿔도 불변).
- 그런데 앵커는 로컬극점을 ATR매칭 귀무의 2.2배로 잡는다(부록 B/C).
=> "어느 쪽" 신호가 아니라 "여기서 결판난다" 신호라는 해석과 정합적이다.

## 이미 닫힌 인접 축과의 차이 (반복 아님)
- `eth_regime_volexpand_early_warning_mixed_result_20260827` = **크기**(변동성 확장) 조기경보. pooled 기각.
- `eth_breakout_continuation_rejected_20260831` = **레벨 돌파 후 지속**. 기각.
- `eth_chop_fade_..._breakout_predictor_20260827` Stream2 = 방향 베팅의 손절 예측. 기각.
여기서 묻는 F1 은 **크기로 나눈 편측성**이라 크기 축과 분리된다. F2 는 브라켓 체결성이다.

## 지표 (전부 출구·비용 없음, 방향 무관)
발동 봉 종가 c0 기준 이후 H봉:  up = (max high - c0)/c0 ,  dn = (c0 - min low)/c0
  F1 결단력   |up - dn| / (up + dn)   in [0,1]. **크기로 나눠 변동성 축과 분리.**
                                       0.5 부근 = 양쪽으로 흔들림(횡보), 1 부근 = 한쪽으로 결판.
  F2 브라켓   max(up,dn) >= K*atr  AND  min(up,dn) < m*atr    (K,m) = (2,1)·(3,1)·(3,1.5)
                                       "한쪽으로 크게, 반대로는 거의 안 감" = 양방향 브라켓이 먹히는 모양.
  F3 크기     (up + dn) / atr        맥락용. 이 축이 예측 가능하다는 건 이미 안다.
H = 12 / 24 / 48.

## 사전 판정 (실행 전 고정)
ATR 십분위 매칭 기준선을 앵커별로 빼고(excess_i = y_i - m_d(i)), **VAL·OOS 두 창 모두**
일군집 부트스트랩 CI 하한 > 0. 현행 first_fire_union 대비 차이도 같은 기준으로 병기.
n >= 30 이고 서로 다른 날 >= 10. 다중도 잣대 = 시간이동 플라시보 R=10.
⭐F1 이 통과하고 F3 만 통과하는 게 아니어야 '갈림길' 주장이 성립한다 —
  F3(크기)만 통과하면 그건 변동성 확장 재발견이고 이미 닫힌 축이다.
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

import research_eth_anchor_combination_screen_20260907 as S  # noqa: E402
import research_eth_bottom_anchor_wd_grid_20260907 as G  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_combo_20260907"
HS = (12, 24, 48)
BRACKETS = ((2.0, 1.0), (3.0, 1.0), (3.0, 1.5))
MIN_N, MIN_DAYS = 30, 10
PLACEBO_R = 10


def fork_metrics(high, low, close, atr, H):
    n = len(close)
    up = (pd.Series(high[::-1]).rolling(H, min_periods=H).max().to_numpy()[::-1] - close) / close
    dn = (close - pd.Series(low[::-1]).rolling(H, min_periods=H).min().to_numpy()[::-1]) / close
    up = np.roll(up, -1); dn = np.roll(dn, -1)          # t+1..t+H
    up[-H - 1:] = np.nan; dn[-H - 1:] = np.nan
    tot = up + dn
    out = {"F1": np.where(tot > 0, np.abs(up - dn) / np.maximum(tot, 1e-12), np.nan),
           "F3": np.where(atr > 0, tot / np.maximum(atr, 1e-12), np.nan)}
    mx, mn = np.maximum(up, dn), np.minimum(up, dn)
    for K, m in BRACKETS:
        out[f"F2_{K:g}_{m:g}"] = ((mx >= K * atr) & (mn < m * atr)).astype(float)
        out[f"F2_{K:g}_{m:g}"][~np.isfinite(mx) | ~np.isfinite(atr)] = np.nan
    return out


def anchors_all(P, fire):
    a = G.build_anchor_sets(P, fire)
    out = {}
    for side in ("bottom", "top"):
        for nm, wc in [("first_fire_union", None), ("any2", 3), ("any3", 1), ("any3", 3), ("any3", 12), ("any4", 12)]:
            out[(f"{side}:{nm}" if nm == "first_fire_union" else f"{side}:{nm}/Wc{wc}")] = a[(side, nm, wc)]
    for nm, wc in [("first_fire_union", None), ("any3", 3), ("any3", 12)]:
        u = np.unique(np.concatenate([a[("bottom", nm, wc)], a[("top", nm, wc)]]))
        keep, last = [], -10 ** 9
        for i in u:                                    # 양측면 합집합도 GAP12 재적용
            if i - last > S.GAP_BARS:
                keep.append(i)
            last = i
        out[("either:" + (nm if nm == "first_fire_union" else f"{nm}/Wc{wc}"))] = np.array(keep, dtype=int)
    return out


def run(P, fire, rng, verbose=True):
    ts, dec, day = P["ts"], P["dec"], P["day"]
    wins = {"TRAIN": (S.TRAIN_START, S.VAL_START - pd.Timedelta(seconds=1)),
            "VAL": (S.VAL_START, S.VAL_END), "OOS": (S.OOS_START, S.OOS_END)}
    pools = {w: np.flatnonzero((ts >= a) & (ts <= b)) for w, (a, b) in wins.items()}
    anc = anchors_all(P, fire)
    rows = []
    for H in HS:
        M = fork_metrics(P["high"], P["low"], P["close"], P["atr"], H)
        for met, y in M.items():
            base = {w: S.decile_baseline(y, dec, pools[w]) for w in wins}
            ff = {}
            for w in ("VAL", "OOS"):
                a = anc["either:first_fire_union"]; p = pools[w]; a = a[(a >= p[0]) & (a <= p[-1])]
                yy, dd, dy = y[a], dec[a], day[a]
                ok = np.isfinite(yy) & np.isfinite(dd)
                ff[w] = (yy[ok] - base[w][dd[ok].astype(int)], dy[ok])
            for name, a0 in anc.items():
                rec = {"H": H, "metric": met, "anchor": name}
                good = True
                for w in wins:
                    p = pools[w]; a = a0[(a0 >= p[0]) & (a0 <= p[-1])]
                    yy, dd, dy = y[a], dec[a], day[a]
                    ok = np.isfinite(yy) & np.isfinite(dd)
                    yy, dd, dy = yy[ok], dd[ok].astype(int), dy[ok]
                    if len(yy) < MIN_N or len(np.unique(dy)) < MIN_DAYS:
                        good = False
                        continue
                    exc = yy - base[w][dd]
                    rec[f"{w}_n"] = len(yy); rec[f"{w}_obs"] = float(yy.mean())
                    rec[f"{w}_null"] = float(base[w][dd].mean()); rec[f"{w}_exc"] = float(exc.mean())
                    if w != "TRAIN":
                        rec[f"{w}_exc_lo"], rec[f"{w}_exc_hi"] = S.day_ci(exc, dy, rng)
                        d2, dd2 = ff[w]
                        rec[f"{w}_vsff"] = float(exc.mean() - d2.mean())
                        rec[f"{w}_vsff_lo"], _ = S.diff_day_ci(exc, dy, d2, dd2, rng)
                rec["evaluable"] = good
                rec["PASS"] = bool(good and rec.get("VAL_exc_lo", -9) > 0 and rec.get("OOS_exc_lo", -9) > 0)
                rows.append(rec)
        if verbose:
            print(f"  H={H} 완료", flush=True)
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(20260907)
    P = S.build_panel()
    sig = pd.read_parquet(S.CACHE)
    for c in ("high", "low", "close"):
        P[c] = sig[c].to_numpy(float)
    P["atr"] = sig["atr_pct"].to_numpy(float)
    print("[1/2] 실제 ...", flush=True)
    real = run(P, P["fire"], rng)
    real.to_parquet(OUT / "fork_real.parquet")
    print("      PASS by metric:", real[real.anchor != "either:first_fire_union"].groupby("metric").PASS.sum().to_dict())
    print(f"[2/2] 플라시보 R={PLACEBO_R} ...", flush=True)
    pl = []
    for r in range(PLACEBO_R):
        pr = run(P, S.placebo_fire(P["fire"], rng), rng, verbose=False)
        pl.append({"rep": r, **pr.groupby("metric").PASS.sum().to_dict()})
        print(f"      rep {r+1}: {pl[-1]}", flush=True)
    pd.DataFrame(pl).to_csv(OUT / "fork_placebo.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
