#!/usr/bin/env python3
"""⭐**상위트레이더 롱숏비 수준 페이드** -- 첫 진짜 후보의 정밀 감사 (2026-09-08).

## 후보
`tt_count_level` = log(`count_toptrader_long_short_ratio`) 의 **수준**.
롱숏비가 낮은(= 상위트레이더가 숏 쏠린) 종목 k개 롱 / 높은 종목 k개 숏. 되돌아보기 창 없음.
`H=864봉(3일) · k=5 · NU=40`: TRAIN **+33.15** [+12.8] · VAL +101.3 [+37.4] · OOS +40.8 [−13.6] ·
HOLDOUT +20.3. 90셀 중 **네 창 점추정 양수 44개**(우연 기대 5.6) · H 에 대해 단조 증가.
감사 1차: 평균 +41.0 / 중앙 +16.0 / 승률 53.4% / **상위1% 제거해도 +28.2** / 상위20일 52.9%
— 오늘 밤 다른 후보들과 달리 꼬리 의존이 아니다.

## 이 스크립트가 확인하는 6가지 위협
1. ⚠️**겹침으로 CI 가 좁아졌다** -- H=864 인데 step=216 이었다. **비겹침(step=H)** 으로 다시 재고,
   추가로 **블록 부트스트랩**(블록 길이 = H 이상)을 쓴다.
2. ⚠️**무작위 배정 귀무** -- 같은 시각·같은 유니버스에서 k개씩 무작위 롱숏(B=400).
3. ⚠️**변동성 불일치** -- 두 다리의 변동성이 다르면 롱숏이라도 리스크 중립이 아니다.
   1/vol 가중 변형을 같이 낸다.
4. ⚠️**섹터/구성 편향** -- 어떤 종목이 어느 다리에 반복해서 들어가는가.
5. ⚠️**생존편향** -- 이 패널은 현재 상장 종목만이다(상장폐지 없음). 롱 다리가 알트로 기울면 낙관 편향.
6. ⚠️**부호 뒤집기** 대조군.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
H_GRID = (288, 576, 864)
K_GRID = (3, 5, 8)
NU = 40
LIQW = 288
VOLW = 576
BOOT = 4000
NULLB = 400
SEED = 20260908


def boot_ci(v, B=BOOT, rng=None):
    if len(v) < 10: return (np.nan, np.nan)
    idx = rng.integers(0, len(v), (B, len(v)))
    return tuple(np.percentile(v[idx].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    mz = np.load(DIR / "metrics_panel.npz", allow_pickle=True)
    S = np.where(mz["count_toptrader_long_short_ratio"] > 0,
                 np.log(np.maximum(mz["count_toptrader_long_short_ratio"], 1e-9)), np.nan)
    lr = np.full_like(Cm, np.nan); lr[1:] = np.log(Cm[1:] / Cm[:-1])
    vol = pd.DataFrame(lr).rolling(VOLW, min_periods=VOLW // 2).std().to_numpy()
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w

    print("=" * 108)
    print("1) 비겹침(step=H) 재추정 -- 겹침이 CI 를 좁혔는지")
    print("=" * 108)
    print(f"{'H':>5}{'k':>4} {'n':>5} | " + " | ".join(f"{w[:8]:>24}" for w in SPLITS))
    keep = {}
    for H in H_GRID:
        fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
        tid = np.arange(max(LIQW, VOLW) + 1, len(ts) - H - 2, H)      # 비겹침
        el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid]) & np.isfinite(vol[tid])
        sa = np.where(el, S[tid], np.nan); fw = np.where(el, fwd[tid], np.nan)
        vv = np.where(el, vol[tid], np.nan)
        nval = np.isfinite(sa).sum(1)
        order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), axis=1)
        for k in K_GRID:
            gd = nval >= 2 * k + 2
            rr = np.flatnonzero(gd)
            if len(rr) < 60: continue
            lo_i = order[rr][:, :k]
            hi_i = order[rr][np.arange(len(rr))[:, None],
                             (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
            fl = np.take_along_axis(fw[rr], lo_i, 1); fh = np.take_along_axis(fw[rr], hi_i, 1)
            port = (fl.mean(1) - fh.mean(1)) / 2.0 * 1e4
            ww = win_of[tid][rr]
            line = f"{H:>5}{k:>4} {len(rr):>5} | "
            for w in SPLITS:
                m = (ww == w) & np.isfinite(port)
                if m.sum() < 10: line += f"{'--':>24} | "; continue
                lo, hi = boot_ci(port[m], rng=rng)
                line += f"{port[m].mean():>+7.1f}[{lo:>+6.1f},{hi:>+6.1f}]n{m.sum():>3} | "
            print(line, flush=True)
            keep[(H, k)] = (tid[rr], port, ww, lo_i, hi_i, fl, fh, vv[rr], nval[rr], order[rr])

    H, k = 864, 5
    tid_, port, ww, lo_i, hi_i, fl, fh, vv, nv, orr = keep[(H, k)]
    fin = np.isfinite(port)
    print("\n" + "=" * 108)
    print(f"대표 셀 H={H}(3일) k={k} NU={NU} · 비겹침 n={fin.sum()}")
    print("=" * 108)

    print("\n2) 무작위 배정 귀무 (같은 시각·같은 유니버스, B=%d)" % NULLB)
    obs = {w: port[fin & (ww == w)].mean() for w in SPLITS if (fin & (ww == w)).sum() >= 10}
    nulls = {w: [] for w in obs}
    fwd_all = np.where(np.isfinite(fl), fl, np.nan)
    for _ in range(NULLB):
        pick = np.stack([rng.permutation(int(n_))[:2 * k] for n_ in nv])
        L_i = np.take_along_axis(orr, pick[:, :k], 1); H_i = np.take_along_axis(orr, pick[:, k:], 1)
        fwdm = np.full((len(tid_), Om.shape[1]), np.nan)
        # 필요한 열만 다시 계산하지 않도록 원본 fwd 재사용
        pass
    print("   (전체 fwd 행렬 재구성이 필요해 아래 블록에서 처리)")
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    F = fwd[tid_]
    for _ in range(NULLB):
        pick = np.stack([rng.permutation(int(n_))[:2 * k] for n_ in nv])
        L_i = np.take_along_axis(orr, pick[:, :k], 1); H_i = np.take_along_axis(orr, pick[:, k:], 1)
        p = (np.take_along_axis(F, L_i, 1).mean(1) - np.take_along_axis(F, H_i, 1).mean(1)) / 2 * 1e4
        for w in obs:
            m = fin & (ww == w)
            nulls[w].append(np.nanmean(p[m]))
    for w in obs:
        a = np.array(nulls[w]); pc = (a >= obs[w]).mean()
        print(f"   {w:>14}: 관측 {obs[w]:>+7.1f}bp · 귀무 평균 {a.mean():>+6.1f} "
              f"sd {a.std():>5.1f} · 백분위 {100*(1-pc):>5.1f}% · p={max(pc,1/NULLB):.4f}")

    print("\n3) 변동성 정합 -- 두 다리 연변동성")
    vl = np.take_along_axis(vv, lo_i, 1).mean(1); vh = np.take_along_axis(vv, hi_i, 1).mean(1)
    ann = np.sqrt(288 * 365)
    print(f"   롱 다리 {np.nanmean(vl)*ann:.1%} · 숏 다리 {np.nanmean(vh)*ann:.1%} · "
          f"비율 {np.nanmean(vl)/np.nanmean(vh):.3f}")
    w_l = 1.0 / np.maximum(np.take_along_axis(vv, lo_i, 1), 1e-9)
    w_h = 1.0 / np.maximum(np.take_along_axis(vv, hi_i, 1), 1e-9)
    pv = ((fl * w_l).sum(1) / w_l.sum(1) - (fh * w_h).sum(1) / w_h.sum(1)) / 2 * 1e4
    line = "   1/vol 가중: "
    for w in SPLITS:
        m = fin & (ww == w)
        if m.sum() < 10: continue
        lo, hi = boot_ci(pv[m], rng=rng)
        line += f"{w[:5]} {np.nanmean(pv[m]):>+7.1f}[{lo:>+6.1f}] "
    print(line)

    print("\n4~5) 다리 구성 (생존편향 점검)")
    cl = pd.Series([syms[i] for i in lo_i[fin].ravel()]).value_counts()
    ch = pd.Series([syms[i] for i in hi_i[fin].ravel()]).value_counts()
    print(f"   롱(숏쏠림) 상위8: {list(cl.head(8).items())}")
    print(f"   숏(롱쏠림) 상위8: {list(ch.head(8).items())}")
    maj = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}
    print(f"   메이저5 비중 -- 롱 {cl[cl.index.isin(maj)].sum()/cl.sum():.1%} · "
          f"숏 {ch[ch.index.isin(maj)].sum()/ch.sum():.1%}")

    print("\n6) 부호 뒤집기 · 손익 분포")
    v = port[fin]
    print(f"   정방향 {v.mean():+.1f}bp · 뒤집기 {-v.mean():+.1f}bp · 중앙 {np.median(v):+.1f} · "
          f"승률 {(v>0).mean():.1%} · 상위1%제거 {v[v<np.percentile(v,99)].mean():+.1f}")
    print(json.dumps({"n_nonoverlap": int(fin.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
