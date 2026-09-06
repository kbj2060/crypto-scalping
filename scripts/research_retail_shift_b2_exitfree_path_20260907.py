#!/usr/bin/env python3
"""B2(롱숏비 급변)의 **청산 없는 경로 방향 측정** (2026-09-07).

B2의 유일한 근거였던 경제성은 결함 트레일링(`sim_exit`)에서 나왔고 2026-09-07 수정으로 무효가 됐다.
그래서 지속 규칙에 09-06에 적용했던 것과 **같은 방법**으로, 청산·비용·배리어를 전부 빼고
**가격 경로만으로** B2가 고른 방향이 실제로 우세한지 잰다.

  모집단  `count_long_short_ratio`(metrics 덤프) d6 = ratio − ratio.shift(6),
          z = (d6 − mean288)/std288 (min_periods 144), **lag1**(봉 T 결정은 T−5분 행까지만 본다),
          |z| ≥ 2.2616 이면 발동, 같은 측면 직전 12행 안 발동 없을 때만(GAP12, 뒤만 봄)
  방향    **−부호**: z ≤ −T → 롱, z ≥ +T → 숏 (개미 쏠림 변화의 반대)
  측정    발동 봉 종가 c0 기준 이후 H봉의 최대 유리/불리 이탈폭. 규칙 방향이 더 멀리 뻗었는가(우세 빈도).
          H = 12/24/48/96/200봉. 일군집 부트스트랩 CI. 0.5를 포함하면 방향 정보 없음.
  귀무    같은 발동 봉·같은 롱숏 비율로 측면을 무작위 배정(B=200) -- 그 구간의 시장 드리프트를 상쇄한다.

⚠️HOLDOUT(≥2026-04-01) 미접촉. 덤프 `create_time` 정렬은 라이브 API 스탬프와 의미가 달라(러너 --parity 참고)
라이브 발동과 1:1 대응은 아니다 -- 여기서 재는 것은 **규칙의 방향 정보량**이다.
"""
from __future__ import annotations

import glob, io, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MET = ROOT / "binance_data/metrics"
OUT = ROOT / "data/research/retail_shift_b2_exitfree_path_20260907"
THRESH_Z, Z_WIN, Z_MINP, DIFF_N, GAP = 2.2616, 288, 144, 6, 12
HORIZONS = (12, 24, 48, 96, 200)
SPLITS = {"TRAIN": ("2024-05-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"), "OOS": ("2026-01-01", "2026-03-31")}
B_BOOT, B_NULL = 2000, 200


def log(m): print(f"[b2path] {m}", flush=True)


def first_fire(raw, gap):
    out = np.zeros(len(raw), bool); last = -10**9
    for i in np.flatnonzero(raw):
        if i - last > gap:
            out[i] = True
        last = i
    return out


def day_boot(x, days, B, rng):
    d = pd.Series(np.asarray(x, float), index=pd.DatetimeIndex(days).normalize())
    u = d.index.unique().to_numpy(); g = d.groupby(level=0)
    s = g.sum().reindex(u).to_numpy(); c = g.count().reindex(u).to_numpy()
    o = np.empty(B)
    for b in range(B):
        k = rng.integers(0, len(u), len(u)); o[b] = s[k].sum() / max(c[k].sum(), 1)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(20260907)
    rows = []
    for f in sorted(glob.glob(str(MET / "ETHUSDT-metrics-*.zip"))):
        z = zipfile.ZipFile(f)
        rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["create_time", "count_long_short_ratio"]))
    m = pd.concat(rows, ignore_index=True)
    m["ts"] = pd.to_datetime(m["create_time"])
    m = m.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    log(f"metrics {len(m):,}행 {m['ts'].iloc[0]} ~ {m['ts'].iloc[-1]} ({time.time()-t0:.0f}s)")

    s = pd.Series(m["count_long_short_ratio"].to_numpy(float), index=m["ts"])
    d6 = s - s.shift(DIFF_N)
    zz = (d6 - d6.rolling(Z_WIN, min_periods=Z_MINP).mean()) / d6.rolling(Z_WIN, min_periods=Z_MINP).std().replace(0, np.nan)
    zl = zz.shift(1)                                        # lag1: 봉 T 결정은 T−5분 행까지
    lf = first_fire(np.nan_to_num(zl.to_numpy() <= -THRESH_Z).astype(bool), GAP)
    sf = first_fire(np.nan_to_num(zl.to_numpy() >= THRESH_Z).astype(bool), GAP)
    F = pd.DataFrame({"ts": zz.index, "long": lf, "short": sf})
    F = F.loc[F["long"] | F["short"]].copy()
    F["dir"] = np.where(F["long"], 1.0, -1.0)
    log(f"발동 {len(F):,} (롱 {int(F['long'].sum()):,} 숏 {int(F['short'].sum()):,})")

    kl = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    kl = kl.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    pos = pd.Series(np.arange(len(kl)), index=kl["timestamp"])
    F["i"] = pos.reindex(F["ts"]).to_numpy()
    F = F.loc[np.isfinite(F["i"])].copy(); F["i"] = F["i"].astype(int)
    h, l, c = (kl[x].to_numpy(float) for x in ("high", "low", "close"))
    F = F.loc[F["i"] + max(HORIZONS) < len(kl)].reset_index(drop=True)
    F["split"] = pd.NA
    for w, (a, b) in SPLITS.items():
        F.loc[(F["ts"] >= a) & (F["ts"] <= pd.Timestamp(b) + pd.Timedelta(days=1)), "split"] = w
    F = F.loc[F["split"].notna()].reset_index(drop=True)
    log(f"창 안 발동 {len(F):,} · " + " ".join(f"{w} {int((F['split']==w).sum()):,}" for w in SPLITS)
        + f" · 일수 {F['ts'].dt.normalize().nunique()}")

    rep = {"threshold_z": THRESH_Z, "gap_rows": GAP, "diff_n": DIFF_N, "z_win": Z_WIN, "lag": 1,
           "holdout_touched": False, "n_fires": int(len(F)), "horizons": list(HORIZONS), "results": {}}
    ii = F["i"].to_numpy(); dr = F["dir"].to_numpy(); c0 = c[ii]; days = F["ts"].to_numpy()
    for H in HORIZONS:
        idx = ii[:, None] + np.arange(1, H + 1)[None, :]
        up = (h[idx].max(axis=1) - c0) / c0
        dn = (c0 - l[idx].min(axis=1)) / c0
        mfe_dir = np.where(dr > 0, up, dn); mfe_opp = np.where(dr > 0, dn, up)
        dom = (mfe_dir > mfe_opp).astype(float)
        rep["results"][f"H{H}"] = {}
        for w in list(SPLITS) + ["ALL"]:
            msk = np.ones(len(F), bool) if w == "ALL" else (F["split"].to_numpy() == w)
            if msk.sum() < 50:
                continue
            lo, hi = day_boot(dom[msk], days[msk], B_BOOT, rng)
            # 측면비율 매칭 무작위 귀무
            p_long = float((dr[msk] > 0).mean()); nulls = np.empty(B_NULL)
            for b in range(B_NULL):
                rd = np.where(rng.random(int(msk.sum())) < p_long, 1.0, -1.0)
                nulls[b] = ((np.where(rd > 0, up[msk], dn[msk])) > (np.where(rd > 0, dn[msk], up[msk]))).mean()
            rep["results"][f"H{H}"][w] = {"n": int(msk.sum()), "dominance": round(float(dom[msk].mean()), 4),
                                          "ci95": [round(lo, 4), round(hi, 4)],
                                          "null_mean": round(float(nulls.mean()), 4),
                                          "null_p95": round(float(np.percentile(nulls, 95)), 4),
                                          "null_percentile": round(float((nulls < dom[msk].mean()).mean() * 100), 1),
                                          "p_long": round(p_long, 3)}
        r = rep["results"][f"H{H}"]
        log(f"H{H:3d} · " + " · ".join(
            f"{w} {r[w]['dominance']:.4f} [{r[w]['ci95'][0]:.4f},{r[w]['ci95'][1]:.4f}] (귀무 {r[w]['null_mean']:.4f}, 백분위 {r[w]['null_percentile']:.0f})"
            for w in ("VAL", "OOS", "ALL") if w in r))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    log(f"저장 {OUT/'report.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
