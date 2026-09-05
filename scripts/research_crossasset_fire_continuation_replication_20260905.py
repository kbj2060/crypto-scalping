#!/usr/bin/env python3
"""**교차자산 지속 규칙 재현** — "첫발동 봉 = 지속 시점"이 ETH 밖에서도 성립하는가 (2026-09-05).

호메로스 「다음 단계」의 **연구 1순위**: R(반전 8종 첫발동의 지속 방향)이 ETH 특이성인지 메커니즘인지.
성립하면 (a) 메커니즘 증거이고 (b) 3자산 분산으로 일 변동성이 줄어 같은 엣지의 샤프가 올라간다.

## 규칙 (ETH R에서 자유도 없이 이식 — 자산별 튜닝 금지)
  발동   `live_evidence_signal_dashboard_20260823.compute_signals()`의 raw 단일봉 발동 (**라이브 정본, 자산 공유**)
         8종 × 바닥/천장. 첫발동 = 같은 신호·같은 측면이 직전 12봉 안에 발동하지 않음. (봉,측면) 중복 제거.
  방향   지속 = 신호 반대 (바닥 발동 → 숏, 천장 발동 → 롱)
  진입   open[i+1] 시장가 · 청산 sim_exit(5.0 SL / 1.5 ARM / 0.1 trail ×ATR14) 200봉 · 비용 10bp · 동시 5
  창     TRAIN 2024-05-01~2025-08-31 · VAL 2025-09-01~2025-12-31 · OOS 2026-01-01~2026-03-31
         **HOLDOUT(≥2026-04-01)은 로드 단계에서 잘라낸다 — 이 스크립트는 볼 수 없다.**

## 주의 (자산 이식의 알려진 함정 — §5.19 포팅 프로토콜)
  · `smt_divergence`는 교차자산 비확인 신호다. 참조 자산을 자산별로 준다(ETH↔BTC, XRP·SOL→BTC). 참조가 자기 자신이면 발동 불가.
  · `funding_df=None`이면 `orthogonal_combo` **바닥 다리가 delta_z 단독 공식으로 degrade**한다(2026-08-27 이전 형태).
    네 자산 **전부 동일하게** None을 주므로 자산 간 비교는 공정하다. ETH 정본(펀딩 포함)과는 다르다는 점만 명시.
  · ETH를 같은 파이프라인으로 함께 돌려 **대조**로 삼는다 — 연구용 oof 발동집합(R 원본) 대신
    compute_signals raw 발동을 쓰므로 수치가 정확히 같을 필요는 없고, **부호·크기**가 맞는지를 본다.

판정: 자산별로 (지속 − 페이드) 일군집 CI 하한 > 0 이고 지속 포트폴리오 exp_bp > 0 이면 "재현".
VAL·OOS 두 창 모두 만족해야 한다. 연구/개발 점수 — 승격은 전진 섀도우.
"""
from __future__ import annotations

import importlib.util
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


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


V2 = _load("hev2_xa", "scripts/research_homer_entry_v2_20260904.py")
DASH = _load("dash_sig", "scripts/live_evidence_signal_dashboard_20260823.py")
C1M = _load("comp1_xa", "scripts/research_eth_composite_direction_trend_pullback_20260905.py")
sim_exit, portfolio, day_boot, stats_of = V2.sim_exit, V2.portfolio, V2.day_boot, V2.stats_of
pf, cand_of = C1M.pf, C1M.cand_of
OUT = ROOT / "data/research/crossasset_fire_continuation_20260905"
SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
ASSETS = {"ETHUSDT": "BTCUSDT", "BTCUSDT": "ETHUSDT", "XRPUSDT": "BTCUSDT", "SOLUSDT": "BTCUSDT"}
CELL, FWD, COST, GAP, CAP = (5.0, 1.5, 0.1), 200, 10.0, 12, 5
SPLITS = {"TRAIN": ("2024-05-01", "2025-09-01"), "VAL": ("2025-09-01", "2026-01-01"), "OOS": ("2026-01-01", "2026-04-01")}
HOLDOUT_START = pd.Timestamp("2026-04-01")
WARMUP_START = pd.Timestamp("2024-01-01")           # 지표 워밍업 (TRAIN 시작 전 4개월)
B_BOOT, B_NULL, NULL_POOL = 1000, 200, 12000
rng = np.random.default_rng(20260905)


def log(m): print(f"[xasset] {m}", flush=True)


def load_kl(sym):
    p = ROOT / f"binance_data/klines/{sym}/{sym}-5m-api.csv"
    d = pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close", "volume", "trades", "taker_buy_base"], parse_dates=["timestamp"])
    d = d.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return d.loc[(d["timestamp"] >= WARMUP_START) & (d["timestamp"] < HOLDOUT_START)].reset_index(drop=True)   # HOLDOUT 차단


def first_fire_mask(fire, gap=GAP):
    """직전 gap봉 안에 같은 (신호,측면) 발동이 없을 때만 True (뒤만 봄)."""
    idx = np.flatnonzero(fire); keep = np.zeros(len(fire), bool); last = -10 ** 9
    for j in idx:
        if j - last > gap:
            keep[j] = True
        last = j
    return keep


def run_asset(sym, ref_sym):
    t0 = time.time()
    kl = load_kl(sym); ref = load_kl(ref_sym)
    r = ref.rename(columns={c: c for c in ref.columns})
    sig = DASH.compute_signals(kl.copy(), btc_df=r, funding_df=None)
    n = len(kl)
    c = kl["close"].to_numpy(float); h = kl["high"].to_numpy(float); l = kl["low"].to_numpy(float); o = kl["open"].to_numpy(float)
    prev = np.r_[np.nan, c[:-1]]; tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr_all = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    ts_all = kl["timestamp"].to_numpy()
    rows = []; per_sig = {}
    for s in SIGNALS:
        for side, sd in (("bottom", 1), ("top", 0)):
            col = f"{side}_{s}"
            if col not in sig.columns:
                continue
            f = sig[col].fillna(False).to_numpy(bool)
            ff = first_fire_mask(f)
            per_sig.setdefault(s, {})[side] = {"raw": int(f.sum()), "first": int(ff.sum())}
            idx = np.flatnonzero(ff)
            rows.append(pd.DataFrame({"i": idx, "is_downside": sd, "signal": s}))
    F = pd.concat(rows, ignore_index=True).sort_values("i")
    F = F.drop_duplicates(["i", "is_downside"]).reset_index(drop=True)
    ok = (F["i"].to_numpy() + 1 + FWD < n) & np.isfinite(atr_all[F["i"].to_numpy()])
    F = F.loc[ok].reset_index(drop=True)
    i = F["i"].to_numpy(); sd = F["is_downside"].to_numpy().astype(int)
    ts = ts_all[i]; atr = atr_all[i]; entry = o[i + 1]
    fade_sign = np.where(sd == 1, 1.0, -1.0); cont_sign = -fade_sign
    st = i + 1; ix = st[:, None] + np.arange(FWD)
    H, L, C = h[ix], l[ix], c[ix]
    cont_ret, cont_ex = sim_exit(entry, atr, cont_sign, H, L, C, *CELL)
    fade_ret, fade_ex = sim_exit(entry, atr, fade_sign, H, L, C, *CELL)
    cont_bp = cont_ret * 1e4 - COST; fade_bp = fade_ret * 1e4 - COST
    tsi = pd.DatetimeIndex(ts)
    rep = {"symbol": sym, "smt_reference": ref_sym, "funding": None, "n_bars": int(n),
           "date_range": [str(kl['timestamp'].iloc[0]), str(kl['timestamp'].iloc[-1])],
           "n_first_fires": int(len(F)), "per_signal_first_fires": per_sig, "windows": {}}
    # 무작위 진입 귀무용 풀 (창별)
    for w, (a, b) in SPLITS.items():
        m = (tsi >= pd.Timestamp(a)) & (tsi < pd.Timestamp(b))
        if m.sum() < 100:
            rep["windows"][w] = {"n": int(m.sum()), "skip": "n<100"}; continue
        R = {"n": int(m.sum()), "n_days": int(tsi[m].normalize().nunique())}
        rc = pf(cand_of(ts[m], i[m] + 1, i[m] + 1 + cont_ex[m], cont_bp[m]))
        rf = pf(cand_of(ts[m], i[m] + 1, i[m] + 1 + fade_ex[m], fade_bp[m]))
        R["cont"] = rc["stats"] if rc else None
        R["fade"] = rf["stats"] if rf else None
        R["row_cont_bp"] = round(float(cont_bp[m].mean()), 2); R["row_fade_bp"] = round(float(fade_bp[m].mean()), 2)
        R["p_fade_gt_cont"] = round(float((fade_bp[m] > cont_bp[m]).mean()), 4)
        lo, hi = day_boot(cont_bp[m] - fade_bp[m], ts[m], B_BOOT, rng)
        R["cont_minus_fade_day_ci95"] = [round(lo, 2), round(hi, 2)]
        R["by_side"] = {}
        for sv, nm in ((1, "bottom_fire(cont=short)"), (0, "top_fire(cont=long)")):
            mm = m & (sd == sv)
            if mm.sum() < 30:
                continue
            lo2, hi2 = day_boot(cont_bp[mm] - fade_bp[mm], ts[mm], B_BOOT, rng)
            R["by_side"][nm] = {"n": int(mm.sum()), "cont_bp": round(float(cont_bp[mm].mean()), 2),
                                "fade_bp": round(float(fade_bp[mm].mean()), 2), "gap_day_ci95": [round(lo2, 2), round(hi2, 2)]}
        # 측면비율 매칭 무작위 진입 귀무 (같은 창의 임의 봉 풀에서)
        wm = (pd.DatetimeIndex(ts_all) >= pd.Timestamp(a)) & (pd.DatetimeIndex(ts_all) < pd.Timestamp(b))
        pool_i = np.flatnonzero(wm & np.isfinite(atr_all) & (np.arange(n) + 1 + FWD < n))
        if len(pool_i) > NULL_POOL:
            pool_i = rng.choice(pool_i, NULL_POOL, replace=False)
        pool_i = np.sort(pool_i)
        pst = pool_i + 1; pix = pst[:, None] + np.arange(FWD)
        for pool_sign, key in ((1.0, "long"), (-1.0, "short")):
            pr, pe = sim_exit(o[pst], atr_all[pool_i], np.full(len(pool_i), pool_sign), h[pix], l[pix], c[pix], *CELL)
            rep.setdefault("_pool", {}).setdefault(w, {})[key] = (pool_i, pr * 1e4 - COST, pe)
        n_long = int((cont_sign[m] > 0).sum()); n_short = int((cont_sign[m] < 0).sum())
        vals = []
        for _ in range(B_NULL):
            parts = []
            for key, cnt in (("long", n_long), ("short", n_short)):
                pi, pp, pe = rep["_pool"][w][key]
                k = rng.choice(len(pi), size=min(cnt, len(pi)), replace=False)
                parts.append(cand_of(ts_all[pi[k]], pi[k] + 1, pi[k] + 1 + pe[k], pp[k]))
            x = pd.concat(parts, ignore_index=True)
            rr = portfolio(x, CAP); vals.append(rr["exp_bp"] if rr else np.nan)
        v = np.asarray(vals, float); obs = R["cont"]["exp_bp"] if R["cont"] else np.nan
        R["null_random_entry"] = {"mean_bp": round(float(np.nanmean(v)), 2), "p95_bp": round(float(np.nanpercentile(v, 95)), 2),
                                  "percentile_of_cont": round(float((v < obs).mean() * 100), 1)}
        rep["windows"][w] = R
    rep.pop("_pool", None)
    log(f"{sym}: 첫발동 {len(F):,} · " + " · ".join(
        f"{w} cont {rep['windows'][w]['cont']['exp_bp'] if rep['windows'][w].get('cont') else 'NA'}bp "
        f"CI{rep['windows'][w].get('cont',{}).get('day_ci95')} n={rep['windows'][w]['n']}" for w in SPLITS) + f" ({time.time()-t0:.0f}s)")
    return rep


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "cell": CELL, "cost_bp": COST, "gap": GAP,
           "max_concurrent": CAP, "holdout_excluded_from_load": str(HOLDOUT_START), "assets": {}}
    for sym, ref in ASSETS.items():
        try:
            out["assets"][sym] = run_asset(sym, ref)
        except Exception as e:
            log(f"⚠️{sym} 실패: {type(e).__name__}: {e}")
            out["assets"][sym] = {"error": f"{type(e).__name__}: {e}"}
    # 판정
    verd = {}
    for sym, r in out["assets"].items():
        if "windows" not in r:
            continue
        okc = []
        for w in ("VAL", "OOS"):
            W = r["windows"].get(w, {})
            okc.append(bool(W.get("cont") and W["cont"]["exp_bp"] > 0 and W.get("cont_minus_fade_day_ci95", [-9])[0] > 0))
        verd[sym] = {"VAL_pass": okc[0], "OOS_pass": okc[1], "replicated": all(okc)}
    out["verdict"] = verd
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    log(f"완료 {time.time()-t0:.0f}s → {OUT/'report.json'}")
    for sym, v in verd.items():
        log(f"  {sym}: 재현={v['replicated']} (VAL {v['VAL_pass']} / OOS {v['OOS_pass']})")


if __name__ == "__main__":
    main()
