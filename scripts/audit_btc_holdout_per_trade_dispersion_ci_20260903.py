#!/usr/bin/env python3
"""BTC 증거신호 HOLDOUT 성과의 **트레이드별 분산 신뢰구간** -- 헤드라인 bp가 0과 구분되는가.

## 왜 필요한가

2026-09-02 HOLDOUT 단일노출에서 `demarker_extreme` **+3.25bp**(n=428) 등이 나왔지만,
보고된 건 **평균뿐**이다. 평균만 보면 "+3.25bp면 통과"로 읽히는데, 트레일링스톱은 소수의
큰 승리가 평균을 만드는 구조여서 **분산을 보지 않으면 0과 구분되는지 알 수 없다**.

⚠️**이것은 HOLDOUT 재노출이 아니다.** 사전등록된 셀로 이미 수행한 단일 노출의 **트레이드
원장을 다른 통계로 요약**할 뿐이다. 새 셀을 고르거나 새 창을 여는 행위가 아니므로
단일노출 원칙을 위반하지 않는다. 계산은 완전 결정론적이라 같은 원장이 재현된다.

## 산출

  · 트레이드별 수익 분포: 평균 / 표준편차 / 중앙값 / t통계량
  · **부트스트랩 95% 신뢰구간**(B=10,000, 트레이드 재표집) -- 하한이 0 위인가
  · **집중도**: 상위 1%/5% 트레이드를 빼면 평균이 얼마나 남는가
    (소수 승리가 전부 만든 성과인지)
  · 측면별(롱/숏) 동일 통계

⚠️새로운 셀 탐색·새 창 노출 없음. 사전등록 셀 고정.
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
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402

_S = importlib.util.spec_from_file_location(
    "btcgate", ROOT / "scripts/gate_btc_evidence_signals_trailing_economics_20260902.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)

_H = importlib.util.spec_from_file_location(
    "btchold", ROOT / "scripts/holdout_btc_evidence_signals_costgate_single_exposure_20260902.py")
_h = importlib.util.module_from_spec(_H)
_H.loader.exec_module(_h)

PREREG = _h.PREREG
HOLDOUT_START = pd.Timestamp("2026-04-01")
OUT = ROOT / "data/research/btc_evidence_signals_costgate_20260902/holdout_per_trade_dispersion.json"
B_BOOT, SEED = 10_000, 20260903

SIG_BY_NAME = {s[0]: s for s in _g.SIGNALS}


def log(m): print(f"[disp] {m}", flush=True)


def holdout_ledger(kl, fires, H, cell):
    """사전등록 셀로 HOLDOUT 창의 트레이드 원장을 재생성한다(결정론적)."""
    ts = kl["timestamp"]
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    kl_ts = ts.to_numpy()
    f_ts = pd.to_datetime(fires["timestamp"]).to_numpy()
    dec = np.searchsorted(kl_ts, f_ts)
    inb = dec < len(kl_ts)
    if not inb.all():
        fires, dec = fires.loc[inb].reset_index(drop=True), dec[inb]
    bad = int((kl_ts[dec] != pd.to_datetime(fires["timestamp"]).to_numpy()).sum())
    if bad:
        raise ValueError(f"타임스탬프 불일치 {bad}건")
    is_long = (fires["side"].astype(str) == "bottom").to_numpy()
    atr = fires["atr_pct"].to_numpy(dtype=float)
    if not np.all(np.diff(dec) >= 0):
        order = np.argsort(dec, kind="stable")
        dec, is_long, atr = dec[order], is_long[order], atr[order]

    end = ts.iloc[-1] + pd.Timedelta(minutes=5)
    el = set(np.flatnonzero(purged_decision_mask(
        ts, start=HOLDOUT_START, end=end, horizon_bars=H)).tolist())
    m = np.array([d in el for d in dec])

    sl, arm, trail = cell
    r = simulate_single_position(
        timestamps=ts, open_px=o, high=h, low=l, close=c,
        decision_indices=dec[m], scores=np.where(is_long, 1.0, -1.0)[m],
        tp_moves=np.full(int(m.sum()), 999.0), sl_moves=(sl*atr)[m],
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H,
        margin_fraction=_g.MARGIN_FRACTION, leverage=_g.LEVERAGE,
        roundtrip_cost_rate=0.001,
        arm_moves=(arm*atr)[m], trail_moves=(trail*atr)[m])
    led = r.ledger.copy()
    led["is_long"] = is_long[m][:len(led)] if len(led) == int(m.sum()) else np.nan
    return led


def stats(x, rng, label=""):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return {"n": n, "note": "표본 부족"}
    bp = x * 1e4
    mean = float(bp.mean()); sd = float(bp.std(ddof=1))
    se = sd / np.sqrt(n)
    tstat = mean / se if se > 0 else float("nan")
    idx = rng.integers(0, n, size=(B_BOOT, n))
    boot = bp[idx].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    srt = np.sort(bp)[::-1]
    k1, k5 = max(1, n // 100), max(1, n // 20)
    return {"n": n, "mean_bp": mean, "median_bp": float(np.median(bp)), "sd_bp": sd,
            "se_bp": float(se), "t_stat": float(tstat),
            "ci95_lo_bp": lo, "ci95_hi_bp": hi, "ci_excludes_zero": bool(lo > 0),
            "win_rate": float((x > 0).mean()),
            "ex_top1pct_mean_bp": float(srt[k1:].mean()),
            "ex_top5pct_mean_bp": float(srt[k5:].mean()),
            "max_bp": float(srt[0]), "min_bp": float(srt[-1])}


def main() -> int:
    t0 = time.time()
    kl = pd.read_csv(_g.KLINES)
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
    kl = kl.sort_values("timestamp").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    log(f"BTC klines {len(kl):,}행  |  부트스트랩 B={B_BOOT:,}  seed={SEED}")
    log("⚠️HOLDOUT 재노출 아님 -- 사전등록 셀의 기존 원장을 다른 통계로 요약")

    out = {"asset": "BTCUSDT", "B_boot": B_BOOT, "seed": SEED,
           "reexposure": False, "note": "사전등록 셀 고정, 새 탐색 없음", "signals": {}}

    for name, pr in PREREG.items():
        spec = SIG_BY_NAME.get(name)
        if spec is None:
            log(f"{name}: SIGNALS에 없음 -- 건너뜀"); continue
        _, rel, builder, prep, kind = spec
        log("")
        log(f"=== {name}  셀 SL={pr['cell'][0]} ARM={pr['cell'][1]} Trail={pr['cell'][2]}  H={pr['H']} ===")
        fires, _frame = _g.build_fires(name, rel, builder, prep, kind)
        fires["timestamp"] = pd.to_datetime(fires["timestamp"])
        if fires["timestamp"].dt.tz is not None:
            fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
        fires = fires.loc[fires["timestamp"] >= HOLDOUT_START].reset_index(drop=True)

        led = holdout_ledger(kl, fires, pr["H"], pr["cell"])
        if not len(led):
            log("  트레이드 0건"); continue
        s = stats(led["trade_return"].to_numpy(), rng)
        star = "✅" if s["ci_excludes_zero"] else "❌"
        log(f"  n={s['n']}  평균 {s['mean_bp']:+.2f}bp  중앙값 {s['median_bp']:+.2f}  "
            f"표준편차 {s['sd_bp']:.1f}bp")
        log(f"  t={s['t_stat']:+.2f}   **95%CI [{s['ci95_lo_bp']:+.2f}, {s['ci95_hi_bp']:+.2f}]bp** {star}")
        log(f"  집중도: 상위1% 제외 {s['ex_top1pct_mean_bp']:+.2f}bp / "
            f"상위5% 제외 {s['ex_top5pct_mean_bp']:+.2f}bp  (최대 {s['max_bp']:+.0f} / 최소 {s['min_bp']:+.0f})")

        sides = {}
        if "is_long" in led.columns and led["is_long"].notna().all():
            for lab, m in (("롱", led["is_long"] == True), ("숏", led["is_long"] == False)):  # noqa: E712
                sub = led.loc[m, "trade_return"].to_numpy()
                if len(sub) < 30:
                    continue
                ss = stats(sub, rng)
                sides[lab] = ss
                log(f"  {lab} n={ss['n']:<4} 평균 {ss['mean_bp']:+.2f}bp  "
                    f"CI [{ss['ci95_lo_bp']:+.2f}, {ss['ci95_hi_bp']:+.2f}] "
                    f"{'✅' if ss['ci_excludes_zero'] else '❌'}")
        out["signals"][name] = {"tier": pr["tier"], "cell": list(pr["cell"]), "H": pr["H"],
                                "reported_fwd_bp": None, "all": s, "sides": sides}

    log("")
    log("=== 종합 (HOLDOUT 평균이 0과 구분되는가) ===")
    for k, v in out["signals"].items():
        s = v["all"]
        log(f"  {k:<26} {s['mean_bp']:+6.2f}bp  n={s['n']:<5} t={s['t_stat']:+5.2f}  "
            f"CI [{s['ci95_lo_bp']:+6.2f},{s['ci95_hi_bp']:+6.2f}]  "
            f"{'✅0 제외' if s['ci_excludes_zero'] else '❌0 포함'}")
    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({out['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
