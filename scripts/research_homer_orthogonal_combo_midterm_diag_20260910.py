#!/usr/bin/env python3
"""호메로스 프로토콜 §2 — `orthogonal_combo` 중기 이식 **라벨 설계 전 진단** (2026-09-10).

사용자: *"기존 규칙으로 라벨부터 만들고 딥러닝하는 로직이 호메로스 메모리에 있을 텐데 참고했나?"*
→ 참고하지 않았음을 확인하고, 이제 `docs/homer/README.md` §재사용 방법론 템플릿 절차를 그대로 따른다.
플래그십 `orthogonal_combo` 하나로 절차를 먼저 검증한다(사용자 승인).

## 프로토콜 §2 체크리스트 (신호마다 재측정 · 복붙 금지)
 1. 원시 hit rate 의 **호라이즌 민감도** 실측
 2. **크기 분포** 확인(hit 상당수가 노이즈 크기인가)
 3. **발동봉 ↔ 실제 극값 어긋남** 직접 측정(넓은 창 argmax/argmin) — 신호마다 부호가 반대일 수 있음
 4. **연속발동 클러스터링** 확인 → 필요시 클러스터 앵커링
 5. (지속성은 단일 스냅샷 아니라 관찰창 전체 유지로 — 라벨 정의 단계에서 반영)
 6. 시각 검증 10건 — 이 스크립트는 수치만, 차트는 별도

라벨 정의(§3, 3개 신호 공통 수렴형): 발동봉 종가=entry → 고정 창 intrabar 고/저 MFE ≥ K×atr_pct → hit=1.
K 는 스윕 후 **균형분포(50/50 근접)** 로 고른다. 이 스크립트는 그 K 를 창별로 찾아 표로 낸다.

타임프레임 5m/15m/1h. 창 후보는 §5.5 대로 **8점 이상 촘촘히**.
출력 tmp/homer_orth_midterm_20260910/diag.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
import research_eth_signals_midterm_timeframe_20260910 as M  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

OUT = ROOT / "tmp/homer_orth_midterm_20260910"
SIGNAL = "orthogonal_combo"
TFS = {"5m": 1, "15m": 3, "1h": 12}
HORIZONS = [6, 9, 12, 16, 20, 24, 30, 36, 48]      # §5.5 촘촘히
K_GRID = np.round(np.arange(0.5, 6.01, 0.25), 2)
MIS_W = 24                                          # 어긋남 측정 창(양방향 봉)


def log(m):
    print(f"[diag {time.strftime('%H:%M:%S')}] {m}", flush=True)


def fires_of(sig: pd.DataFrame, side: str) -> np.ndarray:
    return np.flatnonzero(sig[f"{side}_{SIGNAL}"].fillna(False).to_numpy(bool))


def mfe_atr(sig: pd.DataFrame, idx: np.ndarray, side: str, H: int) -> np.ndarray:
    """발동봉 종가 entry, 이후 H봉 intrabar 유리방향 최대이동 / atr_pct (§3 라벨 정의)."""
    c = sig.close.to_numpy(float)
    fav = (sig.high if side == "bottom" else sig.low).to_numpy(float)
    a = sig.atr_pct.to_numpy(float)
    sgn = 1.0 if side == "bottom" else -1.0
    out = np.full(len(idx), np.nan)
    n = len(c)
    for j, i in enumerate(idx):
        if i + H >= n:
            continue
        seg = sgn * (fav[i + 1:i + 1 + H] - c[i]) / c[i]
        out[j] = seg.max() / max(a[i], 1e-9)
    return out


def misalignment(sig: pd.DataFrame, idx: np.ndarray, side: str) -> dict:
    """§2-3 발동봉↔실제 극값 어긋남. 양수=극값이 발동 **이후**(지연), 음수=이전(선행)."""
    ext = (sig.low if side == "bottom" else sig.high).to_numpy(float)
    n = len(ext); offs = []
    for i in idx:
        lo, hi = max(0, i - MIS_W), min(n, i + MIS_W + 1)
        seg = ext[lo:hi]
        k = int(np.argmin(seg) if side == "bottom" else np.argmax(seg))
        offs.append((lo + k) - i)
    o = np.array(offs, float)
    return {"n": int(len(o)), "median_bars": float(np.median(o)), "mean_bars": float(o.mean()),
            "frac_after": float((o > 0).mean()), "frac_at": float((o == 0).mean()),
            "q25": float(np.percentile(o, 25)), "q75": float(np.percentile(o, 75))}


def clustering(idx: np.ndarray) -> dict:
    """§2-4 연속발동 클러스터링. gap<=1 봉이면 같은 클러스터."""
    if len(idx) < 2:
        return {"n_fires": int(len(idx)), "n_clusters": int(len(idx)), "max_run": int(len(idx))}
    gaps = np.diff(idx)
    breaks = np.flatnonzero(gaps > 1)
    n_cl = len(breaks) + 1
    runs = np.diff(np.concatenate([[-1], breaks, [len(idx) - 1]]))
    return {"n_fires": int(len(idx)), "n_clusters": int(n_cl), "max_run": int(runs.max()),
            "mean_run": float(runs.mean()), "median_gap_bars": float(np.median(gaps))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    eth, btc = BD.load_klines("eth", "ETHUSDT"), BD.load_klines("btc", "BTCUSDT")
    rep = {"signal": SIGNAL, "horizons": HORIZONS, "k_grid": K_GRID.tolist(), "tfs": {}}
    for tf, mult in TFS.items():
        e, b = M.resample(eth, mult), M.resample(btc, mult)
        sig = compute_signals(e, btc_df=b, funding_df=None)
        d = {"bars": int(len(sig)), "span": [str(sig.timestamp.iloc[0]), str(sig.timestamp.iloc[-1])],
             "atr_pct_median_bp": float(np.median(sig.atr_pct.dropna()) * 1e4), "sides": {}}
        for side in ("bottom", "top"):
            idx = fires_of(sig, side)
            idx = idx[idx >= 900]
            s = {"n_fires": int(len(idx)),
                 "per_day": float(len(idx) / max(1, (sig.timestamp.iloc[-1] - sig.timestamp.iloc[0]).days)),
                 "misalignment": misalignment(sig, idx, side), "clustering": clustering(idx),
                 "horizon": {}}
            for H in HORIZONS:
                m = mfe_atr(sig, idx, side, H)
                ok = np.isfinite(m)
                if ok.sum() < 50:
                    continue
                mv = m[ok]
                # §3: 균형분포(50/50) 에 가장 가까운 K
                rates = {float(k): float((mv >= k).mean()) for k in K_GRID}
                kbal = min(rates, key=lambda k: abs(rates[k] - 0.5))
                s["horizon"][H] = {
                    "n": int(ok.sum()), "hit_rate_at_K1.0": rates.get(1.0),
                    "hit_rate_at_K2.0": rates.get(2.0),
                    "K_balanced": kbal, "hit_rate_at_K_balanced": rates[kbal],
                    "K_balanced_bp": float(kbal * d["atr_pct_median_bp"]),
                    "mfe_median": float(np.median(mv)), "mfe_q75": float(np.percentile(mv, 75)),
                    "frac_mfe_below_1atr": float((mv < 1.0).mean()),
                }
            d["sides"][side] = s
            ms = s["misalignment"]
            log(f"{tf} {side}: 발동 {s['n_fires']:,}({s['per_day']:.2f}/일) · 클러스터 {s['clustering']['n_clusters']:,} "
                f"(최대연속 {s['clustering']['max_run']}) · 어긋남 중앙 {ms['median_bars']:+.0f}봉 "
                f"(이후 비중 {ms['frac_after']:.2f}) · ATR중앙 {d['atr_pct_median_bp']:.0f}bp")
        rep["tfs"][tf] = d
        (OUT / "diag.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log("=" * 100)
    for tf in TFS:
        for side in ("bottom", "top"):
            h = rep["tfs"][tf]["sides"][side]["horizon"]
            row = " · ".join(f"H{H} K={h[H]['K_balanced']:.2f}({h[H]['hit_rate_at_K_balanced']:.2f})" for H in HORIZONS if H in h)
            log(f"{tf} {side} 균형K: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
