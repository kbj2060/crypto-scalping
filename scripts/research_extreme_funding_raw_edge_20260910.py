#!/usr/bin/env python3
"""극단 펀딩 → 후속 수익 **원시 엣지 사전점검** (2026-09-10). 모델 없음.

사용자: *"어차피 딥러닝으로 다시 학습할 텐데 우선 엣지가 있는지 없는지만 점검해줘."*
→ 저장소 표준인 **raw lift 사전점검**(2026-09-02 문헌 라운드가 쓴 것과 같은 관문). 통과해야 파이프라인으로 보낸다.

근거 문헌: AIJMR 2026 `10.62127/aijmr.2026.v04i04.1493` — 바이낸스 BTCUSDT 무기한 펀딩을
**직전 180일 분포 내 분위**로 극단 판정(하위/상위 십분위), 정산 후 15분·1h·4h·8h·24h 수익 측정.
후보 등급표: `docs/eth_midterm_signal_literature_survey_20260910.md` (B등급).

## 가설(방향 고정, 사후 반전 금지)
극단 **양(+)** 펀딩 = 롱 쏠림(롱이 숏에 지불) → 되돌림 기대 → **숏**
극단 **음(−)** 펀딩 = 숏 쏠림 → **롱**
09-08 쏠림 페이드([[xsec_crowding_fade...]])와 같은 논리지만 그건 횡단면·롱숏비, 이건 시계열·펀딩이다.
🔴저장소 **펀딩 캐리 0/45 실패**와 구분: 그건 캐리 *수취*, 이건 극단 *상태 후 가격 되돌림*.

## 판정 기준 (결과 보기 전 고정)
1차  **양측 평균 초과분**이 순환이동 귀무 95% CI 를 넘고, **비용 10bp(테이커 왕복)를 넘는가**
     — 한쪽만 보면 잔존 베타를 신호로 오독한다(저장소 반복 함정).
보조  측면별 부호 · 분위 단조성(극단에서만인가 전 구간인가) · IS/OOS 안정성 · 독립 관측 수
귀무  순환이동(발동 간격·군집·측면 보존, 가격 정렬만 파괴) B=600
출력 tmp/extreme_funding_edge_20260910/report.json
"""
from __future__ import annotations

import glob
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402

OUT = ROOT / "tmp/extreme_funding_edge_20260910"
FR_DIR = "/home/kbj20/crypto-scalping/binance_data/funding_rate"
SYMS = {"ETHUSDT": "eth", "BTCUSDT": "btc"}
LOOKBACK_D = 180                      # 문헌 규격: 직전 180일 분포
DECILES = [0.10, 0.20]                # 극단 문턱(십분위 · 이십분위)
HORIZ_MIN = {"15m": 15, "1h": 60, "4h": 240, "8h": 480, "24h": 1440}
COST_BP = 10.0
B_NULL = 600
SPLIT = pd.Timestamp("2025-09-01")


def log(m):
    print(f"[fund {time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_funding(sym: str) -> pd.DataFrame:
    rows = []
    for z in sorted(glob.glob(f"{FR_DIR}/{sym}-fundingRate-*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                rows.append(pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0]))))
        except Exception:                                  # 손상/부분 아카이브는 건너뛴다
            continue
    d = pd.concat(rows, ignore_index=True)
    d["ts"] = pd.to_datetime(d["calc_time"], unit="ms")
    d = d.dropna(subset=["ts", "last_funding_rate"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return d[["ts", "last_funding_rate"]].rename(columns={"last_funding_rate": "fr"})


def pct_rank_causal(fr: np.ndarray, ts: pd.Series, days: int) -> np.ndarray:
    """직전 `days` 일 분포 내 자기 분위. **자기 자신 포함, 미래 미포함**(ex ante)."""
    out = np.full(len(fr), np.nan)
    t = ts.to_numpy()
    for i in range(len(fr)):
        lo = t[i] - np.timedelta64(days, "D")
        w = fr[(t >= lo) & (t <= t[i])]
        if len(w) >= 60:                                   # 최소 20일치(3/일)
            out[i] = (w <= fr[i]).mean()
    return out


def fwd_returns(kl: pd.DataFrame, ev_ts: pd.Series) -> dict:
    """정산 시각 이후 각 지평의 종가 수익. 정산 시각 **직후 봉 종가**를 기준가로(체결 가능)."""
    t = pd.DatetimeIndex(kl["timestamp"]); c = kl["close"].to_numpy(float)
    base = t.searchsorted(pd.DatetimeIndex(ev_ts), side="right")   # 정산 후 첫 봉
    out = {}
    for name, mins in HORIZ_MIN.items():
        step = mins // 5
        b = np.clip(base, 0, len(c) - 1); f = np.clip(base + step, 0, len(c) - 1)
        r = np.where((base < len(c) - step) & (base > 0), c[f] / c[b] - 1.0, np.nan)
        out[name] = r
    out["_base_ok"] = (base > 0) & (base < len(c) - max(HORIZ_MIN.values()) // 5)
    return out


def cyc_null(sig_ret: np.ndarray, idx: np.ndarray, b=B_NULL, seed=0):
    """순환이동 귀무: 발동 인덱스를 통째로 밀어 군집·측면·드리프트를 보존."""
    n = len(sig_ret); rng = np.random.default_rng(seed)
    out = []
    for s in rng.integers(1, n, b):
        v = sig_ret[(idx + s) % n]
        v = v[np.isfinite(v)]
        if len(v):
            out.append(v.mean() * 1e4)
    return np.array(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"lookback_days": LOOKBACK_D, "deciles": DECILES, "cost_bp": COST_BP, "b_null": B_NULL,
           "hypothesis": "극단 + 펀딩 → 숏 / 극단 − 펀딩 → 롱 (사후 반전 금지)",
           "criterion": "양측 평균 초과분이 순환이동 CI 초과 AND 비용 10bp 초과", "syms": {}}
    for sym, short in SYMS.items():
        fr = load_funding(sym)
        kl = BD.load_klines(short, sym)
        pr = pct_rank_causal(fr["fr"].to_numpy(), fr["ts"], LOOKBACK_D)
        fw = fwd_returns(kl, fr["ts"])
        ok0 = np.isfinite(pr) & fw["_base_ok"]
        log(f"[{sym}] 펀딩 {len(fr):,}건 {fr.ts.min().date()}~{fr.ts.max().date()} · 분위 유효 {int(ok0.sum()):,}")
        d = {"n_events": int(len(fr)), "n_usable": int(ok0.sum()),
             "span": [str(fr.ts.min()), str(fr.ts.max())], "cells": {}, "decile_profile": {}}
        is_m = (fr["ts"] < SPLIT).to_numpy(); oos_m = ~is_m
        for q in DECILES:
            for H in HORIZ_MIN:
                r = fw[H]
                hi = ok0 & (pr >= 1 - q)          # 극단 양(+) → 숏
                lo = ok0 & (pr <= q)              # 극단 음(−) → 롱
                cells = {}
                for side, m, sgn in (("pos_short", hi, -1.0), ("neg_long", lo, +1.0)):
                    v = sgn * r
                    idx = np.flatnonzero(m & np.isfinite(v))
                    if len(idx) < 40:
                        continue
                    obs = float(np.mean(v[idx]) * 1e4)
                    nulls = cyc_null(v, idx, seed=abs(hash((sym, q, H, side))) % 9999)
                    cells[side] = {
                        "n": int(len(idx)), "obs_bp": obs, "acc": float(np.mean(v[idx] > 0)),
                        "null_mean_bp": float(nulls.mean()),
                        "null_ci95": [float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))],
                        "excess_bp": obs - float(nulls.mean()),
                        "beats_null": bool(obs > np.percentile(nulls, 97.5)),
                        "is_bp": float(np.mean(v[np.flatnonzero(m & is_m & np.isfinite(v))]) * 1e4)
                                 if (m & is_m & np.isfinite(v)).sum() >= 20 else None,
                        "oos_bp": float(np.mean(v[np.flatnonzero(m & oos_m & np.isfinite(v))]) * 1e4)
                                  if (m & oos_m & np.isfinite(v)).sum() >= 20 else None,
                    }
                if len(cells) == 2:
                    both = (cells["pos_short"]["excess_bp"] + cells["neg_long"]["excess_bp"]) / 2
                    cells["_two_sided_mean_excess_bp"] = both
                    cells["_net_after_cost_bp"] = both - COST_BP
                    cells["_both_beat_null"] = cells["pos_short"]["beats_null"] and cells["neg_long"]["beats_null"]
                d["cells"][f"q{int(q*100)}|{H}"] = cells
        # 분위 단조성(H=8h, 십분위 버킷)
        r8 = fw["8h"]
        prof = {}
        for k in range(10):
            m = ok0 & (pr >= k / 10) & (pr < (k + 1) / 10) & np.isfinite(r8)
            if m.sum() >= 30:
                prof[f"d{k+1}"] = {"n": int(m.sum()), "mean_bp": float(np.mean(r8[m]) * 1e4)}
        d["decile_profile"] = prof
        rep["syms"][sym] = d
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
        for q in DECILES:
            for H in HORIZ_MIN:
                c = d["cells"].get(f"q{int(q*100)}|{H}")
                if c and "_two_sided_mean_excess_bp" in c:
                    log(f"  {sym} q{int(q*100)} {H:>3}: 양측평균 초과 {c['_two_sided_mean_excess_bp']:+7.1f}bp "
                        f"순 {c['_net_after_cost_bp']:+7.1f} | 숏측 {c['pos_short']['excess_bp']:+7.1f}"
                        f"(n={c['pos_short']['n']},{'통과' if c['pos_short']['beats_null'] else '—'}) "
                        f"롱측 {c['neg_long']['excess_bp']:+7.1f}(n={c['neg_long']['n']},"
                        f"{'통과' if c['neg_long']['beats_null'] else '—'})")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
