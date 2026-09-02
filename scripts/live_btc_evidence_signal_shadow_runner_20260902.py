#!/usr/bin/env python3
"""BTC 증거신호 **섀도우 러너** -- 주문 없이 발동·확률·사후결과만 가상 원장에 기록.

## 이게 무엇이고, 무엇이 아닌가

**관측 도구다.** ETH의 경제라벨 섀도우(`live_eth_v_rebound_econ_shadow_runner_20260902.py`)는
VAL/OOS/HOLDOUT을 통과한 **검증된 후보**를 돌리지만, BTC는 아직 경제성을 통과한 모델이 없다:

  · 2026-09-02 BTC 경제라벨 시도 -- 손익비 하한 없이는 승률 84~95%/손익비 0.089라는
    변동성매도 프로파일이 만들어져 OOS에서 -22.18bp(뒤집기 +19.65bp)로 반전.
    손익비 하한(0.25)을 걸면 **VAL에서조차 수익 조합이 없다**(최선 -13.72bp).
  · 원인: BTC ATR 중앙 16.0bp(ETH ~23bp)로 30% 작은데 비용은 10bp로 동일 --
    비용/ATR이 62%(ETH 43%)라 5분봉 BTC 스캘핑이 수수료 바닥에 훨씬 가깝다.
    무작위 진입 기준선도 -1.04bp(ETH 약 +2.6bp)로 음수.

그래서 이 러너는 **가상 매매 성과를 주장하지 않는다.** 신호가 발동한 시점의 메타라벨 확률과
그 이후 실제 가격 전개(각 신호 자신의 HIT 정의 기준)를 기록해, **라이브에서 학습시점 검증치가
재현되는지**를 관측한다. 재현되면 그때 경제성을 다시 물을 근거가 생긴다.

## 기록 내용

봉마다: 발동 신호, 측면, 메타라벨 확률, 그 시점 ATR/가격.
호라이즌 경과 후: 해당 신호의 HIT 정의로 실제 hit 여부 판정 -> 원장에 확정.
집계: 신호별 **라이브 hit률 vs 학습 hit률 vs HOLDOUT AUC** 대조.

⚠️주문 없음. `trading_bot.py` 미배선. 실행 인자로도 주문을 켤 수 없다.

Usage:
    python scripts/live_btc_evidence_signal_shadow_runner_20260902.py --loop
    python scripts/live_btc_evidence_signal_shadow_runner_20260902.py --report
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_SIG = importlib.import_module("live_btc_evidence_signal_metalabel_20260902")
compute_btc_evidence_signals = _SIG.compute_btc_evidence_signals
SYMBOL = _SIG.SYMBOL

STATE = ROOT / "data/live/btc_evidence_signal_shadow_state.json"
MARK_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
LOOP_SECONDS = 60

# 신호별 HIT 판정 규격 (동결 컨텍스트 리포트의 btc_params와 동일 출처)
HIT_SPEC = {
    "taker_delta_climax":       {"horizon": 6,  "k": 2.0,  "mode": "close_at_h"},
    "liquidity_sweep":          {"horizon": 20, "k": 2.0,  "mode": "touch"},
    "kalman_deviation_meanrev": {"horizon": 10, "k": 3.5,  "mode": "touch"},
    "short_term_return_z":      {"horizon": 6,  "k": 2.0,  "mode": "touch"},
    "orthogonal_combo":         {"horizon": 8,  "k": 2.0,  "mode": "touch"},
    "demarker_extreme":         {"horizon": 8,  "k": 0.70, "mode": "touch"},
    "fib_extension_exhaustion": {"horizon": 10, "k": 2.75, "mode": "close_at_h"},
}
HOLDOUT_AUC = {"demarker_extreme": 0.7286, "kalman_deviation_meanrev": 0.6709,
               "short_term_return_z": 0.6443, "taker_delta_climax": 0.6276,
               "orthogonal_combo": 0.5933, "fib_extension_exhaustion": 0.5657,
               "liquidity_sweep": 0.5214}


def log(m: str) -> None:
    print(f"[btc-evshadow {datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:                                      # noqa: BLE001
            log("⚠️상태 파싱 실패 -- 새로 시작")
    return {"pending": [], "ledger": [], "cycles": 0,
            "started_utc": datetime.now(timezone.utc).isoformat(), "version": 1}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str))
    tmp.replace(STATE)


def fetch_bars(limit: int = 60) -> pd.DataFrame | None:
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": SYMBOL, "interval": "5m", "limit": limit}, timeout=15)
        r.raise_for_status()
        raw = r.json()
    except Exception:                                          # noqa: BLE001
        return None
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv",
            "trades", "tbb", "tq", "ignore"]
    d = pd.DataFrame(raw, columns=cols)
    for c in ("open", "high", "low", "close"):
        d[c] = d[c].astype(float)
    d["timestamp"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    now_ms = int(time.time() * 1000)
    if len(d) and int(d.iloc[-1]["close_time"]) >= now_ms:
        d = d.iloc[:-1].reset_index(drop=True)
    return d


def resolve(s: dict[str, Any], bars: pd.DataFrame) -> None:
    """호라이즌이 지난 pending을 각 신호 자신의 HIT 정의로 판정해 원장에 확정한다."""
    if bars is None or bars.empty:
        return
    ts = bars["timestamp"].to_numpy()
    keep = []
    for p in s["pending"]:
        spec = HIT_SPEC.get(p["signal"])
        if spec is None:
            continue
        t0 = np.datetime64(pd.Timestamp(p["bar_utc"]).tz_convert("UTC").tz_localize(None))
        pos = int(np.searchsorted(ts.astype("datetime64[ns]"), t0))
        if pos >= len(bars) or str(bars["timestamp"].iloc[pos])[:16] != str(p["bar_utc"])[:16]:
            keep.append(p); continue                            # 해당 봉이 아직 창 안에 없음
        end = pos + spec["horizon"]
        if end >= len(bars):
            keep.append(p); continue                            # 아직 미완
        seg = bars.iloc[pos + 1:end + 1]
        entry, atr, k = p["entry"], p["atr"], spec["k"]
        if spec["mode"] == "close_at_h":
            px = float(bars["close"].iloc[end])
            hit = (px >= entry + k * atr) if p["side"] == "bottom" else (px <= entry - k * atr)
        else:                                                   # touch (intrabar MFE)
            if p["side"] == "bottom":
                hit = bool((seg["high"].to_numpy() >= entry + k * atr).any())
            else:
                hit = bool((seg["low"].to_numpy() <= entry - k * atr).any())
        s["ledger"].append({**p, "hit": int(hit),
                            "resolved_utc": datetime.now(timezone.utc).isoformat()})
        log(f"  확정 {p['signal']} {p['side']} p={p['proba']:.4f} -> hit={int(hit)}")
    s["pending"] = keep


def record(s: dict[str, Any], out: dict[str, Any], bars: pd.DataFrame) -> None:
    if not out.get("warmed_up") or bars is None or bars.empty:
        return
    latest = str(bars["timestamp"].iloc[-1])
    atr_map = {}
    for name, v in out["signals"].items():
        for side, d in (v.get("fired") or {}).items():
            if "error" in d or d.get("latest_proba") is None:
                continue
            bar = d["latest_bar_utc"]
            if any(q["signal"] == name and q["side"] == side and q["bar_utc"] == bar
                   for q in s["pending"]) or \
               any(q["signal"] == name and q["side"] == side and q["bar_utc"] == bar
                   for q in s["ledger"][-200:]):
                continue                                        # 같은 봉 중복 방지
            m = bars.loc[bars["timestamp"].astype(str) == bar]
            if m.empty:
                continue
            entry = float(m["close"].iloc[0])
            atr = atr_map.get(bar)
            if atr is None:
                hi, lo, cl = (bars[c].to_numpy() for c in ("high", "low", "close"))
                tr = np.maximum(hi[1:] - lo[1:],
                                np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
                atr = float(pd.Series(tr).rolling(14, min_periods=14).mean().iloc[-1])
                atr_map[bar] = atr
            if not np.isfinite(atr) or atr <= 0:
                continue
            s["pending"].append({"signal": name, "side": side, "bar_utc": bar,
                                 "proba": float(d["latest_proba"]), "entry": entry, "atr": atr,
                                 "recorded_utc": datetime.now(timezone.utc).isoformat()})
            log(f"  [가상기록] {name} {side} p={d['latest_proba']:.4f} @{entry:.1f} bar={bar}")


def report(s: dict[str, Any]) -> None:
    led = s["ledger"]
    log(f"원장 {len(led)}건 / 대기 {len(s['pending'])}건 / 사이클 {s.get('cycles', 0)}")
    if not led:
        log("아직 확정된 기록 없음"); return
    df = pd.DataFrame(led)
    ctx = json.loads((ROOT / "data/labels/btc_5m_evidence_signal_live_contexts_20260902"
                      / "contexts_report.json").read_text())["signals"]
    log(f"  {'신호':26s} {'n':>4s} {'라이브hit':>9s} {'학습hit':>8s} {'차이':>7s} {'HOLDOUT AUC':>11s}")
    for name, g in df.groupby("signal"):
        tr = ctx.get(name, {}).get("hit_rate")
        live = float(g["hit"].mean())
        d = (live - tr) if tr is not None else float("nan")
        log(f"  {name:26s} {len(g):>4d} {live:>9.4f} {tr if tr else float('nan'):>8.4f} "
            f"{d:>+7.4f} {HOLDOUT_AUC.get(name, float('nan')):>11.4f}")
    log("  ⚠️표본이 수십 건 미만이면 해석하지 말 것 -- 이 러너는 관측 도구이지 성과 주장이 아니다.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    s = load_state()
    if a.report:
        report(s); return 0
    log(f"⚠️섀도우 -- 주문을 내지 않습니다. 신호 {len(HIT_SPEC)}종 관측")
    while True:
        try:
            bars = fetch_bars()
            out = compute_btc_evidence_signals()
            if out.get("error"):
                log(f"  스코어러 오류: {out['error']}")
            else:
                record(s, out, bars)
            resolve(s, bars)
            s["cycles"] = int(s.get("cycles", 0)) + 1
            save_state(s)
            if s["cycles"] % 30 == 0:
                report(s)
        except Exception as e:                                  # noqa: BLE001
            log(f"  사이클 오류: {type(e).__name__}: {e}")
        if not a.loop:
            break
        time.sleep(LOOP_SECONDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
