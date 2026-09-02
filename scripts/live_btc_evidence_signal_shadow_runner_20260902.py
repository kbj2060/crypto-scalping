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
# 2026-09-02: 60 -> 300. 5분봉 신호라 봉당 1회면 충분하다(GPU 절감).
# 이 러너의 해상 판정은 봉 인덱스(pos + horizon) 기준이라 주기와 무관하다.
LOOP_SECONDS = 300

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


# 조회 창. 최장 호라이즌은 liquidity_sweep의 20봉이라 평시엔 60봉으로도 충분했지만,
# 러너가 창보다 오래 멈추면 신호 봉이 밀려나 pending이 **영구 미해소**가 된다.
# 500봉 = 약 41시간까지의 다운타임을 견딘다(Binance 상한 1500).
FETCH_BARS = 500


def fetch_bars(limit: int = FETCH_BARS) -> pd.DataFrame | None:
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


def _finalize(s: dict[str, Any], p: dict[str, Any], hit: int, bars_to_result: int,
              early: bool) -> None:
    s["ledger"].append({**p, "hit": int(hit),
                        "bars_to_result": int(bars_to_result),
                        "resolved_early": bool(early),
                        "resolved_utc": datetime.now(timezone.utc).isoformat()})
    log(f"  확정 {p['signal']} {p['side']} p={p['proba']:.4f} -> hit={int(hit)} "
        f"({bars_to_result}봉{', 조기' if early else ''})")


def resolve(s: dict[str, Any], bars: pd.DataFrame) -> None:
    """pending을 각 신호 자신의 HIT 정의로 판정해 원장에 확정한다.

    ⭐**2026-09-03 (1) touch 모드 조기 확정.** 예전엔 호라이즌이 다 지나야만 확정해서,
    3번째 봉에서 이미 목표에 닿았어도 20봉이 끝날 때까지(최대 100분) "대기"로 남았다.
    기록되는 결과는 어차피 같지만(구간 어디서든 닿으면 hit=1) **표시 상태가 사실과 달랐다**.
    이제 닿는 즉시 hit=1로 확정한다 -- 결과는 동일하고 대기 목록만 정확해진다.
    ⚠️`close_at_h`(taker/fib)는 **H봉 종가**로 판정하므로 정의상 조기 확정이 불가능하다.
      중간에 목표를 스쳐도 종가가 미달이면 hit=0이다. 이건 결함이 아니라 그 신호의 정의다.

    ⭐**(2) 창 밖 pending 만료.** 러너가 조회 창보다 오래 멈추면 신호 봉이 밀려나
    `pos` 판정에 실패해 pending에 **영원히** 남았다. 이제 `expired`로 분리 기록한다.
    ⚠️원장(`ledger`)이 아니라 별도 리스트에 넣는다 -- hit률 집계를 오염시키면 안 된다.
    """
    if bars is None or bars.empty:
        return
    # tz-aware Series에 .to_numpy()를 쓰면 object 배열(Timestamp)이 나오고, astype이
    # "no explicit representation of timezones" 경고를 낸다. UTC이므로 tz를 명시적으로
    # 떼서 같은 값을 얻는다 -- 의도가 드러나고 경고도 사라진다.
    ts = bars["timestamp"].dt.tz_localize(None).to_numpy()
    first = pd.Timestamp(bars["timestamp"].iloc[0])
    keep = []
    for p in s["pending"]:
        spec = HIT_SPEC.get(p["signal"])
        if spec is None:
            continue
        bar_ts = pd.Timestamp(p["bar_utc"])
        t0 = np.datetime64(bar_ts.tz_convert("UTC").tz_localize(None))
        pos = int(np.searchsorted(ts, t0))
        in_win = (pos < len(bars)
                  and str(bars["timestamp"].iloc[pos])[:16] == str(p["bar_utc"])[:16])
        if not in_win:
            if bar_ts < first:                                  # 창 밖으로 밀려남
                s.setdefault("expired", []).append(
                    {**p, "hit": None, "reason": "out_of_window",
                     "expired_utc": datetime.now(timezone.utc).isoformat(),
                     "note": f"조회 창({len(bars)}봉) 밖 -- 러너 다운타임 추정"})
                s["expired"] = s["expired"][-200:]
                log(f"  ⚠️만료 {p['signal']} {p['side']} bar={str(p['bar_utc'])[:16]} "
                    f"-- 조회 창 밖(다운타임 추정), hit 집계에서 제외")
            else:
                keep.append(p)                                  # 아직 창에 안 들어옴
            continue

        end = pos + spec["horizon"]
        avail = min(end, len(bars) - 1)                         # 지금까지 확보된 봉까지만
        seg = bars.iloc[pos + 1:avail + 1]
        k, entry, atr = spec["k"], p["entry"], p["atr"]

        if spec["mode"] == "touch" and len(seg):
            if p["side"] == "bottom":
                mask = seg["high"].to_numpy() >= entry + k * atr
            else:
                mask = seg["low"].to_numpy() <= entry - k * atr
            if mask.any():                                      # ⭐닿는 즉시 확정
                _finalize(s, p, 1, int(np.argmax(mask)) + 1, early=(end >= len(bars)))
                continue

        if end >= len(bars):
            keep.append(p)                                      # 아직 미완
            continue

        if spec["mode"] == "close_at_h":
            px = float(bars["close"].iloc[end])
            hit = (px >= entry + k * atr) if p["side"] == "bottom" else (px <= entry - k * atr)
            _finalize(s, p, int(hit), spec["horizon"], early=False)
        else:
            _finalize(s, p, 0, spec["horizon"], early=False)    # 창 전체에서 미도달
    s["pending"] = keep


def _atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """봉별 ATR. 인덱스가 `bars`의 행 인덱스와 정렬된다(TR이 1봉 밀리는 걸 보정)."""
    hi, lo, cl = (bars[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    tr = np.maximum(hi[1:] - lo[1:],
                    np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])))
    a = pd.Series(tr).rolling(period, min_periods=period).mean()
    # tr[i-1]이 bar i에 대응하므로 앞에 NaN 하나를 붙여 bars 인덱스와 맞춘다
    return pd.concat([pd.Series([np.nan]), a], ignore_index=True)


def record(s: dict[str, Any], out: dict[str, Any], bars: pd.DataFrame) -> None:
    if not out.get("warmed_up") or bars is None or bars.empty:
        return
    # ⚠️2026-09-03 수정: 예전엔 `rolling(14).mean().iloc[-1]`로 **항상 최신 봉의 ATR**을 썼다.
    # `atr_map`이 봉별로 캐시하는 모양이었지만 계산값은 봉과 무관하게 동일했다.
    # 이 ATR은 HIT 판정 문턱(`k x atr`)에 그대로 들어가므로 **신호 봉의 ATR**이어야 한다.
    # 신호는 보통 발동 후 1~2사이클 안에 기록돼 오차가 작았지만, 러너가 지연되거나
    # 변동성이 급변한 구간에서는 문턱이 눈에 띄게 어긋난다.
    atr_all = _atr_series(bars)
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
            bpos = int(m.index[0])
            entry = float(bars["close"].iloc[bpos])
            atr = float(atr_all.iloc[bpos])          # ⭐그 신호 봉의 ATR
            if not np.isfinite(atr) or atr <= 0:
                continue
            s["pending"].append({"signal": name, "side": side, "bar_utc": bar,
                                 "proba": float(d["latest_proba"]), "entry": entry, "atr": atr,
                                 "recorded_utc": datetime.now(timezone.utc).isoformat()})
            log(f"  [가상기록] {name} {side} p={d['latest_proba']:.4f} @{entry:.1f} bar={bar}")


def report(s: dict[str, Any]) -> None:
    led = s["ledger"]
    exp = s.get("expired") or []
    log(f"원장 {len(led)}건 / 대기 {len(s['pending'])}건 / 만료 {len(exp)}건 "
        f"/ 사이클 {s.get('cycles', 0)}")
    if exp:
        log(f"  ⚠️만료 {len(exp)}건 -- 러너가 조회 창({FETCH_BARS}봉)보다 오래 멈춘 흔적. "
            f"hit 집계에는 넣지 않는다.")
    if not led:
        log("아직 확정된 기록 없음"); return
    df = pd.DataFrame([r for r in led if r.get("hit") is not None])
    if df.empty:
        log("확정 hit 없음"); return
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
