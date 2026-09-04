#!/usr/bin/env python3
"""ETH 증거신호 발동 봉 **지속(continuation) 규칙** 섀도우 러너 -- 가상 체결만 기록한다. 주문을 내지 않는다.

## 규칙 (사전등록: docs/experiments/eth_fire_cont_shadow_prereg_20260904.md, 근거: 호메로스 §5.23)

    모집단  8종 증거신호 raw 단일봉 발동(compute_signals, 라이브 정본)의 **첫발동**
            (같은 신호·같은 측면이 직전 12봉 안에 발동하지 않았을 때). 뒤만 보므로 인과적.
    방향    **신호 반대 방향** -- 바닥(bottom) 발동 → 숏, 천장(top) 발동 → 롱
    진입    발동 봉 마감 직후 첫 사이클의 마크가격(= 다음 봉 시가 근사). **직전 완결봉 발동만** 본다 --
            사이클을 놓쳐 더 오래된 봉의 발동이 남아 있어도 쫓지 않는다(백로그 일괄 진입 금지, 09-04 원장 분석 교훈).
    청산    손절 entry ∓ 5.0×ATR · +1.5×ATR 도달 시 무장 · 트레일 0.1×ATR · 200봉 시간청산 (F0 경제라벨 셀 상속)
            ATR = 트루레인지 14봉 단순이동평균(라벨 프레임과 동일 정의, 파리티 검증됨)
    한도    동시보유 5 · 같은 봉에서 바닥·천장이 **둘 다** 첫발동이면 둘 다 스킵(백테스트 0.74%)
    비용    원장에 총수익률·테이커 10bp·메이커 7.8bp(실측) 순손익을 전부 기록. 판정은 10bp.

배리어 판정은 **완결 봉 고가/저가**로, `sim_exit`과 같은 순서((1)불리한 쪽 스톱 → (2)best → (3)무장 → (4)트레일)로 한다
(`live_eth_v_rebound_econ_shadow_runner_20260902.py::manage()` 원문 -- 층 게이트 L2P가 `trail_single`과 300경로 0.00bp 일치 확인).

## 왜 이 러너가 가벼운가
TabPFN/GPU 없음. 발동 계산(`compute_signals`)과 ATR만 쓴다. 무거운 피쳐 빌더 import 체인(torch)을 타지 않아 로컬 스모크 가능.

## 산출
`data/live/fire_cont_shadow_state.json` -- 포지션/원장/스킵 사유/놓친 봉 수. 진입 시각·마크가·다음 봉 시가(사후 채움)·슬리피지·
발동 신호 목록·발동 측면·ATR·레짐(대시보드 API, 최선 노력)을 전부 남겨 2차 분석(측면별·레짐·ATR 하한·셀 재시뮬)을
원장만으로 할 수 있게 한다.

⚠️이 스크립트는 어떤 주문도 내지 않는다. 공개 API 조회와 가상 원장 기록만 한다.

Usage:
    python scripts/live_eth_fire_cont_shadow_runner_20260904.py --once [--state PATH]
    python scripts/live_eth_fire_cont_shadow_runner_20260904.py --loop
    python scripts/live_eth_fire_cont_shadow_runner_20260904.py --report
    python scripts/live_eth_fire_cont_shadow_runner_20260904.py --selftest
"""
from __future__ import annotations

import argparse
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
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402  (라이브 정본, 가벼움)

SYMBOL, BTC_SYMBOL = "ETHUSDT", "BTCUSDT"
FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
MARK_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
REGIME_URL = "http://127.0.0.1:8787/api/regime-wide24"      # 대시보드(서버 로컬) ETH 레짐 S12_K3 -- 최선 노력 태그
FETCH_LIMIT = 1500                                          # 1500봉 창의 발동 = 8000봉 창과 동일(09-04 실측 0 결측/0 유령)
SIGNALS = ["taker_delta_z_climax", "short_term_return_z", "liquidity_sweep", "orthogonal_combo",
           "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
BRACKET = {"sl_atr": 5.0, "arm_atr": 1.5, "trail_atr": 0.1}
MAX_CONCURRENT, MAX_HOLD_BARS, GAP_BARS, ATR_N = 5, 200, 12, 14
COST_TAKER_BP, COST_MAKER_BP = 10.0, 7.8
BAR_SECONDS, WAKE_OFFSET_SEC = 300, 12                      # 5분 경계 + 12초에 깨어나 방금 마감한 봉만 판단
BARS_RETURNED = 60
STATE_DEFAULT = ROOT / "data/live/fire_cont_shadow_state.json"
RULE_ID = "fire_cont_v1_gap12_cell5.0-1.5-0.1_cap5"
BACKTEST_REF = {"VAL": {"exp_bp": 4.44, "per_day": 24.4, "win_rate": 0.758, "payoff": 0.35},
                "OOS": {"exp_bp": 6.78, "per_day": 24.3, "win_rate": 0.747, "payoff": 0.39}}
STATE = STATE_DEFAULT


def log(m: str) -> None:
    print(f"[fire-cont-shadow {datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


# ----------------------------------------------------------------------------- 데이터
def fetch_klines(symbol: str, retries: int = 3) -> pd.DataFrame | None:
    """`live_eth_sweep_v_rebound_signal_20260829._fetch_klines`와 같은 스키마·같은 형성봉 제거."""
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv", "trades",
            "taker_buy_base", "tq", "ignore"]
    for k in range(retries):
        try:
            r = requests.get(FUTURES_KLINES_URL, params={"symbol": symbol, "interval": "5m", "limit": FETCH_LIMIT}, timeout=15)
            r.raise_for_status(); raw = r.json()
            df = pd.DataFrame(raw, columns=cols)
            for c in ("open", "high", "low", "close", "volume", "taker_buy_base"):
                df[c] = df[c].astype(float)
            df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
            df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            if len(df) and int(df.iloc[-1]["close_time"]) >= int(time.time() * 1000):
                df = df.iloc[:-1].reset_index(drop=True)          # 형성 중인 봉 제거
            return df
        except Exception as e:                                     # noqa: BLE001
            log(f"⚠️klines {symbol} 실패({k+1}/{retries}): {type(e).__name__}: {e}"); time.sleep(2 * (k + 1))
    return None


def mark_price() -> float | None:
    try:
        r = requests.get(MARK_URL, params={"symbol": SYMBOL}, timeout=10); r.raise_for_status()
        return float(r.json()["price"])
    except Exception:                                              # noqa: BLE001
        return None


def atr_series(kl: pd.DataFrame) -> pd.Series:
    """라벨 프레임의 `atr`(build_eth_5m_sweep_followthrough_v2_labels_20260829::ATR_N=14, TR 단순평균)과 동일."""
    prev = kl["close"].shift(1)
    tr = pd.concat([kl["high"] - kl["low"], (kl["high"] - prev).abs(), (kl["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_N, min_periods=ATR_N).mean()


def regime_tag() -> dict[str, Any] | None:
    try:
        r = requests.get(REGIME_URL, timeout=4); r.raise_for_status(); j = r.json()
        if not j.get("warmed_up"):
            return None
        probs = {k: float(j[f"{k}_prob"]) for k in ("bull", "bear", "chop") if j.get(f"{k}_prob") is not None}
        return {"label": max(probs, key=probs.get) if probs else None, **probs, "latest_bar_utc": j.get("latest_bar_utc")}
    except Exception:                                              # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- 발동
def first_fire_mask(fire: np.ndarray, gap: int) -> np.ndarray:
    """raw 발동 중 직전 gap봉에 같은 신호·측면 발동이 없던 것. 뒤만 본다(§5.22 규약, 백테스트와 동일)."""
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def last_bar_first_fires(sig: pd.DataFrame) -> list[dict[str, str]]:
    """마지막 완결 봉의 첫발동 목록 [{signal, fire_side}]."""
    out = []
    n = len(sig)
    for s in SIGNALS:
        for side in ("bottom", "top"):
            col = f"{side}_{s}"
            if col not in sig.columns:
                continue
            f = sig[col].fillna(False).to_numpy(bool)
            if f[-1] and not f[max(0, n - 1 - GAP_BARS):n - 1].any():
                out.append({"signal": s, "fire_side": side})
    return out


# ----------------------------------------------------------------------------- 대시보드 표시용 상태 (표시 전용)
CONT_WINDOW_BARS, CONT_LOOKBACK_BARS = 12, 48      # 지속 창 12봉(지속 트레이드 승자 중앙 보유 7봉·54%가 12봉 내 청산) / 되돌림 대기 48봉
CONT_BACKTEST_REF = {"VAL_bp": 4.44, "OOS_bp": 6.78, "train_p_fade_gt_cont": 0.446}


def regime_label_from_payload(p: dict | None) -> str | None:
    """대시보드 /api/regime-wide24 payload(bull_prob/bear_prob/chop_prob) -> argmax 라벨. 없으면 None."""
    if not isinstance(p, dict) or not p.get("warmed_up"):
        return None
    probs = {k: p.get(f"{k}_prob") for k in ("bull", "bear", "chop")}
    probs = {k: float(v) for k, v in probs.items() if v is not None}
    return max(probs, key=probs.get) if probs else None


def continuation_state(sig: pd.DataFrame, regime_payload: dict | None = None, f0_calls: list[dict] | None = None,
                       window: int = CONT_WINDOW_BARS, lookback: int = CONT_LOOKBACK_BARS) -> dict[str, Any]:
    """증거신호 패널 표시용: 최근 lookback봉 안의 **가장 최근 첫발동 사건**(8종 합집합, GAP 규약은 러너와 동일)과
    그 사건 기준 국면(지속 창 / 되돌림 대기), 레짐 방향 일치, F0 경제모델(섀도우) 최근 호출을 돌려준다.
    순수 함수(네트워크 없음). 발동 조건·확률·net_score는 건드리지 않는다 -- 해석 한 줄을 더할 뿐이다."""
    n = len(sig)
    out: dict[str, Any] = {"active": False, "window_bars": window, "lookback_bars": lookback, "gap_bars": GAP_BARS,
                           "backtest_ref": CONT_BACKTEST_REF}
    if n == 0:
        return out
    events: dict[int, dict[str, list[str]]] = {}
    for s in SIGNALS:
        for side in ("bottom", "top"):
            col = f"{side}_{s}"
            if col not in sig.columns:
                continue
            ff = first_fire_mask(sig[col].fillna(False).to_numpy(bool), GAP_BARS)
            for i in np.flatnonzero(ff):
                if i >= n - lookback:
                    events.setdefault(int(i), {"bottom": [], "top": []})[side].append(s)
    if not events:
        return out
    i = max(events); ev = events[i]; bars_since = n - 1 - i
    sides = [sd for sd in ("bottom", "top") if ev[sd]]
    both = len(sides) == 2
    fire_side = "both" if both else sides[0]
    cont_side = None if both else ("short" if fire_side == "bottom" else "long")
    fade_side = None if both else ("long" if cont_side == "short" else "short")
    phase = "continuation" if bars_since <= window else "fade_watch"
    regime_label = regime_label_from_payload(regime_payload)
    consistency = None
    if cont_side and regime_label in ("bull", "bear", "chop"):
        if regime_label == "chop":
            consistency = "neutral"
        else:
            consistency = "match" if (cont_side == "short") == (regime_label == "bear") else "conflict"
    ts_str = sig["timestamp"].astype(str).to_numpy()
    pos_of = {t: k for k, t in enumerate(ts_str)}
    calls = []
    for c in (f0_calls or []):
        k = pos_of.get(str(c.get("entry_utc")))
        if k is None or k < i or c.get("side") not in ("long", "short"):
            continue
        calls.append({"side": c["side"], "bars_ago": int(n - 1 - k), "entry_utc": str(c.get("entry_utc")),
                      "proba": (round(float(c["proba"]), 4) if c.get("proba") is not None else None)})
    calls.sort(key=lambda x: x["bars_ago"])
    f0_fade = next((c for c in calls if c["side"] == fade_side), None) if fade_side else None
    f0_cont = next((c for c in calls if c["side"] == cont_side), None) if cont_side else None
    out.update({"active": True, "event_bar_utc": str(ts_str[i]), "bars_since": int(bars_since), "fire_side": fire_side,
                "cont_side": cont_side, "fade_side": fade_side, "phase": phase, "skip_both_sides": both,
                "signals": {**{s: "bottom" for s in ev["bottom"]}, **{s: "top" for s in ev["top"]}},
                "regime_label": regime_label, "regime_consistency": consistency,
                "f0_fade_call": f0_fade, "f0_cont_call": f0_cont, "f0_calls_recent": calls[:3],
                "n_events_lookback": len(events)})
    return out



def continuation_history(sig: pd.DataFrame, lookback: int = CONT_LOOKBACK_BARS, window: int = CONT_WINDOW_BARS) -> list[str]:
    """특화감지기 카드의 봉별 국면 띠(표시 전용, 순수 함수). 각 봉 k에서 **그 봉까지의** 가장 최근 첫발동 사건이
    지속 창(≤window봉) 안이면 지속 방향 톤(good=롱 지속, bad=숏 지속), 양측 동시 첫발동이면 warn, 아니면 neutral.
    첫발동 규약은 continuation_state/러너와 같은 first_fire_mask(GAP_BARS). 미래 봉은 보지 않는다."""
    n = len(sig)
    if n == 0:
        return []
    ev_side: dict[int, set[str]] = {}
    for s_ in SIGNALS:
        for side in ("bottom", "top"):
            col = f"{side}_{s_}"
            if col not in sig.columns:
                continue
            for i in np.flatnonzero(first_fire_mask(sig[col].fillna(False).to_numpy(bool), GAP_BARS)):
                ev_side.setdefault(int(i), set()).add(side)
    tones: list[str] = []
    last_i = -10**9; last_tone = "neutral"
    for k in range(n):
        if k in ev_side:
            sd = ev_side[k]; last_i = k
            last_tone = "warn" if len(sd) == 2 else ("bad" if "bottom" in sd else "good")
        tones.append(last_tone if (k - last_i) <= window else "neutral")
    return tones[-lookback:]


def continuation_levels(kl: pd.DataFrame, event_bar_utc: str, cont_side: str, bars_since: int) -> dict[str, Any]:
    """특화감지기 카드의 가격선(표시 전용, 순수 함수). 러너의 실제 회계와 같은 산식: 진입 = 사건 다음 봉 시가(fill_next_open),
    ATR = atr_series(ATR_N=14, 가격단위)의 사건 봉 값, 손절 = 진입 ∓ sl_atr·ATR, 무장 = 진입 ± arm_atr·ATR(도달 후 best 대비
    trail_atr·ATR 트레일), 만기 = MAX_HOLD_BARS. 다음 봉이 아직 없으면 진입 미확정(pending)으로 사건 봉 종가를 참고가로 준다."""
    if cont_side not in ("long", "short") or kl is None or len(kl) == 0:
        return {}
    ts = kl["timestamp"].astype(str).to_numpy()
    hit = np.flatnonzero(ts == str(event_bar_utc))
    if len(hit) == 0:
        return {}
    i = int(hit[-1]); n = len(kl)
    atr = atr_series(kl).iloc[i]
    if not (np.isfinite(atr) and atr > 0):
        return {}
    sgn = 1.0 if cont_side == "long" else -1.0
    ref_close = float(kl["close"].iloc[i])
    if i + 1 < n:
        entry = float(kl["open"].iloc[i + 1]); basis = "next_open"
    else:
        entry = ref_close; basis = "pending_next_open"
    out = {"entry": round(entry, 2), "entry_basis": basis, "atr": round(float(atr), 2), "atr_pct": round(float(atr / ref_close), 5),
           "stop": round(entry - sgn * BRACKET["sl_atr"] * atr, 2), "arm": round(entry + sgn * BRACKET["arm_atr"] * atr, 2),
           "trail_dist": round(BRACKET["trail_atr"] * atr, 2), "bracket": dict(BRACKET), "max_hold_bars": MAX_HOLD_BARS,
           "bars_left": int(max(MAX_HOLD_BARS - int(bars_since), 0))}
    return out


def shadow_summary(state: dict[str, Any] | None) -> dict[str, Any]:
    """섀도우 원장 요약(표시 전용, 순수 함수). 러너 상태파일(load_state 스키마)에서 마감 건수·건당 bp(테이커/메이커)·승률·MDD·
    미결 포지션·가동일·최근 30일 건당 bp를 계산한다. 주문은 없다 -- 숫자는 사전등록(30일 계측/90일 판정) 전까지 참고값."""
    st = state or {}
    ledger = st.get("ledger") if isinstance(st.get("ledger"), list) else []
    positions = st.get("positions") if isinstance(st.get("positions"), list) else []
    pnls, pnls_mk, exits = [], [], []
    for r in ledger:
        try:
            pnls.append(float(r["pnl_bp"])); pnls_mk.append(float(r.get("pnl_maker_bp", r["pnl_bp"]))); exits.append(str(r.get("exit_utc") or ""))
        except (KeyError, TypeError, ValueError):
            continue
    n = len(pnls); run = peak = mdd = 0.0
    for v in pnls:
        run += v; peak = max(peak, run); mdd = min(mdd, run - peak)
    days = None; started = st.get("started_utc")
    if started:
        try:
            days = max((datetime.now(timezone.utc) - datetime.fromisoformat(str(started))).total_seconds() / 86400.0, 0.0)
        except (TypeError, ValueError):
            days = None
    cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=30)).isoformat()
    recent = [v for v, e in zip(pnls, exits) if e >= cutoff]
    wins = [v for v in pnls if v > 0]
    return {"rule": st.get("rule") or RULE_ID, "days_running": (round(days, 2) if days is not None else None), "closed_trades": n,
            "exp_bp": (round(sum(pnls) / n, 2) if n else None), "exp_maker_bp": (round(sum(pnls_mk) / n, 2) if n else None),
            "win_rate": (round(len(wins) / n, 3) if n else None), "total_bp": (round(sum(pnls), 1) if n else None),
            "max_dd_bp": (round(mdd, 1) if n else None), "last30d_trades": len(recent), "last30d_exp_bp": (round(sum(recent) / len(recent), 2) if recent else None),
            "trades_per_day": (round(n / days, 2) if (n and days) else None), "open_positions": len(positions),
            "open_sides": sorted({str(p.get("side")) for p in positions}), "missed_bars": st.get("missed_bars"),
            "skipped": st.get("skipped"), "backtest_ref": BACKTEST_REF}

# ----------------------------------------------------------------------------- 상태
def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:                                          # noqa: BLE001
            log("⚠️상태 파싱 실패 -- 새로 시작")
    return {"version": 1, "rule": RULE_ID, "started_utc": datetime.now(timezone.utc).isoformat(), "positions": [], "ledger": [],
            "skipped": {"both_sides": 0, "slots_full": 0, "no_mark": 0, "dup": 0}, "missed_bars": 0, "last_decided_bar_utc": None,
            "consec_loss": 0}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str)); tmp.replace(STATE)


# ----------------------------------------------------------------------------- 포지션 관리 (경제라벨 러너 manage() 원문)
def manage(s: dict[str, Any], bars: list[dict]) -> None:
    if not bars:
        log("⚠️봉 데이터 없음 -- 포지션 갱신 건너뜀"); return
    keep = []
    for p in s["positions"]:
        sgn = 1.0 if p["side"] == "long" else -1.0; a = p["atr"]; last = p.get("last_bar_utc")
        todo = [b for b in bars if last is None or b["timestamp_utc"] > last]
        closed = False
        for b in todo:
            adv = b["low"] if sgn > 0 else b["high"]                  # (1) 불리한 쪽 먼저
            if (adv <= p["stop"]) if sgn > 0 else (adv >= p["stop"]):
                _close(s, p, p["stop"], "stop", b["timestamp_utc"]); closed = True; break
            fav = b["high"] if sgn > 0 else b["low"]                  # (2) best
            if sgn * (fav - p["best"]) > 0:
                p["best"] = fav
            if not p["armed"] and sgn * (p["best"] - p["entry"]) >= BRACKET["arm_atr"] * a:
                p["armed"] = True                                     # (3) 무장
            if p["armed"]:                                            # (4) 트레일(한 방향)
                ns = p["best"] - sgn * BRACKET["trail_atr"] * a
                if sgn * (ns - p["stop"]) > 0:
                    p["stop"] = ns
            p["bars_held"] = int(p.get("bars_held", 0)) + 1; p["last_bar_utc"] = b["timestamp_utc"]
            if p["bars_held"] >= MAX_HOLD_BARS:
                _close(s, p, b["close"], "timeout", b["timestamp_utc"]); closed = True; break
        if not closed:
            keep.append(p)
    s["positions"] = keep


def _close(s: dict[str, Any], p: dict[str, Any], exit_px: float, reason: str, bar_utc: str) -> None:
    sgn = 1.0 if p["side"] == "long" else -1.0
    gross = sgn * (exit_px - p["entry"]) / p["entry"] * 1e4
    rec = {k: p.get(k) for k in ("entry_utc", "opened_utc", "side", "fire_side", "signals", "entry", "next_open", "entry_slip_bp",
                                 "decision_lag_sec", "atr", "atr_pct", "regime")}
    rec.update({"exit_utc": bar_utc, "recorded_utc": datetime.now(timezone.utc).isoformat(), "exit": exit_px,
                "gross_bp": round(gross, 2), "pnl_bp": round(gross - COST_TAKER_BP, 2), "pnl_maker_bp": round(gross - COST_MAKER_BP, 2),
                "bars_held": int(p.get("bars_held", 0)), "reason": reason, "exit_basis": "bar_high_low", "rule": RULE_ID})
    s["ledger"].append(rec)
    s["consec_loss"] = 0 if rec["pnl_bp"] > 0 else s["consec_loss"] + 1
    log(f"  청산 {p['side']} {rec['pnl_bp']:+.2f}bp ({reason}, {rec['bars_held']}봉, {p.get('fire_side')} {','.join(p.get('signals', []))}) 연속손실 {s['consec_loss']}")


def fill_next_open(s: dict[str, Any], kl: pd.DataFrame) -> None:
    """발동 봉 다음 봉의 시가(백테스트 진입가)를 사후에 채운다 -- 진입 규약 격차 측정용."""
    ts = kl["timestamp"].astype(str).to_numpy(); op = kl["open"].to_numpy(float); pos = {t: i for i, t in enumerate(ts)}
    for coll in (s["positions"], s["ledger"][-200:]):
        for p in coll:
            if p.get("next_open") is None and p.get("entry_utc") in pos and pos[p["entry_utc"]] + 1 < len(ts):
                nxt = float(op[pos[p["entry_utc"]] + 1]); sgn = 1.0 if p["side"] == "long" else -1.0
                p["next_open"] = nxt; p["entry_slip_bp"] = round(sgn * (p["entry"] - nxt) / nxt * 1e4, 2)   # +면 백테스트보다 불리


def enter(s: dict[str, Any], bar_ts: str, fires: list[dict], px: float, atr: float, close: float) -> None:
    sides = {"short" if f["fire_side"] == "bottom" else "long" for f in fires}
    if len(sides) == 2:
        s["skipped"]["both_sides"] += 1; log(f"  스킵(양측 첫발동) @{bar_ts}"); return
    side = sides.pop()
    if any(p["entry_utc"] == bar_ts for p in s["positions"]) or any(t["entry_utc"] == bar_ts for t in s["ledger"][-50:]):
        s["skipped"]["dup"] += 1; return
    if len(s["positions"]) >= MAX_CONCURRENT:
        s["skipped"]["slots_full"] += 1; log(f"  스킵(한도) @{bar_ts} {side}"); return
    if not (np.isfinite(atr) and atr > 0):
        log("  스킵(ATR 없음)"); return
    sgn = 1.0 if side == "long" else -1.0
    bar_close_utc = pd.Timestamp(bar_ts) + pd.Timedelta(minutes=5)
    lag = (pd.Timestamp.now(tz="UTC") - bar_close_utc).total_seconds()
    s["positions"].append({"entry_utc": bar_ts, "opened_utc": datetime.now(timezone.utc).isoformat(), "side": side,
                           "fire_side": fires[0]["fire_side"], "signals": sorted({f["signal"] for f in fires}),
                           "entry": px, "ref_close": close, "next_open": None, "entry_slip_bp": None, "decision_lag_sec": round(lag, 1),
                           "atr": float(atr), "atr_pct": float(atr / close), "stop": px - sgn * BRACKET["sl_atr"] * atr, "best": px,
                           "armed": False, "bars_held": 0, "last_bar_utc": bar_ts, "regime": regime_tag()})
    log(f"  [가상진입] {side} @{px:.2f} ({fires[0]['fire_side']} 발동: {','.join(sorted({f['signal'] for f in fires}))}) atr={atr:.2f} "
        f"stop={px - sgn*BRACKET['sl_atr']*atr:.2f} lag={lag:.0f}s")


def cycle(s: dict[str, Any]) -> None:
    kl = fetch_klines(SYMBOL); btc = fetch_klines(BTC_SYMBOL)
    if kl is None or len(kl) < 900:
        log("⚠️ETH klines 부족 -- 사이클 건너뜀"); return
    sig = compute_signals(kl, btc_df=btc)
    atr = atr_series(kl).to_numpy(float)
    tail = kl.tail(BARS_RETURNED)
    bars = [{"timestamp_utc": str(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
            for t, o, h, l, c in zip(tail["timestamp"], tail["open"], tail["high"], tail["low"], tail["close"])]
    manage(s, bars); fill_next_open(s, kl)
    last_ts = str(kl["timestamp"].iloc[-1]); prev = s.get("last_decided_bar_utc")
    n_fires = 0
    if prev != last_ts:
        if prev:
            gap = int((pd.Timestamp(last_ts) - pd.Timestamp(prev)).total_seconds() // BAR_SECONDS) - 1
            if gap > 0:
                s["missed_bars"] += gap; log(f"⚠️놓친 봉 {gap}개 (쫓지 않음)")
        fires = last_bar_first_fires(sig); n_fires = len(fires)
        if fires:
            px = mark_price()
            if px is None:
                s["skipped"]["no_mark"] += 1; log("⚠️마크가격 실패 -- 진입 보류")
            else:
                enter(s, last_ts, fires, px, float(atr[-1]), float(kl["close"].iloc[-1]))
        s["last_decided_bar_utc"] = last_ts
    tot = sum(t["pnl_bp"] for t in s["ledger"])
    log(f"봉 {last_ts[:16]} 첫발동 {n_fires} · 포지션 {len(s['positions'])}/{MAX_CONCURRENT} · 원장 {len(s['ledger'])}건 {tot:+.0f}bp · 놓친봉 {s['missed_bars']}")


def sleep_to_next_bar() -> None:
    now = time.time(); nxt = (int(now // BAR_SECONDS) + 1) * BAR_SECONDS + WAKE_OFFSET_SEC
    time.sleep(max(1.0, nxt - now))


# ----------------------------------------------------------------------------- 보고
def report(s: dict[str, Any]) -> None:
    led = s["ledger"]
    start = pd.Timestamp(s.get("started_utc")); days = max((pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400, 1e-9)
    log(f"=== 지속 규칙 섀도우 원장 ({RULE_ID}) · 가동 {days:.2f}일 · 놓친 봉 {s.get('missed_bars')} · 스킵 {s.get('skipped')} ===")
    if not led:
        log("원장 비어있음"); return
    df = pd.DataFrame(led)
    def block(d, name):
        if not len(d):
            return
        p = d["pnl_bp"].to_numpy(float); w = p > 0; eq = np.cumsum(p); dd = (eq - np.maximum.accumulate(eq)).min()
        po = p[w].mean() / -p[~w].mean() if w.any() and (~w).any() else float("nan")
        log(f"  {name:>16s} n {len(p):4d} ({len(p)/days:.1f}/일) 기대값 {p.mean():+.2f}bp (메이커 {d['pnl_maker_bp'].mean():+.2f}) 누적 {p.sum():+.0f} "
            f"승률 {w.mean()*100:.1f}% 손익비 {po:.2f} 최대DD {dd:+.0f} 지연 중앙 {d['decision_lag_sec'].median():.0f}s 슬리피지 평균 {d['entry_slip_bp'].mean():+.2f}bp")
    block(df, "전체"); block(df[df["side"] == "short"], "숏(바닥 발동)"); block(df[df["side"] == "long"], "롱(천장 발동)")
    for s_ in SIGNALS:
        m = df["signals"].apply(lambda x: s_ in (x or []))
        if m.any():
            block(df[m], s_[:16])
    log(f"  [백테스트 참조] VAL {BACKTEST_REF['VAL']} · OOS {BACKTEST_REF['OOS']}")


def selftest() -> int:
    # 첫발동 = 직전 raw 발동으로부터 GAP(12)봉 초과 -- 백테스트 `causal_first_fire`와 동일 의미(클러스터 안에서는 첫 봉만)
    f = np.zeros(45, bool); f[[5, 6, 10, 18, 30, 31, 44]] = True                  # 5→6→10→18→30→31 은 전부 12봉 이내 연쇄
    ff = first_fire_mask(f, GAP_BARS); assert list(np.flatnonzero(ff)) == [5, 44], np.flatnonzero(ff)   # 44-31=13>12
    f2 = np.zeros(45, bool); f2[[5, 25, 40]] = True
    assert list(np.flatnonzero(first_fire_mask(f2, GAP_BARS))) == [5, 25, 40]
    sig = pd.DataFrame({f"bottom_{s}": False for s in SIGNALS} | {f"top_{s}": False for s in SIGNALS}, index=range(40))
    sig.loc[39, "bottom_demarker_extreme"] = True; sig.loc[26, "bottom_demarker_extreme"] = True    # 39-26=13>12 -> 첫발동
    assert last_bar_first_fires(sig) == [{"signal": "demarker_extreme", "fire_side": "bottom"}]
    sig.loc[30, "bottom_demarker_extreme"] = True                                                      # 39-30=9 -> 첫발동 아님
    assert last_bar_first_fires(sig) == []
    # 마스크 함수와 마지막 봉 판정의 동치성 (무작위)
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = rng.random(60) < 0.15; m1 = first_fire_mask(g, GAP_BARS)[-1]
        m2 = bool(g[-1] and not g[max(0, 59 - GAP_BARS):59].any()); assert m1 == m2
    # continuation_state: 합성 프레임 -- 바닥 첫발동(taker, 4봉 전) + 레짐 bear -> 지속=숏, 일치; F0 롱 호출은 페이드 쪽
    ts = pd.date_range("2026-09-04 00:00", periods=60, freq="5min", tz="UTC")
    sig2 = pd.DataFrame({"timestamp": ts, **{f"bottom_{s}": False for s in SIGNALS}, **{f"top_{s}": False for s in SIGNALS}})
    sig2.loc[55, "bottom_taker_delta_z_climax"] = True; sig2.loc[40, "bottom_taker_delta_z_climax"] = True   # 55-40=15>12 -> 첫발동
    sig2.loc[30, "top_demarker_extreme"] = True
    calls = [{"entry_utc": str(ts[57]), "side": "long", "proba": 0.83}, {"entry_utc": str(ts[20]), "side": "short", "proba": 0.9}]
    c = continuation_state(sig2, {"warmed_up": True, "bull_prob": 0.1, "bear_prob": 0.7, "chop_prob": 0.2}, calls)
    assert c["active"] and c["phase"] == "continuation" and c["cont_side"] == "short" and c["fire_side"] == "bottom", c
    assert c["bars_since"] == 4 and c["regime_consistency"] == "match" and c["signals"] == {"taker_delta_z_climax": "bottom"}, c
    assert c["f0_fade_call"]["side"] == "long" and c["f0_fade_call"]["bars_ago"] == 2 and c["f0_cont_call"] is None, c
    c2 = continuation_state(sig2.iloc[:57].reset_index(drop=True).assign(), {"warmed_up": True, "bull_prob": 0.6, "bear_prob": 0.2, "chop_prob": 0.2}, [])
    assert c2["bars_since"] == 1 and c2["regime_consistency"] == "conflict", c2
    sig3 = sig2.copy(); sig3.loc[55, "top_kalman_deviation_meanrev"] = True
    c3 = continuation_state(sig3, None, None); assert c3["skip_both_sides"] and c3["cont_side"] is None and c3["regime_consistency"] is None, c3
    sig4 = sig2.copy(); sig4.loc[55, "bottom_taker_delta_z_climax"] = False; sig4.loc[40, "bottom_taker_delta_z_climax"] = True
    c4 = continuation_state(sig4, None, None); assert c4["active"] and c4["phase"] == "fade_watch" and c4["bars_since"] == 19, c4
    assert not continuation_state(sig4.iloc[:5].reset_index(drop=True), None, None)["active"]
    # 카드용 순수 함수 (2026-09-04): 국면 띠는 과거만 본다 / 가격선은 러너 회계와 같은 산식
    kl_t = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=40, freq="5min"), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    sig_t = pd.DataFrame({"timestamp": kl_t["timestamp"]}); sig_t[f"bottom_{SIGNALS[0]}"] = False; sig_t[f"top_{SIGNALS[1]}"] = False
    sig_t.loc[20, f"bottom_{SIGNALS[0]}"] = True                      # 바닥 첫발동 → 숏 지속(bad) 20..32, 그 뒤 neutral
    h = continuation_history(sig_t, lookback=40)
    assert h[19] == "neutral" and h[20] == "bad" and h[32] == "bad" and h[33] == "neutral", h[18:35]
    sig_t.loc[20, f"top_{SIGNALS[1]}"] = True                         # 같은 봉 양측 → warn
    assert continuation_history(sig_t, lookback=40)[25] == "warn"
    lv = continuation_levels(kl_t, str(kl_t["timestamp"].iloc[20]), "short", bars_since=19)
    assert lv["entry"] == 100.0 and lv["entry_basis"] == "next_open" and abs(lv["atr"] - 2.0) < 1e-9, lv
    assert abs(lv["stop"] - (100.0 + BRACKET["sl_atr"] * 2.0)) < 1e-9 and abs(lv["arm"] - (100.0 - BRACKET["arm_atr"] * 2.0)) < 1e-9 and lv["bars_left"] == MAX_HOLD_BARS - 19
    assert continuation_levels(kl_t, str(kl_t["timestamp"].iloc[39]), "long", 0)["entry_basis"] == "pending_next_open"
    sm = shadow_summary({"ledger": [{"pnl_bp": 10.0, "pnl_maker_bp": 12.2, "exit_utc": "2099-01-01T00:00:00+00:00"}, {"pnl_bp": -4.0, "exit_utc": "2000-01-01T00:00:00+00:00"}], "positions": [{"side": "long"}], "started_utc": "2026-09-04T00:00:00+00:00"})
    assert sm["closed_trades"] == 2 and sm["exp_bp"] == 3.0 and sm["exp_maker_bp"] == 4.1 and sm["win_rate"] == 0.5 and sm["max_dd_bp"] == -4.0 and sm["last30d_trades"] == 1 and sm["open_sides"] == ["long"], sm
    print("selftest ok"); return 0


def main() -> int:
    global STATE
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--state", type=str, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.state:
        STATE = Path(a.state)
    s = load_state()
    if a.report:
        report(s); return 0
    log(f"⚠️섀도우 모드 -- 주문 없음. 규칙 {RULE_ID} · 한도 {MAX_CONCURRENT} · 발동창 {FETCH_LIMIT}봉 · 상태 {STATE}")
    if not a.loop:
        cycle(s); save_state(s); report(s); return 0
    while True:
        try:
            cycle(s); save_state(s)
        except KeyboardInterrupt:
            save_state(s); log("중단"); return 0
        except Exception as e:                                     # noqa: BLE001
            log(f"⚠️사이클 예외: {type(e).__name__}: {e}")
        sleep_to_next_bar()


if __name__ == "__main__":
    raise SystemExit(main())
