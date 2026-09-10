#!/usr/bin/env python3
"""포지션 청산 감시자 워커 (2026-09-10) -- 사용자가 연 실계좌 ETH 포지션을 5분봉마다 보고
**지금 익절·부분익절·손절·감축해야 하는가**를 판정한다. 주문은 내지 않는다.

설계: docs/eth_position_exit_advisor_research_20260910.md (§8-2 종합 모델)
판정 = 종합 청산 모델 + 생존 규칙 두 겹:
  T0 생존(규칙, 학습 대상이 아니다): 청산가까지 5분 ATR 3배 안 · 24h 변동성 전망 「위험」에서 하루 1σ 안.
  종합 모델: 대시보드 입력(증거신호 8종·극점 탐지기 p1/p2·24h 변동성 전망·돌파/되돌림 봉피쳐·포지셔닝
    메트릭·시각) + 포지션 상태(원시 이동·MFE·MAE·보유봉) -> p = P(지금 청산이 24봉 상한 보유보다 낫다).
    p ≥ τ → 익절(이익 중)/손절(손실 중), τ_partial ≤ p < τ → 부분익절/감축. 피쳐 식은 exit_synth_features 한 벌
    (학습 빌더와 동일 함수). 아티팩트 data/live/eth_exit_synth_artifact (research_eth_exit_synth_model_20260910.py).
  24봉(사용자 결정) 넘긴 포지션은 모델 정의역 밖 -> 「시간 상한」으로 청산 검토를 낸다.
규칙 v1(decide) 은 원장 비교용으로 `rule_verdict` 에 같이 기록한다.
**가격을 권고하지 않고 시점만 권고한다** -- 시장이 떠난 자리의 스톱을 체결시킨 09-07 결함 재발 방지.
어휘는 다른 카드와 겹치지 않게 포지션 행동(익절/부분익절/손절/감축/보유)·긴급도(즉시/권고/참고).

산출: data/live/position_exit_advisor_state.json (현재 판정) · ..._ledger.jsonl (매 5분 체크포인트).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import load_dotenv  # noqa: E402
from live_binance_account_20260910 import fetch_account  # noqa: E402
import exit_synth_features_20260910 as EF  # noqa: E402
import live_eth_extreme_detector_20260909 as X  # noqa: E402
import live_eth_vol_forecast_20260910 as VF  # noqa: E402
import live_eth_breakout_reversal_shadow_runner_20260908 as BR  # noqa: E402

load_dotenv(ROOT / ".env")

LIVE = ROOT / "data" / "live"
STATE = LIVE / "position_exit_advisor_state.json"
LEDGER = LIVE / "position_exit_advisor_ledger.jsonl"
MODEL_DIR = LIVE / "eth_exit_synth_artifact"
EXTREME_STATE = LIVE / "eth_extreme_detector_state.json"
VOL_STATE = LIVE / "eth_vol_forecast_state.json"
BURST_STATE = LIVE / "liq_burst_state.json"
SYMBOL = "ETHUSDT"                 # ETH 먼저. 다른 코인은 ETH가 자리 잡은 뒤(사용자 결정 4번)
BAR_SECONDS = 300
WAKE_OFFSET_SEC = 90               # 극점 워커(60초 주기)가 새 봉을 채점한 뒤에 읽는다
RULES = {
    "time_cap_bars": 24,           # 사용자 지정(2시간) = 모델 라벨 지평
    "invalidation_bars": 12,       # 극점 탐지기 라벨 창 그대로 -- 발동 봉 극값을 12봉 안에 깨면 빗나감
    "liq_atr_mult": 3.0,           # 청산가까지 5분 ATR 3배 안이면 생존 경고
    "arm_atr": 2.0, "trail_atr": 1.0,   # 규칙 v1 정점 반납(원장 비교용 rule_verdict 에만 쓴다)
    "atr_n": 14,
}
VERDICT_TONE = {"익절": "good", "부분익절": "good", "손절": "bad", "감축": "warn", "보유": "neutral"}
_CACHE: dict[str, Any] = {}


def log(m: str) -> None:
    print(f"[exit-advisor {datetime.now(timezone.utc):%m-%d %H:%M:%S}Z] {m}", flush=True)


def load_model() -> dict | None:
    if "model" not in _CACHE:
        try:
            import joblib
            _CACHE["model"] = {"models": joblib.load(MODEL_DIR / "model.joblib"),
                               "meta": json.loads((MODEL_DIR / "meta.json").read_text())}
        except Exception as e:  # noqa: BLE001 -- 아티팩트 없으면 규칙만으로 돈다
            log(f"⚠️종합 모델 아티팩트 없음: {type(e).__name__}: {e}")
            _CACHE["model"] = None
    return _CACHE["model"]


def read_json(p: Path, max_age_min: float) -> dict:
    """워커 상태 파일. 없거나 낡았으면 빈 dict -- 낡은 신호로 판정하지 않는다."""
    try:
        d = json.loads(p.read_text())
        ts = d.get("updated_utc") or d.get("updated_at")
        age = (time.time() - datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()) / 60
        return d if age <= max_age_min else {}
    except Exception:  # noqa: BLE001
        return {}


def market_frame() -> pd.DataFrame | None:
    """3000봉 ETH/BTC + 메트릭 + DVOL -> 학습과 같은 봉 피쳐 프레임(마지막 행이 최신 완결 봉)."""
    kl, btc = X._fetch(SYMBOL), X._fetch(X.BTC_SYMBOL)
    if kl is None or btc is None or len(kl) < 2300:
        return None
    met, met_ts = BR.fetch_metrics(limit=500)
    dv = VF._fetch_dvol(hours=240)
    if dv is not None:
        dv = dv.iloc[:-1]                                   # 형성 중인 시간봉 제외(학습과 동일)
    ext, costw, vol = EF.load_artifacts()
    return EF.bar_features(kl.rename(columns={"tq_": "taker_buy_quote"}), btc, met, met_ts, dv, ext, costw, vol)


def snapshot(pos: dict, kl: pd.DataFrame, atr: float, extreme: dict, vol: dict, burst: dict, mem: dict) -> dict:
    """포지션 하나의 판정 입력을 스칼라로 모은다. 가격 변동은 원시 비율(레버·명목 곱하지 않음)."""
    s = 1.0 if pos["side"] == "LONG" else -1.0
    e, m = float(pos["entry_price"]), float(pos["mark_price"])
    entry_ts = pd.Timestamp(pos["entry_at"]).tz_convert(None) if pos.get("entry_at") else None
    held = kl[kl["timestamp"] >= entry_ts] if entry_ts is not None else kl.iloc[0:0]
    fav = held["high"] if s > 0 else held["low"]
    adv = held["low"] if s > 0 else held["high"]
    u = s * (m - e) / e
    liq = float(pos.get("liquidation_price") or 0.0)
    snap = {
        "side": pos["side"], "s": s, "u": u,
        "mfe": max(float((s * (fav - e) / e).max()) if len(held) else 0.0, u),
        "mae": min(float((s * (adv - e) / e).min()) if len(held) else 0.0, u),
        "atr": atr, "n": int(len(held)),
        "liq_dist": (s * (m - liq) / m) if liq > 0 else None,
        "vol_grade": vol.get("grade") if vol.get("available") else None,
        "vol_sigma_1d": (float(vol["rv_fwd_pred"]) / 100 / 365 ** 0.5) if vol.get("rv_fwd_pred") else None,   # 연율% -> 하루 1σ
        "burst_adverse": bool(burst.get("hawkes_active"))
                         and burst.get("crisis_type") == ("LONG_CRISIS" if s > 0 else "SHORT_CRISIS"),
        "latest_close": float(kl["close"].iloc[-1]),
        "extreme_grade": None, "extreme_opposite": False, "entry_call_px": mem.get("entry_call_px"),
    }
    if extreme.get("available") and extreme.get("latest_ts_utc"):
        latest_ok = pd.Timestamp(extreme["latest_ts_utc"]).tz_convert(None) == kl["timestamp"].iloc[-1]
        call_long = extreme.get("tone") == "good"
        if latest_ok and extreme.get("grade") and not extreme.get("gated_now"):
            snap["extreme_grade"] = extreme["grade"]
            snap["extreme_opposite"] = call_long != (s > 0)
        if snap["entry_call_px"] is None and entry_ts is not None:
            want = "good" if s > 0 else "bad"
            for t, tone in zip(extreme.get("times") or [], extreme.get("history") or []):
                tt = pd.Timestamp(t).tz_convert(None)
                if tone == want and abs((tt - entry_ts).total_seconds()) <= 2 * BAR_SECONDS:
                    bar = kl[kl["timestamp"] == tt]
                    if len(bar):
                        snap["entry_call_px"] = float(bar["low" if s > 0 else "high"].iloc[0])
                        mem["entry_call_px"], mem["entry_call_ts"] = snap["entry_call_px"], str(t)
    return snap


def survival(x: dict) -> tuple[str, str, str] | None:
    """T0 생존 규칙 -- 학습 대상이 아니다. 모델보다 먼저 본다."""
    a = x["atr"]
    bp = lambda v: f"{v * 1e4:+.0f}bp"  # noqa: E731
    if x["liq_dist"] is not None and x["liq_dist"] <= RULES["liq_atr_mult"] * a:
        return "감축", "즉시", f"청산가까지 {bp(x['liq_dist'])}: 5분 ATR {RULES['liq_atr_mult']:.0f}배 안"
    if x["liq_dist"] is not None and x["vol_grade"] == "위험" and x["vol_sigma_1d"] \
            and x["liq_dist"] <= x["vol_sigma_1d"]:
        return "감축", "권고", f"24h 변동성 위험 등급: 하루 1σ({bp(x['vol_sigma_1d'])}) 안에 청산가"
    return None


def model_verdict(p: float, u: float, n: int, thr_hi: float, thr_mid: float) -> tuple[str, str, str]:
    """종합 모델 p -> 판정. p 는 '지금 청산이 24봉 상한 보유보다 낫다' 확률.
    선택적 모드(2026-09-10 스윕 후): 확신 상위에서만 말한다 -- p ≥ thr_hi(VAL 상위 2.5%) 권고, p ≥ thr_mid(상위 10%) 참고.
    표본외 정밀도 hi 58.4%(OOS)/58.0%(HOLDOUT), mid 56.5%/55.6% -- 전체 커버리지 AUC 는 0.535 다."""
    bp = f"{u * 1e4:+.0f}bp"
    if n >= RULES["time_cap_bars"]:
        return ("익절" if u > 0 else "손절"), "참고", f"{n}봉 보유: 24봉 상한(모델 지평) 초과, 미실현 {bp}"
    if p >= thr_hi:
        return ("익절" if u > 0 else "손절"), "권고", f"종합 모델 청산확률 {p:.2f} ≥ {thr_hi:.2f}(확신 상위), 미실현 {bp}"
    if p >= thr_mid:
        return ("부분익절" if u > 0 else "감축"), "참고", f"종합 모델 청산확률 {p:.2f} ≥ {thr_mid:.2f}, 미실현 {bp}"
    return "보유", "-", f"종합 모델 청산확률 {p:.2f} · 미실현 {bp} · {n}봉"


def decide(x: dict) -> tuple[str, str, str]:
    """규칙 v1 (원장 비교용). 위에서 아래로 첫 매치."""
    a, u, n = x["atr"], x["u"], x["n"]
    bp = lambda v: f"{v * 1e4:+.0f}bp"  # noqa: E731
    sv = survival(x)
    if sv:
        return sv
    if x["entry_call_px"] is not None and n <= RULES["invalidation_bars"] \
            and x["s"] * (x["latest_close"] - x["entry_call_px"]) < 0:
        return "손절", "즉시", "진입 근거였던 극점 봉의 극값을 종가로 이탈: 논거 무효화"
    if x["burst_adverse"] and u < 0:
        return "감축", "권고", f"역방향 청산 캐스케이드 진행 중, 미실현 {bp(u)}"
    if n >= RULES["time_cap_bars"] and u <= 0:
        return "손절", "권고", f"{n}봉 보유에 미실현 {bp(u)}: 시간 상한"
    if x["extreme_opposite"] and u > 0 and x["extreme_grade"] == "강":
        return "익절", "권고", f"반대 방향 극점 탐지(등급 강), 미실현 {bp(u)}"
    if x["extreme_opposite"] and u > 0 and x["extreme_grade"] == "중":
        return "부분익절", "권고", f"반대 방향 극점 탐지(등급 중), 미실현 {bp(u)}"
    if x["mfe"] >= RULES["arm_atr"] * a and (x["mfe"] - u) >= RULES["trail_atr"] * a:
        return "익절", "권고", f"정점 {bp(x['mfe'])}에서 {bp(x['mfe'] - u)} 반납"
    if n >= RULES["time_cap_bars"] and u > 0:
        return "익절", "참고", f"{n}봉 보유 시간 상한, 미실현 {bp(u)}"
    return "보유", "-", f"미실현 {bp(u)} · 정점 {bp(x['mfe'])} · 최저 {bp(x['mae'])} · {n}봉"


def score(model: dict, F_last: pd.Series, x: dict) -> float:
    meta = model["meta"]; cap = int(meta["cap_bars"])
    hold = min(x["n"], cap - 1)                               # 모델은 k<24 만 봤다
    pf = EF.position_features(np.array([x["s"]]), np.array([hold]), cap, np.array([x["u"]]),
                              np.array([x["mfe"]]), np.array([x["mae"]]), np.array([x["atr"]]))
    row = {**{k: float(v[0]) for k, v in pf.items()}, **{c: float(F_last[c]) for c in meta["dash_cols"]}}
    Xr = np.array([[row[c] for c in meta["feature_cols"]]], np.float32)
    return float(np.mean([m.predict_proba(Xr)[0, 1] for m in model["models"]]))


def write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, default=str))
    os.replace(tmp, p)


async def _account() -> dict:
    async with aiohttp.ClientSession() as s:
        return await fetch_account(s, [SYMBOL])


def cycle(prev: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    acct = asyncio.run(_account())
    model = load_model()
    if not acct.get("ok"):
        return {"available": False, "error": acct.get("error"), "positions": [], "rules": RULES, "updated_utc": now}
    kl = X._fetch(SYMBOL)
    F = market_frame() if any(p["symbol"] == SYMBOL for p in acct["positions"]) else None
    if kl is None or (F is None and acct["positions"]):
        return {"available": False, "error": "market_fetch_failed", "positions": [], "rules": RULES, "updated_utc": now}
    atr = float(F["atr"].iloc[-1]) if F is not None else float("nan")
    extreme, vol, burst = read_json(EXTREME_STATE, 15), read_json(VOL_STATE, 30), read_json(BURST_STATE, 5)
    prev_pos = {p["key"]: p for p in prev.get("positions", [])}
    out = []
    for pos in acct["positions"]:
        if pos["symbol"] != SYMBOL:
            continue
        key = f"{pos['symbol']}:{pos['side']}:{pos.get('entry_at')}"
        old = prev_pos.get(key, {})
        mem = dict(old.get("mem") or {})
        x = snapshot(pos, kl, atr, extreme, vol, burst, mem)
        rule = decide(x)
        p = score(model, F.iloc[-1], x) if model is not None else None
        sv = survival(x)
        if sv:
            verdict, urgency, reason = sv
        elif p is not None:
            sl = model["meta"].get("selective") or {}
            verdict, urgency, reason = model_verdict(p, x["u"], x["n"], sl.get("hi", {}).get("p_threshold", model["meta"]["tau"]),
                                                    sl.get("mid", {}).get("p_threshold", model["meta"]["tau_partial"]))
        else:
            verdict, urgency, reason = rule
        row = {
            "key": key, "symbol": pos["symbol"], "side": pos["side"], "leverage": pos["leverage"],
            "entry_price": pos["entry_price"], "mark_price": pos["mark_price"], "entry_at": pos.get("entry_at"),
            "entry_at_truncated": pos["symbol"] in (acct.get("trades_truncated") or []),
            "hold_bars": x["n"], "move_bp": round(x["u"] * 1e4, 1), "mfe_bp": round(x["mfe"] * 1e4, 1),
            "mae_bp": round(x["mae"] * 1e4, 1), "atr_bp": round(x["atr"] * 1e4, 1),
            "liq_dist_bp": (round(x["liq_dist"] * 1e4, 1) if x["liq_dist"] is not None else None),
            "verdict": verdict, "urgency": urgency, "reason": reason, "tone": VERDICT_TONE[verdict],
            "p_exit": (round(p, 4) if p is not None else None),
            "rule_verdict": rule[0], "rule_reason": rule[2],
            "since_utc": old.get("since_utc", now) if old.get("verdict") == verdict else now,
            "context": {"extreme_grade": x["extreme_grade"], "extreme_opposite": x["extreme_opposite"],
                        "vol_grade": x["vol_grade"], "burst_adverse": x["burst_adverse"],
                        "entry_call_ts": mem.get("entry_call_ts")},
            "mem": mem,
        }
        out.append(row)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": now, "bar_utc": str(kl["timestamp"].iloc[-1]),
                                **{k: v for k, v in row.items() if k != "mem"},
                                "features": ({c: float(F.iloc[-1][c]) for c in model["meta"]["dash_cols"]} if model and F is not None else None)},
                               ensure_ascii=False, default=str) + "\n")
    mm = (model or {}).get("meta") or {}
    return {"available": True, "error": None, "updated_utc": now, "latest_bar_utc": str(kl["timestamp"].iloc[-1]),
            "symbol": SYMBOL, "positions": out, "rules": RULES,
            "model": ({"rule_id": mm.get("rule_id"), "tau": mm.get("tau"), "tau_partial": mm.get("tau_partial"),
                       "selective": mm.get("selective"), "decision_mode": mm.get("decision_mode"),
                       "auc": mm.get("auc"), "policy": mm.get("policy"), "gate": mm.get("gate"),
                       "excluded_inputs": mm.get("excluded_inputs")} if mm else None),
            "inputs": {"extreme": bool(extreme), "vol_forecast": bool(vol), "liq_burst": bool(burst)}}


def sleep_to_next_bar() -> None:
    now = time.time()
    time.sleep(max(1.0, (int(now // BAR_SECONDS) + 1) * BAR_SECONDS + WAKE_OFFSET_SEC - now))


def selftest() -> None:
    base = {"side": "LONG", "s": 1.0, "u": 0.001, "mfe": 0.001, "mae": -0.0005, "atr": 0.002, "n": 3,
            "liq_dist": 0.05, "vol_grade": "안정", "vol_sigma_1d": 0.03, "burst_adverse": False,
            "latest_close": 100.0, "extreme_grade": None, "extreme_opposite": False, "entry_call_px": None}
    assert survival(base) is None
    assert survival({**base, "liq_dist": 0.005})[:2] == ("감축", "즉시")                       # T0 3ATR
    assert survival({**base, "vol_grade": "위험", "liq_dist": 0.02})[:2] == ("감축", "권고")    # T0 24h σ
    assert model_verdict(0.9, 0.002, 5, 0.7, 0.6)[:2] == ("익절", "권고")
    assert model_verdict(0.9, -0.002, 5, 0.7, 0.6)[:2] == ("손절", "권고")
    assert model_verdict(0.65, 0.002, 5, 0.7, 0.6)[:2] == ("부분익절", "참고")
    assert model_verdict(0.65, -0.002, 5, 0.7, 0.6)[:2] == ("감축", "참고")
    assert model_verdict(0.2, 0.002, 5, 0.7, 0.6)[0] == "보유"
    assert model_verdict(0.2, 0.002, 24, 0.7, 0.6)[:2] == ("익절", "참고")                    # 상한 초과 = 정의역 밖
    # 규칙 v1 (원장 비교용)
    assert decide(base)[0] == "보유"
    assert decide({**base, "entry_call_px": 100.5})[0] == "손절"
    assert decide({**base, "s": -1.0, "side": "SHORT", "entry_call_px": 99.5})[0] == "손절"
    assert decide({**base, "burst_adverse": True, "u": -0.001})[0] == "감축"
    assert decide({**base, "n": 24, "u": -0.001})[:2] == ("손절", "권고")
    assert decide({**base, "extreme_opposite": True, "extreme_grade": "강"})[0] == "익절"
    assert decide({**base, "mfe": 0.005, "u": 0.002})[0] == "익절"
    # snapshot: 헤지 롱·숏 동시 보유, 진입 이후 봉만 MFE/MAE, 진입 봉 극값 기억
    ts = pd.date_range("2026-09-10 00:00", periods=30, freq="5min")
    kl = pd.DataFrame({"timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    kl.loc[20, ["high", "low"]] = [103.0, 98.0]
    iso = [t.tz_localize("UTC").isoformat() for t in ts]
    ext = {"available": True, "latest_ts_utc": iso[-1], "tone": "bad", "grade": "강", "gated_now": False,
           "times": iso, "history": ["good" if i == 20 else "neutral" for i in range(30)]}
    long_ = {"side": "LONG", "entry_price": 100.0, "mark_price": 100.5, "liquidation_price": 90.0, "entry_at": iso[21]}
    short = {**long_, "side": "SHORT", "liquidation_price": 110.0}
    mem: dict = {}
    xl = snapshot(long_, kl, 0.002, ext, {}, {}, mem)
    assert xl["n"] == 9 and abs(xl["mfe"] - 0.01) < 1e-9 and abs(xl["mae"] + 0.01) < 1e-9
    assert xl["extreme_opposite"] and xl["extreme_grade"] == "강" and xl["entry_call_px"] == 98.0 and mem["entry_call_px"] == 98.0
    xs = snapshot(short, kl, 0.002, ext, {}, {}, {})
    assert xs["u"] < 0 and not xs["extreme_opposite"] and xs["entry_call_px"] is None
    assert abs(xs["liq_dist"] - (110 - 100.5) / 100.5) < 1e-12
    # 모델 채점: 아티팩트가 있으면 학습 피쳐 열과 위치 피쳐가 빠짐없이 맞물리는지
    model = load_model()
    if model is not None:
        F_last = pd.Series({c: 0.1 for c in model["meta"]["dash_cols"]})
        p = score(model, F_last, base)
        assert 0.0 <= p <= 1.0
        assert set(model["meta"]["feature_cols"]) == set(model["meta"]["pos_cols"]) | set(model["meta"]["dash_cols"])
    print("selftest ok", "(model loaded)" if model else "(no model artifact)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    prev = read_json(STATE, 1e9)
    while True:
        try:
            prev = cycle(prev)
            write_json(STATE, prev)
            log(" · ".join(f"{p['side']} {p['verdict']}({p['urgency']}) p={p['p_exit']} {p['move_bp']:+.0f}bp" for p in prev["positions"])
                or (prev.get("error") or "포지션 없음"))
        except Exception as e:  # noqa: BLE001 -- 워커는 죽지 않는다
            log(f"⚠️사이클 실패: {type(e).__name__}: {e}")
        if not a.loop:
            return 0
        sleep_to_next_bar()


if __name__ == "__main__":
    raise SystemExit(main())
