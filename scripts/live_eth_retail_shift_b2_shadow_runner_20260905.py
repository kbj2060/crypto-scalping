#!/usr/bin/env python3
"""ETH **retail_shift 지속 신호(B2)** 섀도우 러너 + 롱숏비 공개 지연(known_ts) 수집기 -- 가상 체결만 기록한다. 주문을 내지 않는다.

## 규칙 (사전등록: docs/experiments/eth_retail_shift_b2_shadow_prereg_20260905.md, 근거: 호메로스 §5.28)

    변수    Binance `globalLongShortAccountRatio`(ETHUSDT, 5m) 의 longShortRatio 시계열(행 공간, 5분 스탬프)
    신호    Δ6 = r_t − r_{t−6행}(30분), z = (Δ6 − 이동평균288)/이동표준편차288 (min 144, ddof=1 -- 연구 `roll_z` 원문)
            |z| ≥ 2.2616 (TRAIN 95분위, 동결) 이면 발동. **방향 = −부호**: z ≤ −T → 롱(계정이 숏으로 쏠림), z ≥ +T → 숏
    첫발동  같은 측면이 직전 12행 안에 발동하지 않았을 때만 (GAP12, 뒤만 봄)
    타이밍  봉 T(시가 T, 마감 C=T+5분)의 결정은 C+12초. **결정 행은 스탬프 == T 인 행**(= 연구의 metrics 1봉 지연 변형 = 라이브
            파생지표 fetcher 규약 '봉 시가 스냅샷'). 스탬프 C 행이 결정 시각에 이미 보였는지는 known_ts 통계로만 기록한다.
            행 T가 결정 시각까지 안 보이면 스킵(row_late) -- 쫓지 않는다. 백필(first_seen 없음) 행으로는 진입하지 않는다.
    진입    결정 시각 마크가격(≈ 다음 봉 시가). 청산·ATR·한도·비용은 지속 규칙 R 러너와 동일
            (SL 5.0×ATR · ARM 1.5×ATR · Trail 0.1×ATR · 200봉 · 완결 봉 고가/저가 · ATR14 · 동시 5 · 테이커 10bp/메이커 7.8bp 병기)

## known_ts 수집
20초마다 최신 3행을 조회해 **행별 첫 관측 시각**(first_seen_utc)과 지연(first_seen − 스탬프)을 `data/live/retail_shift_b2_lsr_rows.jsonl` 에
남긴다. 기존 서버 수집기(oi_lsratio.duckdb)는 배치 시각만 남겨 지연 상한(≤ 수집 주기)밖에 못 준다. 6시간마다 500행 백필로 결측을 메운다(백필 행은 first_seen 없음).

## 산출
`data/live/retail_shift_b2_state.json` -- 포지션/원장/스킵 사유/known_ts 요약. `--report` 는 B2 단독 + known_ts + R 원장과의 합집합 일별 짝비교.
`--parity` 는 과거 metrics 덤프(binance_data/metrics)로 행 공간 신호를 재계산해 연구 스크린의 lag1 발동과 대조한다(L1 성격).

⚠️이 스크립트는 어떤 주문도 내지 않는다.

Usage:
    python scripts/live_eth_retail_shift_b2_shadow_runner_20260905.py --loop
    python scripts/live_eth_retail_shift_b2_shadow_runner_20260905.py --once
    python scripts/live_eth_retail_shift_b2_shadow_runner_20260905.py --report [--r-state PATH]
    python scripts/live_eth_retail_shift_b2_shadow_runner_20260905.py --selftest | --parity
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

import live_eth_fire_cont_shadow_runner_20260904 as R  # noqa: E402  (fetch_klines, atr_series, mark_price, regime_tag, fill_next_open, first_fire_mask)

SYMBOL = "ETHUSDT"
LSR_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
THRESH_Z, Z_WIN, Z_MINP, DIFF_N, GAP_ROWS = 2.2616, 288, 144, 6, 12       # 연구 report.json axes.retail_shift.threshold (TRAIN q95, 동결)
BRACKET, MAX_CONCURRENT, MAX_HOLD_BARS, ATR_N = R.BRACKET, R.MAX_CONCURRENT, R.MAX_HOLD_BARS, R.ATR_N
COST_TAKER_BP, COST_MAKER_BP = R.COST_TAKER_BP, R.COST_MAKER_BP
BAR_SECONDS, WAKE_OFFSET_SEC, POLL_SEC, BACKFILL_SEC = 300, 12, 20, 6 * 3600
STATE_DEFAULT = ROOT / "data/live/retail_shift_b2_state.json"
ROWS_DEFAULT = ROOT / "data/live/retail_shift_b2_lsr_rows.jsonl"
RULE_ID = "retail_shift_b2_v1_z2.2616_gap12_lag1_cell5.0-1.5-0.1_cap5"
BACKTEST_REF = {"lag1": {"TRAIN": 4.89, "VAL": 12.68, "OOS": 12.28, "per_day": 3.4}, "lag0": {"VAL": 10.07, "OOS": 13.26},
                "union_minus_R_bp_per_day_lag1": {"TRAIN": 3.25, "VAL": 6.05, "OOS": 7.43}}
STATE, ROWS = STATE_DEFAULT, ROWS_DEFAULT


def log(m: str) -> None:
    print(f"[retail-b2 {datetime.now(timezone.utc):%m-%d %H:%M:%S}] {m}", flush=True)


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


# ----------------------------------------------------------------------------- 롱숏비 행 저장소 (known_ts)
class LsrRows:
    """ts(UTC naive, 5분 스탬프) -> {ratio, long, short, first_seen_utc|None, delay_sec|None}. JSONL 에 append."""

    def __init__(self, path: Path):
        self.path = path; self.rows: dict[pd.Timestamp, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    d = json.loads(line); self.rows[pd.Timestamp(d["ts"])] = d
                except Exception:                                  # noqa: BLE001
                    continue

    def add(self, ts: pd.Timestamp, ratio: float, long_: float, short_: float, seen: pd.Timestamp | None) -> bool:
        if ts in self.rows:
            return False
        d = {"ts": str(ts), "ratio": float(ratio), "long": float(long_), "short": float(short_),
             "first_seen_utc": (seen.isoformat() if seen is not None else None),
             "delay_sec": (round((seen.tz_localize(None) - ts).total_seconds(), 1) if seen is not None else None)}
        self.rows[ts] = d
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(d) + "\n")
        return True

    def series(self) -> pd.Series:
        if not self.rows:
            return pd.Series(dtype=float)
        s = pd.Series({t: r["ratio"] for t, r in self.rows.items()}).sort_index()
        return s[~s.index.duplicated()]

    def seen_at(self, ts: pd.Timestamp) -> pd.Timestamp | None:
        r = self.rows.get(ts)
        return pd.Timestamp(r["first_seen_utc"]) if r and r.get("first_seen_utc") else None

    def delay_stats(self) -> dict[str, Any]:
        d = np.array([r["delay_sec"] for r in self.rows.values() if r.get("delay_sec") is not None], float)
        if not len(d):
            return {"n": 0}
        return {"n": int(len(d)), "p05": round(float(np.percentile(d, 5)), 1), "p50": round(float(np.percentile(d, 50)), 1),
                "p95": round(float(np.percentile(d, 95)), 1), "max": round(float(d.max()), 1),
                "share_le_12s": round(float((d <= WAKE_OFFSET_SEC).mean()), 3), "share_le_300s": round(float((d <= 300).mean()), 3),
                "n_backfill": int(sum(1 for r in self.rows.values() if r.get("delay_sec") is None))}


def fetch_lsr(limit: int, retries: int = 3) -> list[dict] | None:
    for k in range(retries):
        try:
            r = requests.get(LSR_URL, params={"symbol": SYMBOL, "period": "5m", "limit": limit}, timeout=15); r.raise_for_status()
            return r.json()
        except Exception as e:                                     # noqa: BLE001
            log(f"⚠️lsr 조회 실패({k+1}/{retries}): {type(e).__name__}: {e}"); time.sleep(2 * (k + 1))
    return None


def poll_rows(store: LsrRows, limit: int = 3, backfill: bool = False) -> int:
    data = fetch_lsr(limit)
    if not data:
        return 0
    seen = now_utc(); n = 0
    for d in data:
        ts = pd.Timestamp(int(d["timestamp"]), unit="ms")
        if store.add(ts, float(d["longShortRatio"]), float(d["longAccount"]), float(d["shortAccount"]), None if backfill else seen):
            n += 1
    return n


# ----------------------------------------------------------------------------- 신호 (행 공간, 연구 roll_z 원문)
def zscore_rows(ratio: pd.Series) -> pd.Series:
    d6 = ratio - ratio.shift(DIFF_N)
    m = d6.rolling(Z_WIN, min_periods=Z_MINP).mean(); s = d6.rolling(Z_WIN, min_periods=Z_MINP).std()
    return (d6 - m) / s.replace(0, np.nan)


def fire_masks(z: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(롱 첫발동, 숏 첫발동) -- 방향 = −부호. GAP12 는 행 단위."""
    zv = z.to_numpy(float)
    long_raw = np.nan_to_num(zv <= -THRESH_Z); short_raw = np.nan_to_num(zv >= THRESH_Z)
    return R.first_fire_mask(long_raw.astype(bool), GAP_ROWS), R.first_fire_mask(short_raw.astype(bool), GAP_ROWS)


def decision_row(store: LsrRows, bar_open: pd.Timestamp) -> dict[str, Any]:
    """봉 T 결정: 스탬프 == T 행이 결정 시각까지 관측됐는지, 그 행이 첫발동인지."""
    s = store.series(); out: dict[str, Any] = {"row_ts": str(bar_open), "available": False, "backfill": False, "side": None, "z": None}
    if bar_open not in s.index:
        return out
    seen = store.seen_at(bar_open)
    out["available"] = True; out["backfill"] = seen is None
    z = zscore_rows(s); lf, sf = fire_masks(z); i = int(np.flatnonzero(s.index == bar_open)[0])
    out["z"] = (round(float(z.iloc[i]), 4) if np.isfinite(z.iloc[i]) else None); out["ratio"] = float(s.iloc[i])
    out["d6"] = (round(float(s.iloc[i] - s.iloc[i - DIFF_N]), 5) if i >= DIFF_N else None)
    out["n_rows"] = int(len(s)); out["delay_sec"] = (store.rows[bar_open].get("delay_sec") if bar_open in store.rows else None)
    out["side"] = "long" if lf[i] else ("short" if sf[i] else None)
    out["row_C_available"] = (bar_open + pd.Timedelta(minutes=5)) in s.index
    return out


# ----------------------------------------------------------------------------- 상태·포지션 (R 러너 manage 원문, 원장 필드만 다름)
def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:                                          # noqa: BLE001
            log("⚠️상태 파싱 실패 -- 새로 시작")
    return {"version": 1, "rule": RULE_ID, "started_utc": datetime.now(timezone.utc).isoformat(), "positions": [], "ledger": [],
            "skipped": {"row_late": 0, "backfill_row": 0, "slots_full": 0, "no_mark": 0, "dup": 0, "no_atr": 0}, "missed_bars": 0,
            "decided_bars": 0, "row_C_available_at_decision": 0, "last_decided_bar_utc": None, "consec_loss": 0, "known_ts": {}}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2, default=str)); tmp.replace(STATE)


def manage(s: dict[str, Any], bars: list[dict]) -> None:
    if not bars:
        log("⚠️봉 데이터 없음 -- 포지션 갱신 건너뜀"); return
    keep = []
    for p in s["positions"]:
        sgn = 1.0 if p["side"] == "long" else -1.0; a = p["atr"]; last = p.get("last_bar_utc")
        todo = [b for b in bars if last is None or b["timestamp_utc"] > last]
        closed = False
        for b in todo:
            adv = b["low"] if sgn > 0 else b["high"]
            if (adv <= p["stop"]) if sgn > 0 else (adv >= p["stop"]):
                _close(s, p, p["stop"], "stop", b["timestamp_utc"]); closed = True; break
            fav = b["high"] if sgn > 0 else b["low"]
            if sgn * (fav - p["best"]) > 0:
                p["best"] = fav
            if not p["armed"] and sgn * (p["best"] - p["entry"]) >= BRACKET["arm_atr"] * a:
                p["armed"] = True
            if p["armed"]:
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
    rec = {k: p.get(k) for k in ("entry_utc", "opened_utc", "side", "metric_ts", "z", "ratio", "d6", "row_delay_sec", "row_C_available",
                                 "entry", "next_open", "entry_slip_bp", "decision_lag_sec", "atr", "atr_pct", "regime")}
    rec.update({"exit_utc": bar_utc, "recorded_utc": datetime.now(timezone.utc).isoformat(), "exit": exit_px,
                "gross_bp": round(gross, 2), "pnl_bp": round(gross - COST_TAKER_BP, 2), "pnl_maker_bp": round(gross - COST_MAKER_BP, 2),
                "bars_held": int(p.get("bars_held", 0)), "reason": reason, "exit_basis": "bar_high_low", "rule": RULE_ID})
    s["ledger"].append(rec); s["consec_loss"] = 0 if rec["pnl_bp"] > 0 else s["consec_loss"] + 1
    log(f"  청산 {p['side']} {rec['pnl_bp']:+.2f}bp ({reason}, {rec['bars_held']}봉, z={p.get('z')}) 연속손실 {s['consec_loss']}")


def enter(s: dict[str, Any], bar_ts: str, dec: dict[str, Any], px: float, atr: float, close: float) -> None:
    side = dec["side"]
    if any(p["entry_utc"] == bar_ts for p in s["positions"]) or any(t["entry_utc"] == bar_ts for t in s["ledger"][-50:]):
        s["skipped"]["dup"] += 1; return
    if len(s["positions"]) >= MAX_CONCURRENT:
        s["skipped"]["slots_full"] += 1; log(f"  스킵(한도) @{bar_ts} {side}"); return
    if not (np.isfinite(atr) and atr > 0):
        s["skipped"]["no_atr"] += 1; log("  스킵(ATR 없음)"); return
    sgn = 1.0 if side == "long" else -1.0
    lag = (now_utc() - (pd.Timestamp(bar_ts, tz="UTC") + pd.Timedelta(minutes=5))).total_seconds()
    s["positions"].append({"entry_utc": bar_ts, "opened_utc": datetime.now(timezone.utc).isoformat(), "side": side,
                           "metric_ts": dec["row_ts"], "z": dec["z"], "ratio": dec.get("ratio"), "d6": dec.get("d6"), "row_delay_sec": dec.get("delay_sec"),
                           "row_C_available": dec.get("row_C_available"), "entry": px, "ref_close": close, "next_open": None, "entry_slip_bp": None,
                           "decision_lag_sec": round(lag, 1), "atr": float(atr), "atr_pct": float(atr / close), "stop": px - sgn * BRACKET["sl_atr"] * atr,
                           "best": px, "armed": False, "bars_held": 0, "last_bar_utc": bar_ts, "regime": R.regime_tag()})
    log(f"  [가상진입] {side} @{px:.2f} z={dec['z']} ratio={dec.get('ratio')} atr={atr:.2f} stop={px - sgn*BRACKET['sl_atr']*atr:.2f} lag={lag:.0f}s")


def decide(s: dict[str, Any], store: LsrRows) -> None:
    kl = R.fetch_klines(SYMBOL)
    if kl is None or len(kl) < 300:
        log("⚠️ETH klines 부족 -- 결정 건너뜀"); return
    kl = kl.copy(); kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    atr = R.atr_series(kl).to_numpy(float); tail = kl.tail(R.BARS_RETURNED)
    bars = [{"timestamp_utc": str(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
            for t, o, h, l, c in zip(tail["timestamp"], tail["open"], tail["high"], tail["low"], tail["close"])]
    manage(s, bars); R.fill_next_open(s, kl)
    last_ts = kl["timestamp"].iloc[-1]; last_str = str(last_ts); prev = s.get("last_decided_bar_utc"); dec = None
    if prev != last_str:
        if prev:
            gap = int((last_ts - pd.Timestamp(prev)).total_seconds() // BAR_SECONDS) - 1
            if gap > 0:
                s["missed_bars"] += gap; log(f"⚠️놓친 봉 {gap}개 (쫓지 않음)")
        dec = decision_row(store, last_ts); s["decided_bars"] += 1
        if dec.get("row_C_available"):
            s["row_C_available_at_decision"] += 1
        if not dec["available"]:
            s["skipped"]["row_late"] += 1; log(f"  행 {last_str[:16]} 미관측(row_late) -- 스킵")
        elif dec["backfill"]:
            s["skipped"]["backfill_row"] += 1
        elif dec["side"]:
            px = R.mark_price()
            if px is None:
                s["skipped"]["no_mark"] += 1; log("⚠️마크가격 실패 -- 진입 보류")
            else:
                enter(s, last_str, dec, px, float(atr[-1]), float(kl["close"].iloc[-1]))
        s["last_decided_bar_utc"] = last_str
        # 2026-09-06 텔레메트리(규칙 무변경): 대시보드가 "지금 z가 임계에서 얼마나 떨어져 있나"를
        # 보여줄 수 있게 마지막 결정을 남긴다. 화면이 "대기"만 말하면 조건이 안 온 것과 러너가
        # 죽은 것을 구분할 수 없다. 결정은 새 봉에서만 하므로 이 블록 안에 둔다.
        s["last_decision"] = {"bar_utc": last_str, "decided_at_utc": datetime.now(timezone.utc).isoformat(),
                              "z": dec.get("z"), "side": dec.get("side"), "thresh_z": THRESH_Z,
                              "available": dec.get("available"), "backfill": dec.get("backfill"),
                              "row_C_available": dec.get("row_C_available"), "delay_sec": dec.get("delay_sec"),
                              "ratio": (round(float(dec["ratio"]), 4) if dec.get("ratio") is not None else None),
                              "d6": dec.get("d6")}
    s["known_ts"] = store.delay_stats()
    tot = sum(t["pnl_bp"] for t in s["ledger"])
    log(f"봉 {last_str[:16]} z={dec.get('z') if dec else '-'} side={dec.get('side') if dec else '-'} rowC={dec.get('row_C_available') if dec else '-'} "
        f"· 포지션 {len(s['positions'])}/{MAX_CONCURRENT} · 원장 {len(s['ledger'])}건 {tot:+.0f}bp · 행 {len(store.rows)} 지연p50 {s['known_ts'].get('p50')}s")


# ----------------------------------------------------------------------------- 보고
def _daily(pnl: np.ndarray, ts: list[str], cap: int) -> pd.Series:
    return pd.Series(np.asarray(pnl, float) / cap, index=pd.DatetimeIndex(pd.to_datetime(ts, utc=True, format="mixed")).tz_localize(None).normalize()).groupby(level=0).sum()


def _day_ci(x: np.ndarray, B: int = 1000, seed: int = 20260905) -> list[float]:
    rng = np.random.default_rng(seed); out = np.empty(B)
    for k in range(B):
        out[k] = x[rng.integers(0, len(x), len(x))].mean()
    return [round(float(np.percentile(out, 2.5)), 2), round(float(np.percentile(out, 97.5)), 2)]


def union_cap_sim(trades: pd.DataFrame, cap: int) -> pd.DataFrame:
    """두 원장을 합쳐 진입 순서대로 동시 cap 슬롯 체결을 재구성(연구 portfolio() 규약)."""
    t = trades.copy()
    for c in ("entry_utc", "exit_utc"):                            # R 원장은 tz-aware 문자열, B2 는 naive -- 둘 다 UTC naive 로 통일
        t[c] = pd.to_datetime(t[c], utc=True, format="mixed").dt.tz_localize(None)
    t = t.sort_values("entry_utc").reset_index(drop=True); open_until: list[pd.Timestamp] = []; keep = []
    for k, row in t.iterrows():
        eb = row["entry_utc"]; open_until = [u for u in open_until if u > eb]
        if len(open_until) < cap:
            open_until.append(row["exit_utc"]); keep.append(k)
    return t.iloc[keep]


def report(s: dict[str, Any], store: LsrRows, r_state: Path | None) -> None:
    led = pd.DataFrame(s["ledger"]); start = pd.Timestamp(s.get("started_utc")); days = max((now_utc() - start).total_seconds() / 86400, 1e-9)
    ks = store.delay_stats()
    log(f"=== retail_shift B2 섀도우 ({RULE_ID}) · 가동 {days:.2f}일 · 결정 봉 {s.get('decided_bars')} · 놓친 봉 {s.get('missed_bars')} · 스킵 {s.get('skipped')} ===")
    log(f"  [known_ts] 행 {ks.get('n')} 지연 p05/p50/p95/max {ks.get('p05')}/{ks.get('p50')}/{ks.get('p95')}/{ks.get('max')}s · ≤12s {ks.get('share_le_12s')} · ≤300s {ks.get('share_le_300s')} "
        f"· 결정 시 스탬프C 행 가용 {s.get('row_C_available_at_decision')}/{s.get('decided_bars')} · 백필 {ks.get('n_backfill')}")
    if not len(led):
        log("  원장 비어있음"); return
    def block(d: pd.DataFrame, name: str) -> None:
        if not len(d):
            return
        p = d["pnl_bp"].to_numpy(float); w = p > 0; eq = np.cumsum(p); dd = (eq - np.maximum.accumulate(eq)).min()
        log(f"  {name:>10s} n {len(p):3d} ({len(p)/days:.2f}/일) 기대값 {p.mean():+.2f}bp (메이커 {d['pnl_maker_bp'].mean():+.2f}) 누적 {p.sum():+.0f} 승률 {w.mean()*100:.1f}% "
            f"최대DD {dd:+.0f} 지연 중앙 {d['decision_lag_sec'].median():.0f}s 슬리피지 {d['entry_slip_bp'].mean():+.2f}bp")
    block(led, "전체"); block(led[led.side == "long"], "롱"); block(led[led.side == "short"], "숏")
    dB = _daily(led["pnl_bp"].to_numpy(), led["exit_utc"].tolist(), MAX_CONCURRENT)
    log(f"  [B2 한계 일손익, 자본 대비 bp/일(슬롯=자본/5), 마감일 기준] 평균 {dB.mean():+.2f} · 일CI {_day_ci(dB.to_numpy())} · 거래일 {len(dB)}")
    rp = r_state or R.STATE_DEFAULT
    if rp.exists():
        rs = json.loads(rp.read_text()); rl = pd.DataFrame(rs.get("ledger", []))
        if len(rl):
            dR = _daily(rl["pnl_bp"].to_numpy(), rl["exit_utc"].tolist(), MAX_CONCURRENT); days_u = dR.index.union(dB.index)
            diff = dB.reindex(days_u, fill_value=0.0).to_numpy()          # (R∪B2) − R = B2 (원장 분리 시 정확히 B2 한계)
            log(f"  [(R∪B2)−R, 원장 분리(각자 cap5)] 평균 {diff.mean():+.2f}bp/일 · 일CI {_day_ci(diff)} · 이긴 날 {(diff > 0).mean():.3f} (일수 {len(diff)})")
            U = pd.concat([rl.assign(src="R"), led.assign(src="B2")], ignore_index=True)[["entry_utc", "exit_utc", "pnl_bp", "src"]]
            Us = union_cap_sim(U, MAX_CONCURRENT); Rs = union_cap_sim(rl.assign(src="R")[["entry_utc", "exit_utc", "pnl_bp", "src"]], MAX_CONCURRENT)
            dU = _daily(Us["pnl_bp"].to_numpy(), Us["exit_utc"].tolist(), MAX_CONCURRENT); dR2 = _daily(Rs["pnl_bp"].to_numpy(), Rs["exit_utc"].tolist(), MAX_CONCURRENT)
            du = (dU.reindex(days_u, fill_value=0.0) - dR2.reindex(days_u, fill_value=0.0)).to_numpy()
            log(f"  [(R∪B2)−R, 합집합 cap5 재구성(2차)] 평균 {du.mean():+.2f}bp/일 · 일CI {_day_ci(du)} · B2 체결 {int((Us.src == 'B2').sum())}/{len(led)}")
    log(f"  [백테스트 참조] {BACKTEST_REF}")


# ----------------------------------------------------------------------------- 파리티 (연구 lag1 발동 vs 행 공간 재계산)
def parity() -> int:
    import glob, io, zipfile
    rows = []
    for f in sorted(glob.glob(str(ROOT / "binance_data/metrics/ETHUSDT-metrics-*.zip"))):
        day = f[-14:-4]
        if not ("2025-06-01" <= day <= "2026-03-31"):
            continue
        z = zipfile.ZipFile(f); rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["create_time", "count_long_short_ratio"]))
    m = pd.concat(rows, ignore_index=True); m["ts"] = pd.to_datetime(m["create_time"]); m = m.drop_duplicates("ts").sort_values("ts")
    s = pd.Series(m["count_long_short_ratio"].to_numpy(float), index=m["ts"]); z = zscore_rows(s); lf, sf = fire_masks(z)
    fires = pd.DataFrame({"ts": s.index, "side": np.where(lf, "long", np.where(sf, "short", ""))}); fires = fires[fires.side != ""]
    tr = ROOT / "data/research/eth_econ_axis_continuation_screen_20260904/triggers_retail_shift.parquet"
    if not tr.exists():
        log("연구 트리거 파일 없음 -- 행 공간 발동 수만 보고"); log(f"행 공간 첫발동 {len(fires)}건 ({len(fires)/max((s.index[-1]-s.index[0]).days,1):.2f}/일)"); return 0
    A = pd.read_parquet(tr); A["timestamp"] = pd.to_datetime(A["timestamp"]); A = A[(A.timestamp >= "2025-06-02") & (A.timestamp < "2026-03-31")]
    # 연구 lag0 발동은 봉 T 에 스탬프 C=T+5 행을 붙였다 -> 행 스탬프 = T+5. lag1 은 행 스탬프 = T. 여기서는 행 스탬프 자체를 비교한다.
    fr = set(zip(fires.ts, fires.side)); a0 = set(zip(A.timestamp + pd.Timedelta(minutes=5), np.where(A.trade_long, "long", "short")))
    inter = len(fr & a0); log(f"행 공간 첫발동 {len(fires)} · 연구 발동(행 스탬프 환산) {len(a0)} · 일치 {inter} ({inter/max(len(a0),1):.3f} of research, {inter/max(len(fires),1):.3f} of rows)")
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0); r = pd.Series(2.5 + np.cumsum(rng.normal(0, 0.01, 600)), index=pd.date_range("2026-01-01", periods=600, freq="5min"))
    r.iloc[400:406] += np.linspace(0.05, 0.4, 6)                     # 급등(계정 롱 쏠림) -> 숏 발동
    z = zscore_rows(r); lf, sf = fire_masks(z)
    assert sf[400:406].sum() == 1 and sf[380:400].sum() == 0 and lf[380:406].sum() == 0, (sf[398:408], z.iloc[398:408].round(2).tolist())
    assert lf[406:414].sum() == 1, "급등이 끝나면 Δ6 이 음전 -> 롱 첫발동 1회 (GAP12 안 중복 없음)"
    assert np.isnan(z.iloc[:Z_MINP + DIFF_N - 1]).all()
    st = LsrRows(Path("/tmp/claude_b2_selftest_rows.jsonl")); st.path.unlink(missing_ok=True); st = LsrRows(st.path)
    for t, v in r.items():
        st.add(t, v, 0.7, 0.3, None)
    t_last = r.index[-1]; assert st.add(t_last + pd.Timedelta(minutes=5), r.iloc[-1] - 0.5, 0.6, 0.4, now_utc())
    d = decision_row(st, t_last + pd.Timedelta(minutes=5)); assert d["available"] and not d["backfill"] and d["side"] == "long" and d["z"] < -THRESH_Z, d
    d2 = decision_row(st, t_last); assert d2["available"] and d2["backfill"] and d2["side"] is None, d2
    assert not decision_row(st, t_last + pd.Timedelta(minutes=10))["available"]
    st.path.unlink(missing_ok=True)
    # 합집합 cap 재구성: 겹치는 3건, cap 2 -> 2건
    U = pd.DataFrame({"entry_utc": ["2026-01-01T00:00", "2026-01-01T00:05", "2026-01-01T00:10"], "exit_utc": ["2026-01-01T02:00"] * 3, "pnl_bp": [1, 2, 3], "src": ["R", "B2", "R"]})
    assert union_cap_sim(U, 2)["pnl_bp"].tolist() == [1, 2]
    print("selftest ok"); return 0


def main() -> int:
    global STATE, ROWS
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--parity", action="store_true")
    ap.add_argument("--state", type=str, default=None); ap.add_argument("--rows", type=str, default=None); ap.add_argument("--r-state", type=str, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.parity:
        return parity()
    if a.state:
        STATE = Path(a.state)
    if a.rows:
        ROWS = Path(a.rows)
    s = load_state(); store = LsrRows(ROWS)
    if a.report:
        report(s, store, Path(a.r_state) if a.r_state else None); return 0
    log(f"⚠️섀도우 모드 -- 주문 없음. 규칙 {RULE_ID} · 상태 {STATE} · 행 {ROWS} ({len(store.rows)}행)")
    n = poll_rows(store, limit=500, backfill=True); log(f"백필 {n}행 (first_seen 없음, 진입 불가·z 워밍업용)")
    if not a.loop:
        poll_rows(store); decide(s, store); save_state(s); report(s, store, None); return 0
    last_backfill = time.time(); next_dec = (int(time.time() // BAR_SECONDS) + 1) * BAR_SECONDS + WAKE_OFFSET_SEC
    while True:
        try:
            poll_rows(store)
            if time.time() - last_backfill > BACKFILL_SEC:
                poll_rows(store, limit=500, backfill=True); last_backfill = time.time()
            if time.time() >= next_dec:
                decide(s, store); save_state(s); next_dec = (int(time.time() // BAR_SECONDS) + 1) * BAR_SECONDS + WAKE_OFFSET_SEC
        except KeyboardInterrupt:
            save_state(s); log("중단"); return 0
        except Exception as e:                                     # noqa: BLE001
            log(f"⚠️사이클 예외: {type(e).__name__}: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
