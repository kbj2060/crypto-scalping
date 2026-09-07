#!/usr/bin/env python3
"""ETH 앵커 **MASHT 방향 섀도우 러너** -- 가상 체결만 기록한다. 주문을 내지 않는다. (2026-09-07)

사용자: *"각 피쳐별과 각 arm 별 최고 조합을 가지고 최고 Top1 을 새도우를 돌려보자"*
       *"조건 없이 그대로 새도우 러너를 만들어줘"*

## 규칙 (동결: `data/live/masht_wbin_shadow_artifact/meta.json`)

    모집단  앵커 `any3/Wc3` -- 8종 증거신호 중 **3종 이상이 3봉 안에 발동**하고 마지막 발동이
            그 봉일 때. GAP 12봉 dedup. 뒤만 보므로 인과적(`build_eth_anchor_label_dataset_20260907
            .py::anchor_index` 원문 재사용).
    피쳐    앵커 봉 포함 **48봉 창 × 8채널** → MultiRocket 2016 + Hydra 768 = **2784열**
            채널: logret · path_atr · dem14 · p_fast · delta_z · ret3_z · kalman_dev_z · hl_range
            앞 7개는 측면정렬(cont_sign = +1 top / -1 bottom), 마지막은 크기
    모델    TabPFN in-context (문맥 3,555 앵커, 2024-01-04 ~ 2026-07-29 동결)
    진입    두 팔 (2026-09-08 되돌림 팔 추가, 사용자 요청):
            **지속 팔**   p >= 0.537178 (표본외 워크포워드 70분위 = 상위 30%)
                          바닥 앵커면 하락 계속(숏), 천장 앵커면 상승 계속(롱)
            **되돌림 팔** p <= 0.521566 (하위 30%) -- 지속의 반대 방향
                          바닥 앵커면 반등(롱), 천장 앵커면 반락(숏)
            체결가 = 다음 봉 시가 (라벨 규약 `open[t+1]`과 동일)
            ⚠️두 팔은 **대칭이 아니다**. 워크포워드 실측:
                지속   n=325 적중 59.69% [54.18, 65.08] 건당 +11.6bp  ✅손익분기 초과
                되돌림 n=325 적중 52.31% [47.01, 57.38] 건당 **−3.2bp** ❌손익분기 미달
            되돌림 팔도 정보는 있다(기저 47.28% → 52.31%, 무작위진입 귀무 p=0.018)지만
            손익분기 53.90% 에 못 미친다. **기대치가 음수인 채로 가동한다** -- 모델 정보가
            한쪽 꼬리에만 있다는 가설을 전방으로 검정하기 위해서다.
    청산    **±1% 대칭 배리어**, 1분봉 first-touch, 최대 48봉(4시간) 보유 후 시간청산
            ⚠️트레일링을 쓰지 않는다 -- 라벨이 고정 배리어이고, 트레일링은 2026-09-07 에
            "걸 수 없는 자리" 결함이 확인된 축이다.
    비용    원장에 총수익률·테이커 10bp·메이커 7.8bp 순손익을 전부 기록. 판정은 10bp.
            손익분기 정확도 = (100+7.8)/200 = 53.90%

## 측정된 근거 (사전등록 값 -- 앞으로 이 숫자와 비교한다)
    워크포워드(월 1회 재학습, 세 창 풀링) 상위30% 진입 정확도 **59.88% [53.94%, 64.56%]**
      · 날블록 귀무 p=0.017 · 풀링 AUC 0.5453 · 건당 +12.0bp
    전방 확인(2026-06-30~07-31, n=25) 64.00% [44.44%, 82.63%] · **무작위진입 귀무 p=0.282**
      → 모순되지 않으나 확증 아님. 그 창은 기저가 57.14%로 손익분기 위였다.
    ⚠️승자의 저주: 87셀에서 고른 Top-1 이다. 기대치는 할인해서 봐야 한다.

## 원장이 남기는 것 (판정은 사람이 나중에)
매 진입에 예측확률·앵커 측면·발동 신호·ATR·진입가·배리어가·해소시각·결과(cont/fade/timeout)와
**같은 봉의 '전건 진입' 결과**를 함께 남긴다. 후자는 게이트가 아니라 **기록**이다 --
전방 확인에서 기저가 손익분기 위로 올라가면 선별이 오히려 손해인 구간이 있음을 봤기 때문에,
나중에 그 비교를 원장만으로 할 수 있어야 한다.

⚠️이 스크립트는 어떤 주문도 내지 않는다. 공개 API 조회와 가상 원장 기록만 한다.

Usage:
    python scripts/live_eth_masht_anchor_shadow_runner_20260907.py --once
    python scripts/live_eth_masht_anchor_shadow_runner_20260907.py --loop
    python scripts/live_eth_masht_anchor_shadow_runner_20260907.py --report
    python scripts/live_eth_masht_anchor_shadow_runner_20260907.py --selftest
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

ART = ROOT / "data/live/masht_wbin_shadow_artifact"
STATE = ROOT / "data/live/masht_anchor_shadow_state.json"
KLINES = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL, BTC_SYMBOL = "ETHUSDT", "BTCUSDT"
K_WIN, H_BARS, GAP_BARS = 48, 48, 12
BARRIER_PCT = 1.0
COST_TAKER_BP, COST_MAKER_BP = 10.0, 7.8
MAX_CONCURRENT = 5
RULE_ID = "masht_wbin_top30_20260907"
THR_FADE_DEFAULT = 0.521566        # 표본외 워크포워드 30분위 (되돌림 팔)

_CACHE: dict[str, Any] = {}


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {m}", flush=True)


def fetch_klines(symbol: str, interval: str = "5m", limit: int = 1500,
                 retries: int = 3) -> pd.DataFrame | None:
    """형성 중인 마지막 봉을 제거한다 -- 라이브는 **완결 봉만** 본다."""
    for k in range(retries):
        try:
            r = requests.get(KLINES, params={"symbol": symbol, "interval": interval,
                                             "limit": limit}, timeout=15)
            r.raise_for_status()
            d = r.json()
            kl = pd.DataFrame(d, columns=["open_time", "open", "high", "low", "close", "volume",
                                          "close_time", "quote_volume", "trades",
                                          "taker_buy_base", "taker_buy_quote", "ignore"])
            for c in ("open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume"):
                kl[c] = kl[c].astype(float)
            kl["trades"] = kl["trades"].astype(int)
            kl["timestamp"] = pd.to_datetime(kl["open_time"], unit="ms")
            now_ms = int(time.time() * 1000)
            kl = kl[kl["close_time"] < now_ms].reset_index(drop=True)   # 형성봉 제거
            return kl
        except Exception as e:
            log(f"⚠️klines {symbol} {interval} 실패({k+1}/{retries}): {type(e).__name__}: {e}")
            time.sleep(2 * (k + 1))
    return None


def load_artifact() -> dict[str, Any]:
    if "art" in _CACHE:
        return _CACHE["art"]
    import joblib
    meta = json.loads((ART / "meta.json").read_text())
    rk = joblib.load(ART / "rocket.joblib")
    ctx = np.load(ART / "context.npz")
    _CACHE["art"] = {"meta": meta, "mr": rk["multirocket"], "hy": rk["hydra"],
                     "X": ctx["X"], "y": ctx["y"], "clf": None}
    log(f"아티팩트 로드: 문맥 {ctx['X'].shape} · 임계 {meta['entry_threshold']:.6f}")
    return _CACHE["art"]


def predict(win: np.ndarray) -> float:
    """win: (1, 8, 48) -> 지속 확률."""
    a = load_artifact()
    if a["clf"] is None:
        from tabpfn import TabPFNClassifier
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        clf = TabPFNClassifier(device=dev, n_estimators=a["meta"]["n_estimators"],
                               random_state=a["meta"]["seed"], ignore_pretraining_limits=True,
                               memory_saving_mode=True)
        clf.fit(a["X"], a["y"])
        a["clf"] = clf
        log(f"TabPFN 문맥 주입 완료 ({dev})")
    F = np.concatenate([np.asarray(a["mr"].transform(win.astype(np.float32))),
                        np.asarray(a["hy"].transform(win.astype(np.float32)))], axis=1)
    return float(a["clf"].predict_proba(F.astype(np.float32))[0, 1])


def build_window(sig: pd.DataFrame, cores: pd.DataFrame, i: int, cont_sign: float) -> np.ndarray | None:
    """앵커 봉 i 를 **포함해** 끝나는 48봉 창. 원 빌더와 같은 채널 정의."""
    import build_eth_anchor_window_tensor_20260907 as WB
    s = i - K_WIN + 1
    if s < 0:
        return None
    w = slice(s, i + 1)
    close = sig["close"].to_numpy(float); high = sig["high"].to_numpy(float)
    low = sig["low"].to_numpy(float); atr = sig["atr_pct"].to_numpy(float)
    bar = {"logret": np.concatenate([[np.nan], np.diff(np.log(close))]),
           "dem14": sig["dem"].to_numpy(float), "p_fast": cores["p_fast"].to_numpy(float),
           "delta_z": cores["delta_z"].to_numpy(float), "ret3_z": cores["ret3_z"].to_numpy(float),
           "kalman_dev_z": sig["kalman_dev_z"].to_numpy(float),
           "hl_range": (high - low) / np.maximum(close, 1e-12)}
    cl = close[w]; a = atr[i]
    X = np.full((1, len(WB.CHANNELS), K_WIN), np.nan, np.float32)
    for c, ch in enumerate(WB.CHANNELS):
        v = (cl - cl[-1]) / max(a * cl[-1], 1e-12) if ch == "path_atr" else bar[ch][w]
        if ch in WB.DIRECTIONAL:
            v = v * cont_sign
        elif ch in WB.CENTERED:
            v = 0.5 + (v - 0.5) * cont_sign
        X[0, c, :] = v
    return X if np.isfinite(X).all() else None


def last_bar_anchor(sig: pd.DataFrame) -> dict[str, Any] | None:
    """직전 완결 봉이 any3/Wc3 앵커인가. 백로그는 쫓지 않는다(09-04 원장 교훈)."""
    import build_eth_anchor_label_dataset_20260907 as B
    n = len(sig)
    for side in ("bottom", "top"):
        fire = np.stack([sig[f"{side}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS], axis=1)
        idx = B.anchor_index(fire, "any3", 3)
        if len(idx) and int(idx[-1]) == n - 1:
            names = [B.ABBR[B.SIGNALS[j]] for j in np.flatnonzero(fire[n - 1])]
            return {"side": side, "bar_i": n - 1, "signals": names,
                    "n_signals": int(fire[n - 1].sum())}
    return None


def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            log("⚠️state 파싱 실패 -- 새로 시작")
    return {"version": 1, "rule": RULE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
            "positions": [], "ledger": [], "skips": [], "seen_bars": []}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    s["updated_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, ensure_ascii=False))
    tmp.replace(STATE)


def resolve(s: dict[str, Any], kl1: pd.DataFrame) -> None:
    """열린 가상 포지션을 1분봉 first-touch 로 해소한다."""
    if kl1 is None or not s["positions"]:
        return
    t1 = kl1["timestamp"].to_numpy()
    hi = kl1["high"].to_numpy(float); lo = kl1["low"].to_numpy(float)
    keep = []
    for p in s["positions"]:
        st = np.datetime64(pd.Timestamp(p["entry_utc"]))
        m = t1 >= st
        if not m.any():
            keep.append(p); continue
        H, L = hi[m], lo[m]
        up, dn = p["barrier_up"], p["barrier_dn"]
        i_up = int(np.argmax(H >= up)) if (H >= up).any() else 10**9
        i_dn = int(np.argmax(L <= dn)) if (L <= dn).any() else 10**9
        n_min = int(m.sum())
        if min(i_up, i_dn) == 10**9:
            if n_min >= H_BARS * 5:          # 시간청산 (48봉 = 240분)
                _close(s, p, "timeout", None, n_min)
            else:
                keep.append(p)
            continue
        cont_up = p["side"] == "top"          # 천장 앵커의 지속 = 상승
        hit_cont = (i_up < i_dn) if cont_up else (i_dn < i_up)
        if min(i_up, i_dn) >= H_BARS * 5:
            _close(s, p, "timeout", None, n_min); continue
        # outcome 은 **시장이 무엇을 했나**(cont/fade)이고, 손익은 팔에 따라 부호가 뒤집힌다.
        _close(s, p, "cont" if hit_cont else "fade", int(min(i_up, i_dn)), n_min)
    s["positions"] = keep


def _close(s: dict[str, Any], p: dict[str, Any], outcome: str,
           minutes: int | None, n_min: int) -> None:
    # 지속 팔은 outcome=="cont" 일 때 이기고, 되돌림 팔은 outcome=="fade" 일 때 이긴다.
    bet = p.get("bet", "cont")
    if outcome == "timeout":
        gross = 0.0
    else:
        won = (outcome == "cont") if bet == "cont" else (outcome == "fade")
        gross = BARRIER_PCT * 100.0 if won else -BARRIER_PCT * 100.0
    rec = dict(p)
    rec.update(outcome=outcome, bet=bet, gross_bp=gross, minutes_to_hit=minutes,
               net_taker_bp=gross - COST_TAKER_BP, net_maker_bp=gross - COST_MAKER_BP,
               closed_utc=datetime.now(timezone.utc).isoformat(), bars_observed=n_min // 5)
    s["ledger"].append(rec)
    log(f"청산 {p['entry_utc']} {p['side']} 팔={bet} → 시장 {outcome} ({gross:+.0f}bp gross)")


def cycle(s: dict[str, Any]) -> None:
    import build_eth_anchor_label_dataset_20260907 as B
    import build_eth_anchor_oscillator_cores_20260907 as OC
    kl = fetch_klines(SYMBOL); btc = fetch_klines(BTC_SYMBOL)
    kl1 = fetch_klines(SYMBOL, interval="1m", limit=1500)
    if kl is None or btc is None or len(kl) < K_WIN + 900:
        log("⚠️klines 부족 -- 사이클 건너뜀"); return
    resolve(s, kl1)

    fund = B._load_funding()
    sig = B.compute_signals(kl, btc_df=btc, funding_df=fund)
    cores = OC.compute_cores(sig, B)
    bar_ts = str(pd.Timestamp(sig["timestamp"].iloc[-1]))
    if bar_ts in s.get("seen_bars", [])[-50:]:
        return
    s.setdefault("seen_bars", []).append(bar_ts)
    s["seen_bars"] = s["seen_bars"][-200:]

    a = last_bar_anchor(sig)
    if a is None:
        return
    art = load_artifact()
    cs = 1.0 if a["side"] == "top" else -1.0
    win = build_window(sig, cores, a["bar_i"], cs)
    if win is None:
        s["skips"].append({"bar": bar_ts, "why": "window_nan"}); return
    p_cont = predict(win)
    thr = art["meta"]["entry_threshold"]
    entry_px = float(sig["close"].iloc[-1])         # 다음 봉 시가는 다음 사이클에 채운다
    atr = float(sig["atr_pct"].iloc[-1])
    rec = {"bar_utc": bar_ts, "side": a["side"], "signals": a["signals"],
           "n_signals": a["n_signals"], "p_cont": p_cont, "threshold": thr,
           "atr_pct": atr, "anchor_close": entry_px}
    thr_f = float(art["meta"].get("entry_threshold_fade", THR_FADE_DEFAULT))
    rec["threshold_fade"] = thr_f
    if p_cont >= thr:
        bet = "cont"
    elif p_cont <= thr_f:
        bet = "fade"
    else:
        rec["why"] = "between_thresholds"
        s["skips"].append(rec)
        log(f"앵커 {bar_ts} {a['side']} p={p_cont:.4f} · {thr_f:.4f}<p<{thr:.4f} → 스킵")
        return
    if len(s["positions"]) >= MAX_CONCURRENT:
        rec["why"] = "max_concurrent"; rec["bet"] = bet; s["skips"].append(rec); return
    up = entry_px * (1 + BARRIER_PCT / 100.0)
    dn = entry_px * (1 - BARRIER_PCT / 100.0)
    # 지속 방향: 천장=상승 · 바닥=하락. 되돌림 팔은 그 반대다.
    cont_up = a["side"] == "top"
    take_up = cont_up if bet == "cont" else (not cont_up)
    s["positions"].append({**rec, "bet": bet,
                           "entry_utc": datetime.now(timezone.utc).isoformat(),
                           "entry_px_provisional": entry_px, "entry_px": None,
                           "barrier_up": up, "barrier_dn": dn,
                           "cont_dir": "up" if cont_up else "down",
                           "trade_dir": "long" if take_up else "short"})
    log(f"⭐진입 {bar_ts} {a['side']} p={p_cont:.4f} · 팔={bet} "
        f"· 방향 {'롱' if take_up else '숏'} · 배리어 ±{BARRIER_PCT}%")


def fill_next_open(s: dict[str, Any], kl: pd.DataFrame) -> None:
    """진입가를 라벨 규약대로 **다음 봉 시가**로 사후 확정한다."""
    ts = kl["timestamp"].astype(str).tolist()
    for p in s["positions"]:
        if p.get("entry_px") is not None:
            continue
        try:
            i = ts.index(str(p["bar_utc"]))
        except ValueError:
            continue
        if i + 1 < len(kl):
            o = float(kl["open"].iloc[i + 1])
            p["entry_px"] = o
            p["barrier_up"] = o * (1 + BARRIER_PCT / 100.0)
            p["barrier_dn"] = o * (1 - BARRIER_PCT / 100.0)
            p["slippage_bp"] = (o - p["entry_px_provisional"]) / p["entry_px_provisional"] * 1e4


def report(s: dict[str, Any]) -> None:
    L = pd.DataFrame(s["ledger"])
    print(f"규칙 {s.get('rule')} · 시작 {s.get('started_utc')}")
    print(f"열린 포지션 {len(s['positions'])} · 마감 {len(L)} · 스킵 {len(s.get('skips', []))}")
    if not len(L):
        print("아직 마감된 가상 거래 없음"); return
    if "bet" not in L.columns:
        L["bet"] = "cont"
    d = L[L.outcome != "timeout"]
    print(f"\n마감 {len(L)} (배리어 {len(d)} · 시간청산 {(L.outcome=='timeout').sum()})")
    print(f"손익분기 {(100+COST_TAKER_BP)/200:.2%}")
    for b in ("cont", "fade"):
        db = d[d.bet == b]
        if not len(db):
            print(f"  {b:<5} 없음"); continue
        won = (db.outcome == "cont") if b == "cont" else (db.outcome == "fade")
        print(f"  {b:<5} n={len(db):>4} 적중 {won.mean():.2%} · 건당 {L[L.bet==b].net_taker_bp.mean():+.2f}bp")
    for c, tag in (("net_taker_bp", "테이커 10bp"), ("net_maker_bp", "메이커 7.8bp")):
        print(f"  {tag:<12} 건당 {L[c].mean():+.2f}bp · 합계 {L[c].sum():+.0f}bp")
    print(f"\n측면별:"); print(L.groupby("side").outcome.value_counts().to_string())


def selftest() -> int:
    """네트워크 없이 아티팩트·창구성·부호 규약만 확인."""
    ok = True
    if not (ART / "meta.json").exists():
        print("🔴아티팩트 없음"); return 1
    m = json.loads((ART / "meta.json").read_text())
    print(f"아티팩트 {m['rule_id']} · 문맥 {m['n_context']} · 임계 {m['entry_threshold']:.6f}")
    print(f"  임계 출처: {m.get('threshold_source', '?')[:70]}")
    if not (0.3 < m["entry_threshold"] < 0.8):
        print("🔴임계가 비정상 범위"); ok = False
    import build_eth_anchor_window_tensor_20260907 as WB
    if m["channels"] != WB.CHANNELS:
        print("🔴채널 정의 불일치"); ok = False
    else:
        print(f"  채널 {len(WB.CHANNELS)}개 일치 · 방향정렬 {len(WB.DIRECTIONAL)} · 중심반전 {len(WB.CENTERED)}")
    print(f"  라벨 규약 {m['label']} · 손익분기 {m['breakeven_acc']:.4f}")
    print(f"  ⚠️사전등록 측정치: 워크포워드 {m['measured']['walkforward_top30_acc']:.4f} "
          f"{m['measured']['ci']} · 귀무 p={m['measured']['dayblock_null_p']}")
    print("selftest " + ("PASS" if ok else "🔴FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    s = load_state()
    if args.report:
        report(s); return 0
    if args.once:
        cycle(s)
        kl = fetch_klines(SYMBOL, limit=200)
        if kl is not None:
            fill_next_open(s, kl)
        save_state(s); return 0
    if args.loop:
        log(f"루프 시작 · 규칙 {RULE_ID} · ⚠️주문 없음(가상 원장만)")
        while True:
            try:
                cycle(s)
                kl = fetch_klines(SYMBOL, limit=200)
                if kl is not None:
                    fill_next_open(s, kl)
                save_state(s)
            except Exception as e:
                log(f"⚠️사이클 예외: {type(e).__name__}: {e}")
            now = time.time()
            time.sleep(max(10, 300 - (now % 300) + 20))
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
