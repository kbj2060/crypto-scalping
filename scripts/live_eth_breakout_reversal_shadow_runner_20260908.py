#!/usr/bin/env python3
"""ETH 앵커 **돌파/되돌림 섀도우 러너** -- 가상 체결만 기록한다. 주문을 내지 않는다. (2026-09-08)

사용자: *"정보량을 일 10건으로 제한하진 말고 first_fire · T=0.75 · H=1시간 ·
        HGB 모델로 새도우 러너와 대시보드에 카드 추가해줘"*
→ **커버리지 상한을 두지 않는다.** 발현한 전 트리거에 판정을 내고 확신 등급만 표시한다.

## 규칙 (동결: data/live/breakout_reversal_shadow_artifact/meta.json)
    모집단  앵커 `first_fire` -- 8종 증거신호 중 어느 하나의 첫 발동(신호별 GAP dedup). 25.2건/일
    발현    앵커 다음 봉 시가 기준 **15분 안에 ±0.75×ATR** 최초 터치.
            그 분이 트리거, 터치한 쪽이 **발현 방향**(관측값이지 예측 대상이 아니다). 22.1건/일
    판정    발현 방향으로 계속 가나(**돌파**) 되돌아오나(**되돌림**) -- 1시간 안에
            진입가 **±0.8×ATR** 중 먼저 닿는 쪽. 시간청산이면 12봉 뒤 종가 부호.
            (2026-09-08 개정 2: 절대 ±0.25% 에서 ATR 상대로. 저ATR 구간의 시간청산 25%→3%,
             고ATR 의 돌파율 불균형 0.334→0.46 이 해소된다.)
    피쳐    69개, 전부 **트리거 봉 bt 의 직전 봉 bt-1** 기준.
            🔴경로 피쳐는 트리거 분 s1 을 **포함하지 않는다**(CLAUDE.md 사건 라벨 경계 계약).
    모델    HGB 5시드 평균. 모델 축은 2026-09-08 종결 -- TabPFN/TabICL/LightGBM/고전GBM 전부
            T2 노이즈 안이었고 회귀는 분류에 2.6~5.1pp 뒤졌다.

## 사전등록 (시드 20개 워크포워드 · 셔플 귀무와 함께 읽는다)
    전건 22.1건/일   VAL .5748  OOS .6046  HOLDOUT .5716
      셔플 귀무           .5457      .5697          .5157   → 초과 +2.9 / +3.5 / +5.6pp
    ⭐시드 20/20 이 세 창 동시에 귀무 위 · B=100 셔플에서 p=0.010
    ⚠️**원시 정확도끼리 비교하지 않는다.** 창마다 클래스 균형이 달라 귀무가 .515~.570 로 움직인다.

## 이 러너가 남기는 것
매 트리거에 예측확률·확신등급·앵커 측면·발동 신호·발현 방향/분·ATR·진입가·배리어·해소시각·
결과(cont/fade/timeout)를 남긴다. **전 트리거를 남기므로 결과선택 편향이 구조상 불가능하다** --
"어느 사건이 해소되는가"를 결과가 정하는 문제(MASHT 에서 실제로 발생)가 여기서는 생기지 않는다.

⚠️주문을 내지 않는다. 공개 API 조회와 가상 원장 기록만 한다.

Usage:
    python scripts/live_eth_breakout_reversal_shadow_runner_20260908.py --once
    python scripts/live_eth_breakout_reversal_shadow_runner_20260908.py --loop
    python scripts/live_eth_breakout_reversal_shadow_runner_20260908.py --report
    python scripts/live_eth_breakout_reversal_shadow_runner_20260908.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import live_eth_breakout_features_20260908 as LF  # noqa: E402

ART = ROOT / "data/live/breakout_reversal_shadow_artifact"
STATE = ROOT / "data/live/breakout_reversal_shadow_state.json"
LEDGER = ROOT / "data/live/breakout_reversal_shadow_ledger.jsonl"
KLINES = "https://fapi.binance.com/fapi/v1/klines"
FDATA = "https://fapi.binance.com/futures/data"
SYMBOL, BTC_SYMBOL = "ETHUSDT", "BTCUSDT"
MET_EP = {"retail": ("globalLongShortAccountRatio", "longShortRatio"),
          "ttc": ("topLongShortAccountRatio", "longShortRatio"),
          "ttp": ("topLongShortPositionRatio", "longShortRatio"),
          "tkv": ("takerlongshortRatio", "buySellRatio")}
COST_TAKER_BP, COST_MAKER_BP = 10.0, 7.8
MAX_OPEN = 40                      # 폭주 가드일 뿐 -- 판정 자체엔 상한을 두지 않는다
MIN_BARS5 = 1200
RULE_ID = "breakout_reversal_ff075_p025_h1h_20260908"
_C: dict[str, Any] = {}


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {m}", flush=True)


def fetch_klines(symbol: str, interval: str = "5m", limit: int = 1500, retries: int = 3):
    """형성 중인 마지막 봉을 제거한다 -- 라이브는 **완결 봉만** 본다."""
    for k in range(retries):
        try:
            r = requests.get(KLINES, params={"symbol": symbol, "interval": interval,
                                             "limit": limit}, timeout=15)
            r.raise_for_status()
            kl = pd.DataFrame(r.json(), columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
            for c in ("open", "high", "low", "close", "volume", "taker_buy_base", "quote_volume"):
                kl[c] = kl[c].astype(float)
            kl["trades"] = kl["trades"].astype(int)
            kl["timestamp"] = pd.to_datetime(kl["open_time"], unit="ms")
            return kl[kl["close_time"] < int(time.time() * 1000)].reset_index(drop=True)
        except Exception as e:
            log(f"⚠️klines {symbol} {interval} 실패({k+1}/{retries}): {type(e).__name__}: {e}")
            time.sleep(2 * (k + 1))
    return None


def fetch_metrics(limit: int = 500):
    """4종 포지셔닝 메트릭을 5분 격자로. 학습은 binance_data/metrics 아카이브를 썼고
    라이브는 같은 값을 주는 futures/data 엔드포인트를 쓴다(컬럼 대응은 MET_EP)."""
    out, idx = {}, None
    for nm, (ep, col) in MET_EP.items():
        try:
            r = requests.get(f"{FDATA}/{ep}", params={"symbol": SYMBOL, "period": "5m",
                                                      "limit": limit}, timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            if df.empty:
                return None, None
            df["ts"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
            df = df.sort_values("ts").drop_duplicates("ts", keep="last")
            s = pd.Series(df[col].astype(float).to_numpy(), index=df["ts"])
            idx = s.index if idx is None else idx.union(s.index)
            out[nm] = s
        except Exception as e:
            log(f"⚠️metrics {nm} 실패: {type(e).__name__}: {e}")
            return None, None
    return {k: v.reindex(idx).ffill().to_numpy(float) for k, v in out.items()}, idx


def load_models():
    if "m" in _C:
        return _C["m"]
    import joblib
    meta = json.loads((ART / "meta.json").read_text())
    _C["m"] = (joblib.load(ART / "model.joblib"), meta)
    log(f"아티팩트 로드: {meta['rule_id']} · 피쳐 {len(meta['features'])} · 시드 {len(meta['seeds'])}개")
    return _C["m"]


def tier_of(conf: float, tiers: dict) -> str:
    return "강" if conf >= tiers["high"] else "중" if conf >= tiers["mid"] else \
           "약" if conf >= tiers["low"] else "미약"


def _strategy_fields(meta, entry, sgn, p, conf) -> dict[str, Any]:
    """사전등록 매매 규칙(meta.strategy)의 진입 정보 -- 종이거래로만 기록한다.

    ⚠️라벨(±0.8×ATR 대칭)을 그대로 매매하면 손익분기 승률이 70%인데 실측 정확도는 56~59%다.
      그래서 매매 브라켓은 **비대칭**(TP 20 / SL 50)이고 확신 게이트가 붙는다.
      라벨 축과 매매 축을 혼동하지 말 것 -- 정확도가 귀무를 이기는 것과 돈이 되는 것은 다르다.
    """
    st = meta.get("strategy")
    if not st:
        return {}
    side = 1.0 if ((p > 0.5) == (sgn > 0)) else -1.0     # 돌파면 발현 방향, 되돌림이면 반대
    # ⭐2026-09-08 게이트 제거(사용자 지정): gate_threshold=0.0 이라 전건이 통과한다.
    #   `strat_conf` 를 함께 남기므로 사후에 **어떤 임계로도** 재계산할 수 있다 --
    #   사전등록 임계는 meta.strategy.gate_threshold_prereg 에 보존돼 있다.
    return {"strat_id": st["id"], "strat_gate": bool(conf >= st["gate_threshold"]),
            "strat_conf": float(conf),
            "strat_gate_prereg": bool(conf >= st.get("gate_threshold_prereg", st["gate_threshold"])),
            "strat_side": "long" if side > 0 else "short",
            "strat_tp_px": entry * (1 + side * st["tp_bp"] / 1e4),
            "strat_sl_px": entry * (1 - side * st["sl_bp"] / 1e4),
            "strat_side_sign": side}


def load_state() -> dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            log("⚠️상태 파일 파손 -- 새로 시작")
    return {"watching": [], "positions": [], "seen_anchor_ts": [], "closed": 0}


def save_state(s: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False))
    tmp.replace(STATE)


def new_anchors(sig: pd.DataFrame, seen: list[str]) -> list[dict[str, Any]]:
    """직전 완결 봉이 first_fire 앵커인가. 백로그는 쫓지 않는다(09-04 원장 교훈)."""
    import build_eth_anchor_label_dataset_20260907 as B
    n = len(sig); out = []
    for side in ("bottom", "top"):
        fire = np.stack([sig[f"{side}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS], axis=1)
        idx = B.anchor_index(fire, "first_fire", 3)
        if len(idx) and int(idx[-1]) == n - 1:
            ts = str(pd.Timestamp(sig["timestamp"].iloc[-1]))
            if f"{ts}|{side}" in seen[-200:]:
                continue
            out.append({"anchor_utc": ts, "side": side, "bar_i": n - 1,
                        "signals": {s: int(fire[n - 1, j]) for j, s in enumerate(B.SIGNALS)},
                        "n_signals": int(fire[n - 1].sum()),
                        "atr_pct": float(sig["atr_pct"].iloc[-1])})
    return out


def append_ledger(rec: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cycle(s: dict[str, Any]) -> None:
    import build_eth_anchor_label_dataset_20260907 as B
    models, meta = load_models()
    kl = fetch_klines(SYMBOL); btc = fetch_klines(BTC_SYMBOL)
    kl1 = fetch_klines(SYMBOL, interval="1m", limit=1500)
    b1 = fetch_klines(BTC_SYMBOL, interval="1m", limit=1500)
    if kl is None or btc is None or kl1 is None or len(kl) < MIN_BARS5:
        log("⚠️klines 부족 -- 사이클 건너뜀"); return
    met, mts = fetch_metrics()
    if met is None:
        log("⚠️메트릭 없음 -- 사이클 건너뜀"); return

    ts5 = kl["timestamp"].to_numpy()
    O5 = kl["open"].to_numpy(float); C5 = kl["close"].to_numpy(float)
    H5 = kl["high"].to_numpy(float); L5 = kl["low"].to_numpy(float)
    V5 = kl["volume"].to_numpy(float); TB = kl["taker_buy_base"].to_numpy(float)
    bt5 = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts5)).ffill().to_numpy(float)
    ts1 = kl1["timestamp"].to_numpy()
    hi1 = kl1["high"].to_numpy(float); lo1 = kl1["low"].to_numpy(float); cl1 = kl1["close"].to_numpy(float)
    bcl = (b1.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts1)).ffill().to_numpy(float)
           if b1 is not None else None)
    F, LV, atr5 = LF.bar_features(C5, H5, L5, V5, TB, bt5)
    XS = LF.metric_features(met, mts, ts5)

    resolve(s, ts1, hi1, lo1, ts5, C5)

    # ---------- 1) 새 앵커 등록 ----------
    fund = B._load_funding()
    sig = B.compute_signals(kl, btc_df=btc, funding_df=fund)
    for a in new_anchors(sig, s["seen_anchor_ts"]):
        s["seen_anchor_ts"].append(f"{a['anchor_utc']}|{a['side']}")
        s["watching"].append({**a, "ref": None, "s0": None})
        log(f"앵커 {a['anchor_utc']} {a['side']} · 신호 {a['n_signals']}종 · 발현 감시 시작")
    s["seen_anchor_ts"] = s["seen_anchor_ts"][-400:]

    # ---------- 2) 발현 감시 (앵커 다음 봉 시가 기준 15분) ----------
    tsl = pd.DatetimeIndex(ts5); ts1i = pd.DatetimeIndex(ts1)
    still = []
    for w in s["watching"]:
        ai = tsl.get_indexer([pd.Timestamp(w["anchor_utc"])])[0]
        if ai < 0 or ai + 1 >= len(ts5):
            still.append(w); continue                       # 다음 봉 아직 미완결
        ref = float(O5[ai + 1])
        s0 = int(ts1i.get_indexer([pd.Timestamp(ts5[ai + 1])])[0])
        if s0 < 0:
            still.append(w); continue
        T = w["atr_pct"] * meta["t_mult"]
        up, dn = ref * (1 + T), ref * (1 - T)
        end = min(s0 + LF.NMOVE, len(ts1))
        seg_u = np.flatnonzero(hi1[s0:end] >= up); seg_d = np.flatnonzero(lo1[s0:end] <= dn)
        tu = int(seg_u[0]) if len(seg_u) else 1 << 30
        td = int(seg_d[0]) if len(seg_d) else 1 << 30
        if tu == td:                                        # 같은 분 양방향 -- 학습과 동일하게 버림
            log(f"   {w['anchor_utc']} 같은 분 양방향 → 폐기"); continue
        if tu == (1 << 30) and td == (1 << 30):
            if end - s0 < LF.NMOVE:
                still.append(w)                             # 창이 아직 안 찼다
            else:
                log(f"   {w['anchor_utc']} 15분 내 미발현 → 폐기")
            continue
        sgn = 1.0 if tu < td else -1.0
        tmin = min(tu, td)
        s1 = s0 + tmin
        bt = int(np.searchsorted(ts5, ts1[s1], side="right") - 1)
        fb = bt - 1
        if fb < 900 or bt >= len(ts5):
            log(f"   {w['anchor_utc']} 피쳐 창 부족(fb={fb}) → 폐기"); continue
        entry = ref * (1 + sgn * T)
        # ⭐2026-09-08 개정 2: 배리어가 **사건별 ATR 상대**다(P = k × atr_at_anchor).
        #   절대 0.25% 는 ATR 구간마다 난이도가 딴판이었다 -- 저ATR 시간청산 25.3% ·
        #   고ATR 돌파율 0.334(불균형). meta.barrier_mode 없으면 옛 절대값으로 되돌아간다.
        if meta.get("barrier_mode") == "atr_relative":
            P = w["atr_pct"] * float(meta["barrier_k_atr"])
        else:
            P = float(meta["barrier_pct"]) / 100.0
        ev = {"T_atr": T, "trig_min": float(tmin), "dir_up": 1.0 if sgn > 0 else 0.0,
              "atr_at_anchor": w["atr_pct"], "n_signals": float(w["n_signals"]),
              "side_bottom": 1.0 if w["side"] == "bottom" else 0.0, "signals": w["signals"]}
        pathf = LF.path_features(hi1, lo1, cl1, bcl, s0, tmin, ref, sgn, w["atr_pct"], T)
        levf = LF.level_features(LV, atr5, fb, entry, sgn)
        try:
            vec, _ = LF.assemble(meta["features"], F, XS, ts5, fb, pathf, levf, ev)
        except KeyError as e:
            log(f"   ⚠️피쳐 조립 실패 {w['anchor_utc']}: {e}"); continue
        X = np.nan_to_num(vec.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        p = float(np.mean([m.predict_proba(X)[0, 1] for m in models]))
        conf = abs(p - 0.5); tier = tier_of(conf, meta["confidence_tiers"])
        pos = {"rule_id": RULE_ID, "anchor_utc": w["anchor_utc"], "side": w["side"],
               "n_signals": w["n_signals"], "signals": [k for k, v in w["signals"].items() if v],
               "trigger_utc": str(pd.Timestamp(ts1[s1])), "trig_min": tmin,
               "dir_up": bool(sgn > 0), "atr_pct": w["atr_pct"], "T_atr": T,
               "entry_px": entry, "ref_px": ref, "p_breakout": p, "confidence": conf,
               "barrier_pct": P * 100,          # 이 사건에 실제로 쓴 배리어(%)
               "tier": tier, "call": "돌파" if p > 0.5 else "되돌림",
               "barrier_up": entry * (1 + P), "barrier_dn": entry * (1 - P),
               "s1_utc": str(pd.Timestamp(ts1[s1])), "bt_utc": str(pd.Timestamp(ts5[bt])),
               **_strategy_fields(meta, entry, sgn, p, conf),
               "deadline_utc": str(pd.Timestamp(ts5[bt]) + pd.Timedelta(minutes=5 * meta["horizon_bars"])),
               "opened_utc": datetime.now(timezone.utc).isoformat()}
        if len(s["positions"]) >= MAX_OPEN:
            log(f"   ⚠️미해소 {MAX_OPEN}건 초과 -- 기록만 하고 추적 생략"); append_ledger({**pos, "outcome": "skipped_max_open"}); continue
        s["positions"].append(pos)
        log(f"⭐판정 {pos['trigger_utc']} {w['side']} · 발현 {'상승' if sgn>0 else '하락'}"
            f"({tmin}분) · {pos['call']} p={p:.4f} [{tier}]")
    s["watching"] = still


def resolve(s: dict[str, Any], ts1, hi1, lo1, ts5, C5) -> None:
    """1분봉 first-touch. 미터치면 트리거 봉 +12봉 종가 부호로 시간청산."""
    _, meta = load_models()
    ts1i = pd.DatetimeIndex(ts1); ts5i = pd.DatetimeIndex(ts5)
    keep = []
    for p in s["positions"]:
        i0 = ts1i.get_indexer([pd.Timestamp(p["s1_utc"])])[0]
        if i0 < 0:
            keep.append(p); continue
        end = min(i0 + meta["horizon_bars"] * 5, len(ts1))
        su = np.flatnonzero(hi1[i0:end] >= p["barrier_up"])
        sd = np.flatnonzero(lo1[i0:end] <= p["barrier_dn"])
        tu = int(su[0]) if len(su) else 1 << 30
        td = int(sd[0]) if len(sd) else 1 << 30
        up_first = tu < td; dn_first = td < tu
        cont = up_first if p["dir_up"] else dn_first
        fade = dn_first if p["dir_up"] else up_first
        if cont or fade:
            k = min(tu, td)
            _close(s, p, "cont" if cont else "fade",
                   p["barrier_up"] if up_first else p["barrier_dn"], str(pd.Timestamp(ts1[i0 + k])),
                   _strategy_outcome(p, hi1, lo1, ts1, C5, ts5, meta))
            continue
        bi = ts5i.get_indexer([pd.Timestamp(p["bt_utc"])])[0]
        if bi >= 0 and bi + meta["horizon_bars"] < len(C5):
            x = bi + meta["horizon_bars"]
            _close(s, p, "timeout", float(C5[x]), str(pd.Timestamp(ts5[x])),
                   _strategy_outcome(p, hi1, lo1, ts1, C5, ts5, meta))
            continue
        keep.append(p)
    s["positions"] = keep


def _strategy_outcome(p, hi1, lo1, ts1, C5, ts5, meta) -> dict[str, Any]:
    """TP/SL 브라켓을 1분봉 first-touch 로 판정. 같은 분에 양쪽이면 **비관적(SL 우선)**."""
    st = meta.get("strategy")
    if not st or p.get("strat_side_sign") is None:
        return {}
    i0 = pd.DatetimeIndex(ts1).get_indexer([pd.Timestamp(p["s1_utc"])])[0]
    if i0 < 0:
        return {}
    end = min(i0 + st["horizon_bars"] * 5, len(ts1))
    side = float(p["strat_side_sign"]); tp_px = p["strat_tp_px"]; sl_px = p["strat_sl_px"]
    tph = (hi1[i0:end] >= tp_px) if side > 0 else (lo1[i0:end] <= tp_px)
    slh = (lo1[i0:end] <= sl_px) if side > 0 else (hi1[i0:end] >= sl_px)
    a = int(np.flatnonzero(tph)[0]) if tph.any() else 1 << 30
    b = int(np.flatnonzero(slh)[0]) if slh.any() else 1 << 30
    if a == (1 << 30) and b == (1 << 30):
        bi = pd.DatetimeIndex(ts5).get_indexer([pd.Timestamp(p["bt_utc"])])[0]
        if bi < 0 or bi + st["horizon_bars"] >= len(C5):
            return {}
        gross = (float(C5[bi + st["horizon_bars"]]) - p["entry_px"]) / p["entry_px"] * 1e4 * side
        oc = "timeout"
    else:
        oc = "tp" if a < b else "sl"          # 동시면 SL -- 비관적
        gross = st["tp_bp"] if a < b else -st["sl_bp"]
    return {"strat_outcome": oc, "strat_gross_bp": float(gross),
            "strat_net_bp": float(gross - st["cost_bp"]),
            "strat_win": bool(gross - st["cost_bp"] > 0)}


def _close(s, p, outcome, exit_px, exit_utc, strat=None) -> None:
    sgn = 1.0 if p["dir_up"] else -1.0
    gross = (exit_px - p["entry_px"]) / p["entry_px"] * 1e4 * sgn
    correct = (outcome == "cont") if p["p_breakout"] > 0.5 else (outcome == "fade")
    if outcome == "timeout":
        correct = (gross > 0) == (p["p_breakout"] > 0.5)
    rec = {**p, "outcome": outcome, "exit_px": exit_px, "exit_utc": exit_utc,
           "gross_bp": gross, "net_taker_bp": gross - COST_TAKER_BP,
           "net_maker_bp": gross - COST_MAKER_BP,
           "label_breakout": int(outcome == "cont") if outcome != "timeout" else int(gross > 0),
           "correct": bool(correct), "closed_utc": datetime.now(timezone.utc).isoformat(),
           **(strat or {})}
    append_ledger(rec)
    s["closed"] = s.get("closed", 0) + 1
    log(f"   해소 {p['trigger_utc']} → {outcome} · 총 {gross:+.1f}bp · "
        f"{'적중' if correct else '빗나감'} (판정 {p['call']} p={p['p_breakout']:.3f})")


def report() -> int:
    if not LEDGER.exists():
        print("원장 없음"); return 0
    R = pd.DataFrame([json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()])
    R = R[R["outcome"].isin(["cont", "fade", "timeout"])]
    if R.empty:
        print("해소된 건 없음"); return 0
    _, meta = load_models()
    pr = meta["prereg"]["cov100"]
    print(f"규칙 {RULE_ID} · 해소 {len(R):,}건 "
          f"({R['trigger_utc'].min()[:16]} ~ {R['trigger_utc'].max()[:16]})")
    days = max((pd.Timestamp(R['trigger_utc'].max()) - pd.Timestamp(R['trigger_utc'].min())).days, 1)
    print(f"   빈도 {len(R)/days:.1f}건/일 (사전등록 {pr['per_day']}건/일)")
    print(f"   적중률 {R['correct'].mean():.4f}  "
          f"(사전등록 VAL {pr['acc']['VAL']:.4f} / OOS {pr['acc']['OOS']:.4f} / HOLD {pr['acc']['HOLDOUT_SPENT']:.4f})")
    print(f"   ⚠️셔플 귀무 {pr['null']['VAL']:.4f}~{pr['null']['OOS']:.4f} -- 정확도는 이것과 함께 읽는다")
    print(f"   해소 내역 {dict(R['outcome'].value_counts())}")
    if "strat_net_bp" in R.columns:
        st = meta.get("strategy", {}); ng = st.get("prereg_nogate") or {}
        pr = (st.get("prereg") or {}).get("OOS", {})
        A = R[R["strat_net_bp"].notna()]
        print(f"\n   ⭐매매규칙 {st.get('id')} "
              f"(TP{st.get('tp_bp')}/SL{st.get('sl_bp')}·비용{st.get('cost_bp')}bp·손익분기승률 85.7%)")
        if st.get("status") == "suspended":
            print(f"      🔴매매 규칙 **보류**(2026-09-08) -- ATR 상대 배리어로 바꾸자 브라켓 격자가 "
                  f"{ng.get('grid_pass', 0)}/{ng.get('grid_total', 0)} 통과로 무너졌다. "
                  f"기록만 하고 판정 근거로 쓰지 않는다(최선칸도 {ng.get('bp_per_trade', 0):+.2f}bp/건).")
        elif st.get("gate_threshold", 0) <= 0:
            print(f"      ⚠️게이트 제거됨(사용자 지정) -- 전 트리거 기록. "
                  f"백테스트 기대 {ng.get('bp_per_trade', 0):+.2f}bp/건")
        if len(A):
            print(f"      [전건] {len(A)}건 · 건당 {A['strat_net_bp'].mean():+.2f}bp "
                  f"(백테스트 {ng.get('bp_per_trade', 0):+.2f}) · 승률 {A['strat_win'].mean()*100:.1f}% "
                  f"(백테스트 {ng.get('winrate', 0)*100:.1f}%) · 일 {A['strat_net_bp'].sum()/days:+.2f}bp")
            print(f"             해소 {dict(A['strat_outcome'].value_counts())}")
        # 사전등록 부분집합은 계속 따로 낸다 -- 게이트를 지워도 그 질문은 살아 있다
        if "strat_gate_prereg" in R.columns and st.get("status") != "suspended":
            S = A[A["strat_gate_prereg"] == True]
            thr = st.get("gate_threshold_prereg")
            print(f"      [사전등록 부분집합 |p-.5|≥{thr:.4f}] {len(S)}건/{len(A)}건 "
                  f"(커버 {len(S)/max(len(A),1)*100:.1f}% · 사전등록 28.0%)")
            if len(S):
                print(f"             건당 {S['strat_net_bp'].mean():+.2f}bp (사전등록 {pr.get('bp_per_trade', 0):+.2f}) · "
                      f"승률 {S['strat_win'].mean()*100:.1f}% · 일 {S['strat_net_bp'].sum()/days:+.2f}bp "
                      f"(사전등록 {pr.get('bp_per_day', 0):+.2f})")
    for t in ("강", "중", "약", "미약"):
        q = R[R["tier"] == t]
        if len(q):
            print(f"   확신 {t}: n={len(q):>4} 적중 {q['correct'].mean():.4f} 총 {q['gross_bp'].mean():+.2f}bp")
    return 0


def selftest() -> int:
    models, meta = load_models()
    assert len(meta["features"]) == 69, meta["features"]
    assert meta["anchor"] == "first_fire" and meta["t_mult"] == 0.75
    assert meta["barrier_mode"] == "atr_relative" and abs(meta["barrier_k_atr"] - 0.8) < 1e-9
    assert meta["horizon_bars"] == 12
    X = np.zeros((1, 69), np.float32)
    p = float(np.mean([m.predict_proba(X)[0, 1] for m in models]))
    assert 0.0 < p < 1.0, p
    for c, want in ((0.20, "강"), (0.06, "중"), (0.03, "약"), (0.001, "미약")):
        got = tier_of(c, meta["confidence_tiers"])
        assert got == want, f"{c} → {got} (기대 {want})"
    print(f"✅셀프테스트 통과 · 피쳐 69 · 영벡터 p={p:.4f} · 등급 임계 {meta['confidence_tiers']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.report:
        return report()
    s = load_state()
    if a.once:
        cycle(s); save_state(s)
        log(f"감시 {len(s['watching'])} · 미해소 {len(s['positions'])} · 누적해소 {s.get('closed',0)}")
        return 0
    if not a.loop:
        log("--once / --loop / --report / --selftest 중 하나를 지정한다"); return 2
    log(f"루프 시작 (주기 {a.interval}초) · 주문 없음")
    while True:
        try:
            cycle(s); save_state(s)
        except KeyboardInterrupt:
            log("중단"); save_state(s); return 0
        except Exception as e:
            log(f"⚠️사이클 예외: {type(e).__name__}: {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
