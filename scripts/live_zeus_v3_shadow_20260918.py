#!/usr/bin/env python3
"""**Zeus Baseline v3 섀도우 기록기** (2026-09-18). 🔴주문을 내지 않는다. 기록만 한다.

사양은 `docs/zeus/README.md §2`(동결) · 판정은 `docs/zeus/shadow_prereg_v3_20260918.md`.
아티팩트 `data/live/zeus_v3_shadow_20260918/`(파리티 편차 0.000e+00 로 내보냄).

⭐**시작 시 파리티 관문**: 라이브 경로로 만든 96열을 연구 프레임과 겹치는 구간에서 대조해
편차가 크면 **뜨지 않는다**. 사전 등록 P 기준(불일치 0건)이 여기서 걸린다.
🔴피쳐는 봇과 «독립적으로» 재계산한다(사용자 결정 2026-09-18) -- 라이브 봇과 격리하기 위해서다.
   대신 위 관문으로 미세 차이를 잡는다.
🔴OI/LSR 은 라이브 API 가 최근 41.7시간만 준다 ⇒ 저장 패널(`data/binance_vision/panel`,
   `live_evr_gate_worker` 가 갱신)에 오늘분을 이어 붙인다. **패널이 멈추면 이 러너도 멈춘다.**

수익은 «판정»이 아니라 «누적»이다 -- 사전 등록대로 체결 436건 전까지 중간 판정을 하지 않는다.
"""
from __future__ import annotations
import argparse, json, os, sys, time, urllib.request
from collections import deque
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(os.environ.get("ZEUS_ROOT") or Path.home() / "crypto-scalping")
for p in (ROOT, ROOT / "scripts", ROOT / "trading_bot_modules"):
    sys.path.insert(0, str(p))
from features.engineering import FeatureEngineer                   # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm             # noqa: E402

ART = ROOT / "data/live/zeus_v3_shadow_20260918"
STATE = ART / "state.json"
LEDGER = ART / "ledger.jsonl"
PANEL = ROOT / "data/binance_vision/panel"
REF_FRAME = ROOT / "tmp/omega461_longwindow_20260917/features_with_regime_2022_2026_realfunding.parquet"
FAPI = "https://fapi.binance.com"
OHLC = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
        "trades", "taker_buy_base", "taker_buy_quote"]
PARITY_TOL = 5e-3        # 표준화 전 원시 열 상대오차. 이보다 크면 뜨지 않는다.


def log(*a): print(time.strftime("[%m-%d %H:%M:%S]"), *a, flush=True)


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _live_tail(sym: str, n: int = 1000) -> pd.DataFrame:
    """최근 n봉 + OI/LSR. 실패는 조용히 넘기지 않는다 -- NaN 이면 그 봉은 판정 불가다."""
    k = _get(f"{FAPI}/fapi/v1/klines?symbol={sym}&interval=5m&limit={n}")
    d = pd.DataFrame(k, columns=["open_time", "open", "high", "low", "close", "volume",
                                 "close_time", "quote_volume", "trades", "taker_buy_base",
                                 "taker_buy_quote", "ignore"])
    d["timestamp"] = pd.to_datetime(d["open_time"], unit="ms")
    for c in OHLC[1:]:
        d[c] = d[c].astype(float)
    out = d[OHLC].iloc[:-1].copy()                    # ⭐마지막 봉은 «미완결» -- 버린다
    for col, (ep, fld) in {
        "sum_open_interest": ("openInterestHist", "sumOpenInterest"),
        "count_long_short_ratio": ("globalLongShortAccountRatio", "longShortRatio"),
        "sum_toptrader_long_short_ratio": ("topLongShortPositionRatio", "longShortRatio"),
    }.items():
        m = pd.DataFrame(_get(f"{FAPI}/futures/data/{ep}?symbol={sym}&period=5m&limit=500"))
        m["timestamp"] = pd.to_datetime(m["timestamp"], unit="ms")
        m[col] = m[fld].astype(float)
        out = out.merge(m[["timestamp", col]], on="timestamp", how="left")
    return out


def _panel(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(PANEL / f"{sym}USDT.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_localize(None)
    d = d.drop_duplicates("timestamp").sort_values("timestamp")
    typ = (d.high + d.low + d.close) / 3.0
    d["open"] = d.close.shift(1).fillna(d.close)
    d["quote_volume"] = d.volume * typ
    d["taker_buy_quote"] = d.get("taker_buy_base", pd.Series(np.nan, index=d.index)) * typ
    return d.reset_index(drop=True)


def build_frame(live: bool, tail_days: int = 30) -> pd.DataFrame:
    """저장 패널 + (라이브면) 오늘분을 잇고 **연구 빌더와 같은 순서로** 피쳐를 만든다."""
    pe, pb = _panel("ETH"), _panel("BTC")
    fresh = (pd.Timestamp.utcnow().tz_localize(None) - pe.timestamp.max()).days
    if fresh > 3:
        raise RuntimeError(f"패널이 {fresh}일 낡았다 (최신 {pe.timestamp.max()}) -- "
                           f"live_evr_gate_worker 를 먼저 확인할 것")
    eth = pe[OHLC + ["sum_open_interest", "sum_toptrader_long_short_ratio",
                     "count_long_short_ratio"]].copy()
    if live:
        eth = pd.concat([eth, _live_tail("ETHUSDT")], ignore_index=True)
        bt = _live_tail("BTCUSDT")[["timestamp", "close", "volume", "quote_volume"]]
    else:
        bt = None
    eth = eth.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    eth = eth[eth.timestamp >= eth.timestamp.max() - pd.Timedelta(days=tail_days)].reset_index(drop=True)
    eth["sum_open_interest_value"] = eth.sum_open_interest * eth.close
    b = pb[["timestamp", "close", "volume", "quote_volume"]].rename(
        columns={"close": "close_btc", "volume": "volume_btc", "quote_volume": "quote_volume_btc"})
    if bt is not None:
        bt = bt.rename(columns={"close": "close_btc", "volume": "volume_btc",
                                "quote_volume": "quote_volume_btc"})
        b = pd.concat([b, bt], ignore_index=True).drop_duplicates("timestamp", keep="last")
    eth = eth.merge(b, on="timestamp", how="left")
    fr = pd.DataFrame(_get(f"{FAPI}/fapi/v1/fundingRate?symbol=ETHUSDT&limit=500"))
    fr["timestamp"] = pd.to_datetime(fr["fundingTime"], unit="ms")
    fr["last_funding_rate"] = fr["fundingRate"].astype(float)
    eth = pd.merge_asof(eth.sort_values("timestamp"),
                        fr[["timestamp", "last_funding_rate"]].sort_values("timestamp"),
                        on="timestamp", direction="backward")
    F = FeatureEngineer().process(
        eth.drop(columns=["close_btc", "volume_btc", "quote_volume_btc"]).copy(),
        eth[["timestamp", "close_btc", "volume_btc", "quote_volume_btc"]].copy())
    if "timestamp" not in F.columns:
        F["timestamp"] = eth["timestamp"].to_numpy()
    return _with_raw_state12(F)


def parity_gate(art) -> float:
    """⭐라이브 경로 96열 vs 연구 프레임. 겹치는 구간에서 대조해 편차가 크면 뜨지 않는다."""
    if not REF_FRAME.exists():
        log("⚠️연구 프레임 없음 -- 파리티 관문 생략(서버에는 있어야 한다)"); return -1.0
    ref = pd.read_parquet(REF_FRAME, columns=["timestamp"] + art["base_cols"])
    ref["timestamp"] = pd.to_datetime(ref["timestamp"])
    cur = build_frame(live=False, tail_days=25)
    j = cur[["timestamp"] + art["base_cols"]].merge(ref, on="timestamp", suffixes=("_l", "_r"))
    j = j[j.timestamp >= j.timestamp.max() - pd.Timedelta(days=10)]
    if len(j) < 500:
        raise RuntimeError(f"파리티 대조 구간이 {len(j)}봉뿐 -- 검증 불가")
    worst, who = 0.0, ""
    for c in art["base_cols"]:
        a, b = j[f"{c}_l"].to_numpy(float), j[f"{c}_r"].to_numpy(float)
        s = max(np.nanstd(b), 1e-9)
        d = float(np.nanmax(np.abs(a - b)) / s)
        if d > worst:
            worst, who = d, c
    log(f"⭐파리티 관문: {len(j):,}봉 · 최대 상대편차 {worst:.2e} ({who})")
    if worst > PARITY_TOL:
        raise RuntimeError(f"파리티 실패 {worst:.2e} > {PARITY_TOL} ({who}) -- 섀도우를 띄우지 않는다")
    return worst


class Model:
    def __init__(self):
        a = torch.load(ART / "model.pt", map_location="cpu", weights_only=False)
        self.art = a
        self.ms = []
        for sd in a["state_dicts"]:
            m = tabm.ThreeHeadTabM(a["input_dim"], cfg=tabm.CFG)
            m.load_state_dict(sd); m.eval(); self.ms.append(m)

    def predict(self, frame: pd.DataFrame):
        x = tabm._standardize_apply(tabm._base_input(frame, self.art["base_cols"]), self.art["scaler"])
        D = Q = None
        with torch.no_grad():
            for m in self.ms:
                o = m(torch.from_numpy(x))
                d = torch.softmax(o["direction"], -1).mean(1).numpy()
                q = torch.softmax(o["quality"], -1).mean(1).numpy()
                D = d if D is None else D + d
                Q = q if Q is None else Q + q
        return D / len(self.ms), Q / len(self.ms)


def load_state(spec):
    if STATE.exists():
        s = json.loads(STATE.read_text())
        s["buf"] = deque(s["buf"], maxlen=spec["rollq_window"])
        return s
    return {"buf": deque(maxlen=spec["rollq_window"]), "pos": None, "last_ts": None,
            "n_fill": 0, "n_reject": 0, "closed": [],
            "started_utc": str(pd.Timestamp.utcnow().tz_localize(None))}


def save_state(s):
    """⭐`ops_watchdog.check_shadow_runner` 스키마를 함께 쓴다 -- 감시기 로직을 고치는 대신
    러너가 규약을 따른다. 감시기는 last_decided_bar_utc·ledger·positions·pending 을 본다."""
    STATE.write_text(json.dumps({
        **s, "buf": list(s["buf"]),
        "last_decided_bar_utc": s.get("last_ts"),
        "ledger": s.get("closed", []),          # 감시기는 len() 만 본다
        "positions": [s["pos"]] if s.get("pos") else [],
        "pending": [],
        "started_utc": s.get("started_utc"),
        "rule": "zeus_v3_shadow_20260918 TP1.5/SL0.7 q0.983 1slot noorders",
    }, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true", help="파리티 관문만 돌리고 종료")
    A = ap.parse_args()
    meta = json.loads((ART / "meta.json").read_text()); spec = meta["spec"]
    assert meta["orders"].startswith("NONE"), "주문 금지 계약이 아티팩트에 없다"
    log(f"Zeus v3 섀도우 · {meta['model_id']} · TP{spec['tp']*100:g}%/SL{spec['sl']*100:g}% · "
        f"q={spec['rollq_q']} 창{spec['rollq_window']} · 1슬롯 · 🔴주문 없음")
    mdl = Model()
    dev = parity_gate(mdl.art)
    if A.dry:
        log(f"dry: 파리티 {dev:.2e} -- 종료"); return 0
    st = load_state(spec)
    ART.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            F = build_frame(live=True)
            row = F.iloc[-1]                      # 방금 «완결된» 봉
            ts = str(row["timestamp"])
            if ts == st.get("last_ts"):
                time.sleep(20); continue
            D, Q = mdl.predict(F.tail(1))
            da = int(D[0].argmax())
            qf = float(Q[0][da] if da > 0 else Q[0][0])
            thr = float(np.quantile(list(st["buf"]), spec["rollq_q"])) if len(st["buf"]) >= 50 else None
            if da != 0:
                st["buf"].append(qf)              # ⭐임계값을 «쓴 뒤에» 넣는다(인과)
            side = 0
            if da != 0 and thr is not None and qf >= thr:
                side = 1 if da == 1 else -1
            rec = {"ts": ts, "close": float(row["close"]), "high": float(row["high"]),
                   "low": float(row["low"]), "da": da, "qf": qf, "thr": thr, "side": side,
                   "pos": bool(st["pos"]), "buf": len(st["buf"])}
            p = st["pos"]
            if p:                                  # 보유 중 -- intrabar 고·저가로 배리어 판정
                e, s_ = p["entry"], p["side"]
                up, dn = (row["high"] - e) / e, (row["low"] - e) / e
                g_hi, g_lo = (up, dn) if s_ > 0 else (-dn, -up)
                hit = "sl" if g_lo <= -spec["sl"] else ("tp" if g_hi >= spec["tp"] else None)
                if hit:
                    r = -spec["sl"] if hit == "sl" else spec["tp"]
                    rec |= {"event": "exit", "reason": hit, "ret_bp": r * 1e4,
                            "hold_bars": p["bars"] + 1, "entry": e, "entry_ts": p["ts"]}
                    st["pos"] = None; st["n_fill"] += 1
                    st.setdefault("closed", []).append({"ts": ts, "ret_bp": r * 1e4, "reason": hit})
                else:
                    p["bars"] += 1; rec |= {"event": "hold", "hold_bars": p["bars"]}
            elif side:
                st["pos"] = {"entry": float(row["close"]), "side": side, "ts": ts, "bars": 0}
                rec |= {"event": "entry"}
            elif da != 0 and thr is not None and qf >= thr:
                st["n_reject"] += 1; rec |= {"event": "reject_slot"}
            with LEDGER.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            st["last_ts"] = ts; save_state(st)
            if rec.get("event") in ("entry", "exit"):
                log(f"{rec['event']} {ts} side={side or (p or {}).get('side')} "
                    f"체결누적 {st['n_fill']} 거절 {st['n_reject']}")
        except Exception as e:                     # 조용히 넘기지 않는다 -- 원장에 남긴다
            log(f"🔴오류 {type(e).__name__}: {e}")
            with LEDGER.open("a") as f:
                f.write(json.dumps({"ts": str(pd.Timestamp.utcnow()), "event": "error",
                                    "err": f"{type(e).__name__}: {e}"}) + "\n")
            if not A.loop:
                return 1
            time.sleep(60)
        if A.once:
            return 0
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
