#!/usr/bin/env python3
"""**E|r| 조건부 사이징 배수** — 큰 움직임이 예상되는 자리에 크게, 아니면 줄인다. (2026-09-15)

근거(실계좌 72왕복 · 진입은 배포 시스템 것 · 09-15):
  🔴현행 명목 ↔ E|r|백분위 스피어만 **−0.426** — **큰 E|r| 에 작게 걸고 있었다.**
  구간별 단위당 순손익: 하위50% +10.34 / 50~80% +14.32 / **상위20% +53.18bp**(단조).
  ⭐**현행 × E|r|백분위**(축소전용): 명목 **42%** 로 손익 **90%**(＄461.53/＄511.79),
    명목당 **+10.84 → +23.54bp**(2.2배), t **1.03 → 4.06**, 최대낙폭 **＄444 → ＄30**(1/15).
  ⚠️**하한을 두면 나빠진다**(×max(q,0.3) ＄376 · ×max(q,0.5) ＄369) — 하한이 나쁜 큰 거래를
    살려두기 때문이다. **하한 없는 순수 ×q 가 맞다.**
  전체 기록: docs/homer/README.md §5.36-R · [[direction_evr_gate_real_ledger_20260915]]

⭐**모델은 줄이기만 한다**(저장소 규약) — 배수는 [0,1] 이고 기존 상한 체계(생존·순자산·정책)의
  **뒤에** 곱한다. 키우려면 사람이 레버리지를 올려야 하고 그건 이 스크립트의 일이 아니다.

🔴방향은 이 모델이 정하지 않는다. 같은 아티팩트의 `dir` 분류기는 실원장에서 적중 **47.2%**(동전
  아래)로 **해로웠다**(모델−사용자 −31.73bp). 여기서는 **`evr` 회귀기만** 쓴다.

데이터: `data.binance.vision` 일별(어제까지) + 바이낸스 API 최근분. `_z` 창이 2016봉(7일)이라
그만큼의 5분 OI·롱숏비가 필요한데 API 는 41.7시간만 주므로 **둘을 이어붙여야** 한다.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data/models/direction_4h_top5_20260915"
PANEL = ROOT / "data/binance_vision/panel/ETHUSDT.parquet"
STATE = ROOT / "data/live/eth_evr_sizing_state.json"
WARM = 8064          # 백분위 확장창 최소 관측(28일)


def _mod(rel: str, name: str):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp)
    sv = sys.argv; sys.argv = [name]
    try:
        sp.loader.exec_module(m)
    except SystemExit:
        pass
    sys.argv = sv
    return m


def recent_from_api(sym: str = "ETHUSDT", hours: int = 40) -> pd.DataFrame | None:
    """API 로 최근 5분봉 + 메트릭. 메트릭은 최근 41.7시간만 주므로 그 안에서만 쓴다."""
    import urllib.request
    end = int(time.time() * 1000); start = end - hours * 3600 * 1000
    try:
        u = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=5m&startTime={start}&limit=1500"
        k = json.load(urllib.request.urlopen(u, timeout=20))
        kd = pd.DataFrame(k).iloc[:, :11]
        kd.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                      "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        kd = kd.astype(float)
        kd["timestamp"] = pd.to_datetime(kd["open_time"], unit="ms")
        u2 = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={sym}&period=5m&limit=500"
        oi = pd.DataFrame(json.load(urllib.request.urlopen(u2, timeout=20)))
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], unit="ms")
        oi["sum_open_interest"] = oi["sumOpenInterest"].astype(float)
        u3 = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={sym}&period=5m&limit=500"
        ls = pd.DataFrame(json.load(urllib.request.urlopen(u3, timeout=20)))
        ls["timestamp"] = pd.to_datetime(ls["timestamp"], unit="ms")
        ls["count_long_short_ratio"] = ls["longShortRatio"].astype(float)
        u4 = f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={sym}&period=5m&limit=500"
        tp = pd.DataFrame(json.load(urllib.request.urlopen(u4, timeout=20)))
        tp["timestamp"] = pd.to_datetime(tp["timestamp"], unit="ms")
        tp["sum_toptrader_long_short_ratio"] = tp["longShortRatio"].astype(float)
        u5 = f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={sym}&period=5m&limit=500"
        tk = pd.DataFrame(json.load(urllib.request.urlopen(u5, timeout=20)))
        tk["timestamp"] = pd.to_datetime(tk["timestamp"], unit="ms")
        tk["sum_taker_long_short_vol_ratio"] = tk["buySellRatio"].astype(float)
        d = kd[["timestamp", "high", "low", "close", "volume", "trades", "taker_buy_base"]]
        for x, c in ((oi, "sum_open_interest"), (ls, "count_long_short_ratio"),
                     (tp, "sum_toptrader_long_short_ratio"), (tk, "sum_taker_long_short_vol_ratio")):
            d = d.merge(x[["timestamp", c]], on="timestamp", how="left")
        d["count_toptrader_long_short_ratio"] = np.nan     # 패널에 있으나 피쳐 미사용
        return d
    except Exception as exc:
        print(f"  API 최근분 실패: {type(exc).__name__} {exc}", file=sys.stderr)
        return None


def score(use_api: bool = True) -> dict:
    import joblib
    X = _mod("scripts/research_direction_event_expansion_20260915.py", "X")
    M = joblib.load(ART / "models.joblib"); evr, cols = M["evr"], M["cols"]
    base = pd.read_parquet(PANEL); base["timestamp"] = pd.to_datetime(base["timestamp"])
    src = "panel"
    if use_api:
        r = recent_from_api()
        if r is not None and len(r):
            base = pd.concat([base, r], ignore_index=True)
            base = base.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
            src = f"panel+api({len(r)}봉)"
    base = base.reset_index(drop=True)
    P = X._features(base, "ETH", False, "2022-01-01")
    Xm = P[cols].to_numpy(np.float32)
    ok = np.isfinite(Xm).all(1)
    pred = np.full(len(P), np.nan); pred[ok] = evr.predict(Xm[ok])
    # 🔴백분위는 **직전까지의** 예측 분포 기준(인과). shift(1) 로 당일 제외 — 검증판과 동일 규약.
    #   ⇒ q[i] 는 pred[**i−1**] 의 순위다. 둘을 **같은 봉에서** 읽어야 보고가 어긋나지 않는다.
    #   🔴최근 봉은 API 메트릭이 아직 안 붙어 pred 가 NaN 일 수 있다 — 유효한 마지막 봉을 쓴다.
    q = pd.Series(pred).expanding(WARM).rank(pct=True).shift(1).to_numpy()
    good = np.flatnonzero(np.isfinite(q) & np.r_[False, np.isfinite(pred[:-1])])
    if not len(good):
        raise RuntimeError("유효한 E|r| 봉이 없다")
    i = int(good[-1])                      # 배수의 근거 봉 = i−1
    src_bar = P["timestamp"].iloc[i - 1]
    lag_min = (pd.Timestamp.utcnow().tz_localize(None) - src_bar).total_seconds() / 60
    mult = float(np.clip(q[i], 0.0, 1.0))         # ⭐줄이기만 한다: [0,1]
    return {"ts": str(P["timestamp"].iloc[i]), "signal_bar": str(src_bar),
            "evr": float(pred[i - 1]), "evr_q": float(q[i]),
            "size_mult": round(mult, 4), "lag_min": round(lag_min, 1),
            "source": src, "bars": int(len(P)),
            "computed_at": pd.Timestamp.utcnow().isoformat()}


def main() -> int:
    ap = argparse.ArgumentParser(description="E|r| 조건부 사이징 배수")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            s = score(use_api=not a.no_api)
            STATE.parent.mkdir(parents=True, exist_ok=True)
            STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1))
            print(f"{s['ts']} · 신호봉 {s['signal_bar']}(지연 {s['lag_min']:.0f}분) · "
                  f"E|r| {s['evr']:.4f} · 백분위 {s['evr_q']:.3f} · "
                  f"**배수 {s['size_mult']:.3f}** · {s['source']}", flush=True)
        except Exception as exc:
            print(f"실패 {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if not a.loop:
            return 0
        time.sleep(a.interval)


def _selfcheck() -> None:
    """실원장 72왕복에서 «×q» 가 문서의 숫자를 재현하는지 — 배수 정의가 맞는지 확인한다."""
    s = score(use_api=False)
    assert 0.0 <= s["size_mult"] <= 1.0, f"배수가 [0,1] 밖: {s['size_mult']}"
    assert np.isfinite(s["evr"]), "evr 이 NaN 인데 배수가 나왔다 — 인덱스 정렬 버그"
    X = _mod("scripts/research_direction_event_expansion_20260915.py", "X")
    import joblib
    M = joblib.load(ART / "models.joblib")
    base = pd.read_parquet(PANEL); base["timestamp"] = pd.to_datetime(base["timestamp"])
    P = X._features(base, "ETH", False, "2022-01-01")
    Xm = P[M["cols"]].to_numpy(np.float32); ok = np.isfinite(Xm).all(1)
    pr = np.full(len(P), np.nan); pr[ok] = M["evr"].predict(Xm[ok])
    P["q"] = pd.Series(pr).expanding(WARM).rank(pct=True).shift(1).to_numpy()
    rows = [json.loads(l) for l in (ROOT / "data/live/account_round_trips.jsonl").read_text().splitlines() if l.strip()]
    tr = [r for r in rows if r.get("closed") and r.get("side") in ("LONG", "SHORT")]
    T = pd.DataFrame([{"ts": pd.to_datetime(r["entry_time"], unit="ms").floor("5min"),
                       "side": 1 if r["side"] == "LONG" else -1,
                       "mv": (float(r["exit_price"]) / float(r["entry_price"]) - 1) * 1e4,
                       "N": float(r["entry_price"]) * float(r["max_qty"])} for r in tr])
    T = T.merge(P[["timestamp", "q"]], left_on="ts", right_on="timestamp", how="left").dropna(subset=["q"])
    rn = T.side * T.mv - 5.88
    cur = float((T.N * rn / 1e4).sum()); gated = float((T.N * T.q * rn / 1e4).sum())
    eff_c = cur / T.N.sum() * 1e4; eff_g = gated / (T.N * T.q).sum() * 1e4
    print(f"자체점검: 배수 {s['size_mult']:.3f}(∈[0,1]) · 원장 {len(T)}건 · "
          f"현행 ${cur:+.2f}({eff_c:+.2f}bp) → ×q ${gated:+.2f}({eff_g:+.2f}bp · 명목 {(T.N*T.q).sum()/T.N.sum():.0%})")
    assert eff_g > eff_c, f"×q 가 명목당 효율을 못 올렸다 {eff_g:.2f} vs {eff_c:.2f}"
    assert 0.35 < (T.N * T.q).sum() / T.N.sum() < 0.50, "명목 축소율이 문서(42%)와 다르다"
    print("자체점검 통과")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
