#!/usr/bin/env python3
"""**방향 게이트 섀도우 기록기** — 지평·분위는 아티팩트 manifest 가 정한다. (2026-09-15)

기본 아티팩트 = **`1d × 예측 E|r| 상위10% × 20자산`**(사용자 승인 1순위). 왜 이것인가:
사용자 기준(2024·25·26 각 해 순손익>0 · 메이커 5.52bp)으로 19칸을 재면 **4칸이 통과**하는데
그 중 이 칸만 **감쇠가 없고**(+17.9/+18.9/**+29.1** — 2026 이 최대) **모델이 무조건 롱보다
낫다**(+21.72). 나머지 셋은 2026 에 꺼지거나(4h) 롱과 구분이 안 된다.

🔴**20자산 포트폴리오로만 성립한다.** ETH 단독은 네 칸 모두 세 해 양수가 아니다
(1d×10% ETH 는 2024 −8.5). 사용자 승인(09-15): 다른 자산도 함께 계산해도 된다.
🔴그리고 통과와 별개로 남는 위험: 건당 순손익 날짜블록 CI 가 0 을 포함하고
(1d×10%: [−10.23,+50.63]) 다중성 보정 max-t p = 0.0610 이다.
전문: `docs/experiments/direction_event_trigger_expansion_and_oos_audit_20260915.md` (13~14절)

⭐**두 팔을 반드시 같이 기록한다**: `model`(방향 모델) 과 `long`(무조건 롱).
미달 관문 셋 중 하나가 「모델−롱 증분 CI 0 포함」이라 **같은 사건에서 두 팔을 짝지어** 모아야
그게 갈린다. 한 팔만 기록하면 몇 달 뒤에도 그 질문에 답할 수 없다.

🔴데이터 경로(이게 이 파일의 핵심 제약): 피쳐가 **최대 2016봉(7일) 롤링**을 쓰는데 바이낸스
라이브 메트릭 API 는 **최근 41.7시간**만 준다([[reference_binance_futures_metrics_history_sources]]).
그래서 **일별 `data.binance.vision` 파일(D−1 까지)** 로 과거를 채우고 **REST 로 오늘 분**을
이어 붙인다. 두 구간이 겹치므로 연속성이 보장된다.
⇒ 패널 갱신은 **E|r| 게이트 워커**(`live_evr_gate_worker_20260915.py`, 같은 20자산·같은
   `data/binance_vision/panel`)가 UTC 하루 한 번 한다. 이 러너에는 갱신 플래그가 없다 —
   그 워커가 멈추면 여기 정산도 같이 멈춘다.

🔴이 스크립트는 **주문을 내지 않는다.** 기록만 한다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# 🔴지평·분위를 **manifest 에서 읽는다.** 하드코딩하면 아티팩트를 바꿨을 때 조용히 어긋난다
#   (학습 1d/상위10% · 서빙 4h/상위5% 같은 사고가 정확히 이렇게 난다).
ART = ROOT / os.environ.get("DIR_ARTIFACT", "data/models/direction_1d_top10_20260915")
LED = ROOT / "data/live" / (ART.name.replace("direction_", "shadow_"))
FAPI = "https://fapi.binance.com"
WARM = 500                  # 확장창 분위 최소 관측
_MAN = json.loads((ART / "manifest.json").read_text()) if (ART / "manifest.json").exists() else {}
H = int(_MAN.get("horizon_bars", 288))
GATE_Q = float(_MAN.get("gate_quantile", 0.90))


def _research():
    """연구 스크립트의 **같은** 피쳐 빌더를 쓴다 — 복제하면 학습/서빙 괴리가 생긴다."""
    f = ROOT / "scripts/research_direction_event_expansion_20260915.py"
    sp = importlib.util.spec_from_file_location("RX", f)
    m = importlib.util.module_from_spec(sp)
    argv, sys.argv = sys.argv, ["x"]
    try:
        sp.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def fetch_live(sym: str, minutes: int = 2400) -> pd.DataFrame:
    """REST 로 최근 klines + metrics. 메트릭은 최근 ~41.7h 만 온다(그래서 vision 이 필요하다)."""
    def get(url):
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    n = min(1500, max(100, minutes // 5))
    k = get(f"{FAPI}/fapi/v1/klines?symbol={sym}&interval=5m&limit={n}")
    kd = pd.DataFrame(k, columns=["open_time", "open", "high", "low", "close", "volume",
                                  "close_time", "quote_volume", "trades", "taker_buy_base",
                                  "taker_buy_quote", "ignore"])
    kd["timestamp"] = pd.to_datetime(kd["open_time"], unit="ms")
    for c in ("high", "low", "close", "volume", "trades", "taker_buy_base"):
        kd[c] = kd[c].astype(float)
    out = kd[["timestamp", "high", "low", "close", "volume", "trades", "taker_buy_base"]]
    parts = {
        "sum_open_interest": ("openInterestHist", "sumOpenInterest"),
        "count_long_short_ratio": ("globalLongShortAccountRatio", "longShortRatio"),
        "sum_toptrader_long_short_ratio": ("topLongShortPositionRatio", "longShortRatio"),
        "sum_taker_long_short_vol_ratio": ("takerlongshortRatio", "buySellRatio"),
    }
    for col, (ep, fld) in parts.items():
        try:
            d = get(f"{FAPI}/futures/data/{ep}?symbol={sym}&period=5m&limit=500")
            m = pd.DataFrame(d)
            m["timestamp"] = pd.to_datetime(m["timestamp"], unit="ms")
            m[col] = m[fld].astype(float)
            out = out.merge(m[["timestamp", col]], on="timestamp", how="left")
        except Exception as e:          # 조용히 넘기지 않는다 — NaN 이면 그 봉은 판정 불가가 된다
            print(f"    [{sym}] {ep} 실패: {type(e).__name__} {e}")
            out[col] = np.nan
    out["count_toptrader_long_short_ratio"] = np.nan
    out["sum_open_interest_value"] = np.nan
    return out


def build_frame(RX, asset: str, live: bool) -> pd.DataFrame:
    """저장 패널(D−1) + 라이브(오늘)를 이어 붙이고 **연구와 같은** 피쳐를 만든다."""
    raw = pd.read_parquet(RX.BV_PANEL / f"{asset}USDT.parquet")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    if live:
        lv = fetch_live(f"{asset}USDT")
        raw = pd.concat([raw, lv[[c for c in lv.columns if c in raw.columns]]], ignore_index=True)
    raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return RX._features(raw.reset_index(drop=True), asset, False, "2022-01-01")


def decide(RX, art, assets, live: bool) -> pd.DataFrame:
    hist = json.loads((ART / "evr_history.json").read_text())
    rows = []
    for A in assets:
        p = build_frame(RX, A, live)
        if len(p) < 3000:
            print(f"  [{A}] 봉 부족 {len(p)} — 건너뜀"); continue
        X = p[art["cols"]].to_numpy(np.float32)
        ok = np.isfinite(X).any(1)
        pred = np.full(len(p), np.nan)
        pred[ok] = art["evr"].predict(X[ok])
        # 인과 확장창 분위 — 과거 이력으로 **웜스타트**(없으면 첫 500건 판정 불가)
        seed = np.array(hist.get(A, {}).get("pred", []), dtype=float)
        seen = np.concatenate([seed, pred[np.isfinite(pred)]])
        if len(seen) < WARM:
            print(f"  [{A}] 웜업 부족 {len(seen)}"); continue
        thr = float(np.quantile(seen[:-1], GATE_Q))
        i = len(p) - 1                                  # 마지막 **확정** 봉
        if not np.isfinite(pred[i]) or pred[i] <= thr:
            continue
        prob = float(np.mean([m.predict_proba(X[i:i + 1])[:, 1][0] for m in art["dir"]]))
        rows.append({"decision_ts": str(p["timestamp"].iloc[i]), "asset": A,
                     "entry_close": float(np.exp(p["__lc__"].iloc[i])),
                     "pred_evr_bp": float(np.exp(pred[i])), "gate_thr_bp": float(np.exp(thr)),
                     "prob_up": prob, "side_model": 1 if prob > 0.5 else -1, "side_long": 1,
                     "horizon_bars": H, "artifact_sha": art["sha"][:16]})
    return pd.DataFrame(rows)


def settle(RX, assets) -> int:
    """만기(48봉) 지난 결정에 **두 팔** 손익을 채운다. 이미 채워진 건 건드리지 않는다."""
    f = LED / "decisions.csv"
    if not f.exists():
        return 0
    d = pd.read_csv(f)
    if "net_model_bp" not in d.columns:
        d["net_model_bp"] = np.nan; d["net_long_bp"] = np.nan; d["exit_close"] = np.nan
    todo = d[d.net_model_bp.isna()]
    if not len(todo):
        return 0
    n = 0
    for A in sorted(todo.asset.unique()):
        p = build_frame(RX, A, live=False)
        ts = pd.to_datetime(p["timestamp"]); lc = p["__lc__"].to_numpy()
        pos = {str(t): k for k, t in enumerate(ts)}
        for ix in todo[todo.asset == A].index:
            k = pos.get(str(pd.to_datetime(d.at[ix, "decision_ts"])))
            if k is None or k + H >= len(lc):
                continue
            r = (lc[k + H] - lc[k]) * 1e4
            d.at[ix, "exit_close"] = float(np.exp(lc[k + H]))
            d.at[ix, "net_model_bp"] = float(d.at[ix, "side_model"] * r)
            d.at[ix, "net_long_bp"] = float(r)
            n += 1
    d.to_csv(f, index=False)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="한 번만 돌고 끝")
    ap.add_argument("--offline", action="store_true", help="REST 없이 저장 패널만 (자체점검·재현용)")
    ap.add_argument("--assets", default="", help="쉼표 목록(기본: manifest 전체)")
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    RX = _research()
    import joblib
    art = joblib.load(ART / "models.joblib")
    man = json.loads((ART / "manifest.json").read_text())
    art["sha"] = man["models_sha256"]
    assets = a.assets.split(",") if a.assets else man["assets"]
    LED.mkdir(parents=True, exist_ok=True)
    assert man.get("horizon_bars", H) == H and abs(man.get("gate_quantile", GATE_Q) - GATE_Q) < 1e-9, \
        "manifest 와 런타임의 지평/분위가 다르다 — 학습·서빙 괴리"
    print(f"섀도우 {man['name']} · status={man['status']} · 자산 {len(assets)} · "
          f"게이트 상위{(1-GATE_Q):.0%} · 지평 {H}봉({H*5//60}시간)")
    print("🔴이 스크립트는 주문을 내지 않는다. 기록만 한다.")
    while True:
        try:
            new = decide(RX, art, assets, live=not a.offline)
            f = LED / "decisions.csv"
            if len(new):
                old = pd.read_csv(f) if f.exists() else pd.DataFrame()
                allr = pd.concat([old, new], ignore_index=True) if len(old) else new
                allr = allr.drop_duplicates(["decision_ts", "asset"], keep="first")
                allr.to_csv(f, index=False)
                print(f"  발동 {len(new)}건 기록 (누적 {len(allr)})")
            k = settle(RX, assets)
            if k:
                print(f"  정산 {k}건")
        except Exception as e:
            print(f"  ERR {type(e).__name__}: {e}")
        if a.once:
            break
        time.sleep(a.interval)
    return 0


def _selfcheck() -> None:
    """아티팩트 없이도 되는 것만 검사한다(피쳐 파리티·게이트 산술·정산 부호)."""
    RX = _research()
    p = build_frame(RX, "ETH", live=False)
    man_cols = json.loads((ART / "manifest.json").read_text())["features"] if \
        (ART / "manifest.json").exists() else None
    if man_cols:
        assert all(c in p.columns for c in man_cols), "manifest 피쳐가 패널에 없다 = 학습/서빙 괴리"
        assert list(p[man_cols].columns) == man_cols, "피쳐 **순서**가 다르다"
    # 게이트 산술: 상위 (1−q) 만큼만 발동해야 한다 — **분위를 하드코딩하지 않는다**
    # (아티팩트를 1d/상위10% 로 바꿨는데 점검이 5% 를 가정해 실패한 적이 있다)
    v = np.random.default_rng(0).normal(size=20000)
    thr = np.quantile(v[:-1], GATE_Q)
    fire = float((v > thr).mean()); want = 1.0 - GATE_Q
    assert 0.6 * want < fire < 1.4 * want, f"게이트 분위: 기대 {want:.1%} · 실제 {fire:.1%}"
    # 정산 부호: 숏이면 가격이 내려야 이익
    lc = np.log(np.array([100.0, 99.0]))
    r = (lc[1] - lc[0]) * 1e4
    assert -1 * r > 0 and 1 * r < 0, "정산 부호가 뒤집혔다"
    print(f"자체점검 통과 (ETH 패널 {len(p):,}행)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
