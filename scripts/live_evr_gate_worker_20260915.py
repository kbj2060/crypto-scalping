#!/usr/bin/env python3
"""**E|r| 게이트 워커** — 「지금 큰 움직임이 예상되는가」만 띄운다. (2026-09-15)

🔴🔴**방향은 말하지 않는다.** 같은 날 다른 세션이 **실계좌 72왕복**으로 갈랐다(호메로스 §5.36-R):
  구간            n   사용자방향   **모델방향**   무조건롱   사용자적중   **모델적중**
  전체           72    +20.32     **−11.41**   +10.55    **83.3%**   **47.2%**
  게이트 상위20%  15    **+53.18**    +2.00     +11.12      87%        53%
**모델−사용자 −31.73bp**이고 격차가 **게이트가 고른 가장 좋은 자리에서 −51.18 로 최대**다 —
게이트가 찾은 기회를 방향 분류기가 정확히 **반대로** 쓴다. 적중 47.2% = 동전 아래.
이건 §5.36 H 의 「고 E|r| 봉 적중 50.54% vs 벽 53.14%」와 「모델−롱 세 구간 전부 음수」를
실원장이 독립 확인한 것이다. ⇒ **아티팩트의 `dir` 모델은 쓰지 않는다. `evr` 만 쓴다.**
살아있는 구조: **방향 ← 사람 · 언제 ← 이 게이트 · 어느쪽 ← 모델 ❌**

⚠️이 저장소 규칙(2026-09-10 실장애): **대시보드 요청 경로에서 모델을 인라인으로 돌리지 않는다.**
여기 모델은 HGB 4개(E|r| 1 + 방향 3)이고 20자산을 돌므로 반드시 워커여야 한다.

⭐무엇을 띄우나 — **사건 트리거 하나**다:
  1. 매 봉 `ê = E|r| 모델 예측`(24시간 뒤 기대 변동폭 bp)
  2. `ê > 인과 확장창 90분위` 면 **발동** (봉의 10%만)
  3. 끝. **방향은 말하지 않는다**(위 참조).
게이트의 근거: 두 독립 설계에서 짝지은 증분 **+8.25 / +7.44bp/일 · CI 둘 다 0배제** ·
2025~26 에서 더 강함(+10.13). MDD 를 −47.4% → −8.4% 로 줄인다. **손실 차단기**이지
수익 생성기가 아니다.
🔴**매매에 연결돼 있지 않다.** 표시 전용이다.

계약  `data/live/direction_gate_state.json` 에 원자적으로(tmp→rename) 쓴다.
성능  패널 전체 재빌드는 자산당 ~15초(20자산 300초)라 주기를 못 맞춘다 ⇒ **꼬리 6,000봉만**
      계산한다. 피쳐 최대 롤링이 2,016봉이라 마지막 행은 전체 계산과 **같아야** 하고,
      `--selfcheck` 가 그 파리티를 실제로 확인한다(이게 이 최적화의 유일한 안전장치다).
신선도 `metrics` 라이브 API 는 최근 41.7시간만 준다 ⇒ 하루 한 번 vision 일별 파일로 패널을
      D−1 까지 메우고(`build_binance_vision_panel_20260915`), 오늘 분은 REST 로 잇는다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/live/evr_gate_state.json"
ART = ROOT / os.environ.get("DIR_ARTIFACT", "data/models/direction_1d_top10_20260915")
TAIL = 6000                 # 피쳐 최대 롤링 2,016봉의 ~3배
DEFAULT_INTERVAL = 300


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {m}", flush=True)


def _mod(rel: str, name: str):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp)
    argv, sys.argv = sys.argv, ["x"]
    try:
        sp.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


RX = _mod("scripts/research_direction_event_expansion_20260915.py", "RX")
SH = _mod("scripts/live_direction_gate_shadow_20260915.py", "SH")


def tail_frame(asset: str, live: bool) -> pd.DataFrame:
    """저장 패널 꼬리 + (선택) 라이브 REST → 연구와 **같은** 피쳐 빌더."""
    raw = pd.read_parquet(RX.BV_PANEL / f"{asset}USDT.parquet").tail(TAIL)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    if live:
        lv = SH.fetch_live(f"{asset}USDT")
        raw = pd.concat([raw, lv[[c for c in lv.columns if c in raw.columns]]], ignore_index=True)
    raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    raw = raw.tail(TAIL).reset_index(drop=True)
    return RX._features(raw, asset, False, "2000-01-01")


def cycle(art, man, hist, live: bool) -> dict:
    H = int(man["horizon_bars"]); GQ = float(man["gate_quantile"])
    cols = art["cols"]
    rows, err = [], []
    for A in man["assets"]:
        try:
            p = tail_frame(A, live)
            x = p[cols].to_numpy(np.float32)[-1:]
            if not np.isfinite(x).any():
                err.append(A); continue
            ehat = float(np.exp(art["evr"].predict(x)[0]))
            seed = np.array(hist.get(A, {}).get("pred", []), dtype=float)
            thr = float(np.exp(np.quantile(seed, GQ))) if len(seed) >= 500 else None
            fired = thr is not None and ehat > thr
            # 🔴`art["dir"]` 는 **의도적으로 호출하지 않는다** — §5.36-R 실원장에서 적중 47.2%
            #   (동전 아래)이고 게이트가 고른 좋은 자리일수록 더 나빴다(−51.18bp).
            rows.append({"asset": A, "ts": str(p["timestamp"].iloc[-1]),
                         "evr_bp": round(ehat, 1), "thr_bp": round(thr, 1) if thr else None,
                         "ratio": round(ehat / thr, 3) if thr else None, "fired": bool(fired)})
        except Exception as e:                      # 자산 하나가 죽어도 나머지는 보인다
            err.append(f"{A}:{type(e).__name__}")
    fired = sorted([r for r in rows if r["fired"]], key=lambda r: -r["ratio"])
    return {"ok": len(rows) > 0, "n_assets": len(rows), "n_fired": len(fired),
            "assets": rows, "fired": fired, "errors": err,
            "direction_note": ("방향은 말하지 않는다 — 분류기는 실계좌 72왕복에서 적중 47.2%"
                               "(§5.36-R). 언제만 말한다."),
            "horizon_bars": H, "gate_quantile": GQ, "artifact": man["name"],
            "artifact_sha": man["models_sha256"][:16], "status": man["status"],
            "tone": "bad" if fired else "neutral",
            # 🔴subText 는 **안정 키**여야 한다(대시보드 규약 §5-1 — 설명 맵의 키가 이 문자열이다).
            #   개수는 meterNote 로 뺀다. 여기에 숫자를 섞으면 설명이 영원히 매칭되지 않는다.
            "subText": "발동" if fired else "미발동",
            "meterNote": (f"{len(rows)}자산 중 {len(fired)}종 · 최대 {fired[0]['asset']} "
                          f"{fired[0]['ratio']:.2f}배") if fired
                         else f"{len(rows)}자산 전부 임계 아래"}


def write_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    out = {**payload, "updated_utc": datetime.now(timezone.utc).isoformat()}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, default=str))
    os.replace(tmp, STATE)          # 원자적 교체 — 대시보드가 반쪽 파일을 읽지 않게


def refresh_panel(man) -> None:
    """vision 일별 파일로 패널을 D−1 까지 메운다(라이브 metrics 는 41.7h 뿐이라 필수)."""
    B = _mod("scripts/build_binance_vision_panel_20260915.py", "BV")
    end = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for A in man["assets"]:
        f = RX.BV_PANEL / f"{A}USDT.parquet"
        last = pd.to_datetime(pd.read_parquet(f, columns=["timestamp"])["timestamp"].iloc[-1])
        start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if start > end:
            continue
        old = pd.read_parquet(f)
        B.OUT = RX.BV_PANEL
        B.build(A, start, end, 12)
        new = pd.read_parquet(f)
        pd.concat([old, new]).drop_duplicates("timestamp", keep="last") \
            .sort_values("timestamp").reset_index(drop=True).to_parquet(f)
    log("패널 갱신 완료")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--offline", action="store_true", help="REST 없이 저장 패널만")
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    import joblib
    art = joblib.load(ART / "models.joblib")
    man = json.loads((ART / "manifest.json").read_text())
    hist = json.loads((ART / "evr_history.json").read_text())
    log(f"{man['name']} · status={man['status']} · 자산 {len(man['assets'])} · "
        f"게이트 상위{(1-man['gate_quantile'])*100:.0f}% · 지평 {man['horizon_bars']}봉 "
        f"· 🔴표시 전용(매매 미연결)")
    day = None
    while True:
        try:
            today = datetime.now(timezone.utc).date()
            if not a.no_refresh and not a.offline and day != today:
                refresh_panel(man); day = today
            t0 = time.time()
            p = cycle(art, man, hist, live=not a.offline)
            write_state(p)
            log(f"발동 {p['n_fired']}/{p['n_assets']} · {time.time()-t0:.1f}s" + (f" · ⚠️{p['errors'][:3]}" if p["errors"] else ""))
        except Exception as e:
            log(f"ERR {type(e).__name__}: {e}")
            write_state({"ok": False, "error": str(e), "tone": "neutral", "subText": "오류"})
        if not a.loop:
            break
        time.sleep(a.interval)
    return 0


def _selfcheck() -> None:
    """⭐꼬리 절단 파리티 — 이 최적화의 **유일한** 안전장치다.

    꼬리 6,000봉으로 계산한 마지막 행이 전체 488,160봉 계산과 같아야 한다. 피쳐 최대 롤링이
    2,016봉이므로 이론상 같아야 하지만, 하나라도 expanding/cumsum 이 섞이면 조용히 어긋난다."""
    import joblib
    man = json.loads((ART / "manifest.json").read_text())
    cols = man["features"]
    full = RX.panel("ETH", since="2022-01-01")
    cut = tail_frame("ETH", live=False)
    assert str(full["timestamp"].iloc[-1]) == str(cut["timestamp"].iloc[-1]), "마지막 봉이 다르다"
    a_ = full[cols].to_numpy(float)[-1]
    b_ = cut[cols].to_numpy(float)[-1]
    bad = [(c, x, y) for c, x, y in zip(cols, a_, b_)
           if not (np.isnan(x) and np.isnan(y)) and not np.isclose(x, y, rtol=1e-6, atol=1e-9)]
    assert not bad, f"절단 파리티 깨짐 {len(bad)}개: {bad[:3]}"
    art = joblib.load(ART / "models.joblib")
    assert list(art["cols"]) == list(cols), "아티팩트 피쳐 순서 ≠ manifest"
    pf = float(art["evr"].predict(full[cols].to_numpy(np.float32)[-1:])[0])
    pc = float(art["evr"].predict(cut[cols].to_numpy(np.float32)[-1:])[0])
    assert abs(pf - pc) < 1e-6, f"예측이 다르다 {pf} vs {pc}"
    print(f"자체점검 통과 — 절단 파리티 {len(cols)}피쳐 일치 · 예측 {np.exp(pf):.1f}bp")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
