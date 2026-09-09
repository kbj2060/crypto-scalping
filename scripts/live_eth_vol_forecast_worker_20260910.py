#!/usr/bin/env python3
"""24시간 변동성 전망 **워커** -- 채점을 대시보드 응답 경로 밖으로 (2026-09-10).

⚠️이 저장소 규칙(2026-09-10 실장애에서 확정): **대시보드 요청 경로에서 모델을 인라인으로
   돌리지 않는다.** 지금 이 모델은 로지스틱 회귀라 싸지만, 아티팩트는 공유 자원이고 다른
   세션이 언제든 무겁게 바꾼다 -- 그날 극점 탐지기가 HGB(1.8MB) -> TabPFN(1.08GB)으로
   바뀌면서 `asyncio.to_thread` 기본 풀 16스레드가 고갈돼 대시보드 전체가 멈췄다.
   그래서 처음부터 워커로 만든다.

계약  `data/live/eth_vol_forecast_state.json` 에 페이로드를 **원자적으로**(tmp -> rename) 쓴다.
      계산은 기존 함수를 그대로 부른다 -- 화면 숫자와 워커 숫자가 갈라질 여지를 만들지 않는다.
주기  이 신호는 **시간봉**이라 5분마다 돌 이유가 없다. 기본 300초.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_vol_forecast_20260910 import compute_eth_vol_forecast  # noqa: E402

STATE = ROOT / "data" / "live" / "eth_vol_forecast_state.json"
DEFAULT_INTERVAL = 300


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {m}", flush=True)


def write_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    out = {**payload, "updated_utc": datetime.now(timezone.utc).isoformat()}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, default=str))
    os.replace(tmp, STATE)          # 원자적 교체 -- 대시보드가 반쪽 파일을 읽지 않게


def cycle() -> dict:
    t0 = time.time()
    p = compute_eth_vol_forecast()
    write_state(p)
    log(f"{p.get('grade') or '-'} · p {p.get('proba')} · DVOL {p.get('dvol')} "
        f"· RV24 {p.get('rv24')} · VRP {p.get('vrp')} · {time.time()-t0:.2f}s"
        + ("" if p.get("available", True) else f" · ⚠️{p.get('error')}"))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    a = ap.parse_args()
    if not a.loop:
        cycle(); return 0
    log(f"변동성 전망 워커 시작 (주기 {a.interval}초)")
    while True:
        try:
            cycle()
        except Exception as e:  # noqa: BLE001 -- 루프를 절대 죽이지 않는다
            log(f"⚠️사이클 실패: {type(e).__name__}: {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
