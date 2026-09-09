#!/usr/bin/env python3
"""극점 탐지기 **워커** -- 채점을 대시보드 응답 경로 밖으로 (2026-09-10, 사용자 선택 1번).

## 왜 옮기나
대시보드가 이 모델을 **인라인으로** 돌리고 있었다(캐시 60초마다 `compute_eth_extreme_detector()`).
이 저장소의 다른 모델 카드는 전부 러너가 상태 파일을 쓰고 대시보드는 읽기만 한다 --
V자·돌파/되돌림·증거신호 전부. 극점만 예외였다.

인라인이 문제가 되는 지점: 모델을 HGB -> TabPFN v3 로 바꾸면 **0.49초 -> 5.14초**(10.6배,
2026-09-10 서버 실측). 그 GPU 를 V자 TabPFN·증거신호 메타라벨이 이미 공유한다.
그런데 이 모델은 **5분봉마다 한 번만** 새 점수가 필요하다 -- 60초 캐시는 봉당 다섯 번 헛돈다.
워커로 옮기면 대시보드 비용이 0 이 되고 TabPFN 이 공짜가 된다.

## 계약
`data/live/eth_extreme_detector_state.json` 에 `compute_eth_extreme_detector()` 페이로드를
**원자적으로**(tmp -> rename) 쓴다. `updated_utc` 를 얹어 대시보드가 신선도를 판단하게 한다.
계산 자체는 기존 함수를 그대로 부른다 -- 화면 숫자와 워커 숫자가 갈라질 여지를 만들지 않는다.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_extreme_detector_20260909 import compute_eth_extreme_detector  # noqa: E402

STATE = ROOT / "data" / "live" / "eth_extreme_detector_state.json"
DEFAULT_INTERVAL = 60


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
    p = compute_eth_extreme_detector()
    write_state(p)
    log(f"{p.get('subText', '-')} · 등급 {p.get('grade') or '-'} "
        f"· p {p.get('proba')} · {time.time() - t0:.2f}s"
        + ("" if p.get("available", True) else f" · ⚠️{p.get('error')}"))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    a = ap.parse_args()
    if not a.loop:
        cycle(); return 0
    log(f"루프 시작 (주기 {a.interval}초) · 상태 {STATE}")
    while True:
        try:
            cycle()
        except Exception as e:                      # noqa: BLE001 -- 워커는 죽지 않는다
            log(f"⚠️사이클 실패: {type(e).__name__}: {e}")
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
