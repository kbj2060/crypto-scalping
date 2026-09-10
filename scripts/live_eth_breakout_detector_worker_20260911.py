#!/usr/bin/env python3
"""횡보→추세 전환 경보·탐지 **워커** (2026-09-11) — 채점을 대시보드 응답 경로 밖으로.

⚠️저장소 규칙(2026-09-10 실장애에서 확정): 대시보드 요청 경로에서 계산을 인라인으로 돌리지
   않는다. 이 신호는 numpy 몇 줄이라 싸지만, 외부 API 를 때리므로 응답 경로에 두면 안 된다.

계약  data/live/eth_breakout_detector_state.json  현재 상태 (원자적 tmp→rename)
      data/live/eth_breakout_detector_log.jsonl   발동 이력 (전진 표본 축적용, append)
      계산은 live_eth_breakout_detector_20260911.compute_signals 를 그대로 부른다 —
      화면 숫자와 워커 숫자가 갈라질 여지를 만들지 않는다.
주기  5분봉 신호다. 봉 마감 직후에 맞춰 돈다(기본 300초, 마감 +20초 정렬).
      ⚠️백테스트는 **마감된 봉** 기준이다 — 진행 중 봉을 쓰면 검증 안 한 체제가 된다.

이력  상태가 바뀔 때만 기록한다(같은 상태 반복은 안 쌓는다). 나중에 이 로그로
      "경보 후 실제 전환이 왔는가"를 전진 검증할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_breakout_detector_20260911 import get_signals  # noqa: E402

STATE = ROOT / "data" / "live" / "eth_breakout_detector_state.json"
LOG = ROOT / "data" / "live" / "eth_breakout_detector_log.jsonl"
DEFAULT_INTERVAL = 300
BAR_OFFSET = 20          # 5분봉 마감 후 몇 초 뒤에 조회할지(마감 직후 REST 반영 지연 여유)


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {m}", flush=True)


def write_state(p: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    out = {**p, "updated_utc": datetime.now(timezone.utc).isoformat()}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, default=str))
    os.replace(tmp, STATE)                      # 반쪽 파일을 읽히지 않게


def append_log(p: dict) -> None:
    """상태가 바뀐 순간만 남긴다 — 전진 검증용 표본."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": p.get("timestamp"), "close": p.get("close"), "state": p.get("state"),
           "volexp": p.get("volexp"), "compressed": p.get("compressed"),
           "alert_lit": p.get("alert", {}).get("lit"),
           "alert_on": [x["name"] for x in p.get("alert", {}).get("lights", []) if x["on"]],
           "detect_on": p.get("detect", {}).get("on"),
           "detect_count": p.get("detect", {}).get("count"),
           "logged_utc": datetime.now(timezone.utc).isoformat()}
    with LOG.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


_LAST = {"state": None, "ts": None}


def cycle() -> dict:
    t0 = time.time()
    p = get_signals(ttl=0)                      # 워커가 유일한 호출자라 캐시 무의미
    if not p.get("ok"):
        log(f"⚠️조회 실패: {p.get('error')}")
        write_state(p)
        return p
    write_state(p)
    changed = (p["state"] != _LAST["state"]) and (p["timestamp"] != _LAST["ts"])
    if changed:
        append_log(p)
        _LAST.update(state=p["state"], ts=p["timestamp"])
    lit = [x["name"] for x in p["alert"]["lights"] if x["on"]]
    log(f"{p['state']:8s} · {p['timestamp'][11:16]} · {p['close']:.2f} "
        f"· volexp {p['volexp']:.2f} · 경보 {p['alert']['lit']}등{lit if lit else ''} "
        f"· 탐지 {p['detect']['count']}/{len(p['detect']['signals'])}"
        f"{' ★기록' if changed else ''} · {time.time()-t0:.1f}s")
    return p


def _sleep_to_bar(interval: int) -> None:
    """다음 5분봉 마감 +BAR_OFFSET 초에 맞춰 잔다(마감된 봉만 쓰기 위해)."""
    if interval != 300:
        time.sleep(interval)
        return
    now = time.time()
    nxt = (int(now) // 300 + 1) * 300 + BAR_OFFSET
    time.sleep(max(nxt - now, 5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    a = ap.parse_args()
    if not a.loop:
        cycle()
        return 0
    log(f"전환 탐지 워커 시작 (주기 {a.interval}초 · 봉 마감 +{BAR_OFFSET}초 정렬)")
    while True:
        try:
            cycle()
        except Exception as e:                  # noqa: BLE001 — 루프를 절대 죽이지 않는다
            log(f"⚠️사이클 실패: {type(e).__name__}: {e}")
        _sleep_to_bar(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
