#!/usr/bin/env python3
"""코인별 RVOL **워커** (2026-09-11 전환 경보·탐지 → 2026-10-18 RVOL 전용) — 계산을 대시보드 응답 경로 밖으로.

🔴2026-10-18 전환 예고·탐지 제거(live_eth_breakout_detector_20260911 머리말). 발동 이력 로그(<coin>_breakout_detector_log.jsonl)도
  더 쓰지 않는다 -- 기록할 상태 변화가 없다. 옛 로그 파일은 지우지 않았다(전진 표본 기록).

⚠️저장소 규칙(2026-09-10 실장애에서 확정): 대시보드 요청 경로에서 계산을 인라인으로 돌리지
   않는다. 이 신호는 numpy 몇 줄이라 싸지만, 외부 API 를 때리므로 응답 경로에 두면 안 된다.

계약  data/live/<coin>_breakout_detector_state.json  현재 RVOL (원자적 tmp→rename)
      계산은 live_eth_breakout_detector_20260911.compute_signals 를 그대로 부른다 —
      화면 숫자와 워커 숫자가 갈라질 여지를 만들지 않는다.
주기  5분봉 신호다. 봉 마감 직후에 맞춰 돈다(기본 300초, 마감 +20초 정렬).
      ⚠️백테스트는 **마감된 봉** 기준이다 — 진행 중 봉을 쓰면 검증 안 한 체제가 된다.
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
import binance_ban_guard  # noqa: E402,F401 -- 2026-09-26 IP 밴 중엔 바이낸스 REST 를 안 내보낸다(공용 가드)
from live_eth_breakout_detector_20260911 import get_signals  # noqa: E402

# 2026-10-31 BO_SYMBOL 은 쉼표 목록(한 프로세스가 심볼을 차례로) -- 상태 파일은 코인별 옛 이름 그대로
SYMBOLS = [s for s in os.getenv("BO_SYMBOL", "ETHUSDT").upper().split(",") if s]


def state_path(sym: str) -> Path:
    return ROOT / "data" / "live" / f"{sym[:-4].lower()}_breakout_detector_state.json"
DEFAULT_INTERVAL = 300
BAR_OFFSET = 20          # 5분봉 마감 후 몇 초 뒤에 조회할지(마감 직후 REST 반영 지연 여유)


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {m}", flush=True)


def write_state(sym: str, p: dict) -> None:
    STATE = state_path(sym)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    out = {**p, "updated_utc": datetime.now(timezone.utc).isoformat()}
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, default=str))
    os.replace(tmp, STATE)                      # 반쪽 파일을 읽히지 않게


def cycle(sym: str) -> dict:
    t0 = time.time()
    p = get_signals(ttl=0, symbol=sym)          # 워커가 유일한 호출자라 캐시 무의미
    if not p.get("ok"):
        log(f"⚠️{sym} 조회 실패: {p.get('error')}")
        write_state(sym, p)
        return p
    write_state(sym, p)
    rv = p.get("rvol") or {}
    log(f"{sym} {p['timestamp'][11:16]} · {p['close']:.2f} · 세션 RVOL {rv.get('session')} {rv.get('session_label')} · {time.time()-t0:.1f}s")
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
        for sym in SYMBOLS:
            cycle(sym)
        return 0
    log(f"RVOL 워커 시작 (주기 {a.interval}초 · 봉 마감 +{BAR_OFFSET}초 정렬)")
    while True:
        for sym in SYMBOLS:
            try:
                cycle(sym)
            except Exception as e:              # noqa: BLE001 — 루프를 절대 죽이지 않는다(한 심볼 실패가 다른 심볼을 막지 않는다)
                log(f"⚠️{sym} 사이클 실패: {type(e).__name__}: {e}")
        _sleep_to_bar(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
