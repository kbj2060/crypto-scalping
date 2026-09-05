#!/usr/bin/env python3
"""V자반등 섀도우 원장 리셋 (2026-09-05, 1회용).

왜: 커밋 336bd36 으로 진입 규약이 바뀌었다(백로그 일괄 진입 금지 -- 직전 완결 봉 호출만 받는다).
09-05 재구성 감사에서 옛 원장 34건 중 4건이 불일치했고(88.2%, 기준 95%), 원인이 전부 묵은 신호를
같은 마크가격으로 일괄 진입시킨 것이었다. 계측 방식이 다른 두 구간을 한 원장에 섞으면 백테스트
대조가 불가능하므로 끊는다. 옛 원장은 data/live/archive/ 에 보존한다(진단용).

⚠️러너가 정지된 상태에서만 실행할 것 -- 돌고 있으면 다음 사이클이 메모리 상태로 덮어쓴다.
사용: python scripts/ops/reset_v_rebound_shadow_ledger_20260905.py
"""
import datetime
import json
import os
import shutil
import sys

P = "data/live/v_rebound_econ_shadow_state.json"
REASON_TAG = "backlog_entry_blocked(336bd36)"


def main() -> int:
    old = json.load(open(P))
    if old.get("positions"):
        print("ABORT: 오픈 포지션 %d건 -- 리셋하면 추적이 끊긴다" % len(old["positions"])); return 1
    if any(h.get("config") == REASON_TAG for h in (old.get("config_history") or [])):
        print("SKIP: 이미 리셋됨 (%s)" % old.get("reset_utc")); return 0
    now = datetime.datetime.now(datetime.timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%SZ"); now_s = now.isoformat()
    os.makedirs("data/live/archive", exist_ok=True)
    arch = "data/live/archive/v_rebound_econ_shadow_state_%s_pre_backlogfix.json" % ts
    shutil.copy(P, arch)
    hist = list(old.get("config_history") or [])
    hist.append({"config": REASON_TAG, "since_utc": now_s, "ledger_len_at_change": 0,
                 "note": "직전 완결봉 호출만 진입. 이전 원장은 백로그 일괄 진입이 섞여 계측 방식이 다르므로 이어 쓰지 않는다."})
    new = {"positions": [], "ledger": [], "consec_loss": 0, "started_utc": now_s, "version": 1,
           "skipped": {"stale_call": 0, "slots_full": 0}, "missed_bars": 0, "last_decided_bar_utc": None,
           "reset_utc": now_s, "archived_ledger": arch, "config_history": hist,
           "reset_reason": "백로그 일괄 진입 금지(336bd36). 진입 규약 변경으로 이전 원장(마감 %d건)과 섞을 수 없다. 옛 원장 보존: %s" % (len(old.get("ledger", [])), arch)}
    tmp = P + ".tmp"
    json.dump(new, open(tmp, "w"), ensure_ascii=False, indent=2)
    os.replace(tmp, P)
    print("ARCHIVED %s (%d bytes, 마감 %d건)" % (arch, os.path.getsize(arch), len(old.get("ledger", []))))
    print("RESET ledger=0 open=0 started=%s" % now_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
