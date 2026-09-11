#!/usr/bin/env python3
"""model_indicator_history 영속화 자체점검 (2026-09-12). 프레임워크 없이 그냥 실행한다.

왜 있나: 이 띠는 프로세스 메모리 deque 에만 있었고, 배포 워처가 main 전진마다 대시보드를
재기동해서(2026-09-11 실측 하루 12회) 48칸 × 5분 = 4시간짜리 띠가 늘 비어 있었다.
사용자 증상은 "새로고침하면 있던 과거가 사라진다" — 브라우저는 라이브 틱으로 40칸까지
누적하는데 새로고침 후 씨앗(서버 deque)은 10칸뿐이었기 때문이다.

여기서 지키는 계약:
  · 파일이 없거나 깨졌으면 [] (기동을 막지 않는다)
  · 창(4h) 밖 샘플은 버린다 (이틀 전 값을 '최근 4시간'이라 그리면 화면이 거짓말한다)
  · maxlen 으로 절단하되 **최신이 끝에** 남는다
  · sampled_at 이 없거나 깨진 행은 건너뛴다
  · 원자적 교체 — .tmp 잔여물이 없다

실행: python test/test_model_indicator_history_persistence_20260912.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("dash_srv", ROOT / "dashboard" / "server.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    sample, cap = m.MODEL_INDICATOR_SAMPLE_SECONDS, m.MODEL_INDICATOR_HISTORY_MAX
    now = datetime.now(timezone.utc)

    def row(age_s: float, i: int) -> dict:
        return {"sampled_at": (now - timedelta(seconds=age_s)).isoformat(),
                "microstructure": {"i": i}, "tail_risk": {}}

    with tempfile.TemporaryDirectory() as td:
        m.MODEL_INDICATOR_HISTORY_PATH = pathlib.Path(td) / "mih.json"
        load, save = m.load_model_indicator_history, m.save_model_indicator_history

        assert load() == [], "파일이 없으면 빈 리스트"
        m.MODEL_INDICATOR_HISTORY_PATH.write_text("{not json")
        assert load() == [], "깨진 파일이면 빈 리스트"

        # 절단만 본다 -- 60행 전부 창 안쪽(최대 6000s)이라 경계가 끼지 않는다.
        # 창 경계(정확히 4h)에 걸친 행을 쓰면 load() 안의 now() 가 몇 us 뒤라 흔들린다.
        inside = [row(100 * i, i) for i in range(60, 0, -1)]
        save(inside)
        assert len(load()) == cap, f"maxlen 절단 실패: {len(load())}"
        assert load()[-1]["microstructure"]["i"] == 1, "최신 샘플이 끝에 와야 한다"

        # 창 필터만 본다 -- 명백히 밖(2일) + 명백히 안(100~300s)
        save([row(2 * 86400, 99)] + inside[-3:])
        assert len(load()) == 3, f"창 밖 샘플 제거 실패: {len(load())}"

        save([{"bad": "no sampled_at"}] + inside[-2:])
        assert len(load()) == 2, "형식 불량 행을 건너뛰어야 한다"

        assert not list(pathlib.Path(td).glob("*.tmp")), "원자적 교체 후 .tmp 가 남으면 안 된다"

    print("자체점검 7/7 통과 — 없음·깨짐·절단·순서·창밖제거·형식불량·tmp정리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
