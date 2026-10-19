"""워커 상태 파일 읽기 -- 신선도 판정까지(worker_payload)와 그 두 도우미.

2026-10-19 server.py 에서 내용 그대로 옮겼다(함수 본문 바이트 동일). server.py 와 orders.py(크기 가늠자)가 같이 쓴다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _age_min(ts: Any) -> float | None:
    """UTC 문자열 -> 지금까지 경과 분. 원장은 tz 표기가 없는 UTC 문자열이다."""
    try:
        dt = datetime.fromisoformat(str(ts).replace(" ", "T"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - dt).total_seconds() / 60.0, 1)


def worker_payload(path: Path, max_age_min: float, *, ts_field: str = "updated_utc",
                   require_ok: bool = False, stamp_available: bool = False,
                   bare_missing: bool = False, extra_missing: dict | None = None) -> dict[str, Any]:
    """워커 상태 파일을 신선도까지 판정해 돌려준다.

    ⚠️인라인 폴백은 두지 않는다. 폴백을 두면 워커가 죽어도 표는 정상으로 보이고 대신
      대시보드가 5초씩 느려진다(그게 이 구조를 만든 이유다) -- 죽었으면 죽었다고 보여야 한다.
    `available` 은 워커가 아니라 **여기서** 찍는다. 신선도를 판정하는 쪽이 찍어야 워커
      버전이 달라도 계약이 안 깨진다.
    `require_ok` 는 워커가 외부 조회에 실패한 사이클(`ok: False`)을 오류로 떨어뜨린다 --
      그때 옛 톤을 그대로 보이면 화면이 «지금 조용하다»고 거짓말을 한다.
    """
    # `extra_missing` 은 **결측/조회실패에만** 얹는다. stale 은 워커가 쓴 값을 그대로 보여주므로
    # 빈 history/times 로 덮으면 안 된다(차분 테스트로 확인한 원 동작).
    base = {"available": False, "error": "worker_state_missing",
            **({} if bare_missing else {"tone": "neutral"}), "subText": "데이터 없음"}
    st = load_json(path)
    if not st:
        return {**base, **(extra_missing or {})}
    if require_ok and not st.get("ok"):
        # bare_missing 계열은 워커 상태를 얹지 않고 결측과 같은 모양으로 떨어뜨린다(원 동작).
        return {**base, **(extra_missing or {})} if bare_missing else {
            **st, **base, **(extra_missing or {}),
            "error": "worker_fetch_failed", "subText": "오류"}
    age = _age_min(st.get(ts_field))
    stale = round(age, 1) if age is not None else None
    if age is not None and age > max_age_min:
        return {**st, **base, "error": "worker_stale", "stale_min": stale}
    return {**st, **({"available": True} if stamp_available else {}), "stale_min": stale}
