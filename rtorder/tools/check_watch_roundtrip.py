"""rtorder 셋째 조각 왕복 확인(2026-11-20): **Rust 가 쓴** 무장 저장본·지정가 대기 파일을 **원본 파이썬 읽는 쪽**(orders.py 의 bracket_load · limit_recover 클로저)으로 읽어
파이썬이 쓴 원본과 같은 뜻인지 본다(형·값까지: 정수/실수 구분 · 실수는 repr 비교라 -0.0·끝자리까지). 반대쪽(파이썬이 쓴 것 → Rust)은 cargo test --test watch_fixture 가 본다.
먼저 cargo test --test watch_fixture 를 돌려 rtorder/target/watch_roundtrip/ 를 만든다. 네트워크·.env 없음.
실행(저장소 뿌리): ( ulimit -v 4000000; TZ=UTC timeout 120 unshare -rn python3 -I rtorder/tools/check_watch_roundtrip.py )
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for _k in ("BINANCE_API_KEY", "BINANCE_SECRET_KEY", "DASHBOARD_MANUAL_EXEC_ENABLED"):
    os.environ.pop(_k, None)
os.environ["DASHBOARD_MANUAL_EXEC_SYMBOL"] = "ETHUSDC"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard import orders  # noqa: E402
from gen_watch_fixture import closure_map, boom  # noqa: E402

RT = Path(__file__).resolve().parents[1] / "target" / "watch_roundtrip"
FX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "watch_fixture.json"


def same(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return repr(a) == repr(b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main() -> int:
    fx = json.loads(FX.read_text("utf-8"))
    expect = {r["scenario"]["name"]: r["ticks"][-1].get("file_raw") for r in fx["scenarios"]}
    tmp = Path(tempfile.mkdtemp())
    ok = bad = 0
    with mock.patch.multiple(orders, LIVE_DIR=tmp, time=types.SimpleNamespace(time=lambda: 1_800_000_200.0)):
        api = orders.register(account_trip_state={"seen": {}}, refresh_tasks={}, swr_locks={}, binance_session=lambda: None,
                              fetch_binance_json=boom, swr_cached=boom, _timed_produce=boom, load_market_history=boom,
                              situation_by={}, NOCACHE={}, STALE_GRACE_SECONDS=60, sec_prices=lambda _a: None)
        cells = closure_map([api.bracket_tick, api.start_wait_cleanup])
        bracket_load, limit_recover = cells["bracket_load"].cell_contents, cells["limit_recover"].cell_contents

        async def no_position():
            return {"ok": True, "positions": []}
        cells["produce_account"].cell_contents = no_position
        for p in sorted(RT.glob("*.json")):
            if p.name.startswith("limit__"):
                dst = tmp / p.name.removeprefix("limit__")
                shutil.copy(p, dst)
                rec = json.loads(p.read_text("utf-8"))
                out = asyncio.run(limit_recover(dst, None))
                good = (out == {"symbol": rec["plan"]["symbol"], "side": rec["plan"]["positionSide"], "armed": False, "wid": rec["plan"]["wid"]}
                        and not dst.exists() and same(rec["plan"]["limit"]["usdt"], 0.30000000000000004) and same(rec["plan"]["bracket"]["tp_level"], 1e-7))
            else:
                shutil.copy(p, tmp / "manual_bracket_state.json")
                raw = expect[p.stem]
                good = same(bracket_load(), orders.bracket_rekey(json.loads(raw)))
            ok, bad = ok + good, bad + (not good)
            if not good:
                print("🔴다름", p.name)
    print(f"Rust 가 쓴 파일 {ok + bad}개 중 파이썬이 같은 뜻으로 읽음 {ok} · 다름 {bad}")
    return 0 if bad == 0 and ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
