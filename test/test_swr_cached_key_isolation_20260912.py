"""swr_cached 가 캐시·락을 key 로 소유해도 원 동작이 유지되는지. 2026-09-12 리팩터 가드.

프레임워크 없이 `python3 test/test_swr_cached_key_isolation_20260912.py` 로 돈다.
swr_cached 는 클로저 안이라 import 로는 실행되지 않는다 -- 소스를 떼어 내 직접 돌린다.
"""
import asyncio, re, time
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")
start = src.index("    def _schedule_refresh(")
end = src.index("    # Shared, connection-pooled session")
chunk = re.sub(r"^    ", "", src[start:end], flags=re.M)          # 클로저 들여쓰기 제거
ns = {"asyncio": asyncio, "time": time, "Any": object, "refresh_tasks": {}}
exec(compile(chunk, "swr_cached", "exec"), ns)
swr_cached = ns["swr_cached"]

async def main() -> None:
    calls = {"a": 0, "b": 0}
    async def produce_a():
        calls["a"] += 1
        return {"who": "a", "n": calls["a"]}
    async def produce_b():
        calls["b"] += 1
        return {"who": "b", "n": calls["b"]}

    # 1) 콜드는 produce 를 기다린다.
    assert (await swr_cached("A", 60.0, produce_a))["n"] == 1
    # 2) 신선하면 produce 를 다시 부르지 않는다.
    assert (await swr_cached("A", 60.0, produce_a))["n"] == 1 and calls["a"] == 1
    # 3) **키가 다르면 캐시가 섞이지 않는다** -- 선언을 걷어내며 새로 생긴 계약.
    assert (await swr_cached("B", 60.0, produce_b))["who"] == "b"
    assert (await swr_cached("A", 60.0, produce_a))["who"] == "a"
    # 4) ttl 0 이면 매번 다시 만든다.
    assert (await swr_cached("A", 0.0, produce_a))["n"] == 2
    # 5) cache= 를 넘기면 그 dict 에 쓴다(swr 밖에서 읽는 세 곳이 의존).
    external = {"ts": 0.0, "payload": None, "frames": "keep"}
    got = await swr_cached("C", 60.0, produce_b, cache=external)
    assert external["payload"] is got and external["frames"] == "keep", external
    # 6) stale 은 즉시 옛 값을 주고 뒤에서 갱신한다.
    n_before = calls["a"]
    served = await swr_cached("A", 0.0, produce_a, max_stale=60.0)
    assert served["n"] == n_before, (served, n_before)
    await asyncio.sleep(0.05)
    assert calls["a"] == n_before + 1, calls
    print("swr_cached 6건 통과 (키 분리·cache= 우회·SWR 갱신 포함)")

asyncio.run(main())
