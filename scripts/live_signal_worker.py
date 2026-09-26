#!/usr/bin/env python3
"""신호 워커 **공용본** -- 인자 없는 compute 함수 하나를 주기적으로 돌려 상태 파일에 쓴다.

## 왜 공용본인가
2026-09-13 기준 이 저장소엔 같은 워커가 네 벌 복사돼 있었다(극점·변동성 전망·돌파 탐지기·
크기 가늠자). 코드 본문을 이름만 바꿔 비교하면 실제로 다른 줄은 **import 경로·상태 경로·
주기·로그 문장** 넷뿐이었다. 다섯 번째 복사본 대신 그 넷을 인자로 받는다.

기존 네 벌은 **건드리지 않는다** -- 살아 있는 supervisor 4개를 갈아끼우는 위험을 사용자에게
보이는 이득 없이 지불할 이유가 없다. 새로 만드는 워커만 이 파일을 쓴다.

## 왜 워커인가 (2026-09-10 실장애에서 확정된 규칙)
대시보드 요청 경로에서 모델을 돌리면 `asyncio.to_thread` 기본 풀(16스레드)을 먹는다.
그날 극점 탐지기가 HGB(1.8MB) -> TabPFN(1.08GB)으로 바뀌며 풀이 고갈돼 대시보드가 멈췄다.
2026-09-13 실측: 대시보드 재시작 직후 첫 요청이 V자 **137초**, 증거신호 **160초**를 기다린다.
🔴이건 「1회 로딩 비용」이 아니다 -- 두 번째가 더 느린 경우가 있다(V자 137->390초, 증거신호
160->299초). 무거운 계산들이 **서로 경합**하고 페이지를 열면 동시에 발동한다. 재시작은 배포
워처 때문에 하루 12회쯤이라 사용자는 그 구간을 매번 만난다(화면의 "웜업"이 이것이다).
워커로 옮기면 그 계산들이 **각자의 주기로 흩어져** 돌고 화면은 파일만 읽는다.

## 계약
`--state` 경로에 compute 결과를 **원자적으로**(tmp -> os.replace) 쓰고 `updated_utc` 를 얹는다.
대시보드는 `worker_payload()` 로 그 파일만 읽고 신선도를 판단한다.
계산은 기존 함수를 그대로 부른다 -- 화면 숫자와 워커 숫자가 갈라질 여지를 만들지 않는다.

## 사용
    python3 scripts/live_signal_worker.py \
      --compute live_eth_sweep_v_rebound_signal_20260829:compute_eth_sweep_v_rebound_signal \
      --state data/live/eth_v_rebound_state.json --interval 60 --loop

`--compute`/`--state` 를 **짝지어 여러 번** 줄 수 있다. 그러면 한 프로세스가 매 사이클
순서대로 돌며 각자의 상태 파일에 쓴다 -- 레짐 3종처럼 가볍고(각 13초) 같은 주기인 것들을
프로세스 하나로 묶기 위한 것이다. 하나가 실패해도 나머지는 돈다.
"""
from __future__ import annotations
import argparse, importlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import binance_ban_guard  # noqa: E402,F401 -- 2026-09-26 IP 밴 중엔 바이낸스 REST 를 안 내보낸다(공용 가드)


def log(m: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {m}", flush=True)


def load_compute(spec: str):
    """"module:function" -> 호출 가능한 객체. 인자 없이 dict 를 돌려주는 함수여야 한다."""
    module_name, _, func_name = spec.partition(":")
    if not module_name or not func_name:
        raise SystemExit(f"--compute 는 'module:function' 형식이어야 합니다: {spec!r}")
    return getattr(importlib.import_module(module_name), func_name)


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**payload, "updated_utc": datetime.now(timezone.utc).isoformat()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, default=str))
    os.replace(tmp, path)     # 원자적 교체 -- 대시보드가 반쪽 파일을 읽지 않게


def cycle(compute, path: Path, label: str = "") -> dict:
    t0 = time.time()
    payload = compute()
    if not isinstance(payload, dict):
        raise TypeError(f"compute 가 dict 가 아닌 {type(payload).__name__} 을 돌려줬습니다")
    write_state(path, payload)
    log(f"{label}{payload.get('subText') or payload.get('tone') or '-'} · {time.time() - t0:.2f}s"
        + ("" if payload.get("warmed_up", True) else f" · ⚠️{payload.get('error')}"))
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", required=True, action="append",
                    help="module:function (인자 없음, dict 반환). --state 와 짝지어 반복 가능")
    ap.add_argument("--state", required=True, action="append",
                    help="상태 파일 경로 (ROOT 기준 상대 가능)")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--loop", action="store_true")
    a = ap.parse_args(argv)

    if len(a.compute) != len(a.state):
        raise SystemExit(f"--compute {len(a.compute)}개와 --state {len(a.state)}개는 짝이 맞아야 합니다")
    jobs = []
    for spec, state in zip(a.compute, a.state):
        path = Path(state)
        if not path.is_absolute():
            path = ROOT / path
        # import 는 여기서 한 번만 -- 사이클마다 다시 부르면 모델이 다시 로딩된다.
        jobs.append((spec.rpartition(":")[2], load_compute(spec), path))

    def run_all() -> None:
        for name, compute, path in jobs:
            label = f"[{name}] " if len(jobs) > 1 else ""
            try:
                cycle(compute, path, label)
            except Exception as e:     # noqa: BLE001 -- 한 짝이 실패해도 나머지는 돈다
                log(f"⚠️{label}사이클 실패: {type(e).__name__}: {e}")

    if not a.loop:
        run_all()
        return 0
    log(f"워커 시작 · {len(jobs)}개 · 주기 {a.interval}초 · " + ", ".join(n for n, _, _ in jobs))
    while True:
        run_all()
        time.sleep(a.interval)


def _self_check() -> None:
    """프레임워크 없이: python3 scripts/live_signal_worker.py --self-check"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "state.json"
        write_state(p, {"tone": "good", "n": 1})
        got = json.loads(p.read_text())
        assert got["tone"] == "good" and "updated_utc" in got, got
        assert not list(Path(tmp).glob("*.tmp")), "tmp 파일이 남았다"

        # 원자성: 쓰는 중에도 옛 판본이 온전해야 한다 -- os.replace 라 반쪽 파일이 안 보인다.
        write_state(p, {"tone": "bad", "n": 2})
        assert json.loads(p.read_text())["n"] == 2

        # dict 가 아니면 즉시 실패해야 한다(조용히 빈 상태를 쓰면 화면이 옛 값을 붙든다).
        try:
            cycle(lambda: "not a dict", p)
        except TypeError:
            pass
        else:
            raise AssertionError("dict 아닌 반환을 통과시켰다")
        assert json.loads(p.read_text())["n"] == 2, "실패한 사이클이 상태를 덮어썼다"

        # module:function 파싱
        assert load_compute("json:dumps") is json.dumps
        for bad in ("nocolon", ":func", "module:"):
            try:
                load_compute(bad)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"잘못된 spec 을 통과시켰다: {bad!r}")

        # --compute/--state 짝이 안 맞으면 즉시 실패해야 한다 -- 조용히 zip 으로 잘라내면
        # 한 신호가 영영 안 갱신되고 화면은 "데이터 없음"만 띄운다.
        try:
            main(["--compute", "json:dumps", "--compute", "json:loads",
                  "--state", str(Path(tmp) / "a.json")])
        except SystemExit:
            pass
        else:
            raise AssertionError("짝이 안 맞는 인자를 통과시켰다")

        # 여러 짝: 한 프로세스가 각자의 파일에 쓴다. 하나가 실패해도 나머지는 돈다.
        a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
        sys.modules["_wk_fixture"] = type(sys)("_wk_fixture")
        sys.modules["_wk_fixture"].ok = lambda: {"tone": "good"}
        sys.modules["_wk_fixture"].boom = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        rc = main(["--compute", "_wk_fixture:boom", "--state", str(a),
                   "--compute", "_wk_fixture:ok", "--state", str(b)])
        assert rc == 0 and not a.exists() and json.loads(b.read_text())["tone"] == "good"
    print("live_signal_worker 자체점검 통과 (8건)")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
        raise SystemExit(0)
    raise SystemExit(main())
