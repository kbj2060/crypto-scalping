#!/usr/bin/env python3
"""상황 카드 통합 정답지(2026-11-22 · 일곱째 조각): «같은 입력 → /api/situation 본문 · SSE 한 줄».

원본 몫은 이미 있는 것을 불러 쓴다(ctx 쪽 도구 파일은 고치지 않는다):
  · _flow_read_ctx · _card30_bars · cctx/flows 대역 = tools/gen_ctx_x_fixture.py 의 정의부(모듈이 맨 위에서 출력 폴더를 지우고 시험을 다 돌리므로
    소스를 읽어 «def cell():» 앞까지만 실행한다 -- 지우기·폴더 만들기 두 줄은 빼고)
  · situation.classify · flow_read.read · flow_read.vol_size_mult = 원본 모듈 그대로
  · situation_payload = server.py 2734~2738 · SSE 한 줄 = 3175~3179(옮겨 적음 -- Read 로 본 줄)
rtserve 에 아직 없는 재료는 원본 쪽 입력도 «없음»으로 맞춘다: candles_day·candles = [] · bybit_bars = {} · flow_scales·flow_scales_all = {} ·
micro payload = {"available": False}(코어 호가가 없을 때와 같게) · imb40 링 = [] · _flow_read_ctx 의 inp.bars = [](코어 Situation 에 안 실림) ·
evidence 의 veto 0 · obi·persist None(래스터 요약 없음) · vol_mult = vol_size_mult([]).
코어 Situation = classify(inp) 결과 그대로(rtcore 가 같은 칸으로 민다) · 풋프린트 봉 칸 = gen_ctx_x_fixture 의 bar_cells 로 만들어 같이 싣는다(Rust 시험이 rtcore flow_nets_tick → Micro.flow 로 흘린다).
    source <(bash rtmodel/tools/server_env.sh <venv> --env) && $PY rtserve/tools/gen_sit_fixture.py     # 저장소 루트에서 · 출력 rtserve/tests/fixtures/gen/sit/sit.json(커밋 안 함)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "rtserve" / "tools" / "gen_ctx_x_fixture.py").read_text()
head = SRC[:SRC.index("\ndef cell():")]
head = head.replace("shutil.rmtree(OUT, ignore_errors=True)\n", "").replace("OUT.mkdir(parents=True)\n", "")
gx: dict = {"__name__": "gen_ctx_x_defs", "__file__": str(ROOT / "rtserve" / "tools" / "gen_ctx_x_fixture.py")}
exec(compile(head, "gen_ctx_x_fixture.py(정의부)", "exec"), gx)   # noqa: S102
rest = SRC[SRC.index("\ndef cell():"):SRC.index("\ndef run(")]
exec(compile(rest, "gen_ctx_x_fixture.py(cell·bar_cells)", "exec"), gx)   # noqa: S102

import dashboard.situation as sit  # noqa: E402
import dashboard.flow_read as fr  # noqa: E402
from types import SimpleNamespace  # noqa: E402

OUT = ROOT / "rtserve" / "tests" / "fixtures" / "gen" / "sit"
R = random.Random(7)


def situation_payload(st, fo_state, mp_state, situation_by):   # server.py 2734~2738
    return {"now": st["now"], "computed_at": st["computed_at"], "streams": {"fo": dict(fo_state), "mp": dict(mp_state)},
            "read": st.get("read"), "vol_mult": st.get("vol_mult"), "vol_mult_by": {a: s.get("vol_mult") for a, s in situation_by.items()}}


def sse(p):   # server.py 3175~3179
    return f"event: situation\ndata: {json.dumps(p, ensure_ascii=False, separators=(',', ':'))}\n\n"


def stream_stat(events):
    return {"connected": R.random() < 0.9, "since": R.choice([None, 1_763_000_000.5]), "last_ms": R.choice([None, 1_763_000_123_456]),
            "events": events, "errors": R.randint(0, 3), "last_error": R.choice([None, "ConnectionResetError"]), "next_funding_ms": R.choice([None, 1_763_028_800_000])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gx["cctx"].clear(); gx["flows"].clear(); gx["okx_bar_net_by"].clear(); gx["flow_all_hold"].clear()
    now = 1_763_700_000.0 + 3600 * 5 + 17.25
    st = {"bybit_bars": {}, "candles_day": [], "candles": []}
    ms = {"payload": {"available": False}, "flow_scales": {}, "flow_scales_all": {}}
    gx["cctx"]["eth"] = SimpleNamespace(situation_state=st, micro_state=ms, fr_bar_net={}, imb40_ring=[])
    b0 = int(now) // 300 * 300 - 60 * 300
    fb = {b0 + 300 * k: gx["bar_cells"]() for k in range(60)}
    okx = {"first_bar": 0}
    ob: dict = {}
    gx["flows"]["eth"] = SimpleNamespace(footprint_state={"bars": fb}, okx_fp=okx, okx_bars=ob)
    by = {a: {"vol_mult": fr.vol_size_mult([], a)} for a in ("eth", "sol", "xrp")}
    steps, prev_dir, px = [], 0, 3000.0
    for s in range(40):
        now += R.choice([1.0, 1.0, 1.0, 17.0, 300.0])
        cur = int(now) // 300 * 300
        if cur not in fb:
            fb[cur] = gx["bar_cells"]()
        for b in sorted(fb)[-2:]:
            if R.random() < 0.5:
                fb[b][R.randrange(9990, 10050)] = gx["cell"]()
        if s == 10:
            okx["first_bar"] = sorted(fb)[5]
        if okx["first_bar"]:
            for b in fb:
                if b > okx["first_bar"] - 600 and b not in ob:
                    ob[b] = gx["bar_cells"]()
        bars = []
        for k in range(R.choice([3, 8, 13])):   # 3 = 완결 봉 부족(ok False)
            px *= 1 + R.gauss(0, 0.003)
            bars.append({"time": cur - 300 * (13 - k), "high": px * 1.001, "low": px * 0.999, "close": px, "delta": R.gauss(0, 200),
                         "liq_long": abs(R.gauss(0, 1e4)), "liq_short": abs(R.gauss(0, 1e4)), "veto": 0, "oi_delta": R.gauss(0, 100)})
        res = sit.classify({"bars": bars, "mid": px, "prev_dir": prev_dir})
        prev_dir = res.get("dir") or 0
        core = json.loads(json.dumps(res))
        res["computed_at"] = now
        x = gx["_flow_read_ctx"](now, {"bars": []}, "eth")
        st["now"], st["computed_at"] = res, now
        st["read"] = fr.read(res.get("evidence") or {}, x)
        st["vol_mult"] = by["eth"]["vol_mult"]
        fo, mp = stream_stat(R.randint(0, 50)), stream_stat(R.choice([0, 0, 7]))
        fo_state = {"connected": fo["connected"], "since": fo["since"], "last_event_ms": fo["last_ms"], "events": fo["events"], "errors": fo["errors"], "last_error": fo["last_error"]}
        mp_state = {"connected": mp["connected"], "since": mp["since"], "last_ms": mp["last_ms"], "events": mp["events"], "errors": mp["errors"], "last_error": mp["last_error"]}
        if mp["events"] > 0:
            mp_state["next_funding_ms"] = mp["next_funding_ms"]
        p = situation_payload(st, fo_state, mp_state, by)
        steps.append({"now": now, "core": core, "streams": {"fo": fo, "mp": mp}, "fbars": [[b, list(c.values())] for b, c in fb.items()],
                      "okx_first_bar": okx["first_bar"], "okx_bars": [[b, list(c.values())] for b, c in ob.items()],
                      "body": json.dumps(p), "sse": sse(p)})
    (OUT / "sit.json").write_text(json.dumps(steps))
    print(f"상황 카드 걸음 {len(steps)} · ok {sum(1 for x in steps if x['core'].get('ok'))}")


if __name__ == "__main__":
    main()
