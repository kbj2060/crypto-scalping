#!/usr/bin/env python3
"""rtcore 풋프린트 바이낸스+OKX 합산(footprint.rs levels add · from) 일치 시험의 입력·정답지 (2026-11-21). 커밋한다(tests/fixtures/fp_okx_sol.json).

정답 = server.py api_footprint(3500~3586)의 봉·칸 조립을 옮겨 적은 것:
- base = 바이낸스 봉 · first_bar 가 있으면 b > first_bar 만 · 마지막 144봉 · `_merged` = 바이낸스 칸 사본에 OKX 칸 더함.
- 칸 = [round(k × 칸 폭, dp)] + round(x, 3).
- since 이어받기 = 같은 창에서 b ≥ since 인 봉.
입력 봉:
- 바이낸스 봉 = coin_flow footprint_add 의 칸 규칙(옮겨 적음). 체결마다 다른 주문이 되게 tid 를 2씩 띄운다 -- 되묶기 규칙은 footprint_parity 가 따로 본다.
- OKX 봉 = gen_okx_fp_fixture.feed(coin_flow 760~796 옮겨 적음).
OKX 는 중간(봉 100)부터 온다 -- first_bar 가 그 봉이 되어 그 앞 봉이 창에서 빠진다(원본 동작).
    python3 rtcore/tools/gen_fp_okx_fixture.py     # 저장소 루트에서
"""
from __future__ import annotations

import json
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_okx_fp_fixture as g  # noqa: E402

cf, SPEC, INST = g.cf, g.SPEC, g.INST
ROOT = g.ROOT
WANT, BS = 144, 300
RETAIL_MAX_USD, WHALE_MIN_USD = g.RETAIL_MAX_USD, g.WHALE_MIN_USD
SEED = 20261126


def merged_payload(bn: dict, okx: dict, okx_from: int, since: int = 0) -> list:
    """server.py api_footprint 봉·칸 몫(옮겨 적음)."""
    bucket, dp = SPEC.bucket, SPEC.price_dp
    base = sorted(bn.items())
    if okx_from:
        base = [(b, c) for b, c in base if b > okx_from]
    recent = base[-WANT:]

    def _merged(bar, cells):
        add = okx.get(bar)
        if not add:
            return cells
        out = {k: list(v) for k, v in cells.items()}
        for k, v in add.items():
            t = out.setdefault(k, [0.0] * 6)
            for j in range(6):
                t[j] += v[j]
        return out
    sent = recent if since <= 0 else [(b, c) for b, c in recent if b >= since]
    return [[bar, [[round(k * bucket, dp)] + [round(x, 3) for x in v] for k, v in sorted(_merged(bar, cells).items())]] for bar, cells in sent]


def main() -> None:
    rng = random.Random(SEED)
    t0 = 1_763_700_000 // BS * BS
    nbars = 170
    bn: dict[int, dict[int, list]] = {}
    ns = SimpleNamespace(okx_fp={"first_bar": 0, "t0_ms": 0}, okx_bars={}, okx_oi_5m={}, okx_liq_events=deque(maxlen=5000))
    events, cps = [], []
    open_order = None
    px, tid, otid = 140.0, 1000, 5000
    for i in range(nbars):
        b = t0 + i * BS
        times = sorted(rng.randint(b * 1000, b * 1000 + BS * 1000 - 1) for _ in range(rng.randint(2, 7)))
        for ms in times:
            px = round(px + rng.gauss(0, 0.05), 2)
            q = round(rng.lognormvariate(1.5 if rng.random() < 0.1 else 0.0, 1.3), 3)
            sell = rng.random() < 0.5
            tid += 2
            events.append(["bn", ms, tid, px, q, sell])
            # coin_flow footprint_add: 총량은 체결마다, 크기 구간은 이 체결로 닫힌 주문(앞 체결)
            k = int(round(px / SPEC.bucket))
            c = bn.setdefault(b, {}).setdefault(k, [0.0] * 6)
            c[1 if sell else 0] += q
            if open_order:
                ob, ok, oq, op, os_ = open_order
                n = op * oq
                oc = bn[ob].setdefault(ok, [0.0] * 6)
                if n >= WHALE_MIN_USD:
                    oc[2 + os_] += oq
                elif n < RETAIL_MAX_USD:
                    oc[4 + os_] += oq
            open_order = (b, k, q, px, 1 if sell else 0)
            if i >= 100 and rng.random() < 0.6:
                rows = []
                for _ in range(rng.choice((1, 2))):
                    otid += 1
                    sz = rng.lognormvariate(0.5, 1.4)
                    rows.append({"instId": INST, "tradeId": str(otid), "px": f"{px + rng.randint(-2, 2) * 0.01:.2f}",
                                 "sz": f"{max(sz, 0.01):.2f}", "side": rng.choice(("buy", "sell")), "ts": str(ms + 3), "count": "1"})
                msg = {"arg": {"channel": "trades", "instId": INST}, "data": rows}
                g.feed(ns, msg, (ms + 3) / 1000)
                events.append(["okx", ms + 3, json.dumps(msg)])
        if i % 7 == 0 or i in (99, 100, 101) or i == nbars - 1:
            ofr = ns.okx_fp["first_bar"]
            last = max(bn)
            # _flow_read_ctx 2316~2318 의 재료: OKX 봉마다 칸 합(원본 = 넣은 차례로 더함)
            sums = [[b, [sum(v[j] for v in c.values()) for j in range(6)]] for b, c in sorted(ns.okx_bars.items()) if ofr and b > ofr]
            cps.append({"after": len(events), "okx_from": ofr, "okx_sums": sums, "full": merged_payload(bn, ns.okx_bars, ofr),
                        "since": last - BS, "inc": merged_payload(bn, ns.okx_bars, ofr, last - BS)})
    p = ROOT / "rtcore" / "tests" / "fixtures" / "fp_okx_sol.json"
    p.write_text(json.dumps({"symbol": SPEC.symbol.lower(), "events": events, "checkpoints": cps}, separators=(",", ":")))
    print(f"{p}: 사건 {len(events)} · 대조 {len(cps)} · 마지막 창 봉 {len(cps[-1]['full'])} · first_bar {cps[-1]['okx_from']} · "
          f"OKX 칸 더한 봉 {sum(1 for b, _ in cps[-1]['full'] if b in ns.okx_bars)} · {p.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
