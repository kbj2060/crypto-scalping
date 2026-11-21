"""QX4 REST 이어 받기(scripts/live_vol4h_qx4_20261025.py `refresh_rows`·`merge_klines`) 요청 열 정답지 (2026-11-21 · tests/qx4.rs qx4_refresh).
원본 함수에 가짜 fetch(합성 5분봉 · 요청과 응답을 적음)를 넘겨 4회차: 처음(코인마다 1000봉 + endTime 으로 거슬러 NEED_BARS+300) → 5분 뒤(최근 1000봉 이어 붙임)
→ 3일 뒤(이어지지 않음 → 처음부터) → 한 코인 응답 빔(False). 네트워크 없음.
    python rtmodel/tools/gen_qx4_rest_fixture.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_vol4h_qx4_20261025 as Q  # noqa: E402

BAR = 300_000


def bars(sym, end, limit):
    """endTime(포함) 이하 마지막 limit 봉 -- 형성 중 봉 포함(close 가 지금 뒤)."""
    last = end // BAR * BAR
    return [[t, "1", f"{1 + (t // BAR) % 97 / 1000:.4f}", "1", f"{1 + (t // BAR) % 89 / 1000:.4f}", "5", t + BAR - 1, "0", 3, "0", "0", "0"]
            for t in range(last - (limit - 1) * BAR, last + 1, BAR)]


def main():
    now0 = 1_795_183_380_000
    rows, runs = {}, []
    for k, now in enumerate((now0, now0 + BAR, now0 + 3 * 86_400_000, now0 + 3 * 86_400_000 + BAR)):
        log = []

        async def fetch(q, now=now, log=log, k=k):
            log.append([[kk, str(v)] for kk, v in q.items()])
            got = [] if (k == 3 and q["symbol"] == "BTCUSDT") else bars(q["symbol"], q.get("endTime", now), q["limit"])
            log[-1] = [log[-1], got]
            return got
        ok = asyncio.run(Q.refresh_rows(fetch, rows))
        runs.append({"now": now, "ok": ok, "requests": log, "rows": {c: [r[0] for r in v] for c, v in rows.items()}})
        print(k, ok, len(log), {c: len(v) for c, v in rows.items()})
    p = ROOT / "rtmodel/tests/fixtures/qx4_rest_fixture.json"; p.write_text(json.dumps({"runs": runs})); print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
