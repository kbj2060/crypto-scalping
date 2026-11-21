#!/usr/bin/env python3
"""재시작 복원 메모리·시간 측정용 합성 체결 테이프(2026-11-21). 8코인 × 24시간 × 코인마다 약 32만 행(서버 ETH 하루 실측 규모를 8코인 모두에).

표 모양은 원본 수집기의 `_init_sqlite`(scripts/live_trade_tape_collector_20260916.py) 로 만든다 -- 그 함수는 넘긴 경로에만 쓴다.
값은 측정용 난수(정답 대조 아님). 실제 data/ 는 안 건드린다: 경로를 인자로 받는다(스크래치 폴더).
    python3 rtcore/tools/gen_bench_tape.py <출력.sqlite> [코인당 행 수=320000]
그다음: RTCORE_BENCH_TAPE=<출력.sqlite> cargo test --release --test restore_bench -- --ignored --nocapture
"""
from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.live_trade_tape_collector_20260916 import BUCKETS, _init_sqlite  # noqa: E402

END = 1_763_683_200        # 합성 끝 초(2025-11-21 00:00 UTC) -- restore_bench.rs 와 같은 값
MIDS = {"ethusdt": 3000.0, "btcusdt": 90000.0, "solusdt": 140.0, "xrpusdt": 2.1, "sndkusdt": 1150.0,
        "soxlusdt": 80.0, "skhynixusdt": 938.0, "muusdt": 796.0}


def main() -> None:
    out = Path(sys.argv[1])
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 320_000
    assert not out.exists(), f"{out} 가 이미 있다"
    _init_sqlite(out)
    con = sqlite3.connect(out)
    rng = random.Random(7)
    for sym, mid in MIDS.items():
        tb = BUCKETS[sym]
        pb = round(mid / tb)
        rows = []
        per_sec = per / 86_400
        acc = 0.0
        for s in range(END - 86_400, END):
            acc += per_sec
            n, acc = int(acc), acc - int(acc)
            pb += rng.choice((-1, 0, 0, 1))
            for j in range(n):
                b, sq = rng.uniform(0, 5), rng.uniform(0, 5)
                rows.append((sym, s, pb + j - n // 2, b, sq, 3, 2, b / 2, sq / 2, b / 3, sq / 3, b / 4, 0.0, 1, 0, 2, 1, 3, 2))
        with con:
            con.executemany(f"INSERT OR REPLACE INTO trade_tape_1s VALUES ({','.join('?' * 19)})", rows)
        print(f"{sym}: {len(rows)}행", flush=True)
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    print(f"{out}: {out.stat().st_size // (1 << 20)}MB")


if __name__ == "__main__":
    main()
