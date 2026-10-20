"""포지션 프로파일 칸 폭(2026-10-20 XRP): bw 를 코인 칸으로 주면 칸 수·평균가가 그 해상도로 나온다.
python -m pytest -q test/test_position_profile_bw_20261020.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import position_profile as pprof  # noqa: E402


def test_xrp_bins_and_avg():
    lo, vals = pprof.bins(np.array([1.3012]), np.array([1.3000]), np.array([100.0]), 0.0003)
    assert lo == 4333 and len(vals) == 5 and abs(vals.sum() - 100.0) < 1e-9   # 1.2999~1.3014 칸 5개(달러 칸이면 1개)
    b = pprof.Book()
    b.push(999_700, 1.30, 1.30, 0.0, 5e5)                   # 첫 봉 = 옛 포지션만
    b.push(1_000_000, 1.3012, 1.3000, 1e6, 6e5)
    p = pprof.payload(b, 1_000_000, 1.3006, bw=0.0003, windows=(1,))
    w = p["win"]["1"]
    assert p["bw"] == 0.0003 and w["avg"] is not None and 1.2999 < w["avg"] < 1.3015 and w["avg"] != round(w["avg"], 2)
