"""QX4 서빙 증분 klines 병합(scripts/live_vol4h_qx4_20261025.merge_klines) — 겹침 교체 · 끊김 감지 · 길이 상한."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from live_vol4h_qx4_20261025 import merge_klines  # noqa: E402

T0 = 1_700_000_000_000


def rows(i0, n, c=1.0):
    return [[T0 + i * 300_000, 0, c, c, c, 1.0, T0 + i * 300_000 + 299_999, 0, 1] for i in range(i0, i0 + n)]


def test_overlap_replaced_by_new():
    have, got = rows(0, 10), rows(8, 5, c=2.0)          # 8·9 겹침 → 새 값
    m = merge_klines(have, got)
    assert [r[0] for r in m] == [T0 + i * 300_000 for i in range(13)]
    assert m[8][4] == 2.0 and m[7][4] == 1.0


def test_gap_returns_none():
    assert merge_klines(rows(0, 10), rows(12, 5)) is None   # 10·11 빠짐 → 처음부터 다시
    assert merge_klines(rows(0, 10), rows(10, 5)) is not None
    assert merge_klines([], rows(0, 5)) is None and merge_klines(rows(0, 5), []) is None


def test_keep_limit():
    assert len(merge_klines(rows(0, 50), rows(49, 5), keep=20)) == 20


class FakeKlines:
    """바이낸스 /fapi/v1/klines 흉내: 코인마다 같은 격자 · limit=1000 · endTime 이하 마지막 1000봉. calls 로 요청 수를 센다."""
    def __init__(self, n):
        self.n, self.calls, self.fail = n, 0, False

    async def __call__(self, q):
        self.calls += 1
        if self.fail:
            return None
        last = self.n - 1 if "endTime" not in q else (q["endTime"] - T0) // 300_000
        lo = max(0, last - 999)
        return rows(lo, last - lo + 1) if last >= 0 else []


def test_refresh_rows_full_then_incremental():
    import asyncio
    from live_vol4h_qx4_20261025 import NEED_BARS, SYMS, refresh_rows
    f, cache = FakeKlines(NEED_BARS + 5000), {}
    assert asyncio.run(refresh_rows(f, cache))
    assert all(len(cache[c]) >= NEED_BARS for c in SYMS) and f.calls > 4 * (NEED_BARS // 1000)
    f.calls, f.n = 0, f.n + 1                                   # 새 봉 1개
    assert asyncio.run(refresh_rows(f, cache)) and f.calls == 4  # 코인마다 1번만
    assert all(cache[c][-1][0] == T0 + (f.n - 1) * 300_000 for c in SYMS)
    f.fail = True
    assert asyncio.run(refresh_rows(f, cache)) is False


def test_end_to_end_with_shipped_artifact():
    """저장 아티팩트가 있으면(개발 PC) 연구 5분봉을 가짜 klines 로 흘려 ok·30일 기준·중앙값형 눈금 확인."""
    import pytest
    root = Path(__file__).resolve().parents[1]; art_p = root / "tmp/vol4h_qx4/qx4_2026-10-23.joblib"
    if not art_p.exists():
        pytest.skip("아티팩트 없음")
    import numpy as np
    import live_vol4h_qx4_20261025 as Q
    art = Q.load_qx4(art_p); assert art is not None
    xs = {s: Q.R.load(s)[0].loc["2026-08-01":] for s in Q.COINS}; rws = {s: Q._rows(x) for s, x in xs.items()}; now = rws["ETH"][-1][6] + 1
    out = Q.qx4_from_klines(rws, art, now)
    assert out["ok"] and out["ref_days"] >= 30 and out["model"] == "QX4"
    assert abs(out["sig4h_mean_pct"] / out["sig4h_pct"] - np.exp(art["coins"]["ETH"]["c"] / 2)) < 2e-3
