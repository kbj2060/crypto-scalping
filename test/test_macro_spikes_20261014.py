"""server.spikes_5m 자체점검(2026-10-14 «⑥» 급변 칩) -- 휴장 갭·미완결 봉·최근 창 경계."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import server  # noqa: E402


def bars(n, jump_at=None, jump_bp=0.0, gap_before_jump=False):
    """5분봉 n 개(봉마다 1분봉 5개, 봉 수익 ±1bp 교대) → [(ts_sec, close)]."""
    rows, px, t0 = [], 100.0, 1_800_000_000 // 300 * 300
    for i in range(n):
        if gap_before_jump and i == jump_at - 1:
            continue
        px *= math.exp(((jump_bp if i == jump_at else (1 if i % 2 else -1))) / 1e4)
        rows += [(t0 + i * 300 + m * 60, px) for m in range(5)]
    return rows, t0


def test_spike_found_and_bounded():
    rows, t0 = bars(400, jump_at=390, jump_bp=10.0)
    ev = server.spikes_5m(rows)["events"]
    assert [e[0] for e in ev] == [t0 + 390 * 300] and ev[0][1] > 4 and abs(ev[0][2] - (math.expm1(10.0 / 1e4) * 100)) < 1e-3
    assert server.spikes_5m(rows, keep_s=300)["events"] == []          # 창 밖(봉이 1시간 전에 끝남)


def test_gap_and_incomplete_bar_excluded():
    rows, _ = bars(400, jump_at=390, jump_bp=10.0, gap_before_jump=True)
    assert server.spikes_5m(rows)["events"] == []                      # 앞 봉 없음 = 휴장 갭 → 그 봉 수익 제외
    rows, t0 = bars(400, jump_at=399, jump_bp=10.0)
    assert server.spikes_5m(rows)["events"] == []                      # 마지막 봉은 뒤 분이 없어 미완결
    assert server.spikes_5m(rows + [(t0 + 400 * 300, rows[-1][1])])["events"][0][0] == t0 + 399 * 300


if __name__ == "__main__":
    test_spike_found_and_bounded(); test_gap_and_incomplete_bar_excluded(); print("ok")
