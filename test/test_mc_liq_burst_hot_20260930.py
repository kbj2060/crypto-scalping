"""청산 급증 쪽별 판정(app.js mcLiqBurstHot/mcLiqBurstSide) -- 본문을 떼어 node 로 돌린다 (2026-09-30 · 10-04 정의 교체).

옛 판은 hawkes_active 하나로 z −0.24 인 숏 막대까지 «급증»이었다.
    python3 -m pytest -q test/test_mc_liq_burst_hot_20260930.py
"""
import json
import pathlib
import re
import subprocess

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"
NOW = 1_790_000_000_000


def run(bu, now_ms=NOW):
    src = APP.read_text("utf-8")
    fns = "".join(re.search(rf"^function {n}\(.*?^}}\n", src, re.S | re.M).group(0)
                  for n in ("mcLiqBurstHot", "mcLiqBurstSide"))
    js = (f"const LIQ_BURST_STALE_MS = 60000;\n{fns}const bu = {json.dumps(bu)};\n"
          f"console.log(JSON.stringify([mcLiqBurstHot(bu, 'long', {now_ms}), mcLiqBurstHot(bu, 'short', {now_ms}),"
          f" mcLiqBurstSide(bu, {now_ms})]));")
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


THR = [338404, 545795]   # 서버 LIQ_BURST_60S_USD(롱 청산, 숏 청산)


def test_burst_is_research_event_not_small_z():
    """2026-10-04 판정 = 직전 60초 그 쪽 청산 합 > 문턱(H2 와 같은 사건). 옛 판은 z≥3 이라 $1,711 에도 켜졌다."""
    ts = NOW / 1000 - 5
    assert run({"long_usd_60s": 1711.0, "short_usd_60s": 0.0, "thr": THR, "ts": ts}) == [False, False, None]   # 작은 청산 = 급증 아님
    assert run({"long_usd_60s": 4e5, "short_usd_60s": 1e5, "thr": THR, "ts": ts}) == [True, False, "long"]
    assert run({"long_usd_60s": 4e5, "short_usd_60s": 6e5, "thr": THR, "ts": ts}) == [True, True, "long"]      # 둘 다면 문턱 대비 큰 쪽(1.18 > 1.10)
    assert run({"long_usd_60s": 3.5e5, "short_usd_60s": 9e5, "thr": THR, "ts": ts}) == [True, True, "short"]


def test_stale_or_missing():
    assert run({"long_usd_60s": 9e5, "short_usd_60s": 0.0, "thr": THR, "ts": NOW / 1000 - 120}) == [False, False, None]   # 2분 낡음
    assert run(None) == [False, False, None]                                    # 다른 코인(서버 None)
