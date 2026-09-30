"""시장 맥락 ③ 청산 급증 쪽별 판정(app.js mcLiqBurstHot/mcLiqBurstSide) -- 본문을 떼어 node 로 돌린다 (2026-09-30).

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


def iso(ms):
    import datetime
    return datetime.datetime.utcfromtimestamp(ms / 1000).isoformat() + "Z"


def test_hawkes_marks_only_dominant_side():
    # 실측(09-30): hawkes 켜짐 · z 롱 0.97 / 숏 −0.23 · 롱 $549k / 숏 $0 -> 롱만 급증
    bu = {"updated_at": iso(NOW - 5000), "hawkes_active": True, "z_long": 0.97, "z_short": -0.23,
          "long_usd_1m": 549084.0, "short_usd_1m": 0.0}
    assert run(bu) == [True, False, "long"]
    # 두 쪽 다 0 이면 hawkes 만으로는 아무 쪽도 아니다
    bu.update(long_usd_1m=0.0)
    assert run(bu) == [False, False, None]


def test_own_z_and_staleness():
    bu = {"updated_at": iso(NOW - 5000), "hawkes_active": False, "z_long": 0.1, "z_short": 3.2,
          "long_usd_1m": 9e5, "short_usd_1m": 1e5}
    assert run(bu) == [False, True, "short"]
    bu["updated_at"] = iso(NOW - 120000)   # 2분 낡은 파일은 급증을 말하지 않는다
    assert run(bu) == [False, False, None]
