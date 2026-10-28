"""형성 봉 VWAP(app.js formingVwap) = 서버 market_ctx.vwap_rows 가 그 봉까지 센 값 (2026-10-29).

서버는 마감봉까지만 VWAP 을 준다. 화면은 직전 봉 누적(vcum·svcum)에 형성 봉을 더해 이어 센다 -- 같은 식이어야 선이 마감 때 튀지 않는다.
    python3 -m pytest -q test/test_forming_vwap_20261029.py
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import market_ctx as mctx  # noqa: E402


def forming(cs):
    src = (ROOT / "dashboard" / "live" / "app.js").read_text("utf-8")
    fn = re.search(r"^function formingVwap\(.*?^}\n", src, re.S | re.M).group(0)
    js = fn + f"const cs = {json.dumps(cs)}; formingVwap(cs); console.log(JSON.stringify(cs[cs.length - 1]));"
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


def bars(ts):
    hi = [2400 + 3 * (i % 5) for i in range(len(ts))]
    lo = [h - 4 - i % 3 for i, h in enumerate(hi)]
    cl = [h - 1.5 for h in hi]
    vol = [100.0 + 37 * (i % 7) for i in range(len(ts))]
    return hi, lo, cl, vol


def check(ts):
    hi, lo, cl, vol = bars(ts)
    r = mctx.vwap_rows(ts, hi, lo, cl, vol, nd=8)
    cs = [{"time": t, "high": h, "low": l, "close": c, "volume": v, **r[t]} for t, h, l, c, v in zip(ts, hi, lo, cl, vol)]
    last = {k: cs[-1][k] for k in ("time", "high", "low", "close")}
    got = forming(cs[:-1] + [dict(last, lvol=vol[-1])])   # 서버 VWAP 없는 형성 봉 · 거래량은 화면이 센 값
    want = r[ts[-1]]
    for k in ("vwap", "vsd", "vcum", "svwap", "svsd", "svcum"):
        assert abs(got[k] - want[k]) < 1e-6, (k, got[k], want[k])
    assert got["sstart"] == want["sstart"] and got["vlive"]


def test_same_as_server_mid_day():
    check([1_793_190_000 + 300 * i for i in range(30)])


def test_day_rollover_restarts():
    d = 1_793_232_000   # UTC 00:00
    check([d - 300 * 6 + 300 * i for i in range(7)])   # 마지막 봉 = 00:00 → 하루·아시아 세션 둘 다 다시


def test_no_volume_extends_flat_and_server_vwap_kept():
    ts = [1_793_190_000 + 300 * i for i in range(3)]
    hi, lo, cl, vol = bars(ts)
    r = mctx.vwap_rows(ts, hi, lo, cl, vol, nd=8)
    cs = [{"time": t, "high": h, "low": l, "close": c, **r[t]} for t, h, l, c in zip(ts[:2], hi, lo, cl)]
    got = forming(cs + [{"time": ts[2], "high": 9e3, "low": 1.0, "close": 5e3}])   # 거래량 0 → 직전 값 그대로 이어 그린다
    assert got["vwap"] == cs[-1]["vwap"]
    srv = dict(cs[-1], time=ts[2], vwap=1.0)
    assert forming(cs + [srv])["vwap"] == 1.0   # 서버가 준 형성 봉 VWAP(SOL·XRP 경로)은 덮지 않는다
