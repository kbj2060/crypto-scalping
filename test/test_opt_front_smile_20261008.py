"""가까운 만기 스마일 맞추기(dashboard/server.py::front_smile) -- 닿음 등고선(2026-10-08)의 입력."""
import math
from datetime import datetime, timezone

from dashboard.server import front_smile

NOW = datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc).timestamp() * 1000   # 10-08 08:00 만기까지 12h
S, C = 2570.0, (3.0, -0.4, 0.30)   # σ(x) = 3x² − 0.4x + 0.30


def _rows(day, strikes, s=S):
    out = []
    for k in strikes:
        x = math.log(k / s)
        iv = (C[0] * x * x + C[1] * x + C[2]) * 100
        out += [{"instrument_name": f"ETH-{day}-{k}-C", "mark_iv": iv, "underlying_price": s},
                {"instrument_name": f"ETH-{day}-{k}-P", "mark_iv": iv, "underlying_price": s}]
    return out


def test_front_smile():
    ks = range(2400, 2760, 20)
    r = front_smile(_rows("7OCT26", ks) + _rows("8OCT26", ks) + _rows("10OCT26", ks), NOW)   # 7일은 지났다 → 8일
    assert r["ok"] and r["exp_ms"] == datetime(2026, 10, 8, 8, tzinfo=timezone.utc).timestamp() * 1000, r
    assert all(abs(a - b) < 1e-9 for a, b in zip(r["coef"], C)) and r["S"] == S, r
    assert r["n"] == sum(1 for k in ks if abs(math.log(k / S)) <= 0.08)
    near = datetime(2026, 10, 8, 7, 58, tzinfo=timezone.utc).timestamp() * 1000          # 만기 5분 안 → 다음 만기
    assert front_smile(_rows("8OCT26", ks) + _rows("10OCT26", ks), near)["exp_ms"] > near + 86_400_000
    assert front_smile(_rows("8OCT26", (2560, 2580, 2600)), NOW) == {"ok": False, "reason": "few_strikes", "exp_ms": r["exp_ms"]}
    assert front_smile([{"instrument_name": "garbage"}], NOW)["ok"] is False
    assert front_smile([], NOW) == {"ok": False, "reason": "no_expiry"}
