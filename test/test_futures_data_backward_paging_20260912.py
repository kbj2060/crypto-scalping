"""`/futures/data/*` 역방향 페이징 가드. 네트워크 없이 돈다.

바이낸스는 이 엔드포인트군에서 startTime 을 **무시하고** endTime 기준 최근 500행만 준다
(2026-09-12 전수 실측: ETH/BTC/SOL/XRP/HYPE x retail/ttc/ttp/tkv = 20/20). 보존은 약 30.9일.
그 동작을 그대로 흉내 내는 가짜 응답으로, 전방 페이징이 1.7일에서 멈추고 역방향 페이징이
요청한 창을 채우는지 확인한다.

`python3 test/test_futures_data_backward_paging_20260912.py`
"""
import importlib.util, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP = 5 * 60 * 1000          # 5분
LIMIT = 500
RETENTION_MS = 31 * 24 * 3600 * 1000
NOW = 1_800_000_000_000
NOW -= NOW % STEP

class FakeResp:
    def __init__(self, rows): self._rows = rows
    def raise_for_status(self): pass
    def json(self): return self._rows

def fake_get(url, params=None, timeout=None):
    """startTime 은 무시, endTime 만 존중, 30일보다 앞은 빈 배열 -- 실제 API 그대로."""
    end = int(params.get("endTime", NOW))
    end -= end % STEP
    oldest_allowed = NOW - RETENTION_MS
    rows = []
    for i in range(LIMIT):
        ts = end - (LIMIT - 1 - i) * STEP
        if ts < oldest_allowed:
            continue
        rows.append({"timestamp": ts, "longShortRatio": "1.5"})
    return FakeResp(rows)

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m

mod = load(ROOT / "scripts" / "live_regime_wide24_signal_20260826.py", "rw_under_test")
mod.requests = types.SimpleNamespace(get=fake_get)
mod.time = types.SimpleNamespace(sleep=lambda *_: None)

DAYS = 15
start = NOW - DAYS * 24 * 3600 * 1000
df = mod._fetch_data_api("/futures/data/globalLongShortAccountRatio", "ETHUSDT",
                         start, NOW, {"longShortRatio": "count_long_short_ratio"})
span_days = (df.timestamp.max() - df.timestamp.min()).total_seconds() / 86400
assert len(df) > 4000, f"역방향 페이징이 창을 못 채웠다: {len(df)}행"
assert span_days > DAYS - 1, f"커버 {span_days:.1f}일 < 요청 {DAYS}일"
assert df.timestamp.is_monotonic_increasing and not df.timestamp.duplicated().any()

# 보존 한계를 넘겨 요청해도 무한루프에 빠지지 않고 있는 만큼만 준다.
far = NOW - 60 * 24 * 3600 * 1000
df2 = mod._fetch_data_api("/futures/data/globalLongShortAccountRatio", "ETHUSDT",
                          far, NOW, {"longShortRatio": "count_long_short_ratio"})
got_days = (df2.timestamp.max() - df2.timestamp.min()).total_seconds() / 86400
assert 29 < got_days < 32, f"보존 한계 처리 이상: {got_days:.1f}일"

print(f"역방향 페이징 OK — 15일 요청 {len(df):,}행/{span_days:.1f}일, "
      f"60일 요청은 보존 한계 {got_days:.1f}일에서 정지")
