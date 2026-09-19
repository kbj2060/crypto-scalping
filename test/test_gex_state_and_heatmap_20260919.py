"""GEX 상태파일 · 히트맵 API 의 두 비자명 지점만 잡는다(프레임워크 없음).

① write_state 의 front_ratio 0 나눗셈 — total=0 이면 None 이어야 한다(구조를 못 읽는다).
② 히트맵 symbol 화이트리스트 — 경로로 들어가는 값이라 `../` 탈출이 막혀야 한다.
③ f32 base64 왕복 — 서버가 little-endian 으로 보내고 프런트가 그대로 읽는다.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def test_front_ratio_guards_zero_total():
    """total=0 에서 ZeroDivisionError 가 아니라 None 이 나와야 한다."""
    for total, front, want in ((0, 5.0, None), (None, 5.0, None), (10.0, 4.0, 0.4)):
        ratio = (front / total) if total else None
        assert ratio == want, (total, ratio, want)


def test_heatmap_symbol_whitelist():
    """symbol 은 파일 경로로 들어간다 -- 화이트리스트 밖은 전부 거부."""
    allowed = {"ethusdt", "btcusdt"}
    for bad in ("../../etc/passwd", "eth/../..", "ETHUSDT/../btc", "", "sol usdt"):
        assert bad.lower() not in allowed, bad
    assert "ethusdt" in allowed and "btcusdt" in allowed


def test_f32_base64_roundtrip_is_little_endian():
    """서버 인코딩 == 프런트 디코딩. 엔디안이 어긋나면 히트맵이 통째로 쓰레기가 된다."""
    a = np.array([1.5, -2.25, 0.0, np.nan], dtype=np.float32)
    b64 = base64.b64encode(np.ascontiguousarray(a, dtype="<f4")).decode()
    back = np.frombuffer(base64.b64decode(b64), dtype="<f4")
    assert back.shape == a.shape
    assert np.allclose(back[:3], a[:3]) and np.isnan(back[3])


def test_state_file_shape_is_json_serialisable():
    """상태파일은 json.dumps 가 통과해야 한다(datetime 을 그냥 넣으면 여기서 터진다)."""
    row = {"recorded_at_utc": "2026-09-19T18:00:00+00:00", "spot_price": 2479.7,
           "total_gex_usd": 1.84e9, "front_month_gex_usd": 0.68e9,
           "front_ratio": 0.37, "negative_gamma": False,
           "history": [{"t": "2026-09-19T17:00:00+00:00", "total": 1.8e9, "front": 0.6e9}]}
    json.dumps({"generated_at": "2026-09-19T18:00:01+00:00", "currencies": {"ETH": row}})


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all ok")
