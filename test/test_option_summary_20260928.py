"""옵션 요약(scripts/collect_deribit_option_gex_20260815.py::options_summary) 계산 점검 -- 네트워크 없이 합성 체인으로.
max pain · 25Δ 리스크 리버설 부호 · 콜/풋 미결제 · 감마 플립 없음(콜만 있는 체인) · 보험 OTM 목록."""
import sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_deribit_option_gex_20260815 as gex  # noqa: E402


def _chain():
    now = datetime.now(timezone.utc)
    exp = (now + timedelta(days=2)).replace(microsecond=0)
    rows = []
    for k in (1800, 1900, 2000, 2100, 2200):
        for t in ("call", "put"):
            iv = 50.0 + (8.0 if t == "put" and k < 2000 else 0.0)        # 풋 쪽 스큐 -> RR 음수
            rows.append({"recorded_at_utc": now, "currency": "ETH", "instrument_name": f"x{k}{t}", "option_type": t,
                         "strike": float(k), "expiration_ts": pd.Timestamp(exp), "days_to_expiry": 2.0,
                         "open_interest": 100.0 if k == 2000 else 10.0, "mark_iv": iv, "underlying_price": 2000.0,
                         "mark_price": 0.01, "volume": 0.0,
                         "gamma_bs": gex._bs_gamma(2000.0, k, iv, 2 / 365)})
    return pd.DataFrame(rows)


def test_options_summary_synthetic(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(gex, "_pub", boom)                 # index/dvol/rv7 조회 실패 -> None, 계산은 계속
    o = gex.options_summary(_chain(), "ETH")
    assert o["index"] is None and o["dvol"] is None and o["rv7"] is None
    e = o["expiries"][0]
    assert e["pain"] == 2000.0, "미결제가 몰린 행사가가 max pain"
    assert e["rr25"] < 0, "풋 IV 가 높으면 리스크 리버설은 음수"
    assert abs(e["call_oi_usd"] - 140 * 2000) < 1e-6 and abs(e["pc"] - 1.0) < 1e-9
    assert o["gamma"]["now_usd"] != 0 and len(o["gamma"]["profile"]) == 25
    assert all(h["type"] in ("P", "C") and h["usd"] > 0 for h in o["hedge"])
    assert {h["type"] for h in o["hedge"]} == {"P", "C"}, "지수 아래 풋 · 위 콜"
    fr = {r[0]: r for r in o["strikes"]["front"]}
    assert set(fr) == {1900.0, 2000.0, 2100.0}, "사다리 = 지수 ±8% 행사가만(2000 기준 1840~2160)"
    assert fr[2000.0][1] == fr[2000.0][2] == 100 * 2000 and fr[2000.0][3] < fr[2100.0][3] + 1e9


def test_gamma_uses_nearest_expiry_only(monkeypatch):
    """2026-09-29: 딜러 감마 = 가장 가까운 만기 하나. 먼 만기에 콜을 잔뜩 쌓아도 now_usd 부호가 안 바뀐다."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    near = _chain()
    near.loc[near.option_type == "call", "open_interest"] = 1.0          # 가까운 만기는 풋 우세 -> 음감마
    far = _chain().assign(expiration_ts=near.expiration_ts.iloc[0] + pd.Timedelta(days=60), days_to_expiry=62.0)
    far = far[far.option_type == "call"].assign(open_interest=1e5)       # 먼 만기 콜 산더미(+)
    o = gex.options_summary(pd.concat([near, far], ignore_index=True), "ETH")
    only = gex.options_summary(near, "ETH")
    assert o["gamma"]["now_usd"] < 0 and abs(o["gamma"]["now_usd"] - only["gamma"]["now_usd"]) < 1e-6
    assert o["gamma"]["exp_ms"] == int(near.expiration_ts.iloc[0].timestamp() * 1000)
