"""SNDK(Binance TradFi 주식 선물) 풋프린트 연결: 현물 없음 · 칸이 테이프 칸의 정수배 · COIN_CONFIG 키."""
from dashboard.coin_flow import FLOW_SPECS
from scripts.coin_config import COIN_CONFIG
from scripts.live_okx_trade_tape_collector_20260923 import BUCKETS as OKX_BUCKETS
from scripts.live_trade_tape_collector_20260916 import BUCKETS as BN_BUCKETS


def test_sndk_flow_spec():
    s = FLOW_SPECS["sndk"]
    assert s.symbol == "SNDKUSDT" and s.okx_inst == "SNDK-USDT-SWAP"
    assert s.spot_url is None and FLOW_SPECS["eth"].spot_url and FLOW_SPECS["xrp"].spot_url   # 현물은 SNDK 만 없다
    for tape in (BN_BUCKETS["sndkusdt"], OKX_BUCKETS["SNDK-USDT-SWAP"]):    # 복원이 칸 경계를 안 찢으려면 정수배
        assert abs(s.bucket / tape - round(s.bucket / tape)) < 1e-9


def test_sndk_coin_config_has_every_key():
    assert set(COIN_CONFIG["sndk"]) == set(COIN_CONFIG["xrp"]) and COIN_CONFIG["sndk"]["binance_symbol"] == "SNDKUSDT"
