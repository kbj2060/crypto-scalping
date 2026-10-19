from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tail_stream_uses_current_binance_market_path() -> None:
    source = (ROOT / "tail_risk_interceptor.py").read_text()
    assert "wss://fstream.binance.com/market/ws/{symbol}@forceOrder" in source
