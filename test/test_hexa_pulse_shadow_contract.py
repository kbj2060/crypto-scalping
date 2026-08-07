from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hexa_pulse_shadow_cannot_submit_orders() -> None:
    source = (ROOT / "scripts/run_hexa_pulse_formula_shadow_20260718.py").read_text()
    assert '"order_submission_supported": False' in source
    assert '"activation_allowed": False' in source
    assert "create_order" not in source
    assert "place_order" not in source
    assert "binance_execution" not in source
    assert "INSERT INTO decisions" not in source
    assert "UPDATE decisions" not in source
    assert "DELETE FROM decisions" not in source


def test_tail_stream_uses_current_binance_market_path() -> None:
    source = (ROOT / "tail_risk_interceptor.py").read_text()
    assert "wss://fstream.binance.com/market/ws/{symbol}@forceOrder" in source
