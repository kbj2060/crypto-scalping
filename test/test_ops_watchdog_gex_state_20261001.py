"""ops_watchdog.check_gex_state -- 옵션 수집기 요약 JSON 신선도(옛 deribit_gex.duckdb 거짓 CRITICAL 대체)."""
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ops_watchdog as w  # noqa: E402


def test_gex_state_freshness():
    d = Path(tempfile.mkdtemp())
    w.LIVE = d
    assert w.check_gex_state().status == "BLOCKED"
    now = dt.datetime.now(dt.timezone.utc)
    for minutes, expected in [(5, "OK"), (45, "WARN"), (3281, "CRITICAL")]:
        (d / "deribit_gex_state.json").write_text(json.dumps({"generated_at": (now - dt.timedelta(minutes=minutes)).isoformat()}))
        assert w.check_gex_state().status == expected, minutes


if __name__ == "__main__":
    test_gex_state_freshness()
    print("ok")
