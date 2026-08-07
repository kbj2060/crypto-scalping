import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_eth_micro_scalp_v4_shadow_bot_20260718.py"
SPEC = importlib.util.spec_from_file_location("eth_micro_scalp_v4_shadow_bot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _decision_db(path: Path, rows: list[tuple]) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE decisions (
                timestamp TIMESTAMP PRIMARY KEY,
                close DOUBLE NOT NULL,
                previous_position INTEGER NOT NULL,
                target_position INTEGER NOT NULL
            )
            """
        )
        connection.executemany("INSERT INTO decisions VALUES (?, ?, ?, ?)", rows)
    finally:
        connection.close()


def test_shadow_bot_has_no_order_capability() -> None:
    source = inspect.getsource(MODULE)
    for forbidden in (
        "binance_execution",
        "BinanceFuturesExecutionAdapter",
        "create_order",
        "cancel_order",
        "place_order",
        "submit_order",
    ):
        assert forbidden not in source
    assert '"activation_allowed": False' in source
    assert '"order_submission_supported": False' in source


def test_shadow_pnl_uses_only_the_following_completed_close(tmp_path: Path) -> None:
    database = tmp_path / "shadow.duckdb"
    _decision_db(
        database,
        [
            ("2026-07-18 04:00:00", 100.0, 0, 1),
            ("2026-07-18 04:01:00", 101.0, 1, 1),
            ("2026-07-18 04:02:00", 99.0, 1, 0),
        ],
    )
    assert MODULE.settle_shadow_pnl(database, (4.5,)) == 2
    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT decision_timestamp, settlement_timestamp, position,
                   price_return, turnover, cost_return, net_return
            FROM shadow_pnl ORDER BY decision_timestamp
            """
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 2
    assert str(rows[0][0]) == "2026-07-18 04:00:00"
    assert str(rows[0][1]) == "2026-07-18 04:01:00"
    assert rows[0][2] == 1
    assert rows[0][3] == pytest.approx(0.01)
    assert rows[0][4] == pytest.approx(1.0)
    assert rows[0][5] == pytest.approx(0.00045)
    assert rows[0][6] == pytest.approx(0.00955)
    assert str(rows[-1][0]) == "2026-07-18 04:01:00"
    assert str(rows[-1][1]) == "2026-07-18 04:02:00"
    assert MODULE.settle_shadow_pnl(database, (4.5,)) == 0


def test_latest_decision_is_never_settled_without_a_future_close(tmp_path: Path) -> None:
    database = tmp_path / "shadow.duckdb"
    _decision_db(
        database,
        [("2026-07-18 04:00:00", 100.0, 0, -1)],
    )
    assert MODULE.settle_shadow_pnl(database, (4.5,)) == 0
    connection = duckdb.connect(str(database), read_only=True)
    try:
        count = connection.execute("SELECT count(*) FROM shadow_pnl").fetchone()[0]
    finally:
        connection.close()
    assert count == 0


def test_exit_cost_is_charged_when_cash_decision_is_settled(tmp_path: Path) -> None:
    database = tmp_path / "shadow.duckdb"
    _decision_db(
        database,
        [
            ("2026-07-18 04:00:00", 100.0, 1, 0),
            ("2026-07-18 04:01:00", 120.0, 0, 0),
        ],
    )
    assert MODULE.settle_shadow_pnl(database, (4.5,)) == 1
    connection = duckdb.connect(str(database), read_only=True)
    try:
        gross, turnover, cost, net = connection.execute(
            "SELECT gross_return, turnover, cost_return, net_return FROM shadow_pnl"
        ).fetchone()
    finally:
        connection.close()
    assert gross == pytest.approx(0.0)
    assert turnover == pytest.approx(1.0)
    assert cost == pytest.approx(0.00045)
    assert net == pytest.approx(-0.00045)


def test_build_report_contract_fails_closed_on_artifact_mismatch() -> None:
    runtime = MODULE.binding.observer.load_runtime(device_name="cpu")
    report = {
        "model_id": MODULE.MODEL_ID,
        "model_sha256": "wrong",
        "fresh_start_utc": str(MODULE.FRESH_START_UTC),
        "published": True,
        "parity": {"pass": True},
        "stream_contract": {"pass": True},
        "order_endpoints_used": False,
        "output": str(MODULE.FEATURE_STREAM_PATH),
    }
    with pytest.raises(RuntimeError, match="model_sha256"):
        MODULE._validate_build_report(report, runtime)


def test_stale_feature_stream_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_stream = tmp_path / "features.csv"
    report_path = tmp_path / "feature_build.json"
    runtime = SimpleNamespace(model_sha256="exact-hash")
    report_path.write_text(
        json.dumps(
            {
                "model_id": MODULE.MODEL_ID,
                "model_sha256": runtime.model_sha256,
                "fresh_start_utc": str(MODULE.FRESH_START_UTC),
                "published": True,
                "parity": {"pass": True},
                "stream_contract": {"pass": True},
                "order_endpoints_used": False,
                "output": str(feature_stream),
            }
        )
    )
    stale = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=10)
    monkeypatch.setattr(
        MODULE.binding.observer,
        "load_feature_stream",
        lambda _path: pd.DataFrame({"timestamp": [stale]}),
    )
    with pytest.raises(RuntimeError, match="stream is stale"):
        MODULE._load_validated_stream(
            runtime,
            max_stream_age_minutes=5.0,
            feature_stream=feature_stream,
            report_path=report_path,
        )
