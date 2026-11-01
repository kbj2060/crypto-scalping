"""SNDK Option 카드 (2026-11-01).   python -m pytest -q test/test_cboe_option_card_20261101.py
CBOE 체인 요약이 Deribit 요약과 같은 키로 /api/gex 에 합쳐지고, 화면이 SNDK 를 Option 코인으로 안다."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import dashboard.server as srv  # noqa: E402
import cboe_option_summary_20261101 as summ  # noqa: E402


def test_summary_selftest():
    summ._selftest()


def test_gex_payload_merges_sndk(tmp_path, monkeypatch):
    f = tmp_path / "cboe.json"
    f.write_text(json.dumps({"currencies": {"SNDK": {"options": {"src": "CBOE 지연", "index": 1.0}}}}))
    monkeypatch.setattr(srv, "CBOE_OPT_STATE_PATH", f)
    assert srv.gex_payload()["currencies"]["SNDK"]["options"]["src"] == "CBOE 지연"
    monkeypatch.setattr(srv, "CBOE_OPT_STATE_PATH", tmp_path / "없음.json")      # 파일이 없어도 깨지지 않는다
    assert "SNDK" not in srv.gex_payload().get("currencies", {})


def test_app_knows_sndk_option_coin():
    assert 'sndk: "SNDK" }[activeSnapshotAsset]' in (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")
