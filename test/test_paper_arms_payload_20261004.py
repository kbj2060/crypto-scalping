"""2026-10-04 /api/paper-arms: 모의 매매 엔진 state.json -> 화면 몫. 파일 없음·멈춤을 조용히 0 으로 보이지 않는지."""
from dashboard import server


def test_paper_arms_payload():
    assert server.paper_arms_payload(None, 0) == {"available": False, "error": "paper_state_missing"}
    assert server.paper_arms_payload({"ts": 1}, 0)["available"] is False   # 옛 엔진(arms 없음)
    st = {"ts": 1000, "arms": {"w80_z0.5_a24_mh15": {"cum": 5}, "w100_z0.5_w60": {"cum": -2}, "other": {}},
          "port": {"p1": {"cum": 1.5}}, "signals": {"bar_s": 999}}
    d = server.paper_arms_payload(st, 1100)
    assert d["available"] and not d["stale"] and d["age_s"] == 100
    assert set(d["arms"]) == set(server.PAPER_ARMS) and d["port"] == {"cum": 1.5} and d["signals"]["bar_s"] == 999
    assert server.paper_arms_payload(st, 1000 + server.PAPER_STALE_S + 1)["stale"] is True
