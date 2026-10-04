"""2026-10-04 /api/paper-arms: 모의 매매 엔진 state.json -> 화면 몫. 파일 없음·멈춤을 조용히 0 으로 보이지 않는지."""
from dashboard import server


def test_paper_arms_payload():
    assert server.paper_arms_payload(None, 0) == {"available": False, "error": "paper_state_missing"}
    assert server.paper_arms_payload({"ts": 1}, 0)["available"] is False   # 옛 엔진(arms 없음)
    st = {"ts": 1000, "arms": {"w80_z0.5_a24_mh15": {"cum": 5}, "w100_z0.5_w60": {"cum": -2}, "other": {}},
          "port": {"p1": {"cum": 1.5}}, "signals": {"bar_s": 999}}
    d = server.paper_arms_payload(st, 1100)
    assert d["available"] and not d["stale"] and d["age_s"] == 100
    assert set(d["arms"]) == set(server.PAPER_ARMS + server.PAPER_NEW_ARMS) and d["port"]
    assert d["arms"]["trend4"] is None and "other" not in d["arms"]   # 재시작 전(새 판 없음)은 None -- 화면이 «재시작부터»로 그린다 == {"cum": 1.5} and d["signals"]["bar_s"] == 999
    assert server.paper_arms_payload(st, 1000 + server.PAPER_STALE_S + 1)["stale"] is True


def test_pick_paper_state():
    old = {"ts": 1000, "arms": {}}                                   # 옛 엔진: 신호 없음
    new = {"ts": 1000, "arms": {}, "signals": {"bar_s": 1}}
    assert server.pick_paper_state([old, new], 1010) is new          # 정식이 신호 전이면 미리보기
    assert server.pick_paper_state([new, dict(new)], 1010) is new    # 정식이 쓰기 시작하면 정식
    assert server.pick_paper_state([old, new], 1000 + server.PAPER_STALE_S + 1) is old   # 미리보기 멈춤 -> 정식(멈춤 표시)
    assert server.pick_paper_state([None, None], 0) is None
