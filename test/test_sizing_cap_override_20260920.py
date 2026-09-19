"""사이징 상한 무시 스위치(2026-09-20 사용자 지시 «우리 사이징 코드는 잠깐 꺼줘»).

🔴실주문 크기를 정하는 경로다. 지켜야 할 것 셋:
  ① 기본값은 **꺼짐**이다 -- 환경변수 없이 저장소를 받아 돌린 사람에게 상한이 살아 있어야 한다.
  ② 상한을 확정하는 곳이 **두 군데**다(effective_cap · assemble_entry_plan 인라인). 한쪽만
     스위치를 보면 «추천 지평은 옛 상한, 실제 주문은 새 상한»으로 갈린다.
  ③ 꺼진 사실이 **화면에 뜬다**. 조용히 넘기면 화면이 «원장 상한이 묶었다»고 거짓말한다.
"""
import importlib
import os
import re
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "dashboard/server.py").read_text(encoding="utf-8")
JS = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")


def _reload(value=None):
    env = dict(os.environ)
    env.pop("DASHBOARD_SIZING_CAP_EQUITY_X", None)
    if value is not None:
        env["DASHBOARD_SIZING_CAP_EQUITY_X"] = value
    with mock.patch.dict(os.environ, env, clear=True):
        from dashboard import server
        return importlib.reload(server).SIZING_CAP_OVERRIDE_X


def test_default_is_off():
    assert _reload() == 0.0, "환경변수가 없으면 상한이 살아 있어야 한다"


def test_env_opens_and_bad_value_falls_back_to_off():
    assert _reload("20") == 20.0
    assert _reload("7.5") == 7.5
    # 오타를 «무제한»으로 읽지 않는다 -- 꺼진 쪽으로 떨어뜨린다.
    assert _reload("twenty") == 0.0
    assert _reload("") == 0.0
    assert _reload("-5") == 0.0
    _reload()          # 다른 테스트가 켜진 모듈을 물려받지 않게 되돌린다


def test_both_cap_sites_honor_the_switch():
    """상한을 확정하는 두 지점이 모두 스위치를 본다."""
    sites = re.findall(r"SIZING_CAP_OVERRIDE_X > 0 and equity > 0", SRC)
    assert len(sites) == 2, f"두 곳이어야 한다 -- 지금 {len(sites)}곳(한쪽만 고치면 갈린다)"
    assert 'return equity * SIZING_CAP_OVERRIDE_X, "override", cap' in SRC
    assert "cap_notional = equity * SIZING_CAP_OVERRIDE_X" in SRC


def test_payload_and_screen_say_it_is_off():
    assert "override_x=SIZING_CAP_OVERRIDE_X if overridden else None" in SRC
    assert 'who = ("override" if overridden' in SRC
    # 위험모델이 죽어도 남는 무조건 경고 + 위험모델 줄의 경고, 둘 다.
    assert "cap.override_x" in JS, "화면이 payload 의 override_x 를 안 읽는다"
    assert '사이징 상한 꺼짐' in JS
    assert 'r.applied_binding === "override"' in JS
