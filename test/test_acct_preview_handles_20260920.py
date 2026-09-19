"""진입 미리보기는 계좌 카드 노드를 **제자리에서** 고친다(applyAcctPreview) -- 렌더와
패처가 `data-pv` 로만 만난다. 손잡이 이름이 한쪽에서 바뀌면 querySelector 가 null 을
돌려주고 패처는 조용히 아무것도 안 한다(`if (!t) return;`). 에러도 안 난다 --
그래서 이 드리프트는 사람이 화면을 봐야만 발견된다. 이 검사가 그 자리를 지킨다.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")
HTML = (ROOT / "dashboard/live/index.html").read_text(encoding="utf-8")
CSS = (ROOT / "dashboard/live/styles.css").read_text(encoding="utf-8")


def test_every_queried_handle_is_emitted():
    emitted = set(re.findall(r'data-pv="([a-z]+)"', JS)) | set(re.findall(r'\$\{tile\("(\w+)"', JS))
    queried = set(re.findall(r'q\("(\w+)"\)', JS)) | set(re.findall(r'setTile\("(\w+)"', JS))
    assert queried, "패처가 손잡이를 안 읽는다 — applyAcctPreview 가 사라졌나?"
    assert queried <= emitted, f"렌더가 안 내는 손잡이를 읽는다: {sorted(queried - emitted)}"
    assert emitted <= queried, f"아무도 안 읽는 손잡이가 남았다: {sorted(emitted - queried)}"
    # 타일 키가 실제로 속성으로 나가는지 -- tile() 이 data-pv 를 안 붙이면 위 교집합이 거짓이 된다.
    assert 'class="acct-tile" data-pv="${key}"' in JS


def test_preview_classes_have_styles():
    """패처가 붙이는 클래스·삽입하는 노드에 스타일이 없으면 «실제 계좌인 척»하게 된다."""
    assert ".acct-tiles.preview .acct-tile" in CSS, "미리보기 점선 테두리가 없다"
    assert ".acct-rail.entry-rail u" in CSS, "유령 눈금 스타일이 없다"
    assert ".acct-rail i, .acct-rail u, .acct-gauge-knob" in CSS, "transition 대상에 유령 눈금이 빠졌다"
    assert 'class="acct-pv-cap' in JS, "«실제 계좌 아님» 라벨이 없다"


def test_entry_proj_block_is_gone():
    """2026-09-20 «지금 넣으면» 4행 삭제 -- 계좌 카드가 같은 값을 그린다. 주석 말고 코드에
    남아 있으면 같은 사실을 두 곳이 말하는 상태다."""
    for dead in ("entryProj\"", "entry-proj", "data-proj="):
        assert dead not in HTML, f"index.html 에 {dead} 가 남았다"
        assert dead not in CSS, f"styles.css 에 {dead} 가 남았다"
    for dead in ("renderEntryProj", "PROJ_SPEC", 'el("entryProj")'):
        assert dead not in JS, f"app.js 에 {dead} 가 남았다"


def test_both_gauges_refresh_the_preview():
    """진입 비율과 **레버리지** 둘 다 크기를 바꾼다 -- 한쪽만 다시 물으면 카드가 굳는다."""
    lev = JS[JS.index('el("snapLevGauge")?.addEventListener("input"'):][:700]
    assert "manualEntryRefreshSize" in lev, "레버리지 게이지가 미리보기를 갱신하지 않는다"
    assert "setEntryProjPreview(plan)" in JS, "갱신 결과가 카드로 안 간다"
