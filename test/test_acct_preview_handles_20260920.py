"""진입 미리보기 → 계좌 카드. 2026-10-14 카드 개편 뒤로는 포지션 줄의 «넣으면 →» 한 마디가 미리보기를 보여 준다
(옛 타일·data-pv 제자리 패처는 카드에서 주문 정보를 줄이며 없어졌다)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "dashboard/live/app.js").read_text(encoding="utf-8")
HTML = (ROOT / "dashboard/live/index.html").read_text(encoding="utf-8")
CSS = (ROOT / "dashboard/live/styles.css").read_text(encoding="utf-8")


def test_card_shows_preview_one_liner():
    """2026-10-14 카드 개편(사용자 시안 C): 타일·제자리 패처(applyAcctPreview)는 없어지고, 진입 미리보기는
    포지션 줄의 «넣으면 →» 한 마디로만 남는다 -- 미리보기가 바뀌면 카드를 다시 그려야 그 한 마디가 따라온다."""
    assert "applyAcctPreview" not in JS, "카드 개편 뒤 남은 옛 패처"
    assert 'class="acct-pv"' in JS and "entryProjPreview ||" in JS, "포지션 줄 «넣으면 →» 미리보기가 없다"
    body = JS[JS.index("function setEntryProjPreview"):][:2500]
    assert "renderSnapshotAccount();" in body, "미리보기가 바뀌어도 카드를 다시 안 그린다"


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
