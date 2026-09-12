"""matplotlib 한글 폰트 등록 헬퍼.

배경: 서버(`llewyn@192.168.1.89`)의 matplotlib 에는 한글 폰트가 전혀 없어서(fc-list :lang=ko 공백,
번들 폰트도 DejaVu/STIX 뿐) 한글 제목·라벨이 두부(□)로 깨졌다. 2026-09-09 사용자 지적으로 확인.

해결: OFL 라이선스인 **Noto Sans KR**(Windows 에 동봉된 `NotoSansKR-VF.ttf`)을 서버 `~/.fonts/`
에 설치하고, 이 헬퍼가 런타임에 matplotlib font_manager 에 등록한다. 저장소에 폰트 바이너리를
커밋하지 않기 위해 파일 자체는 각 머신의 `~/.fonts/` 에 두고 여기서는 경로만 탐색한다.

사용법:
    from plot_korean_font_20260909 import use_korean_font
    use_korean_font()          # rcParams 설정까지 수행, 실패하면 경고만 내고 진행

주의: 한글 폰트를 쓰면 마이너스 기호가 깨질 수 있어 `axes.unicode_minus=False` 를 같이 건다.
"""
from __future__ import annotations

from pathlib import Path

CANDIDATES = [
    Path.home() / ".fonts/NotoSansKR-VF.ttf",
    Path.home() / ".fonts/NanumGothic.ttf",
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/mnt/c/Windows/Fonts/NotoSansKR-VF.ttf"),   # WSL 로컬 렌더링용
    Path("/mnt/c/Windows/Fonts/malgun.ttf"),
]


def find_korean_font() -> Path | None:
    for p in CANDIDATES:
        if p.exists():
            return p
    return None


def use_korean_font(verbose: bool = True) -> str | None:
    """찾은 한글 폰트를 matplotlib 에 등록하고 기본 family 로 설정. 폰트명 반환(없으면 None)."""
    import matplotlib
    import matplotlib.font_manager as fm

    p = find_korean_font()
    if p is None:
        if verbose:
            print("[font] ⚠️ 한글 폰트를 찾지 못했습니다 — 한글이 깨질 수 있습니다. "
                  "~/.fonts/ 에 NotoSansKR-VF.ttf 를 두세요.", flush=True)
        return None
    fm.fontManager.addfont(str(p))
    name = fm.FontProperties(fname=str(p)).get_name()
    matplotlib.rcParams["font.family"] = name
    matplotlib.rcParams["axes.unicode_minus"] = False
    if verbose:
        print(f"[font] 한글 폰트 등록: {name}  ({p})", flush=True)
    return name
