"""풋프린트 백필 깊이가 차트의 가장 넓은 창을 덮는다 (2026-09-23). 실행: python test/<이 파일>

왜: 백필 창이 FOOTPRINT_BARS(=12, 1시간)로 박혀 있었다. 차트에 1h 창 하나뿐이던 시절의
상수인데 그 뒤 2h/4h/12h 가 붙었다. 그래서 재시작 뒤 최근 1시간만 보장되고, 그 밖의 봉은
화면에서 셀 없는 맨 캔들로 남았다(사용자 신고). 세 상수가 같이 움직여야 하는 계약이라
파일 두 개를 걸쳐 검사한다 -- 24h 버튼을 새로 붙이면 여기서 먼저 걸린다.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
server = (ROOT / "dashboard" / "server.py").read_text(encoding="utf-8")
appjs = (ROOT / "dashboard" / "live" / "app.js").read_text(encoding="utf-8")


def const(src: str, name: str) -> int:
    m = re.search(rf"^{name}\s*=\s*(\d+)", src, re.M)
    assert m, f"{name} 을 못 찾았다"
    return int(m.group(1))


KEEP = const(server, "FOOTPRINT_KEEP_BARS")
MAXWIN = const(server, "FOOTPRINT_MAX_WINDOW_BARS")

m = re.search(r"const CHART_WINDOW_BARS\s*=\s*\[([^\]]*)\]", appjs)
assert m, "app.js 의 CHART_WINDOW_BARS 를 못 찾았다"
windows = [int(x) for x in re.findall(r"\d+", m.group(1))]
assert windows, "창 목록이 비었다"
widest = max(windows)

# 1. 백필 루프가 실제로 MAX_WINDOW 를 쓴다 (주석이 아니라 코드에서)
fi = server.index("async def footprint_backfill(")
body = server[fi:server.index("\n    async def ", fi + 10)]
code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
assert "range(FOOTPRINT_MAX_WINDOW_BARS)" in code, \
    "백필 루프가 FOOTPRINT_MAX_WINDOW_BARS 를 안 쓴다 -- 창 밖 봉이 맨 캔들로 남는다"
assert "FOOTPRINT_MAX_WINDOW_BARS - 1" in code, \
    "window_floor 가 백필 범위와 다른 상수를 쓴다 -- 루프는 도는데 전부 건너뛴다"

# 2. 백필 깊이가 가장 넓은 차트 창을 덮는다
assert MAXWIN >= widest, (
    f"차트 최대 창 {widest}봉인데 백필/스냅샷은 {MAXWIN}봉 -- "
    f"{widest - MAXWIN}봉이 셀 없는 캔들로 남는다")

# 3. 링이 그 깊이를 담을 수 있다
assert KEEP >= MAXWIN, f"링 {KEEP}봉 < 백필 {MAXWIN}봉 -- 받아도 바로 버린다"

# 4. ready 를 깊은 꼬리가 아니라 최근 구간에서 올린다 (콜드스타트에 «수집 중»이 안 굳는다)
assert "if idx == FOOTPRINT_BARS:" in code, \
    "ready 를 최근 FOOTPRINT_BARS 이후에 올리지 않는다 -- budget 소진 시 영영 False 다"

print(f"OK  차트 창 {windows} · 백필/스냅샷 {MAXWIN}봉 · 링 {KEEP}봉")
sys.exit(0)
