"""증거신호 metalabel 을 워커에서 읽어올 때의 **봉 일치 계약**.

프레임워크 없이: python3 test/test_evidence_metalabel_bar_match_20260914.py

2026-09-14: metalabel(TabPFN) 레그만 워커로 뺐다. 워커가 뒤처졌을 때 옛 확률을 새 봉의 값인
것처럼 붙이면 화면이 조용히 틀린다 -- 그래서 나이가 아니라 «같은 봉인가»로 게이팅한다.
server.py 를 통째로 import 하면 무거우므로 함수 소스만 떼어 돌린다(swr 시험과 같은 방식).
"""
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")
start = src.index("def _same_bar(")
end = src.index("\n\n\n", start)
ns: dict = {}
exec(compile(src[start:end], "_same_bar", "exec"), ns)
same_bar = ns["_same_bar"]

# 같은 순간인데 표기가 다른 경우 -- 워커는 pandas(+00:00), 대시보드는 utc_iso(Z)를 쓴다.
assert same_bar("2026-09-13T16:10:00Z", "2026-09-13T16:10:00+00:00") is True
assert same_bar("2026-09-13T16:10:00+00:00", "2026-09-13T16:10:00.000000+00:00") is True
# 한 봉(5분) 어긋나면 병합하면 안 된다.
assert same_bar("2026-09-13T16:10:00Z", "2026-09-13T16:15:00Z") is False
# 깨진/빈 입력은 절대 True 가 아니다 -- 실패는 «미발동»으로 읽혀야 한다.
for bad in ("", None, "garbage", "2026-13-99T99:99:99Z"):
    assert same_bar(bad, "2026-09-13T16:10:00Z") is False, bad
    assert same_bar("2026-09-13T16:10:00Z", bad) is False, bad

print("metalabel 봉 일치 계약 통과 (11건)")
