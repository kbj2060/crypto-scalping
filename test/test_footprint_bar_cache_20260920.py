"""풋프린트 봉별 캐시의 «틀리면 조용히 어긋나는» 세 자리, 2026-09-20.

풋프린트 모드는 체결이 올 때마다 캔들 SVG 를 다시 그리는데(400ms 게이트) 바뀌는 건 맨
오른쪽 봉 하나다. 그래서 봉마다 `<g>` 를 캐시하고 안 바뀐 봉은 그 노드를 다시 붙인다.
실측(소크 20초, 틱 스트림): ScriptDuration −36%(1h) / −43%(4h).

🔴이 최적화는 **틀려도 예외가 안 난다** -- 낡은 픽셀이 남을 뿐이다. 그래서 세 자리를 못박는다.

① **캐시 키에 기하 전체가 들어가야 한다.**
   «마지막 봉만 그리면 된다»는 그냥은 틀리다. 창의 고저가 바뀌면 yMin/yMax 가 움직여 모든
   셀이 자리를 옮기고, 음영은 maxBuy/maxSell 로 정규화되며, 행 크기는 ATR 로 정해진다.
   (실측으로 기하가 렌더 124회 중 0회 바뀌긴 하지만, 그건 «가격이 창 안에 머물 때»의 얘기다.
    돌파하면 바뀌고, 그때 전부 무효가 되는 것이 정확성의 근거다.)

② **델타 라벨과 POC 점은 캐시 밖이어야 한다.**
   델타 y 는 앞선 봉들의 충돌회피 결과(deltaBoxes)에 의존한다 -- 중간 봉 하나만 바뀌어도
   뒤쪽 라벨이 전부 틀어지는 사슬이라, 캐시에 들이면 조용히 어긋난다.
   POC 점은 봉을 가로지르는 폴리라인이 쓰므로 캐시를 건너뛴 봉에서도 나와야 한다.

③ **창 밖 봉은 캐시에서 지워야 한다.** 안 지우면 탭을 켜둔 채로 며칠이면 노드가 쌓인다.

⚠️진짜 검증은 브라우저다. 이 시험은 **구조가 무너지지 않았는지**만 본다. 실제 등가성은
  «캐시 경로 == 캐시 비우고 다시 그린 경로»를 DOM 으로 비교해서 확인했다(아래 재현법):
      renderSnapshotChart(); const a = svg.outerHTML;
      renderCandleSvg._barCache = new Map(); renderSnapshotChart();
      a === svg.outerHTML   // 12봉·48봉 모두 true, 기하를 흔들면 무효화되고 되돌리면 복구
  그 하네스는 라이브 픽스처가 필요해 커밋하지 않는다(docs/dashboard_pipeline_audit_20260920.md §9).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "dashboard/live/app.js"

# ①에서 키에 반드시 들어가야 하는 것들. 하나라도 빠지면 그 축이 바뀔 때 낡은 픽셀이 남는다.
GEOM_KEYS = ("w", "h", "mt", "ch", "ml", "cw", "bw", "yMin", "yMax", "candles.length",
             "rowSize", "rowPx", "maxBuy", "maxSell", "half", "fontPx", "showQty", "INK_OPACITY")


def _src() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _footprint_branch(src: str) -> str:
    """풋프린트 셀을 그리는 구간(geomSig 선언 ~ 봉 루프 끝)."""
    start = src.index("const geomSig = [")
    end = src.index("// ── 봉별 POC 선", start)
    return src[start:end]


class BarCacheKeyTest(unittest.TestCase):
    def test_geometry_signature_covers_every_axis(self) -> None:
        sig = re.search(r"const geomSig = \[(.*?)\]\.join", _src(), re.S)
        self.assertIsNotNone(sig, "geomSig 를 못 찾았다 -- 이름이 바뀌었나?")
        body = re.sub(r"\s+", "", sig.group(1))
        for key in GEOM_KEYS:
            with self.subTest(key=key):
                self.assertIn(key.replace(" ", ""), body,
                              f"{key} 가 캐시 키에 없다 -- 그 축이 바뀌면 낡은 셀이 남는다")

    def test_cache_entry_also_pins_the_bar_identity_and_ohlc(self) -> None:
        """레벨 배열의 **객체 동일성**(증분 폴링이 보장)과 캔들 OHLC(꼬리·몸통)까지 키에 있어야 한다."""
        branch = _footprint_branch(_src())
        self.assertIn("prev.levels === levelsRef", branch, "봉 데이터 신원이 키에 없다")
        self.assertIn("prev.geom === geomSig", branch, "기하가 키에 없다")
        self.assertIn("prev.i === i", branch, "봉의 x 위치(i)가 키에 없다")
        for f in ("prev.o === c.open", "prev.h === c.high",
                  "prev.l === c.low", "prev.c === c.close"):
            with self.subTest(f=f):
                self.assertIn(f, branch, f"{f} -- 꼬리/몸통이 캐시 키에 없으면 굳는다")


class BarCacheBoundaryTest(unittest.TestCase):
    def test_delta_label_is_drawn_outside_the_cached_group(self) -> None:
        """②델타 라벨은 앞선 봉들에 의존하는 사슬이라 캐시하면 안 된다."""
        branch = _footprint_branch(_src())
        self.assertIn("svg.appendChild(dTxt);", branch,
                      "델타 라벨이 봉 그룹(barG)으로 들어갔다 -- 충돌회피 사슬이 조용히 어긋난다")
        self.assertNotIn("barG.appendChild(dTxt)", branch)

    def test_poc_point_is_collected_outside_the_cache(self) -> None:
        """②POC 점은 봉을 가로지르는 폴리라인이 쓴다 -- 캐시를 건너뛴 봉에서도 나와야 한다."""
        branch = _footprint_branch(_src())
        push = branch.index("pocPts.push(")
        reuse = branch.index("if (!reuse) {")
        self.assertLess(push, reuse,
                        "pocPts.push 가 캐시 블록 안이다 -- 재사용된 봉의 POC 선이 끊긴다")

    def test_cells_and_candle_outline_go_into_the_cached_group(self) -> None:
        """반대로, 캐시해야 할 것들은 확실히 그룹 안이어야 한다(안 그러면 이득이 0)."""
        branch = _footprint_branch(_src())
        for needle in ("barG.appendChild(rect)", "barG.appendChild(poc)",
                       "barG.appendChild(wick)", "barG.appendChild(body)"):
            with self.subTest(needle=needle):
                self.assertIn(needle, branch, f"{needle} 가 없다 -- 셀이 캐시 밖이면 절감이 사라진다")

    def test_cache_is_evicted(self) -> None:
        """③창 밖 봉을 안 지우면 노드가 무한히 쌓인다."""
        branch = _footprint_branch(_src())
        self.assertIn("barCache.delete(", branch, "캐시 축출이 없다")
        self.assertIn("seenBars", branch)

    def test_single_caller_assumption_still_holds(self) -> None:
        """캐시를 함수에 달아 뒀다 -- 같은 <g> 를 두 svg 에 붙이면 **옮겨간다**(appendChild 는 이동).
        호출부가 둘이 되면 svg 별로 갈라야 한다."""
        src = _src()
        calls = re.findall(r"^\s*renderCandleSvg\(", src, re.M)
        self.assertEqual(len(calls), 1,
                         "renderCandleSvg 호출부가 늘었다 -- _barCache 를 svg 별로 갈라라")


if __name__ == "__main__":
    unittest.main()
