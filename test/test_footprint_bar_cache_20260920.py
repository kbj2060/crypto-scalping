"""renderCandleSvg 의 «틀리면 조용히 어긋나는» 자리들, 2026-09-20 · 2026-10-20 캔버스로 계약이 옮겨갔다.

2026-10-20 사용자 «4번·5번도 고쳐서»: 칸·캔들·델타·POC·청산 원·레인 막대·밀도 지도가 svg 밑 캔버스 두 장(fpCv)으로
옮겨가며 옛 봉별 <g> 캐시(_barCache)·델타 노드 캐시는 사라졌다(노드를 안 만드니 캐시할 것이 없다). 대신 **형성 봉만
다시 칠하는 길(fpCv.live)** 이 생겼고, 옛 캐시가 지키던 «틀리면 낡은 픽셀이 남는» 자리를 이제 이 길이 지킨다:

① **전체 렌더로 넘길 조건.** 형성 봉 하나만 다시 칠해도 되는 건 축(yMin/yMax)·행 크기(ATR)·밀도 쓸림이 그대로일 때뿐이다.
   셋 중 하나라도 바뀌면 다른 봉·svg 꼬리표가 전부 자리를 옮겨야 하므로 false(전체 렌더)를 돌려야 한다.
② **음영 기준·델타 «전부 아니면 전무»는 fg 통째로.** 둘 다 형성 봉이 바꿀 수 있고(창 최대 칸, 밀려난 라벨 비율)
   바뀌면 모든 봉의 칠이 바뀐다 -- 형성 봉 자리만 칠하면 조용히 어긋난다.
③ **사슬은 형성 봉 앞에서 끊어 다시 잇는다.** 델타 y 는 앞 봉들의 충돌회피(deltaBoxes) 결과라 형성 봉 직전 길이로
   되돌린 뒤 다시 계산해야 하고, POC 선도 형성 봉 점만 갈아 끼워야 한다.

⚠️진짜 검증은 브라우저다(전/후 픽셀 비교 · 형성 봉 갱신 = 전체 렌더와 같은 픽셀). 이 시험은 구조가 무너지지 않았는지만 본다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

APP_JS = Path(__file__).resolve().parents[1] / "dashboard/live/app.js"


def _src() -> str:
    return APP_JS.read_text(encoding="utf-8")


def _live(src: str) -> str:
    """형성 봉 갱신 본문(fpCv.live = () => { ... })."""
    i = src.index("fpCv.live = () => {")
    return src[i:src.index("\n    };\n", i)]


class LiveFrameFallbackTest(unittest.TestCase):
    def test_full_render_when_axis_row_or_density_moves(self) -> None:
        body = _live(_src())
        for needle in ("if (y0 !== yMin || y1 !== yMax) return false;",          # 축
                       "!== rowSize) return false;",                              # 행 크기(ATR)
                       "!densitySweep.has(b.price)",                             # 밀도 쓸림
                       "tfT(last.time) !== c.time"):                             # 새 봉
            with self.subTest(needle=needle):
                self.assertIn(needle, body, f"{needle} -- 이 축이 바뀌어도 형성 봉만 칠해 낡은 그림이 남는다")

    def test_shade_and_delta_crowd_repaint_whole_fg(self) -> None:
        body = _live(_src())
        self.assertIn("mb !== maxBuy || ms !== maxSell", body, "음영 기준(창 최대 칸) 변화를 안 본다")
        self.assertIn("on !== dState.on", body, "델타 «전부 아니면 전무» 판정 변화를 안 본다")
        self.assertIn("wide ? ml :", body, "기준이 바뀌었는데 fg 를 통째로 안 칠한다")

    def test_chains_are_cut_before_the_forming_bar(self) -> None:
        body = _live(_src())
        self.assertIn("deltaBoxes.length = live.dbN;", body, "델타 사슬을 형성 봉 앞에서 안 끊는다 -- 라벨이 계속 밀려난다")
        self.assertIn("pocPts.length = live.pocN;", body, "POC 선에 형성 봉 점이 쌓인다")
        self.assertLess(body.index("deltaBoxes.length = live.dbN;"), body.index("deltaOf(iL, inf)"))

    def test_live_frame_never_touches_svg(self) -> None:
        """형성 봉 갱신은 svg 를 안 건드린다 -- 건드리면 호버 툴팁·십자선이 깜빡이고 비용이 전체 렌더로 돌아간다."""
        body = _live(_src())
        for bad in ("createElementNS", "innerHTML", "appendChild", "renderSnapshotChart("):
            with self.subTest(bad=bad):
                self.assertNotIn(bad, body)

    def test_single_caller_assumption_still_holds(self) -> None:
        """캔버스 두 장·형성 봉 갱신(fpCv)은 차트 하나에 달려 있다 -- 호출부가 둘이 되면 svg 별로 갈라야 한다."""
        calls = re.findall(r"^\s*renderCandleSvg\(", _src(), re.M)
        self.assertEqual(len(calls), 1, "renderCandleSvg 호출부가 늘었다 -- fpCv 를 svg 별로 갈라라")


class ChartLayerCacheTest(unittest.TestCase):
    """가격 플롯 **바깥** 층(격자·x눈금·레짐 리본·구간 줄·OI/청산 레인·청산밀도)의 캐시.

    실측(정상 상태에서 렌더당 만들어지는 노드): 풋프린트 4h 372 -> 27 · 청산맵 574 -> 169.
    🔴여기서 틀리는 방법은 둘뿐이고 둘 다 조용하다:
      ① 층의 draw() 안에서 `svg` 에 직접 붙이면 그 노드는 **캐시를 우회**해 매 렌더 쌓인다.
      ② 층의 키에 그 층의 데이터 신원이 빠지면 새 값이 와도 **낡은 그림이 남는다**.
    ⚠️현재가·진행봉 OHLC 에 의존하는 것(가격 라벨·이벤트 삼각형)은 **캐시하면 안 된다** --
      틱마다 바뀌므로 캐시가 무의미할 뿐 아니라, 키에 안 넣으면 그대로 굳는다.
    """

    @staticmethod
    def _layers(src: str) -> dict[str, str]:
        """cachedLayer("name", sig, (g) => { ... }) 의 이름 -> 본문."""
        out = {}
        for m in re.finditer(r'cachedLayer\("(\w+)",\s*(.*?),\s*\((\w+)\) => \{', src, re.S):
            name, sig, var = m.group(1), m.group(2), m.group(3)
            i = src.index("{", m.end() - 1)
            depth, j = 0, i
            while j < len(src):
                if src[j] == "{":
                    depth += 1
                elif src[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out[name] = (sig, var, src[i:j])
        return out

    def test_helper_pins_geometry_and_bar_times(self) -> None:
        src = _src()
        helper = re.search(r"const cachedLayer = .*?\n  \};", src, re.S)
        self.assertIsNotNone(helper, "cachedLayer 를 못 찾았다")
        self.assertIn("baseGeomSig", helper.group(0), "층 키에 기하가 없다")
        self.assertIn("timesSig", helper.group(0), "층 키에 봉 시각이 없다")

    def test_every_layer_draws_only_into_its_own_group(self) -> None:
        """①`svg.appendChild` 가 층 본문에 있으면 그 노드는 캐시를 우회한다."""
        layers = self._layers(_src())
        self.assertGreaterEqual(len(layers), 2, f"층이 너무 적다: {sorted(layers)}")   # 2026-10-20 밀도·레인·청산 원은 캔버스(fpCv)로 -- 격자(x눈금)·호가 띠가 남는다
        for name, (_sig, var, body) in sorted(layers.items()):
            with self.subTest(layer=name):
                self.assertNotIn("svg.appendChild", body,
                                 f"{name} 층이 svg 에 직접 붙인다 -- 캐시를 우회한다")
                # 직접 붙이든(`g.appendChild`) 헬퍼에 넘기든(`drawLaneTrack(y, g)`) 상관없다.
                # 지켜야 할 것은 «그 그룹을 실제로 쓴다»이다 -- 안 쓰면 아무 데도 안 그려진다.
                self.assertIn(var, body, f"{name} 층이 제 그룹을 안 쓴다")

    def test_data_backed_layers_carry_their_identity(self) -> None:
        """②격자만 기하로 충분하다. 나머지는 제 데이터 신원이 키에 있어야 한다."""
        layers = self._layers(_src())
        for name, (sig, _v, _b) in sorted(layers.items()):
            with self.subTest(layer=name):
                if name == "grid":
                    continue          # 격자·x눈금은 기하와 봉 시각만 본다
                self.assertIn("objToken(", sig,
                              f"{name} 층 키에 데이터 신원이 없다 -- 새 값이 와도 안 바뀐다")

    def test_price_labels_and_event_markers_stay_uncached(self) -> None:
        """⚠️현재가·봉 고저에 의존하는 것은 캐시하면 굳는다."""
        layers = self._layers(_src())
        joined = " ".join(b for _s, _v, b in layers.values())
        self.assertNotIn("priceLabels.forEach", joined, "가격 라벨이 층 캐시에 들어갔다")
        self.assertNotIn("(cm.events || []).forEach", joined, "이벤트 삼각형이 층 캐시에 들어갔다")

    def test_density_history_is_memoized(self) -> None:
        """밀도 이력은 렌더마다 ~1,000개 객체를 새로 만들고 있었다. 신원이 안정돼야 층 키도 선다."""
        src = _src()
        self.assertIn("_densityMemo", src, "liquidationDensityHistory 가 memoize 되지 않았다")
        self.assertIn("if (_densityMemo.src === map) return _densityMemo.out;", src)


if __name__ == "__main__":
    unittest.main()
