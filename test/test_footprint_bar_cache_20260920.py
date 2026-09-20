"""renderCandleSvg 캐시의 «틀리면 조용히 어긋나는» 자리들, 2026-09-20.

봉별 캐시(아래 ①~③)와 **계층 캐시**(맨 아래 ChartLayerCacheTest)를 함께 본다.

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
        self.assertGreaterEqual(len(layers), 5, f"층이 너무 적다: {sorted(layers)}")
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
