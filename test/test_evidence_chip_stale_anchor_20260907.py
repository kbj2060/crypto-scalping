"""증거신호 칩: 목표 도달 후 재발동 시 캐시 무효화 테스트, 2026-09-07.

지키려는 계약은 하나다 -- **앵커의 익절가가 이미 닿았으면 그 사건은 끝난 것이고, 같은 측면
원시 재발동은 새 사건이므로 그 봉에서 다시 추론해야 한다.** 그 전까지 `cache_valid`는 측면과
경과 봉수만 봐서 옛 앵커의 확률/익절가/fire_pos를 그대로 내보냈다(화면엔 "바닥 발동"과 이미
닿은 옛 익절가가 동시에 떴다).

세 갈래를 함께 못박는다. 하나만 보면 반대 방향으로 과교정하기 쉽다.
  1. 도달 후 재발동  -> 재추론 (fire_pos가 새 봉, tp_price 갱신, tp_touched=False)
  2. 미도달 재발동   -> 캐시 재사용 (기존 동작 -- 여기까지 무효화하면 매 봉 GPU 추론이 된다)
  3. 애프터글로우    -> 캐시 유지 (도달했어도 앵커를 들고 있어야 "목표 도달 · 종료"를 말한다)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import live_evidence_signal_metalabel_20260829 as metalabel  # noqa: E402

KLINES = ROOT / "data/eth_5m_1year.csv"
SIGNAL = "demarker_extreme"          # H=8, K=0.70 -- 가장 짧은 호라이즌이라 시나리오가 짧다
BCOL, ACOL = f"bottom_{SIGNAL}", f"bottom_{SIGNAL}_active"


@unittest.skipUnless(KLINES.exists(), f"{KLINES} 없음")
class StaleAnchorCacheTests(unittest.TestCase):
    """실제 klines 꼬리를 써서 피쳐가 전부 유한하게 만든 뒤, 발동 컬럼만 직접 심는다."""

    @classmethod
    def setUpClass(cls) -> None:
        raw = pd.read_csv(KLINES, parse_dates=["timestamp"]).tail(2000).reset_index(drop=True)
        cls.base = raw[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()

    def setUp(self) -> None:
        metalabel._LAST_FIRE_CACHE.clear()
        self.calls: list[int] = []                      # _predict_proba가 호출된 봉의 close 값

        def fake_proba(signal_name, feature_row):       # GPU/TabPFN 없이 캐시 경로만 검증
            self.calls.append(float(feature_row["close"]))
            return 0.5
        self._real = metalabel._predict_proba
        metalabel._predict_proba = fake_proba

    def tearDown(self) -> None:
        metalabel._predict_proba = self._real

    # ---------------------------------------------------------------- helpers
    def _cycle(self, end: int, fire: bool, touch_after: int | None = None):
        """봉 `end`까지 확정된 상태로 한 사이클 호출. touch_after: 그 위치의 고가를 +5% 띄운다."""
        df = self.base.iloc[: end + 1].copy().reset_index(drop=True)
        if touch_after is not None:
            df.loc[touch_after, "high"] = df.loc[touch_after, "high"] * 1.05
        sig = pd.DataFrame({"timestamp": df["timestamp"]})
        sig[BCOL] = False
        if fire:
            sig.loc[len(sig) - 1, BCOL] = True
        sig[ACOL] = sig[BCOL]                            # 켜진 칩만 active (fired 오버라이드 경로 통과)
        return metalabel.compute_evidence_signal_metalabels(df, sig)[SIGNAL], len(df) - 1

    # ---------------------------------------------------------------- tests
    def test_refire_after_target_hit_reinfers_at_new_bar(self) -> None:
        anchor = 1900
        o1, pos1 = self._cycle(anchor, fire=True)
        self.assertTrue(o1["fired"])
        self.assertEqual(o1["bars_since_fire"], 0)
        self.assertEqual(len(self.calls), 1)
        anchor_tp = o1["tp_price"]

        # 앵커 다음 봉에서 익절가를 넘기고, 5봉 뒤(호라이즌 8 안)에 같은 측면 재발동
        o2, pos2 = self._cycle(anchor + 5, fire=True, touch_after=anchor + 1)
        self.assertTrue(o2["fired"])
        self.assertEqual(len(self.calls), 2, "도달 후 재발동인데 재추론하지 않았다")
        self.assertEqual(o2["bars_since_fire"], 0, "fire_pos가 옛 앵커에 머물러 있다")
        self.assertNotAlmostEqual(o2["tp_price"], anchor_tp, places=6,
                                  msg="이미 닿은 옛 익절가를 그대로 내보냈다")
        self.assertFalse(o2["tp_touched"], "새 발동인데 목표 도달로 표시됐다")

    def test_refire_without_target_hit_still_reuses_cache(self) -> None:
        """과교정 방지 -- 미도달 재발동까지 무효화하면 매 봉 TabPFN을 새로 돌리게 된다."""
        anchor = 1900
        o1, _ = self._cycle(anchor, fire=True)
        o2, _ = self._cycle(anchor + 5, fire=True)       # 고가 조작 없음 = 미도달
        self.assertFalse(o2["tp_touched"])
        self.assertEqual(len(self.calls), 1, "미도달 재발동인데 캐시를 버렸다")
        self.assertEqual(o2["bars_since_fire"], 5)
        self.assertAlmostEqual(o2["tp_price"], o1["tp_price"], places=9)

    def test_afterglow_keeps_resolved_anchor(self) -> None:
        """발동이 없는 봉에서는 도달한 앵커를 계속 들고 있어야 화면이 '목표 도달 · 종료'를 말한다."""
        anchor = 1900
        o1, _ = self._cycle(anchor, fire=True)
        o2, _ = self._cycle(anchor + 5, fire=False, touch_after=anchor + 1)
        self.assertEqual(len(self.calls), 1, "애프터글로우에서 불필요하게 재추론했다")
        self.assertEqual(o2["bars_since_fire"], 5)
        self.assertTrue(o2["tp_touched"])
        self.assertAlmostEqual(o2["tp_price"], o1["tp_price"], places=9)
        self.assertFalse(o2["fired"], "_active가 꺼졌으면 fired도 꺼져야 한다")


if __name__ == "__main__":
    unittest.main()
