"""app.js가 참조하는 칩 id가 index.html에 실제로 있는가, 2026-09-07.

이 대시보드는 칩을 `el(id)`로 찾아 갱신하는데, **요소가 없으면 그냥 건너뛴다** --
renderModelIndicatorList의 `if (chip)` 가드 때문에 예외도, 콘솔 오류도 나지 않는다.
그래서 "app.js에는 배선했는데 index.html에 요소를 안 넣은" 실수가 조용히 통과한다.
실제로 2026-09-07 앵커 방향(MASHT)이 MODEL_CHIP_IDS와 DIRECTIONAL_MODEL_CHIP_KEYS에는
들어갔지만 index.html에는 없어서 상단 요약에 아무것도 안 떴다(사용자 신고).

두 방향을 다 본다 -- 없는 요소를 가리키는 것도, 아무도 안 쓰는 요소가 남는 것도 문제다
(후자는 2026-09-07 지속규칙/B2 제거 때 실제로 정리 대상이었다).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "dashboard/live/app.js"
INDEX = ROOT / "dashboard/live/index.html"


def chip_map(const_name: str) -> dict[str, str]:
    src = APP_JS.read_text()
    m = re.search(rf"const {const_name} = \{{(.*?)\n\}};", src, re.S)
    if not m:
        raise AssertionError(f"{const_name} 를 app.js에서 찾지 못했다 -- 이름이 바뀌었나?")
    return dict(re.findall(r'(\w+)\s*:\s*"([^"]+)"', m.group(1)))


class ChipIdWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html_ids = set(re.findall(r'id="([^"]+)"', INDEX.read_text()))

    def test_model_chip_ids_all_exist_in_html(self) -> None:
        mapping = chip_map("MODEL_CHIP_IDS")
        self.assertIn("masht_anchor", mapping, "앵커 방향 배선이 사라졌다")
        for key, element_id in sorted(mapping.items()):
            with self.subTest(key=key):
                self.assertIn(element_id, self.html_ids,
                              f"app.js가 {key} -> #{element_id} 를 쓰는데 index.html에 그 요소가 없다")

    def test_evidence_chip_ids_all_exist_in_html(self) -> None:
        for key, element_id in sorted(chip_map("EVIDENCE_STRIP_CHIP_IDS").items()):
            with self.subTest(key=key):
                self.assertIn(element_id, self.html_ids, f"{key} -> #{element_id} 가 index.html에 없다")

    def test_no_orphan_model_chip_elements_in_html(self) -> None:
        """반대 방향 -- index.html에만 남은 칩은 영원히 "-"로 남는다."""
        wired = set(chip_map("MODEL_CHIP_IDS").values())
        orphans = {i for i in self.html_ids if i.startswith("modelChip")} - wired
        self.assertEqual(orphans, set(), f"app.js가 안 쓰는 칩 요소가 남아 있다: {sorted(orphans)}")

    def test_directional_keys_are_a_subset_of_chip_keys(self) -> None:
        """방향 화살표(▲/▼)를 붙이는 집합이 칩 키를 벗어나면 그 항목은 아무 데도 안 쓰인다."""
        src = APP_JS.read_text()
        m = re.search(r"const DIRECTIONAL_MODEL_CHIP_KEYS = new Set\(\[(.*?)\]\);", src, re.S)
        self.assertIsNotNone(m, "DIRECTIONAL_MODEL_CHIP_KEYS 를 찾지 못했다")
        directional = set(re.findall(r'"(\w+)"', m.group(1)))
        self.assertIn("masht_anchor", directional, "앵커 방향은 롱/숏이 있으므로 방향 집합에 있어야 한다")
        self.assertTrue(directional <= set(chip_map("MODEL_CHIP_IDS")),
                        f"칩 키에 없는 방향 키: {sorted(directional - set(chip_map('MODEL_CHIP_IDS')))}")


if __name__ == "__main__":
    unittest.main()
