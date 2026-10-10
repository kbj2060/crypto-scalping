"""자동 규칙 크기식(2026-10-10 동일위험 → 10-11 총명목 3배·손절 5%) -- 화면(app.js commitRuleL)과 서버(live_manual_peg_entry.commit_rule_l)가 같은 L·손절폭을 내는가.
화면은 떠 있는 버튼의 «첫 $ · 손절 가격»을 이 식으로 3초마다 그린다 -- 서버와 어긋나면 버튼이 실제 주문과 다른 크기를 말한다.
실행: python -m pytest -q test/test_rule_size_parity_20261010.py   (node 필요)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.live_manual_peg_entry_20260912 import RULE_LOSS_AT_STOP, RULE_STOP_PCT, commit_rule_l  # noqa: E402


class RuleSizeParityTest(unittest.TestCase):
    def test_client_matches_server(self) -> None:
        js = (ROOT / "dashboard/live/app.js").read_text("utf-8")
        src = re.search(r"const RULE_LOSS_AT_STOP = [\s\S]*?\nfunction commitRuleL[\s\S]*?\n}\n", js).group(0)
        grid = ["LONG", "SHORT"]   # 2026-10-11 σ 를 안 쓴다 -- 방향만 다르다(청산 안전 식)
        out = subprocess.run(["node", "-e", src + f"console.log(JSON.stringify({json.dumps(grid)}.map((s) => commitRuleL(s))))"],
                             capture_output=True, text=True, check=True).stdout
        for side, c in zip(grid, json.loads(out)):
            py = commit_rule_l(position_side=side)
            self.assertAlmostEqual(c["l"], py["l"], places=9, msg=side)
            self.assertAlmostEqual(c["sd"], py["stop_pct"], places=12, msg=side)
            self.assertAlmostEqual(py["l"], 3.0, places=12, msg=side)            # 총명목 3배(청산 안전이 안 묶는다)
        self.assertIn(f"const RULE_LOSS_AT_STOP = {RULE_LOSS_AT_STOP};", js)
        self.assertIn(f"const RULE_STOP_PCT = {RULE_STOP_PCT};", js)
        html = (ROOT / "dashboard/live/index.html").read_text("utf-8")
        self.assertEqual(html.count(f"손절 시 순자산 −{100 * RULE_LOSS_AT_STOP:g}%"), 2)   # 카드 칩 옆 · 떠 있는 버튼 띠


if __name__ == "__main__":
    unittest.main()
