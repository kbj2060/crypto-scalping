"""자동 규칙 크기식(2026-10-10 동일위험 → 10-11 재생 ×15.2 식 + 손절 3%) -- 화면(app.js commitRuleL)과 서버(live_manual_peg_entry.commit_rule_l)가 같은 L·손절폭을 내는가.
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

from scripts.live_manual_peg_entry_20260912 import RULE_STOP_PCT, commit_rule_l  # noqa: E402


class RuleSizeParityTest(unittest.TestCase):
    def test_client_matches_server(self) -> None:
        js = (ROOT / "dashboard/live/app.js").read_text("utf-8")
        src = re.search(r"const RULE_STOP_PCT = [\s\S]*?\nfunction commitRuleL[\s\S]*?\n}\n", js).group(0)
        grid = [(side, s4, vm) for side in ("LONG", "SHORT") for s4 in (5.0, 20.0, 55.21, 102.06, 250.0, 400.0) for vm in (0.2, 0.458, 1.0, 3.0)]
        out = subprocess.run(["node", "-e", src + f"console.log(JSON.stringify({json.dumps(grid)}.map(([s, v, m]) => commitRuleL(s, v, m))))"],
                             capture_output=True, text=True, check=True).stdout
        for (side, s4, vm), c in zip(grid, json.loads(out)):
            py = commit_rule_l(position_side=side, sigma24_bp=s4 * 6 ** 0.5, vol_mult=vm)     # 서버: σ24 = σ(4h) × √6
            self.assertAlmostEqual(c["l"], py["l"], places=9, msg=(side, s4, vm))
            self.assertAlmostEqual(c["sd"], py["stop_pct"], places=12, msg=(side, s4, vm))
        today = commit_rule_l(position_side="LONG", sigma24_bp=55.21 * 6 ** 0.5, vol_mult=0.458)
        self.assertAlmostEqual(today["l"], 5.696, delta=1e-3)                          # 10-11 오늘 값 재현(12.435 × 0.458 = 5.6953)
        self.assertIn(f"const RULE_STOP_PCT = {RULE_STOP_PCT};", js)
        html = (ROOT / "dashboard/live/index.html").read_text("utf-8")
        self.assertEqual(html.count('class="rule-risk"'), 2)   # 카드 칩 옆 · 떠 있는 버튼 띠 -- «손절 시 순자산 −X%»는 L × 3% 로 app.js 가 채운다


if __name__ == "__main__":
    unittest.main()
