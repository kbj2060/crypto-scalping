"""deploy_watcher: 대시보드 재시작 판정 테스트, 2026-09-07.

지키려는 계약 -- **대시보드가 import하는 모듈이 바뀌면 재시작한다.** 2026-09-07 이전에는
`^dashboard/` 경로만 봤다. 그래서 `scripts/live_*.py`(server.py가 import하는 것)를 고쳐 배포하면
워처가 머지하고 last_deployed_sha를 갱신하고 `deploy OK`까지 찍은 뒤 재시작을 안 했고, 대시보드는
**살아서 옛 코드를 계속 돌렸다** -- 포트 체크도 curl도 초록이라 어느 확인 절차로도 안 잡혔다.

반대 방향(과교정)도 같이 못박는다. 대시보드가 import하지 않는 섀도우 러너나 리서치 스크립트까지
재시작을 걸면, 파일 하나 고칠 때마다 라이브 대시보드가 내려간다 -- trading_bot_modules/odyssey_*
를 제외한 것과 같은 판단이다.

워처 파일에서 함수 원문을 그대로 떼어다 실행하므로 복사본이 아니라 **실제 배포되는 코드**를 본다.
"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "scripts/ops/deploy_watcher.sh"
SERVER = ROOT / "dashboard/server.py"


def _extract_function(name: str) -> str:
    src = WATCHER.read_text()
    m = re.search(rf"^{re.escape(name)}\(\) \{{$.*?^\}}$", src, re.S | re.M)
    if not m:
        raise AssertionError(f"{name}() 를 {WATCHER} 에서 찾지 못했다 -- 이름이 바뀌었나?")
    return m.group(0)


class DashboardRestartDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fn = _extract_function("dashboard_imports_changed")

    def _restart_decision(self, *changed: str) -> int:
        """워처의 두 조건(^dashboard/ + import 감지)을 그대로 재현해 restart 여부를 돌려준다."""
        script = "\n".join([
            "set -u",
            f'ROOT={ROOT!s}',
            'log() { :; }',
            'changed_files="$(printf "%s\\n" "$@")"',
            'affects() { echo "$changed_files" | grep -qE "$1"; }',
            self.fn,
            'd=0',
            "affects '^dashboard/' && d=1",
            'dashboard_imports_changed && d=1',
            'echo "$d"',
        ])
        out = subprocess.run(["bash", "-c", script, "bash", *changed],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip())

    def test_imported_live_module_restarts(self) -> None:
        self.assertEqual(1, self._restart_decision("scripts/live_evidence_signal_metalabel_20260829.py"))

    def test_every_scripts_import_of_server_py_is_covered(self) -> None:
        """server.py가 import하는 scripts/ 모듈은 **전부** 재시작을 걸어야 한다 (목록 낡음 방지)."""
        mods = sorted(set(re.findall(r"^(?:from|import) scripts\.([A-Za-z0-9_]+)", SERVER.read_text(), re.M)))
        self.assertGreater(len(mods), 5, "import 파싱이 깨졌다")
        for mod in mods:
            with self.subTest(mod=mod):
                self.assertEqual(1, self._restart_decision(f"scripts/{mod}.py"))

    def test_dashboard_path_still_restarts(self) -> None:
        self.assertEqual(1, self._restart_decision("dashboard/live/app.js"))

    def test_unimported_script_does_not_restart(self) -> None:
        """과교정 방지 -- 대시보드가 안 쓰는 파일로 라이브 대시보드를 내리면 안 된다."""
        for path in ("scripts/live_btc_evidence_signal_shadow_runner_20260902.py",
                     "scripts/research_eth_foo_20260907.py",
                     "scripts/ops/deploy_watcher.sh",
                     "other/scripts/coin_config.py"):
            with self.subTest(path=path):
                self.assertEqual(0, self._restart_decision(path))

    def test_no_change_does_not_restart(self) -> None:
        self.assertEqual(0, self._restart_decision("docs/foo.md"))


if __name__ == "__main__":
    unittest.main()
