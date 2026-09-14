"""**죽은 CLI 플래그 검출기** — argparse 로 선언만 되고 아무 데서도 안 읽히는 인자를 찾는다.

## 왜 (2026-09-14 사고)
`research_eth_rl_gym_direction_ppo_20260914.py` 의 `train()` 이 `ppo_update()` 를 부르며
`epochs`/`mb` 를 안 넘겼다. `--mb 1024` 로 돌린 두 팔이 조용히 기본값 8192 로 돌았고,
`report_real_bandit_mb1024.json` 이 `report_real_bandit.json` 과 **비트 단위로 같았다**
(거래 105/118/259, 5씨드 log_mult 전부 일치). 오류도 경고도 없었다 — 두 리포트를 나란히
놓고 비교하기 전까지 아무 신호가 없었다. 전말: docs/experiments/eth_rl_gym_architecture_review_20260914.md §11.

## 규칙
인자의 `dest` 가 소스 어디에서도 `.dest` 또는 `["dest"]` 로 읽히지 않으면 죽은 플래그다.
네임스페이스 변수명이 파일마다 다르고(`a`/`args`/`opt`) `parse_args()` 를 헬퍼가 `return` 하기도 해서
**변수를 추적하지 않는다** -- 오탐 대신 미탐 쪽으로 기운 설계다(변수 추적판은 오탐 547건을 냈다).

ponytail: 「값이 실제로 소비되는가」까지는 안 본다. `a.mb` 를 읽어 전역에 넣고 그 전역을 안 쓰면
이 검사는 통과한다. 업그레이드 경로는 전역까지 따라가는 도달성 분석이지만, 이번 사고 부류
(«읽히지도 않음»)는 이 한 줄로 잡힌다.

    python3 scripts/audit_dead_cli_flags_20260914.py ['glob']   # 기본 scripts/*.py
"""
from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 사고 당시 커밋(수정 직전) -- 검출기를 고치면 이 픽스처부터 다시 통과시킨다.
FIXTURE = ("6ece35f", "scripts/research_eth_rl_gym_direction_ppo_20260914.py", {"mb", "epochs"})


def dead_flags(src: str) -> list[str]:
    out = []
    for n in ast.walk(ast.parse(src)):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument"):
            continue
        for a in n.args:
            if not (isinstance(a, ast.Constant) and str(a.value).startswith("--")):
                continue
            dest = next((k.value.value for k in n.keywords
                         if k.arg == "dest" and isinstance(k.value, ast.Constant)), None) \
                or str(a.value)[2:].replace("-", "_")
            if not re.search(r"\.%s\b" % re.escape(dest), src) \
               and not re.search(r"""['"]%s['"]\s*\]""" % re.escape(dest), src):
                out.append(dest)
    return out


def selftest() -> None:
    """실제 사고를 재현한 커밋에서 --mb/--epochs 를 잡아야 한다."""
    sha, path, want = FIXTURE
    old = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"],
                         capture_output=True, text=True)
    assert old.returncode == 0, f"픽스처 커밋 {sha} 를 못 읽는다 -- 히스토리 확인"
    got = set(dead_flags(old.stdout))
    assert got >= want, f"검출기가 실제 사고를 못 잡는다: {got} ⊉ {want}"
    # 음성 대조: 수정본에서는 그 둘이 안 나와야 한다(안 그러면 검사가 항상 참을 낸다)
    now = set(dead_flags((ROOT / path).read_text()))
    assert not (now & want), f"수정본에서도 잡힌다 -- 검출기가 무조건 참을 낸다: {now & want}"
    print(f"자체점검 ✓ {sha} 에서 {sorted(want)} 검출 · 수정본에서는 미검출")


def main() -> int:
    selftest()
    pat = sys.argv[1] if len(sys.argv) > 1 else "*.py"
    bad = 0
    for f in sorted((ROOT / "scripts").glob(pat)):
        try:
            d = dead_flags(f.read_text())
        except SyntaxError as e:
            print(f"⚠️ {f.name}: 파싱 실패 {e}"); continue
        if d:
            bad += len(d); print(f"🔴 {f.name}: " + ", ".join("--" + x for x in d))
    print(f"죽은 플래그 {bad}개 ({pat})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
