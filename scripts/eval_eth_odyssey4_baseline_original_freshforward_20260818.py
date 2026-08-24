#!/usr/bin/env python3
"""버그수정 전(원본 6/29·6/30 학습) h48qual/zig075 번들의 Fresh-Forward 6창 평가 -- posfix
평가(eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818.py)와 동일 방법론, 동일
프레임/예측생성/replay/게이트 코드를 그대로 재사용(BUNDLES/OUT_DIR만 오버라이드) -- 순수하게
"어떤 번들을 평가하는가"만 다른 두 실행이 되도록, 로직 자체는 단 한 글자도 새로 안 짬.

원본 번들의 base_cols(102개)는 canonical sweep.load_frame 프레임에 100% 포함(cmamba/risk도
전혀 안 씀, 직접 확인) -- posfix 평가 때 필요했던 zero-col 예외 처리 자체가 불필요.
각 번들 자신의 진짜 risk sidecar(sweep.COMPONENTS[name]["sidecar_pkl"])를 그대로 씀 -- posfix
평가처럼 다른 번들 것을 빌려쓰는 근사치가 아니라 완전한 정식 평가.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

ev.OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_baseline_original_freshforward_20260818"


def _cfg(name: str) -> dict:
    # sweep.COMPONENTS uses the key "quality_threshold"; ev.BUNDLES/generate_predictions expects
    # "threshold" -- map it explicitly rather than spreading sweep.COMPONENTS directly (that KeyErrors).
    c = dict(sweep.COMPONENTS[name])
    c["threshold"] = c.pop("quality_threshold")
    c["exit_threshold"] = sweep.BASELINE_EXIT_THRESHOLD
    return c


ev.BUNDLES = {"h48qual": _cfg("h48qual"), "zig075": _cfg("zig075")}
# original bundles use zero cmamba/risk columns (confirmed: 102 base_cols fully covered by
# sweep.load_frame with nothing missing) -- no exceptions needed, but leave the mechanism in place
# harmlessly in case that ever changes.
ev._EXPECTED_ZERO_COLS = set()

if __name__ == "__main__":
    result = ev.main()
    # Patch the report's risk_sizing_source note to reflect that this run uses each bundle's OWN
    # real sidecar, not a borrowed one -- ev.main() already wrote report.json before we can touch
    # it, so rewrite the field directly.
    import json
    report_path = ev.OUT_DIR / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["risk_sizing_source"] = (
        "each_bundle_own_real_trained_sidecar -- sweep.COMPONENTS[name]['sidecar_pkl'], the "
        "genuine sidecar each original bundle was actually paired with at certification time. "
        "Full live-parity sizing, not an approximation."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(result)
