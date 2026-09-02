#!/usr/bin/env python3
"""라이브/섀도우 경로 **결함 유형 일괄 점검** -- BTC 증거신호에서 찾은 4종을 전 경로로 확대.

## 배경

2026-09-02~03에 BTC 증거신호 섀도우에서 4종의 결함을 찾았다. 사용자 질문
*"지금 비트코인 증거신호만 모두 점검한거야?"*에 답하기 위해 **같은 유형을 전 경로로** 훑는다.

## 점검하는 결함 유형

  P1 **이벤트 봉이 아닌 최신 봉의 피쳐**를 쓴다
     예) `rolling(14).mean().iloc[-1]`을 신호 봉 ATR인 것처럼 사용 -> HIT 문턱이 어긋난다
     (BTC 증거신호 `record()`에서 실제 발생, 합성 검증에서 12.3% 오차 확인)

  P2 **배리어/청산을 폴링 가격 한 점**으로 판정한다(백테스트는 봉 고가/저가)
     -> 봉 안에서 스톱을 스치고 되돌아온 wick을 놓쳐 손실 트레이드가 사라진다
     (ETH V자반등 섀도우에서 실제 발생, 원장 9건 전부 양수 +69bp = HOLDOUT의 10배)

  P3 **영구 미해소 상태** -- 조회 창보다 오래 멈추면 pending이 영원히 남는다
     (BTC 증거신호에서 실제 발생, limit=60봉=5시간)

  P4 **루프 주기에 의존하는 시간 계산** -- 틱/사이클 개수를 시간으로 환산해 쓴다
     -> 주기를 바꾸면 보유한도 등이 배수로 틀어진다
     (ETH V자반등 섀도우 `ticks >= MAX_HOLD_BARS * 5`에서 실제 발생)

⚠️이 스크립트는 **정적 스캔**이다. 히트는 "사람이 읽어봐야 하는 후보"이지 판정이 아니다.
   각 히트를 직접 읽고 판단한 결과는 리포트의 `verdict`에 사람이 채운다.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/research/live_shadow_defect_audit_20260903/report.json"

# 실제로 서버에서 돌고 있거나 대시보드가 서빙하는 경로만 본다(연구 스크립트 제외)
TARGETS = [
    "scripts/live_eth_v_rebound_econ_shadow_runner_20260902.py",
    "scripts/live_eth_v_rebound_econ_autotrade_signal_20260902.py",
    "scripts/live_btc_evidence_signal_shadow_runner_20260902.py",
    "scripts/live_btc_evidence_signal_metalabel_20260902.py",
    "scripts/live_evidence_signal_metalabel_20260829.py",
    "scripts/live_eth_sweep_v_rebound_signal_20260829.py",
    "scripts/live_evidence_signal_dashboard_20260823.py",
    "scripts/maker_fill_shadow_worker.py",
    "scripts/run_btc_multislot_shadow_loop_20260807.py",
    "scripts/live_eth_odyssey4_zig075_entry_veto_shadow_cleanroom_20260816.py",
    "scripts/live_eth_odyssey4_zig075_entry_veto_shadow_20260814.py",
    "l2_anomaly_snapshot_collector.py",
    "liq_magnet_collector.py",
    "scripts/live_xrp_evidence_signal_metalabel_20260903.py",
    "scripts/live_xrp_evidence_signal_shadow_runner_20260903.py",
    "dashboard/server.py",
]

PATTERNS = {
    "P1_latest_bar_feature": [
        r"rolling\([^)]*\)\.(?:mean|std|sum)\(\)\.iloc\[-1\]",
        r"\batr\w*\s*=\s*.*\.iloc\[-1\]",
    ],
    "P2_poll_price_barrier": [
        r"ticker/price",
        r"\bmark_price\s*\(",
    ],
    "P3_unresolvable_state": [
        r"limit\s*[=:]\s*\d{1,3}\b",
        r"keep\.append\(",
    ],
    "P4_cycle_dependent_time": [
        r"\bticks\b",
        r"LOOP_SECONDS\s*\*",
        r"\bcycles?\b\s*[><>=]{1,2}",
    ],
}


def log(m): print(f"[defect-audit] {m}", flush=True)


def main() -> int:
    rep = {"targets": len(TARGETS), "files": {}}
    total = 0
    for rel in TARGETS:
        f = ROOT / rel
        if not f.exists():
            log(f"⚠️없음: {rel}")
            rep["files"][rel] = {"missing": True}
            continue
        text = f.read_text()
        lines = text.splitlines()
        hits: dict[str, list] = {}
        for cls, pats in PATTERNS.items():
            for pat in pats:
                for i, ln in enumerate(lines, 1):
                    if re.search(pat, ln):
                        hits.setdefault(cls, []).append({"line": i, "text": ln.strip()[:120]})
        n = sum(len(v) for v in hits.values())
        total += n
        rep["files"][rel] = {"n_hits": n, "hits": hits}
    rep["total_hits"] = total

    log(f"대상 {len(TARGETS)}개 파일, 정적 히트 {total}건")
    log("")
    for rel, d in rep["files"].items():
        if d.get("missing"):
            continue
        if not d["n_hits"]:
            log(f"  {'':2}{Path(rel).name:<58} 히트 없음")
            continue
        cls_s = "  ".join(f"{c.split('_')[0]}:{len(v)}" for c, v in d["hits"].items())
        log(f"  ⚑{Path(rel).name:<58} {cls_s}")
    log("")
    log("⚠️정적 히트는 후보일 뿐이다 -- 각 항목을 직접 읽고 판단해야 한다.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
