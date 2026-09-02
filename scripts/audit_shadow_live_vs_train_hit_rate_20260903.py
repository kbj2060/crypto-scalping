#!/usr/bin/env python3
"""섀도우 원장의 **라이브 hit률 대 학습 hit률** 대조 -- 서빙 경로가 학습과 같은 걸 재느냐.

## 왜 필요한가

2026-09-02 BTC에서 `HIT_SPEC` 모드 2건이 틀려(giveback/mae_capped를 plain touch로 뭉갬)
**라이브 hit률이 2.6배 과대평가**된 사고가 있었다. XRP는 배포 전 800건 대조로 막았지만,
그건 *오프라인 재현* 검증이고 **실제 라이브 경로가 같은 값을 내는지**는 원장이 쌓여야 안다.

## 판정

각 신호에 대해 라이브 해상 건수 n과 hit 수 h로 **Wilson 95% 신뢰구간**을 만들고,
학습 hit률이 그 안에 들어오는지 본다.

  · 학습률이 CI 안 → 일치(구분 불가). 서빙 경로 정상으로 본다.
  · 학습률이 CI 밖 → **불일치**. HIT_SPEC 모드/H/K 또는 피쳐 시점(이벤트봉 vs 최신봉)을 의심한다.
  · n이 작아 CI가 넓으면 → **판정 불가**. 필요 표본을 같이 출력한다.

⚠️이 검정은 hit률(라벨 재현)만 본다. **경제성과 무관**하다 -- 2026-09-03 지연확정 감사에서
증거신호 경제성게이트 수치는 앵커 미래참조로 전부 무효 판정됐다(README 5.16절).
여기서 확인하는 건 "표시되는 확률이 학습된 것과 같은 사건을 가리키는가"뿐이다.

필요 표본 추정: 학습률 p를 ±0.10 폭으로 잡으려면 대략 n >= 4*p*(1-p)/0.10^2.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEDGERS = {
    "BTC": ROOT / "data/live/btc_evidence_signal_shadow_state.json",
    "XRP": ROOT / "data/live/xrp_evidence_signal_shadow_state.json",
}

# 학습(TRAIN) hit률 -- 각 자산의 메타라벨 리포트/라벨 사전에 기록된 값
# BTC: dashboard/live/app.js::BTC_EVIDENCE_SIGNAL_KO, 각 research_btc_*_metalabel_tabpfn 리포트
# XRP: data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_metalabel_report.json
TRAIN_HIT = {
    "BTC": {"demarker_extreme": 0.9003, "kalman_deviation_meanrev": 0.1425,
            "short_term_return_z": 0.3163, "taker_delta_climax": 0.1388,
            "orthogonal_combo": 0.4271, "liquidity_sweep": 0.1022,
            "fib_extension_exhaustion": 0.1928},
    "XRP": {"demarker_extreme": 0.2455, "short_term_return_z": 0.6016,
            "taker_delta_climax": 0.0899, "orthogonal_combo": 0.4198},
}
TARGET_HALFWIDTH = 0.10


def log(m): print(f"[hitcmp] {m}", flush=True)


def wilson(h, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def need_n(p, hw=TARGET_HALFWIDTH):
    return int(math.ceil(4 * p * (1 - p) / (hw * hw))) if 0 < p < 1 else 0


def main() -> int:
    out = {"target_halfwidth": TARGET_HALFWIDTH, "assets": {}}
    for asset, path in LEDGERS.items():
        log("")
        log(f"################ {asset} ################")
        if not path.exists():
            log(f"  원장 없음: {path}")
            continue
        st = json.loads(path.read_text())
        led = st.get("ledger", [])
        log(f"  원장 {len(led)}건 | pending {len(st.get('pending', []))} | "
            f"cycles {st.get('cycles')} | 시작 {st.get('started_utc')}")
        cnt = collections.Counter(r["signal"] for r in led)
        hit = collections.Counter(r["signal"] for r in led if r.get("hit") == 1)
        res = {}
        for name, tp in sorted(TRAIN_HIT.get(asset, {}).items()):
            n, h = cnt.get(name, 0), hit.get(name, 0)
            lo, hi = wilson(h, n)
            nn = need_n(tp)
            if n == 0:
                verdict = "표본 0"
            elif not (lo <= tp <= hi):
                verdict = "⚠️불일치"
            elif (hi - lo) / 2 > TARGET_HALFWIDTH:
                verdict = "판정불가(CI 넓음)"
            else:
                verdict = "✅일치"
            live = f"{h/n:.4f}" if n else "—"
            ci = f"[{lo:.3f}, {hi:.3f}]" if n else "—"
            log(f"  {name:<26} 학습 {tp:.4f} | 라이브 {live} (n={n:>3}, hit={h:>3}) "
                f"CI95 {ci:<18} {verdict}   필요n≈{nn}")
            res[name] = {"train_hit": tp, "n": n, "hits": h,
                         "live_hit": (h / n) if n else None,
                         "ci95": [lo, hi] if n else None,
                         "verdict": verdict, "needed_n": nn}
        out["assets"][asset] = {"n_ledger": len(led), "started_utc": st.get("started_utc"),
                                "cycles": st.get("cycles"), "signals": res}

    log("")
    log("=== 종합 ===")
    for asset, a in out["assets"].items():
        sig = a["signals"]
        dec = [k for k, v in sig.items() if v["verdict"] in ("✅일치", "⚠️불일치")]
        bad = [k for k, v in sig.items() if v["verdict"] == "⚠️불일치"]
        short = sum(max(0, v["needed_n"] - v["n"]) for v in sig.values())
        log(f"  {asset}: 판정가능 {len(dec)}/{len(sig)}종  불일치 {len(bad)}  "
            f"추가 필요 해상건수 합계 ≈{short:,}")
        if bad:
            log(f"     ⚠️불일치: {bad}")
    OUTP = ROOT / "data/research/shadow_live_vs_train_hit_20260903.json"
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    OUTP.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUTP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
