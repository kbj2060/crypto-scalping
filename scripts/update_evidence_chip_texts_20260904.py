#!/usr/bin/env python3
"""증거신호 칩 상세 텍스트의 [검증]/[경제성] 줄을 **정직한 수치**로 교체한다 (2026-09-04).

왜: app.js `EVIDENCE_SIGNAL_KO`의 [검증] AUC는 클러스터 앵커 모집단 수치(예: demarker 0.753/0.716)인데 라이브 칩은 raw 발동에서
호출되며 그 모집단에서의 실제 순위 품질은 낮다(demarker 0.603/0.626). [경제성]의 트레일링스톱 bp는 5.16/5.20절에서 **앵커 미래참조로
전부 무효** 판정된 숫자다. 이 스크립트는 (1) 인과 모집단 VAL/OOS AUC·캘리브레이션(배포 컨텍스트 또는 교체 컨텍스트), (2) 발동 봉
페이드 vs 반대(지속) 경제성(F0 셀, VAL+OOS)으로 두 줄을 다시 쓴다. 다른 줄([조건]/[신뢰도])은 건드리지 않는다.

입력: tmp/eth_chip_accuracy_upgrade_20260904/report_tabpfn.json, data/research/eth_evidence_fire_continuation_econ_20260904/per_signal_econ.json,
      --decisions '{"taker_delta_z_climax": {"arm": "F0", "report": "livepop"}, ...}' (교체된 칩은 새 컨텍스트 수치를 쓴다)
사용: python scripts/update_evidence_chip_texts_20260904.py --app dashboard/live/app.js [--decisions JSON] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "tmp/eth_chip_accuracy_upgrade_20260904/report_tabpfn.json"
LIVEPOP = ROOT / "tmp/eth_chip_accuracy_upgrade_20260904/livepop_report.json"
ECON = ROOT / "data/research/eth_evidence_fire_continuation_econ_20260904/per_signal_econ.json"
SIGNALS = ["orthogonal_combo", "liquidity_sweep", "short_term_return_z", "taker_delta_z_climax", "smt_divergence",
           "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]


def fmt_auc(m):
    return f"{m['auc']:.3f}" if m and m.get("auc") is not None else "-"


def build_lines(decisions):
    rep = json.loads(REP.read_text())["signals"]; econ = json.loads(ECON.read_text())["signals"]
    live = json.loads(LIVEPOP.read_text()) if LIVEPOP.exists() else {}
    out = {}
    for s in SIGNALS:
        dec = decisions.get(s)
        if dec and dec.get("arm") and live.get(s):
            arm = dec["arm"]; m = live[s]["arms"][arm]; src = f"인과 재학습 컨텍스트({arm}, 09-04 교체)"
        elif live.get(s):
            m = live[s]["arms"]["D"]; src = "현재 컨텍스트"
        else:
            m = rep[s]["arms"]["D_deployed_ctx"]; src = "현재 컨텍스트"
        v, o = m["VAL"], m["OOS"]
        slope = v.get("calib_slope"); cal = ("확률 과신(기울기 %.2f — 표시 확률이 실제보다 높음)" % slope) if slope is not None and slope < 0.8 else ("확률 과소(기울기 %.2f)" % slope if slope is not None and slope > 1.25 else "확률 대체로 정직")
        verify = (f"[검증] 라이브 발동 모집단(raw 발동, 앵커 없음) VAL {fmt_auc(v)} / OOS {fmt_auc(o)} — {src}, {cal}. "
                  f"⚠️이전 표기(앵커 모집단 AUC)는 실제 라이브 호출 품질보다 높게 나온 값이라 09-04 정정.")
        e = econ[s]["VAL_OOS"]; bs = e["by_side"]
        econl = (f"[경제성] 발동 봉에서 신호 방향(페이드) 진입은 VAL+OOS n={e['n']:,} 평균 {e['fade_bp']:+.1f}bp, 반대 방향(지속)은 {e['cont_bp']:+.1f}bp"
                 f"(바닥 발동 페이드 {bs['bottom']['fade_bp']:+.1f}/지속 {bs['bottom']['cont_bp']:+.1f} · 천장 발동 페이드 {bs['top']['fade_bp']:+.1f}/지속 {bs['top']['cont_bp']:+.1f}; "
                 f"5.0/1.5/0.1 ATR 트레일, 10bp 차감). ⛔이전 트레일링스톱 수익 표기는 발동 앵커 미래참조로 무효(호메로스 5.16·5.20절). "
                 f"첫 발동 봉은 지속 구간이고 되돌림은 그 뒤(5.23절) — 자동매매 근거 아님.")
        out[s] = (verify, econl)
    return out


def apply(app: Path, lines: dict, dry: bool):
    src = app.read_text(); n_changed = 0
    start = src.index("const EVIDENCE_SIGNAL_KO = {"); end = src.index("const BTC_EVIDENCE_SIGNAL_KO", start)
    block = src[start:end]
    for s, (verify, econl) in lines.items():
        m = re.search(r"(  " + re.escape(s) + r": \{\n)(.*?)(\n  \},)", block, re.S)
        assert m, s
        body = m.group(2)
        # 마지막 detail 줄은 `"...",` 로 끝나고 중간 줄은 `"...\n" +` 로 끝난다 -- 줄 단위로 재조립
        detail_src = body[body.index("detail:"):]
        lines_js = re.findall(r'"((?:[^"\\]|\\.)*)"', detail_src)
        keep = [l.rstrip("\\n") + "\\n" for l in lines_js if not (l.startswith("[검증]") or l.startswith("[경제성]"))]
        new = keep + [verify + "\\n", econl]
        # 문자열 리터럴 안전화
        new = [l.replace('"', '\\"') if '"' in l.replace('\\"', '') else l for l in new]
        rebuilt = "    name: " + re.search(r'name: (".*?"),', body).group(1) + ",\n    detail: " + " +\n      ".join(f'"{l}"' for l in new) + ","
        # 기존 body에서 name 줄 앞의 주석 등은 없음(형식 고정) -- 통째 교체
        block = block.replace(m.group(0), m.group(1) + rebuilt + m.group(3))
        n_changed += 1
    out = src[:start] + block + src[end:]
    if not dry:
        app.write_text(out)
    return n_changed, out[start:start + 2600]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--app", required=True); ap.add_argument("--decisions", default="{}"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); lines = build_lines(json.loads(a.decisions)); n, preview = apply(Path(a.app), lines, a.dry_run)
    print(f"signals updated: {n}{' (dry-run)' if a.dry_run else ''}"); print(preview)
