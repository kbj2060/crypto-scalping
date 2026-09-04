#!/usr/bin/env python3
"""live_evidence_signal_metalabel_20260829.py의 METALABEL_SIGNALS[<sig>]["train_context"]를 인과 컨텍스트로 교체한다 (2026-09-04).
사용: python scripts/patch_evidence_chip_contexts_20260904.py --module <path> --map '{"taker_delta_z_climax": "data/labels/.../x.csv"}'
교체 줄 바로 위에 근거 주석을 남긴다. feature_columns/seed/k/horizon_bars는 건드리지 않는다(F0 팔 전용)."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--module", required=True); ap.add_argument("--map", required=True); a = ap.parse_args()
p = Path(a.module); s = p.read_text(); n = 0
for sig, path in json.loads(a.map).items():
    m = re.search(r'(    "' + re.escape(sig) + r'": \{\n)(.*?)(        "train_context": ROOT / ")([^"]+)(",\n)', s, re.S)
    assert m, sig
    note = ('        # 2026-09-04 인과 모집단 컨텍스트로 교체 -- 라이브는 raw 단일봉 발동에서 호출되는데 이전 컨텍스트는 클러스터 앵커 봉 학습이라\n'
            '        # 확률이 과신(캘리브레이션 기울기 <0.6)이었다. 라이브 결정 모집단(같은 측면 raw 발동이 직전 horizon_bars 안에 없는 봉)의\n'
            '        # TRAIN(<2025-09-01)만으로 재학습. 근거/수치: docs/experiments/eth_evidence_chip_accuracy_upgrade_20260904.md\n'
            f'        # 이전: {m.group(4)}\n')
    s = s[:m.start(3)] + note + m.group(3) + path + m.group(5) + s[m.end(5):]; n += 1
p.write_text(s); print(f"patched {n} contexts in {p}")
