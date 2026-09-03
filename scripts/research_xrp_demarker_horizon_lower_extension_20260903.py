#!/usr/bin/env python3
"""XRP `demarker_extreme`의 HORIZON 격자를 **아래로 한 칸 더** 넓힌다 (H=1 추가).

## 왜 -- 3번 넓히고도 여전히 하단 경계다

2026-09-03 XRP demarker 격자 이력:

    1차  H[6..20]  K[0.4..2.0]  -> H=6  K=2.0   ⚠️K 상단 경계
    2차  H[6..20]  K[0.4..4.0]  -> H=6  K=2.0   ⚠️H 하단 경계
    3차  H[2..20]  K[0.4..4.0]  -> H=2  K=1.5   ⚠️H=2 **여전히 하단 경계**  (HOLDOUT 0.6759)

3차에서도 승자가 격자 하단에 붙었는데, 문서는 "H=2는 구조적 하한"이라고 **주장**하고 멈췄다.
그러나 라벨 정의(`bars[fire+1 : fire+H+1]`의 intrabar 고가/저가가 K*atr에 닿는가)는
**H=1에서도 완전히 정의된다** -- "바로 다음 봉 하나 안에 닿는가". 구조적 하한은 H=1이지 H=2가
아니다. 호메로스 README **5.6절**(격자 경계 규칙)은 경계면 넓히라고 못박는다.

⚠️이 저장소는 같은 실수의 전례가 있다 -- ETH demarker의 진짜 최적 K=0.70을, 격자를 아래로
넓히기 전까지 K≈2.0으로 잘못 수렴했었다(README 5.6).

## 설계

원본 스크립트의 `run_grid_screen` / `select_horizon_k`를 **그대로 import**해서 쓴다
(재구현하면 lift 정의·베이스라인 추출·적격성 게이트가 조용히 달라진다).
`HORIZON_GRID`만 `[1]`을 앞에 붙여 재실행하고, 기존 선택(H=2/K=1.5)과 비교한다.

⚠️프레임은 XRP 후보 CSV를 **직접** 읽는다 -- 원본 `load_tier0`는 그 모듈 자신의 경로를 읽어
다른 자산 데이터가 들어온 전례가 있다(2026-09-03 BTC 오염 사고). 행수 가드를 건다.
⚠️Phase A(격자 스크린)만 돈다. VAL까지만 보고 OOS/HOLDOUT은 건드리지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_S = importlib.util.spec_from_file_location(
    "xrpdem", ROOT / "scripts/research_xrp_demarker_extreme_metalabel_tabpfn_20260903.py")
_d = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_d)

CAND_CSV = (ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
            / "xrp_5m_evidence_signal_candidates_tier0.csv")
EXPECTED_ROWS = 272_490
OUT = ROOT / "data/research/xrp_demarker_horizon_lower_extension_20260903.json"

HORIZON_GRID_EXT = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
PREV_CHOICE = {"horizon": 2, "k": 1.5, "holdout_auc": 0.6759}


def log(m): print(f"[dem-ext] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    f = pd.read_csv(CAND_CSV)
    f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True).dt.tz_localize(None)
    f = _d.add_missing_features(f)
    if abs(len(f) - EXPECTED_ROWS) > 200:
        raise RuntimeError(f"행수 {len(f):,} != XRP 기대치 {EXPECTED_ROWS:,} -- 다른 자산 데이터")
    log(f"XRP 프레임 {len(f):,}행 (자산 가드 통과)")

    orig = list(_d.HORIZON_GRID)
    log(f"기존 격자 H {orig}")
    log(f"확장 격자 H {HORIZON_GRID_EXT}   (K 격자 {_d.K_GRID})")
    _d.HORIZON_GRID = HORIZON_GRID_EXT
    try:
        grid = _d.run_grid_screen(f)
        h, k, info = _d.select_horizon_k(grid)
    finally:
        _d.HORIZON_GRID = orig

    log("")
    log(f"⭐확장 격자 선택: H={h} K={k}")
    log(f"   기존 선택:     H={PREV_CHOICE['horizon']} K={PREV_CHOICE['k']} "
        f"(HOLDOUT AUC {PREV_CHOICE['holdout_auc']})")
    changed = (h != PREV_CHOICE["horizon"]) or (abs(k - PREV_CHOICE["k"]) > 1e-9)
    log(f"   ⇒ {'⚠️**선택이 바뀐다** -- 기존 H=2는 격자가 짧아서 나온 값이었다' if changed else '✅선택 불변 -- H=2가 진짜 최적'}")
    at_edge = (h == HORIZON_GRID_EXT[0]) or (h == HORIZON_GRID_EXT[-1])
    log(f"   경계 점검: {'⚠️여전히 하단 경계(H=1) -- 라벨 정의상 더는 못 내려간다' if h == 1 else ('⚠️상단 경계' if at_edge else '✅내부값')}")

    # H별 최선 셀 요약 (H=1이 실제로 경쟁력이 있는지 보이게)
    log("")
    log(f"{'H':>3} {'최선K':>6} {'TRAIN lift':>11} {'VAL lift':>10} {'VAL hit':>9} {'n_train':>9} {'적격셀':>6}")
    per_h = []
    for hh in HORIZON_GRID_EXT:
        sub = grid[grid["horizon"] == hh]
        if not len(sub):
            continue
        elig = sub[sub["eligible"]] if "eligible" in sub.columns else sub
        base = elig if len(elig) else sub
        pick = base.sort_values("lift_val", ascending=False).iloc[0]
        ntr = int(pick["n_train_bottom"]) + int(pick["n_train_top"])
        per_h.append({"horizon": int(hh), "k": float(pick["k"]),
                      "lift_train": float(pick["lift_train"] or float("nan")),
                      "lift_val": float(pick["lift_val"] or float("nan")),
                      "val_hitrate": float(pick["val_hitrate_pooled"]),
                      "n_train": ntr, "n_eligible": int(len(elig))})
        log(f"{hh:>3} {pick['k']:>6.2f} {float(pick['lift_train'] or 0):>11.3f} "
            f"{float(pick['lift_val'] or 0):>10.3f} "
            f"{float(pick['val_hitrate_pooled']):>9.4f} "
            f"{ntr:>9,} {len(elig):>6}")

    rep = {"prev_choice": PREV_CHOICE, "horizon_grid_ext": HORIZON_GRID_EXT,
           "k_grid": list(_d.K_GRID), "chosen": {"horizon": int(h), "k": float(k)},
           "choice_changed": bool(changed), "chosen_at_edge": bool(at_edge),
           "per_horizon_best": per_h, "select_info": info,
           "holdout_touched": False, "oos_touched": False,
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    grid.to_csv(OUT.with_suffix(".grid.csv"), index=False)
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
