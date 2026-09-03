#!/usr/bin/env python3
"""지연확정 대조군의 **비용 민감도** -- "경제성 기준을 낮추면 통과하나?"에 수치로 답한다.

## 왜

사용자 질문: "경제성 테스트 기준이 어떻게 되지? 좀 낮추는건 어때?"

현행 기준 중 **유일하게 정당하게 조정 가능한 축은 비용 가정**이다. 나머지(방향뒤집기·
무작위진입 귀무·지연확정)는 **성능 문턱이 아니라 타당성 검정**이라 낮추면 "틀린 결과를
받아들이는" 것이 된다.

  · 현행: 왕복 **10bp** (taker 왕복, 수수료 우대 가정 금지 -- 저장소 규칙)
  · 지정가(maker) 진입이면 실제 비용이 더 낮다. README 5.18 기록에 **7bp 가정** 사례가 있다.
  · 그렇다면 **비용을 낮추면 지연확정 대조군을 통과하는가?**

⇒ 비용을 [10, 7, 5, 2, 0]bp로 훑어 **손익분기 비용**을 구한다. 그 값이 현실적으로 달성
가능한지(maker 리베이트 포함해도 0bp는 불가능)를 보면 "낮추면 되는가"에 답이 나온다.

## 설계

`audit_xrp_btc_delayed_anchor_control_20260903.py`의 지연확정 로직을 그대로 import해서
**비용만 바꿔** 다시 돌린다. 앵커·셀·분할 전부 동일 -- 비용 외 변수를 고정한다.

⚠️이건 "기준을 낮춘 결과"가 아니라 **"낮추면 어떻게 되는지의 측정"**이다.
채택 여부는 별개이고, 낮춘 비용을 쓰려면 **체결 모델 검증**이 선행 조건이다
(README 5.18: "체결 가정 미검증 -- peg-maker 섀도우 실제 체결통계와 대조 필요").

⚠️HOLDOUT 미터치. VAL+OOS만.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_D = importlib.util.spec_from_file_location(
    "delayed", ROOT / "scripts/audit_xrp_btc_delayed_anchor_control_20260903.py")
_d = importlib.util.module_from_spec(_D)
_D.loader.exec_module(_d)

OUT = ROOT / "data/research/delayed_anchor_cost_sensitivity_20260903.json"
COSTS_BP = [10.0, 7.0, 5.0, 2.0, 0.0]


def log(m): print(f"[costsens] {m}", flush=True)


def evaluate_at_cost(g, kl, fires, H, cell, cost_rate, dec_override=None):
    """⚠️`_d.evaluate`를 그대로 쓰면 안 된다 -- 그 함수는 내부에서
    `g.ROUNDTRIP_COST_RATE = 0.001`로 **덮어써서** 바깥 패치를 무효화한다.
    비용을 인자로 받는 복제본을 쓴다(나머지 로직은 동일)."""
    save_grid = (g.SL_GRID, g.ARM_GRID, g.TRAIL_GRID)
    save_cost = g.ROUNDTRIP_COST_RATE
    g.SL_GRID, g.ARM_GRID, g.TRAIL_GRID = [cell[0]], [cell[1]], [cell[2]]
    g.ROUNDTRIP_COST_RATE = cost_rate
    try:
        f = fires.copy()
        if dec_override is not None:
            f["timestamp"] = dec_override
        cells, _ns = g.run_grid(kl, f, H)
    finally:
        g.SL_GRID, g.ARM_GRID, g.TRAIL_GRID = save_grid
        g.ROUNDTRIP_COST_RATE = save_cost
    c = cells[0]
    return {"val_fwd": c["val_fwd_bp"], "val_flip": c["val_flip_bp"], "val_n": c["val_n"],
            "oos_fwd": c["oos_fwd_bp"], "oos_flip": c["oos_flip_bp"], "oos_n": c["oos_n"]}


def main() -> int:
    t0 = time.time()
    rep = {"costs_bp": COSTS_BP, "holdout_touched": False,
           "note": "지연확정 대조군(앵커 동일, 진입만 클러스터 확정봉으로 지연)에서 비용만 변주",
           "assets": {}}

    for asset, (modname, relpath, reppath) in _d.GATES.items():
        spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        rep_in = json.loads((ROOT / reppath).read_text())["signals"]
        kl = pd.read_csv(g.KLINES)
        kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
        kl = kl.sort_values("timestamp").reset_index(drop=True)
        log("")
        log("#" * 74)
        log(f"{asset}")
        log("#" * 74)
        res = {}

        for name, rel, builder, prep, kind in g.SIGNALS:
            v = rep_in.get(name, {})
            g1 = v.get("genuine_arm_ge_1") or []
            if not g1:
                continue
            best = max(g1, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
            cell = (best["sl"], best["arm"], best["trail"])
            H = v["horizon_bars"]

            store, conflicts = {}, []
            mod = g.load_mod(rel)
            _d.install_capture(mod, store, conflicts)
            orig_load = g.load_mod
            g.load_mod = lambda r, _m=mod: _m
            try:
                fires, frame = g.build_fires(name, rel, builder, prep, kind)
            finally:
                g.load_mod = orig_load
            if not store or "pos" not in fires.columns:
                continue

            for df_ in (fires, frame):
                df_["timestamp"] = pd.to_datetime(df_["timestamp"])
                if df_["timestamp"].dt.tz is not None:
                    df_["timestamp"] = df_["timestamp"].dt.tz_localize(None)
            frame_ts = frame["timestamp"].to_numpy()
            pos = fires["pos"].to_numpy(dtype=np.int64)
            conf = np.array([store.get(int(p), -1) for p in pos], dtype=np.int64)
            ok = (conf >= 0) & (conf < len(frame_ts))
            f_ok = fires.loc[ok].reset_index(drop=True)
            conf_ts = pd.Series(frame_ts[conf[ok]])
            keep = ((f_ok["timestamp"] < g.HOLDOUT_START)
                    & (conf_ts < g.HOLDOUT_START)).to_numpy()
            f_ok = f_ok.loc[keep].reset_index(drop=True)
            conf_ts = conf_ts.loc[keep].reset_index(drop=True)

            log("")
            log(f"=== {name}  셀 SL={cell[0]} ARM={cell[1]} Trail={cell[2]} H={H} ===")
            log(f"{'비용':>6} {'A 연구앵커 VAL/OOS':>26} {'D 지연확정 VAL/OOS':>26}  D부호")
            rows = []
            for cb in COSTS_BP:
                a = evaluate_at_cost(g, kl, f_ok, H, cell, cb / 1e4)
                dd = evaluate_at_cost(g, kl, f_ok, H, cell, cb / 1e4, dec_override=conf_ts)
                both_pos = dd["val_fwd"] > 0 and dd["oos_fwd"] > 0
                rows.append({"cost_bp": cb, "A": a, "D": dd, "D_both_positive": bool(both_pos)})
                log(f"{cb:>5.1f}bp {a['val_fwd']:>+12.2f}/{a['oos_fwd']:>+12.2f} "
                    f"{dd['val_fwd']:>+12.2f}/{dd['oos_fwd']:>+12.2f}  "
                    f"{'✅양수' if both_pos else '❌'}")
            # 손익분기 비용: D가 VAL·OOS 둘 다 양수가 되는 최대 비용
            passing = [r["cost_bp"] for r in rows if r["D_both_positive"]]
            be = max(passing) if passing else None
            log(f"  ⇒ 지연확정 손익분기 비용: "
                f"{'**' + str(be) + 'bp 이하**' if be is not None else '**0bp에서도 통과 못함**'}")
            res[name] = {"cell": list(cell), "H": H, "by_cost": rows, "breakeven_cost_bp": be}
        rep["assets"][asset] = res

    log("")
    log("=" * 78)
    log("종합 -- 비용을 낮추면 지연확정을 통과하는가")
    log("=" * 78)
    log(f"{'자산':<5} {'신호':<26} {'손익분기 비용':>14}")
    n_any = 0
    for asset, res in rep["assets"].items():
        for name, v in res.items():
            be = v["breakeven_cost_bp"]
            if be is not None:
                n_any += 1
            log(f"{asset:<5} {name:<26} {(str(be) + 'bp 이하') if be is not None else '0bp에서도 실패':>14}")
    log("")
    log(f"⇒ 비용을 0bp까지 낮춰도 통과하는 신호: **{n_any}종**")
    log("  (0bp는 수수료·슬리피지가 전혀 없다는 뜻으로 현실에 존재하지 않는다.")
    log("   maker 리베이트를 최대로 잡아도 왕복 2~7bp가 하한이다.)")
    rep["n_passing_at_any_cost"] = n_any
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
