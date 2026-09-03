#!/usr/bin/env python3
"""ETH demarker/kalman **격자 경계 확장** -- XRP 절차(포팅 프로토콜 §5-A)를 ETH에.

## 왜

2026-09-03 XRP에서 "튜닝이 덜 됐다"의 **진짜 원인은 상속 상수가 아니라 격자 경계**였다.
XRP 레짐은 S·K 둘 다 상단 경계에 걸려 있었고, 확장하니 배포본을 네 축 전부 이기는
셀(S96_K9)이 나왔다. 같은 점검을 BTC에 이어 ETH에도 건다.

배포 중인 ETH demarker/kalman 셀의 경계 상태 (gridscreen_results.csv 실측):

| 신호 | HORIZON_GRID | 선택 H | GAP_GRID | 선택 GAP | 경계 |
|---|---|---|---|---|---|
| `demarker_extreme` | [**8**,12,...,48] | **8** | [3,6,**12**] | **12** | ⚠️⚠️H 하단 + GAP 상단 |
| `kalman_deviation_meanrev` | [8,**12**,...,48] | 12 | [3,6,**12**] | **12** | ⚠️GAP 상단 |

K는 이미 확장했다 -- README §5.6: K_GRID 하단(1.0)에 걸려서 확장했더니 진짜 정점이
**K=0.70**이었다(smt_divergence 경계 버그의 교훈). ⇒ **H와 GAP만 남았다.**

## 설계 -- 재구현 금지

`research_eth_kalman_demarker_gridscreen_20260831.py`의 `main()`을 그대로 재실행하고
`HORIZON_GRID`/`GAP_GRID`/`GBM_SEED`/출력경로만 patch한다.

  ① **먼저 rng(GBM_SEED) 안정성** -- 원본 격자 고정, 시드만 8회 변경.
     XRP에서 배운 순서다: 승자가 흔들리면 "확장했더니 바뀌었다"를 경계 문제로 오독한다.
  ② 그 다음 경계 확장: H 하단으로 [3,4,6], GAP 상단으로 [18,24,36].

⚠️**이 계보의 선택 규칙은 `min(VAL, OOS)`라 OOS가 선택 안에 들어 있다**(원본 설계).
확장에서도 같은 규칙을 쓴다 -- 규칙을 바꾸면 배포본과 비교가 성립하지 않는다.
⇒ 그래서 여기서 나온 "승자"는 **교체 근거가 아니라 재점검 결과**다. 교체하려면
   별도의 미사용 창 단일 노출이 필요하다(XRP S96_K9에서 한 것처럼).

⚠️HOLDOUT(2026-04-01~) 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

OUTDIR = ROOT / "tmp/eth_kalman_demarker_grid_boundary_20260903"
OUT = ROOT / "data/research/eth_kalman_demarker_grid_boundary_20260903.json"
SRC = "research_eth_kalman_demarker_gridscreen_20260831.py"

SEEDS = [20260831, 811453, 30011, 947, 260317, 5387291, 68041, 1299709]
DEPLOYED = {"demarker_extreme": (8, 12), "kalman_deviation_meanrev": (12, 12)}

EXT_H = [3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 48]
EXT_GAP = [3, 6, 12, 18, 24, 36]
# ⭐2차 확장 -- 1차에서 kalman이 **H 하단·GAP 상단에 또 걸렸다**(H=3/GAP=36).
# 경계값 불신 원칙은 "한 번 늘려봤다"로 끝나지 않는다: 늘린 격자에서 또 경계면 또 늘린다.
EXT2_H = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30, 36, 48]
EXT2_GAP = [3, 6, 12, 18, 24, 36, 48, 72]


def log(m): print(f"[eth-bnd] {m}", flush=True)


def load_mod():
    sp = importlib.util.spec_from_file_location("m_eth_gs", ROOT / "scripts" / SRC)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


def run_once(mod, tag, seed=None, h=None, gap=None):
    """gridscreen main()을 돌리고 결과 CSV를 tag별 디렉토리로 옮긴다."""
    dst = OUTDIR / tag / "gridscreen_results.csv"
    if dst.exists():
        log(f"    (재사용 {tag})")
        return pd.read_csv(dst)
    saves = {}
    if seed is not None:
        saves["GBM_SEED"] = mod.GBM_SEED; mod.GBM_SEED = seed
    if h is not None:
        saves["HORIZON_GRID"] = list(mod.HORIZON_GRID); mod.HORIZON_GRID = list(h)
    if gap is not None:
        saves["GAP_GRID"] = list(mod.GAP_GRID); mod.GAP_GRID = list(gap)
    try:
        rc = mod.main()
    finally:
        for k, v in saves.items():
            setattr(mod, k, v)
    if rc != 0:
        return None
    src = ROOT / "tmp/eth_kalman_demarker_gridscreen_20260831/gridscreen_results.csv"
    df = pd.read_csv(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return df


def winner(df, signal):
    sub = df[(df.signal == signal) & (df.variant == "plain")]
    if sub.empty:
        return None
    r = sub.sort_values("min_val_oos", ascending=False).iloc[0]
    return {"horizon": int(r.horizon), "gap": int(r.gap), "val_auc": float(r.val_auc),
            "oos_auc": float(r.oos_auc), "min_val_oos": float(r.min_val_oos),
            "hit_rate": float(r.hit_rate), "n_train": int(r.n_train)}


def main() -> int:
    t0 = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mod = load_mod()
    log(f"원본 HORIZON_GRID={mod.HORIZON_GRID}  GAP_GRID={mod.GAP_GRID}  GBM_SEED={mod.GBM_SEED}")
    log(f"선택 규칙: min(VAL,OOS) AUC (원본 설계 -- ⚠️OOS가 선택 안에 있다)")
    rep = {"asset": "ETHUSDT", "seeds": SEEDS, "holdout_touched": False,
           "deployed": {k: list(v) for k, v in DEPLOYED.items()},
           "selection_rule": "min(VAL,OOS) AUC (원본 gridscreen 규칙 그대로)",
           "order": "① rng(GBM_SEED) 안정성 → ② 경계 확장", "signals": {}}

    # ---------- ① rng 안정성 ----------
    log(""); log("#" * 76); log("① GBM_SEED 재시드 안정성 (원본 격자 고정)"); log("#" * 76)
    per_seed = {k: [] for k in DEPLOYED}
    for s in SEEDS:
        df = run_once(mod, f"seed{s}", seed=s)
        if df is None:
            log(f"  seed={s}: 실패"); continue
        for sig in DEPLOYED:
            w = winner(df, sig)
            if w:
                per_seed[sig].append({"seed": s, **w})
                log(f"  seed={s:<9} {sig:<26} H={w['horizon']:<3} GAP={w['gap']:<3} "
                    f"min(VAL,OOS)={w['min_val_oos']:.4f}")

    for sig, rows in per_seed.items():
        cells = Counter((r["horizon"], r["gap"]) for r in rows)
        dw = cells.get(DEPLOYED[sig], 0)
        log("")
        log(f"  {sig}: 승자 분포({len(rows)}회) " +
            "  ".join(f"H{c[0]}/G{c[1]}×{n}" for c, n in cells.most_common()))
        log(f"    ⇒ 배포셀 H{DEPLOYED[sig][0]}/G{DEPLOYED[sig][1]} 승률 **{dw}/{len(rows)}**  "
            f"{'✅안정' if len(cells) == 1 else f'⚠️{len(cells)}종으로 흔들림'}")
        rep["signals"][sig] = {"stability": {
            "per_seed": rows, "distinct": len(cells),
            "winner_counts": [{"cell": list(c), "n": n} for c, n in cells.most_common()],
            "deployed_win_rate": dw / max(1, len(rows))}}

    # ---------- ② 경계 확장 ----------
    log(""); log("#" * 76)
    log(f"② 격자 경계 확장  HORIZON {mod.HORIZON_GRID} -> {EXT_H}")
    log(f"                  GAP      {mod.GAP_GRID} -> {EXT_GAP}")
    log("#" * 76)
    ext = run_once(mod, "EXT", h=EXT_H, gap=EXT_GAP)
    if ext is None:
        log("  ⚠️확장 실행 실패")
        rep["extension_error"] = "main failed"
    else:
        base = run_once(mod, f"seed{SEEDS[0]}", seed=SEEDS[0])
        for sig in DEPLOYED:
            w, b = winner(ext, sig), winner(base, sig) if base is not None else None
            log("")
            log(f"  {sig}")
            log(f"    확장 선택 : H={w['horizon']:<3} GAP={w['gap']:<3} "
                f"VAL {w['val_auc']:.4f} OOS {w['oos_auc']:.4f} min {w['min_val_oos']:.4f} "
                f"hit {w['hit_rate']:.3f} n_train {w['n_train']}")
            if b:
                log(f"    원본(seed {SEEDS[0]}): H={b['horizon']:<3} GAP={b['gap']:<3} "
                    f"min {b['min_val_oos']:.4f}")
                if w["min_val_oos"] < b["min_val_oos"] - 1e-9:
                    log(f"    ⚠️**상위집합인데 목적함수가 내려갔다**"
                        f"({b['min_val_oos']:.4f} → {w['min_val_oos']:.4f}) -- rng 측정 노이즈")
            edges = []
            if w["horizon"] == EXT_H[0]: edges.append("H 하단")
            if w["horizon"] == EXT_H[-1]: edges.append("H 상단")
            if w["gap"] == EXT_GAP[0]: edges.append("GAP 하단")
            if w["gap"] == EXT_GAP[-1]: edges.append("GAP 상단")
            log(f"    경계 재점검: {'⚠️' + ', '.join(edges) if edges else '✅내부값'}")
            dep = DEPLOYED[sig]
            changed = (w["horizon"], w["gap"]) != dep
            log(f"    배포셀 대비: {'⚠️**바뀜**' if changed else '✅동일'}")
            # 배포셀이 확장 격자 안에서 몇 등인지
            sub = ext[(ext.signal == sig) & (ext.variant == "plain")].sort_values(
                "min_val_oos", ascending=False).reset_index(drop=True)
            hit = sub.index[(sub.horizon == dep[0]) & (sub.gap == dep[1])].tolist()
            rank = (int(hit[0]) + 1) if hit else None
            log(f"    배포셀 순위: {rank}/{len(sub)}" if rank else "    배포셀이 확장격자에 없음")
            rep["signals"][sig]["extension"] = {
                "chosen": w, "baseline_seed0": b, "still_at_edge": edges,
                "changed_vs_deployed": changed, "deployed_rank": rank, "n_cells": int(len(sub)),
                "top8": sub.head(8)[["horizon", "gap", "val_auc", "oos_auc",
                                     "min_val_oos", "hit_rate", "n_train"]].to_dict("records")}

    # ---------- ③ 2차 확장 (1차에서 또 경계에 걸린 경우) ----------
    need2 = [sig for sig, v in rep["signals"].items()
             if (v.get("extension") or {}).get("still_at_edge")]
    if need2:
        log(""); log("#" * 76)
        log(f"③ **2차** 경계 확장 -- 1차에서 또 경계에 걸린 신호: {need2}")
        log(f"   HORIZON {EXT_H} -> {EXT2_H}")
        log(f"   GAP     {EXT_GAP} -> {EXT2_GAP}")
        log("#" * 76)
        ext2 = run_once(mod, "EXT2", h=EXT2_H, gap=EXT2_GAP)
        if ext2 is None:
            log("  ⚠️2차 확장 실패")
        else:
            for sig in DEPLOYED:
                w = winner(ext2, sig)
                e1 = (rep["signals"][sig].get("extension") or {}).get("chosen") or {}
                log("")
                log(f"  {sig}")
                log(f"    2차 확장 선택: H={w['horizon']:<3} GAP={w['gap']:<3} "
                    f"VAL {w['val_auc']:.4f} OOS {w['oos_auc']:.4f} min {w['min_val_oos']:.4f} "
                    f"hit {w['hit_rate']:.3f} n_train {w['n_train']}")
                if e1:
                    log(f"    1차 확장     : H={e1['horizon']:<3} GAP={e1['gap']:<3} "
                        f"min {e1['min_val_oos']:.4f}")
                    if w["min_val_oos"] < e1["min_val_oos"] - 1e-9:
                        log(f"    ⚠️**상위집합인데 목적함수가 내려갔다** -- rng/측정 노이즈")
                edges = []
                if w["horizon"] == EXT2_H[0]: edges.append("H 하단")
                if w["horizon"] == EXT2_H[-1]: edges.append("H 상단")
                if w["gap"] == EXT2_GAP[0]: edges.append("GAP 하단")
                if w["gap"] == EXT2_GAP[-1]: edges.append("GAP 상단")
                log(f"    경계 재점검  : {'⚠️' + ', '.join(edges) if edges else '✅내부값 -- 여기서 멈춘다'}")
                dep = DEPLOYED[sig]
                sub = ext2[(ext2.signal == sig) & (ext2.variant == "plain")].sort_values(
                    "min_val_oos", ascending=False).reset_index(drop=True)
                hit = sub.index[(sub.horizon == dep[0]) & (sub.gap == dep[1])].tolist()
                rank = (int(hit[0]) + 1) if hit else None
                log(f"    배포셀 순위  : {rank}/{len(sub)}")
                rep["signals"][sig]["extension2"] = {
                    "chosen": w, "still_at_edge": edges, "deployed_rank": rank,
                    "n_cells": int(len(sub)), "grid": {"H": EXT2_H, "GAP": EXT2_GAP},
                    "top8": sub.head(8)[["horizon", "gap", "val_auc", "oos_auc",
                                         "min_val_oos", "hit_rate", "n_train"]].to_dict("records")}

    log(""); log("=" * 80); log("종합 -- ETH demarker/kalman 격자 경계"); log("=" * 80)
    for sig, v in rep["signals"].items():
        st, ex = v["stability"], v.get("extension") or {}
        c = ex.get("chosen")
        log(f"  {sig:<26} 배포셀승률 {st['deployed_win_rate']*100:>3.0f}%  승자{st['distinct']}종  "
            + (f"확장승자 H{c['horizon']}/G{c['gap']}"
               f"{'  ⚠️변경' if ex.get('changed_vs_deployed') else '  ✅동일'}"
               f"  (배포셀 {ex.get('deployed_rank')}위/{ex.get('n_cells')})" if c else "확장실패"))
    log("")
    log("⚠️확장 승자는 **교체 근거가 아니다** -- 선택 규칙에 OOS가 들어 있어 자기참조다.")
    log("   교체하려면 미사용 창 단일 노출이 별도로 필요하다(XRP S96_K9 절차).")
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
