#!/usr/bin/env python3
"""XRP·BTC 증거신호 경제성게이트에 **지연확정(delayed-anchor) 대조군**을 소급 적용.

## 왜 이걸 돌리는가 -- 게이트 통과 수치가 무효일 수 있다

2026-09-02 ETH 자동매매 승격감사 결론(README **5.16절**):

> `cluster_dedup(idx, extremeness, gap)`은 GAP 안 연속발동을 클러스터로 묶고 **최극단 봉**을
> 앵커로 남긴다. **어떤 봉이 최극단인지는 클러스터가 끝나야 안다** → 그 봉에선 모른다 = 미래참조.
> ⇒ `cluster_dedup`/`CLUSTER_GAP_MERGE` 발동집합 위 **모든 경제성게이트 수치**가 승격근거로 무효.

⚠️**BTC 게이트(2026-09-02)는 이 감사를 한 번도 받지 않았다** -- 두 작업이 같은 날 병렬로
진행돼 서로 만나지 않았다. XRP 게이트(2026-09-03)도 마찬가지다. 두 게이트 모두 같은 빌더를
쓰므로 같은 결함을 상속한다(확인: XRP 5종·BTC 5종 전부 dedup 호출).

⚠️⚠️**방향뒤집기도 무작위진입 귀무도 이 함정을 못 잡는다.** ETH는 flip 28/28 + DSR/PBO
저장소 최초 통과 + 홀드아웃 24/28 양수를 전부 달성하고도 무효였다. 오염된 발동집합 안에서는
real이 flip을 정상적으로 이기기 때문이다.

## 판정도구 (ETH `research_eth_causal_anchor_variants_20260902.py::v_delayed_anchor` 그대로)

앵커는 **그대로 두고 진입만** 앵커가 확정되는 봉(= 클러스터 마지막 트리거 + gap)으로 미룬다.
앵커 봉이 100% 동일하므로 "앵커가 옳은가"와 "늦춰도 남는가"가 분리된다.
ETH top2 결과: 연구(앵커선택) **+11.73/+14.08/+7.97** → 지연확정 **−5.66/+1.57/−6.44**.

## 구현 -- 극단값 의미를 몰라도 된다

모듈마다 dedup 시그니처가 4가지로 다르다(`cluster_dedup` 3인자/4인자,
`cluster_dedup_gap`, `cluster_dedup_oscillator`). 하지만 **클러스터 경계는 `idx`와 `gap`만으로
결정**되므로, 원본 함수를 래핑해 (a)원래 앵커를 그대로 반환하고 (b)각 앵커가 속한 클러스터의
마지막 트리거+gap을 확정봉으로 기록하면 된다. 극단성 컬럼이 무엇인지 알 필요가 없다.

⚠️HOLDOUT 미터치. VAL+OOS만 본다(사전등록 셀 고정, 새 격자탐색 없음).
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

GATES = {
    "XRP": ("xrpgate", "scripts/gate_xrp_evidence_signals_trailing_economics_20260903.py",
            "data/research/xrp_evidence_signals_costgate_20260903/report.json"),
    "BTC": ("btcgate", "scripts/gate_btc_evidence_signals_trailing_economics_20260902.py",
            "data/research/btc_evidence_signals_costgate_20260902/report.json"),
}
OUT = ROOT / "data/research/delayed_anchor_control_20260903/report.json"

DEDUP_NAMES = ("cluster_dedup", "cluster_dedup_gap", "cluster_dedup_oscillator")
GAP_CONSTS = ("CLUSTER_GAP", "CLUSTER_GAP_MERGE", "GAP")


def log(m): print(f"[delayed] {m}", flush=True)


def _clusters(idx, gap):
    """`cluster_dedup`과 **동일한** 경계 규칙(diff > gap)으로 묶고, 각 클러스터의 마지막 트리거를 준다."""
    idx = np.sort(np.asarray(idx, dtype=np.int64))
    if len(idx) == 0:
        return {}
    ends, cur_last = {}, idx[0]
    cur = [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] > gap:
            for j in cur:
                ends[int(j)] = int(cur[-1])
            cur = [i]
        else:
            cur.append(i)
    for j in cur:
        ends[int(j)] = int(cur[-1])
    return ends


def install_capture(mod, store, conflicts):
    """모듈의 dedup 함수를 래핑한다. 원래 반환값은 그대로 두고 확정봉만 기록한다."""
    installed = []
    for fname in DEDUP_NAMES:
        orig = getattr(mod, fname, None)
        if orig is None or getattr(orig, "_delayed_wrapped", False):
            continue

        def make(orig=orig, fname=fname):
            def wrapped(*args, **kwargs):
                keep = orig(*args, **kwargs)
                try:
                    idx = np.asarray(args[0], dtype=np.int64)
                    gap = kwargs.get("gap")
                    if gap is None:
                        for a in reversed(args[1:]):
                            if isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                                gap = int(a); break
                    if gap is None:
                        for c in GAP_CONSTS:
                            if hasattr(mod, c):
                                gap = int(getattr(mod, c)); break
                    if gap is None:
                        return keep
                    ends = _clusters(idx, gap)
                    for a in np.asarray(keep, dtype=np.int64):
                        cf = ends.get(int(a))
                        if cf is None:
                            continue
                        cf = cf + gap
                        prev = store.get(int(a))
                        if prev is not None and prev != cf:
                            conflicts.append((int(a), prev, cf))
                        store[int(a)] = cf
                except Exception as e:                                  # noqa: BLE001
                    conflicts.append(("capture_error", fname, repr(e)))
                return keep
            wrapped._delayed_wrapped = True
            return wrapped

        setattr(mod, fname, make())
        installed.append(fname)
    return installed


def evaluate(g, kl, fires, H, cell, dec_override=None):
    """사전등록 셀 1개만 평가. `dec_override`가 있으면 진입 봉을 그것으로 바꾼다."""
    save_grid = (g.SL_GRID, g.ARM_GRID, g.TRAIL_GRID)
    g.SL_GRID, g.ARM_GRID, g.TRAIL_GRID = [cell[0]], [cell[1]], [cell[2]]
    g.ROUNDTRIP_COST_RATE = 0.001
    try:
        f = fires.copy()
        if dec_override is not None:
            f["timestamp"] = dec_override
        cells, ns = g.run_grid(kl, f, H)
    finally:
        g.SL_GRID, g.ARM_GRID, g.TRAIL_GRID = save_grid
    c = cells[0]
    return {"val_fwd": c["val_fwd_bp"], "val_flip": c["val_flip_bp"], "val_n": c["val_n"],
            "oos_fwd": c["oos_fwd_bp"], "oos_flip": c["oos_flip_bp"], "oos_n": c["oos_n"]}


def main() -> int:
    t0 = time.time()
    out = {"holdout_touched": False, "tool": "delayed-anchor control (README 5.16)",
           "note": "앵커 동일, 진입만 클러스터 확정봉으로 지연", "assets": {}}

    for asset, (modname, relpath, reppath) in GATES.items():
        spec = importlib.util.spec_from_file_location(modname, ROOT / relpath)
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        rep_in = json.loads((ROOT / reppath).read_text())["signals"]

        kl = pd.read_csv(g.KLINES)
        kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True).dt.tz_localize(None)
        kl = kl.sort_values("timestamp").reset_index(drop=True)
        log("")
        log(f"################ {asset} ################")
        res = {}

        for name, rel, builder, prep, kind in g.SIGNALS:
            v = rep_in.get(name, {})
            g1 = v.get("genuine_arm_ge_1") or []
            if not g1:
                log(f"{name}: 원 게이트 통과 셀 없음 -- 건너뜀")
                continue
            best = max(g1, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
            cell = (best["sl"], best["arm"], best["trail"])
            H = v["horizon_bars"]
            log("")
            log(f"=== {name}  셀 SL={cell[0]} ARM={cell[1]} Trail={cell[2]}  H={H} ===")

            store, conflicts = {}, []
            mod = g.load_mod(rel)
            inst = install_capture(mod, store, conflicts)
            # build_fires가 load_mod를 다시 부르면 래핑이 날아간다 -> 같은 모듈 객체를 반환하도록 고정
            orig_load = g.load_mod
            g.load_mod = lambda r, _m=mod: _m
            try:
                fires, frame = g.build_fires(name, rel, builder, prep, kind)
            finally:
                g.load_mod = orig_load
            log(f"  dedup 래핑: {inst}  포착 앵커 {len(store):,}개  충돌 {len(conflicts)}")
            if not store:
                log("  ⚠️확정봉을 하나도 못 잡았다 -- 이 신호는 판정 불가")
                res[name] = {"error": "no anchors captured"}
                continue

            fires["timestamp"] = pd.to_datetime(fires["timestamp"])
            if fires["timestamp"].dt.tz is not None:
                fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            if frame["timestamp"].dt.tz is not None:
                frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
            frame_ts = frame["timestamp"].to_numpy()

            if "pos" not in fires.columns:
                log("  ⚠️fires에 pos 없음 -- 판정 불가")
                res[name] = {"error": "no pos column"}
                continue
            pos = fires["pos"].to_numpy(dtype=np.int64)
            conf = np.array([store.get(int(p), -1) for p in pos], dtype=np.int64)
            ok = (conf >= 0) & (conf < len(frame_ts))
            cov = float(ok.mean())
            delay = conf[ok] - pos[ok]
            log(f"  확정봉 매핑 {ok.sum():,}/{len(pos):,} ({cov*100:.1f}%)  "
                f"지연 중앙값 {int(np.median(delay))}봉 / 평균 {delay.mean():.1f}봉 / 최대 {delay.max()}봉")

            f_ok = fires.loc[ok].reset_index(drop=True)
            conf_ts = pd.Series(frame_ts[conf[ok]])
            base_mask = f_ok["timestamp"] < g.HOLDOUT_START
            dl_mask = conf_ts < g.HOLDOUT_START
            keep = (base_mask & dl_mask).to_numpy()
            f_ok, conf_ts = f_ok.loc[keep].reset_index(drop=True), conf_ts.loc[keep].reset_index(drop=True)

            a = evaluate(g, kl, f_ok, H, cell)                       # A) 연구 앵커(원본)
            d = evaluate(g, kl, f_ok, H, cell, dec_override=conf_ts)  # D) 지연확정
            log(f"  A 연구앵커  VAL {a['val_fwd']:+7.2f} (뒤 {a['val_flip']:+7.2f}) n={a['val_n']:<5} | "
                f"OOS {a['oos_fwd']:+7.2f} (뒤 {a['oos_flip']:+7.2f}) n={a['oos_n']}")
            log(f"  D 지연확정  VAL {d['val_fwd']:+7.2f} (뒤 {d['val_flip']:+7.2f}) n={d['val_n']:<5} | "
                f"OOS {d['oos_fwd']:+7.2f} (뒤 {d['oos_flip']:+7.2f}) n={d['oos_n']}")
            survives = (d["val_fwd"] > 0 and d["oos_fwd"] > 0
                        and d["val_fwd"] > d["val_flip"] and d["oos_fwd"] > d["oos_flip"])
            log(f"  ⇒ {'✅**지연확정 생존**' if survives else '❌지연확정 붕괴 -- 앵커선택이 만든 성과'}")
            res[name] = {"cell": list(cell), "H": H, "coverage": cov,
                         "delay_bars": {"median": int(np.median(delay)),
                                        "mean": float(delay.mean()), "max": int(delay.max())},
                         "A_research_anchor": a, "D_delayed": d, "survives": bool(survives)}
        out["assets"][asset] = res

    log("")
    log("=== 종합 (지연확정 생존 여부) ===")
    for asset, res in out["assets"].items():
        for name, v in res.items():
            if "error" in v:
                log(f"  {asset} {name:<26} ⚠️{v['error']}"); continue
            a, d = v["A_research_anchor"], v["D_delayed"]
            log(f"  {asset} {name:<26} A {a['val_fwd']:+6.2f}/{a['oos_fwd']:+6.2f} → "
                f"D {d['val_fwd']:+6.2f}/{d['oos_fwd']:+6.2f}  "
                f"{'✅생존' if v['survives'] else '❌붕괴'}")
    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({out['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
