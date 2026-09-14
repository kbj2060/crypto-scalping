"""**라벨 재생성 스윕** — 손절폭 × 보유지평이 방향의 손익분기를 어디까지 내리는가 (2026-09-14).

사용자: *"−3% 손절과 peg maker 등 새로운 기술들로 라벨 정답 데이터를 새로 만들어서 방향 예측"*.

🔴출발점 정정: 왕복 5.88bp 는 **이미 메이커 바닥**이다(`ENTRY_BP=2.95` 는 지정가 진입,
`PEG_EXIT_BP=2.93` 은 peg 청산 — 둘 다 메이커 2.0bp + 큐/스프레드). 그래서 집행을 더 좋게
해서 손익분기를 내릴 여지는 없다. 손익분기 IC = 비용/(E|y|·0.798) 의 **분모**를 움직이는 것은
**손절폭과 보유지평**뿐이다. 이 스크립트는 그 격자를 재라벨링해서 분모를 직접 잰다.

무엇을 재구현하지 않는가: 체결·손절·예산 사다리·만기·펀딩·크기는 `rl_gym_direction_env_20260914`
(= 배포 함수 import)를 그대로 쓴다. 여기서 바꾸는 건 **모듈 전역 `STOP_LOSS_PCT`** 와
**`hold_bars` 반환값** 둘뿐이다.

라벨: `y = ½(r_long − r_short)·1e4` (배포 스택이 청산할 때까지 굴린 순손익 차의 절반).
`y_gross` 는 각 다리의 실현 비용을 되돌려 더한 값 — **손익분기 분모는 gross 로 재야 한다**
(y 는 비용이 상쇄된 값이라 분모에 쓰면 비용을 두 번 세거나, 손절로 비용이 비대칭인 구간에서
어긋난다).

--selftest 는 «3% × 배포 선택기» 셀이 캐시된 `direction_labels.npz` 와 1e-9 안에서 같은지를 본다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_rl_gym_direction_ppo_20260914 as P  # noqa: E402
import rl_gym_direction_env_20260914 as G  # noqa: E402

OUT = ROOT / "data/research/eth_direction_label_breakeven_20260914"
STOPS = {"2%": 0.02, "3%": 0.03, "5%": 0.05, "none": 1.0}      # 1.0 = 사실상 무손절
HOLDS = {"selector": None, "60m": 12, "240m": 48, "480m": 96, "1440m": 288}   # 봉 수. sm 표의 키(분)
SQ2PI = float(np.sqrt(2.0 / np.pi))                            # 0.7979


class _FixedHold(G.DirectionGym):
    """지평 선택기 대신 고정 지평. 크기(`sm[(hb*5, side)]`)는 그 지평의 배포 안전MAE 를 쓴다."""

    def __init__(self, *a, hb: int, **kw):
        super().__init__(*a, **kw)
        self._hb = int(hb)

    def hold_bars(self, i, side):
        return self._hb


def leg(d, sm, i: int, side: int, hb_fixed: int | None):
    """봉 i 에서 한 측면만 넣었을 때의 (순수익 r, 실현 비용 bp, 손절여부). 배포 스택 그대로."""
    gym = (G.DirectionGym(d, sm, i, i + 1) if hb_fixed is None
           else _FixedHold(d, sm, i, i + 1, hb=hb_fixed))
    hb = gym.hold_bars(i, "LONG" if side == 1 else "SHORT")
    if hb is None:
        return None
    gym.hi = i + hb + 1
    res = gym.run(lambda j: ((side if j == i else 0), None))
    if not res["_trades"]:
        return None
    t = res["_trades"][0]
    # 사다리로 전량 닫힌 거래는 r=0.0·cost_bp 없음(기존 `trade_outcome` 규약 유지).
    return float(t["r"]), float(t.get("cost_bp", gym.entry_bp + gym.peg_bp)), bool(t["stopped"])


def label_window(d, sm, lo: int, hi: int, stride: int, hb_fixed):
    idx, y, ygross, cost, stopped, drift = [], [], [], [], [], []
    for i in range(lo, hi, stride):
        if not sm["ok"][i]:
            continue
        a = leg(d, sm, int(i), 1, hb_fixed); b = leg(d, sm, int(i), 2, hb_fixed)
        if a is None or b is None:
            continue
        idx.append(i)
        y.append(0.5e4 * (a[0] - b[0]))
        ygross.append(0.5e4 * ((a[0] + a[1] / 1e4) - (b[0] + b[1] / 1e4)))
        drift.append(0.5e4 * (a[0] + b[0]))          # r_long = m + y · r_short = m − y
        cost.append(0.5 * (a[1] + b[1]))
        stopped.append(a[2] or b[2])
    return (np.array(idx), np.array(y), np.array(ygross), np.array(cost), np.array(stopped),
            np.array(drift))


def breakeven(ygross: np.ndarray, cost: np.ndarray) -> dict:
    """부호 베팅 기대수익 ≈ IC·E|y|·√(2/π) ⇒ 손익분기 IC = 비용/(E|y|·0.798)."""
    e = float(np.mean(np.abs(ygross)))
    c = float(np.mean(cost))
    return {"n": int(len(ygross)), "e_abs_y_bp": e, "cost_bp": c,
            "breakeven_ic": c / (e * SQ2PI) if e > 0 else float("nan"),
            "breakeven_acc": 0.5 + c / (2 * e) if e > 0 else float("nan")}


def readout(idx: np.ndarray, y: np.ndarray, m: np.ndarray, pred: np.ndarray, block_bars: int) -> dict:
    """부호 베팅의 **순손익**과 **블록 t**.

    🔴IC 만 보면 09-13 의 함정에 그대로 빠진다 -- 지평이 길수록 초과는 커지지만 독립 블록이 그보다
    빨리 줄어 t 가 무너진다(H144 는 t=2 에 225년이 필요했다). 블록 = 보유기간이고, **블록 평균**을
    표본 단위로 쓴다(블록당 첫 거래만 뽑으면 임의 선택이라 순익과 t 의 부호가 어긋난다).

    항등식: 다리 두 개에서 r_long = m + y, r_short = m − y 이므로 측면 s 로 베팅한 실현 순손익은
    정확히 `m + s·y` 다(비용·펀딩·손절 전부 포함된 값)."""
    s = np.sign(pred); s[s == 0] = 1.0
    net = m + s * y
    b = (idx // max(block_bars, 1)).astype(np.int64)
    bm = np.array([net[b == k].mean() for k in np.unique(b)])
    t = float(bm.mean() / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 and bm.std(ddof=1) > 0 else float("nan")
    return {"net_bp": float(net.mean()), "n": int(len(net)), "blocks": int(len(bm)),
            "block_t": t, "hit": float((np.sign(y) == s).mean())}


def run_cell(d, sm, win, S, cols, stop: float, hb_fixed, train_stride: int, seeds: list[int]) -> dict:
    G.STOP_LOSS_PCT = stop                       # 🔴 gym 은 모듈 전역을 본다(라인 157·185·227)
    lab = {k: label_window(d, sm, *win[k], 12 if k != "TRAIN" else train_stride, hb_fixed)
           for k in ("TRAIN", "VAL", "OOS", "TEST")}
    cell = {"windows": {k: breakeven(v[2], v[3]) | {"stop_rate": float(v[4].mean())}
                        for k, v in lab.items()}}
    tr_idx, tr_y = lab["TRAIN"][0], lab["TRAIN"][1]
    # ① 이 라벨에서 도달 가능한 정보량의 바닥선 — 단일 피쳐 최대 |학습창 IC|
    ics = [abs(spearmanr(S[tr_idx, j], tr_y, nan_policy="omit").statistic) for j in range(S.shape[1])]
    ics = np.nan_to_num(np.array(ics))
    cell["single_feature"] = {"max_abs_ic": float(ics.max()), "col": cols[int(ics.argmax())],
                              "n_valid": int(np.isfinite(ics).sum())}
    # ② dev 모델(HGB) — 짝지은 씨드. 표본외 IC 를 창마다 잰다.
    per = {k: [] for k in ("VAL", "OOS", "TEST")}; preds = {}
    for sd in seeds:
        mdl = P.twin_fit_predict(S, tr_idx, tr_y, sd)
        preds[sd] = {}
        for k in per:
            i_, y_ = lab[k][0], lab[k][1]
            pr = mdl.predict(S[i_]); preds[sd][k] = pr
            per[k].append(float(spearmanr(pr, y_, nan_policy="omit").statistic))
    cell["hgb_oos_ic"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                              "seeds": v} for k, v in per.items()}
    # ②-b 같은 모델의 **순손익** 읽기값 — 씨드를 짝지어 창별로.
    bb = hb_fixed if hb_fixed is not None else 48
    cell["readout"] = {}
    for k in ("VAL", "OOS", "TEST"):
        rs = [readout(lab[k][0], lab[k][1], lab[k][5], preds[sd][k], bb) for sd in seeds]
        cell["readout"][k] = {f: {"mean": float(np.mean([r[f] for r in rs])),
                                  "se": float(np.std([r[f] for r in rs], ddof=1) / np.sqrt(len(rs)))}
                              for f in ("net_bp", "block_t", "hit")} | {
            "blocks": rs[0]["blocks"], "n": rs[0]["n"]}
    # ③ 판정 — 관측 IC 가 필요 IC 의 몇 %인가(OOS 기준)
    need = cell["windows"]["OOS"]["breakeven_ic"]
    got = max(cell["single_feature"]["max_abs_ic"], cell["hgb_oos_ic"]["OOS"]["mean"])
    cell["ratio_oos"] = float(got / need) if need > 0 else float("nan")
    return cell


def selftest(d, sm, win) -> None:
    """«3% × 배포 선택기» = 09-14 캐시 라벨. 다른 경로로 다시 계산해 1e-9 안에서 같아야 한다."""
    z = np.load(P.OUT / "direction_labels.npz", allow_pickle=True)
    G.STOP_LOSS_PCT = 0.03
    idx, y = label_window(d, sm, *win["OOS"], 12, None)[:2]
    ref_i, ref_y = z["OOS_idx"], z["OOS_y"]
    assert np.array_equal(idx, ref_i), f"인덱스 불일치: {len(idx)} vs {len(ref_i)}"
    err = float(np.max(np.abs(y - ref_y)))
    assert err < 1e-9, f"라벨 재구성 오차 {err:.3e} — 스택을 건드렸다는 뜻이다"
    print(f"자체점검 통과: OOS {len(idx):,}개 라벨이 캐시와 최대 {err:.1e} 차이")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-stride", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stops", default="", help="쉼표 구분, 비우면 전부")
    ap.add_argument("--holds", default="", help="쉼표 구분, 비우면 전부")
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--dump", default="", help="셀 하나(예: none/480m)를 npz 로 저장하고 끝낸다")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    d, sm, win, S, cols, _ = P.prepare(G.DEFAULT_FAMILIES)
    if a.selftest:
        selftest(d, sm, win); return 0
    if a.dump:
        sname, hname = a.dump.split("/")
        G.STOP_LOSS_PCT = STOPS[sname]; hb = HOLDS[hname]
        OUT.mkdir(parents=True, exist_ok=True)
        z = {"names": np.array(["TRAIN", "VAL", "OOS", "TEST"]), "frame_len": len(d),
             "stop": STOPS[sname], "hold_bars": -1 if hb is None else hb}
        ts = d.timestamp.to_numpy()
        for k in ("TRAIN", "VAL", "OOS", "TEST"):
            i_, y_, _, _, _, m_ = label_window(d, sm, *win[k], 12 if k != "TRAIN" else a.train_stride, hb)
            z[f"{k}_idx"], z[f"{k}_y"], z[f"{k}_m"] = i_, y_, m_
            z[f"{k}_ts"] = ts[i_].astype("int64")
            print(f"  {k:>5}: {len(i_):,}개 · E|y| {np.abs(y_).mean():.1f}bp · 표류 {m_.mean():+.2f}bp")
        f = OUT / f"labels_{sname.replace('%','pct')}_{hname}.npz"
        np.savez_compressed(f, **z); print(f"저장: {f}")
        return 0
    seeds = [int(x) for x in np.random.default_rng(20260914).integers(1, 1_000_000, size=a.seeds)]
    selftest(d, sm, win)                      # 스윕 전에 항상 재구성부터 통과시킨다
    res = {"train_stride": a.train_stride, "seeds": seeds, "cells": {}}
    print(f"\n{'셀':>14} {'n(OOS)':>7} {'E|y|bp':>8} {'비용bp':>7} {'손절률':>7} "
          f"{'필요IC':>7} {'손익분기%':>9} {'단일최대':>8} {'HGB OOS':>9} {'비율':>6} "
          f"{'순손익bp':>9} {'블록t':>7} {'블록수':>6} {'적중':>6}")
    stops = {k: v for k, v in STOPS.items() if not a.stops or k in a.stops.split(",")}
    holds = {k: v for k, v in HOLDS.items() if not a.holds or k in a.holds.split(",")}
    for sname, sval in stops.items():
        for hname, hb in holds.items():
            key = f"{sname}/{hname}"
            c = run_cell(d, sm, win, S, cols, sval, hb, a.train_stride, seeds)
            res["cells"][key] = c
            w = c["windows"]["OOS"]; h = c["hgb_oos_ic"]["OOS"]; ro = c["readout"]["OOS"]
            print(f"{key:>14} {w['n']:>7,} {w['e_abs_y_bp']:>8.1f} {w['cost_bp']:>7.2f} "
                  f"{w['stop_rate']:>7.1%} {w['breakeven_ic']:>7.3f} {w['breakeven_acc']:>8.1%} "
                  f"{c['single_feature']['max_abs_ic']:>8.3f} {h['mean']:>+6.3f}±{h['se']:.3f} "
                  f"{c['ratio_oos']:>6.0%} {ro['net_bp']['mean']:>+8.2f} {ro['block_t']['mean']:>+7.2f} "
                  f"{ro['blocks']:>6,} {ro['hit']['mean']:>6.1%}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    best = max(res["cells"].items(), key=lambda kv: kv[1]["ratio_oos"])
    print(f"\n최선 셀: {best[0]} — 관측/필요 = {best[1]['ratio_oos']:.0%} "
          f"(필요 {best[1]['windows']['OOS']['breakeven_ic']:.3f})")
    print(f"저장: {OUT/(a.tag+'.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
