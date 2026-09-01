#!/usr/bin/env python3
"""미포착 사건 정량 분석 -- "모양이 신호와 맞아 보인다"는 육안 관찰을 수치로 검증.

## 배경

매 봉 스코어링 재설계의 전제는 **"9트리거가 놓친 사건(별개 사건의 29~44%)이 진짜다"**였다.
2026-09-01 육안검증 20예시에서 미포착과 포착(대조군)의 모양 품질이 구분되지 않았고, 사용자도
"미포착인데 신호와 맞는 게 꽤 많지 않냐"고 지적했다. 그런데 지금까지 확인한 건 **모양**뿐이고,
정작 중요한 질문 두 개가 비어 있다:

  Q1. 재학습한 **배포 모델이 실제로 미포착 사건을 잡는가?**
      -- 전제가 맞아도 모델이 못 잡으면 recall 이득은 장부상 숫자일 뿐이다. 한 번도 안 쟀다.
  Q2. 미포착 사건은 **정말 별개 사건인가, 아니면 같은 V를 옆 봉에서 라벨한 중복인가?**
      -- 9트리거 중 local_extreme은 ±30분 내 최저/최고면 발동한다. 깨끗한 V의 바닥이라면
      거의 항상 발동해야 정상인데 발동하지 않았다는 건, 라벨이 **진짜 극값이 아닌 봉**에
      붙었다는 뜻일 수 있다. 그렇다면 "미포착 사건"의 상당수가 근처 포착 사건과 같은 V이고,
      recall 이득은 부분적으로 중복 계산이 된다.

## 재는 것

사건 단위(GAP=12봉으로 클러스터링, 육안검증 스크립트와 동일):
  - 포착/미포착별 **모델 포착률**: 사건 내 최대 model_proba >= 0.60인 비율  <- Q1
  - 포착/미포착별 라벨 품질: fast_mult, giveback, 절대 이동폭(bp), atr_pct, 사건 길이
  - 미포착 사건이 **가장 가까운 포착 사건까지의 거리**(봉) 분포                <- Q2
  - 미포착 사건의 첫 봉이 local_extreme 창에서 극값과 몇 틱 떨어져 있는지      <- Q2

VAL과 OOS 각각. **진단 전용** -- 이 결과로 모델/라벨/임계값을 재선택하지 않는다.
⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_uncovered_event_capture_analysis_20260901.py
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

S1 = ROOT / "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py"
_spec = importlib.util.spec_from_file_location("vreb_s1_cap", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_vs = _s1._vs
_feas = _s1._feas

AUDIT = ROOT / "scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py"
_aspec = importlib.util.spec_from_file_location("vreb_audit_cap", AUDIT)
_audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(_audit)

FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
ALL9 = _feas.ALL9
W = 6                 # LOCAL_EXTREME_W
GAP = 12              # 육안검증/사건감사와 동일
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
THRESHOLD = 0.60
CONTEXT_N, SEED = 18000, 20260829

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

OUT_JSON = ROOT / "data/research/eth_v_rebound_label_grid_stage1_20260901/uncovered_event_capture.json"


def log(msg: str) -> None:
    print(f"[cap] {msg}", flush=True)


def pct(a) -> str:
    return "n/a" if not len(a) else f"{np.mean(a)*100:.1f}%"


def q(a, ps=(25, 50, 75)) -> str:
    if not len(a):
        return "n/a"
    v = np.nanpercentile(a, ps)
    return "/".join(f"{x:.2f}" for x in v)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}  (진단 전용 -- 재선택 없음)")

    _s1.VAL_END = OOS_END
    log("building frame...")
    sig, feat, eth = _s1.build_sig()

    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    lab = long.loc[long["label"].notna()]
    tr = lab.loc[lab["split"] == "TRAIN"]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    clf.fit(tr.iloc[idx][FEATURE_COLUMNS], tr.iloc[idx]["label"].to_numpy())
    log(f"배포 설정으로 fit 완료 (컨텍스트 {CONTEXT_N}행)")

    # 봉 x side 확률을 timestamp/side 키로 조회 가능하게
    proba = {}
    for sp in ("VAL", "OOS"):
        s = long.loc[long["split"] == sp].copy()
        s["model_proba"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]
        for ts, side, pr in zip(s["timestamp"], s["side"], s["model_proba"]):
            proba[(ts, side)] = float(pr)
        log(f"  {sp} 채점 {len(s):,}행")

    # local_extreme 및 트리거(포착 여부 판정용)
    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    atr = sig["atr"].to_numpy()
    close = sig["close"].to_numpy()
    ts_arr = sig["timestamp"].to_numpy()
    lo_flag = np.zeros(n, dtype=bool); hi_flag = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            lo_flag[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            hi_flag[i] = True

    results = {}
    for side, is_down in (("bottom", True), ("top", False)):
        status = sb if is_down else st
        trig = np.any([sig[f"{side}_{nm}"].fillna(False).to_numpy()
                       for nm in ALL9 if nm != "local_extreme"], axis=0)
        trig = trig | (lo_flag if is_down else hi_flag)
        fm = _audit.fast_mult_array(sig, is_down)
        v_idx = np.flatnonzero(status == "v_rebound")

        for ev in _audit.cluster_events(v_idx, GAP):
            first, last = int(ev[0]), int(ev[-1])
            ts0 = pd.Timestamp(ts_arr[first])
            split = "TRAIN" if ts0 < TRAIN_END else ("VAL" if ts0 < VAL_END else "OOS")
            if split == "TRAIN":
                continue
            covered = bool(trig[ev].any())
            # 사건 내 최대 확률 = "모델이 이 사건을 잡았는가"
            ps = [proba.get((pd.Timestamp(ts_arr[i]), side)) for i in ev]
            ps = [x for x in ps if x is not None]
            if not ps:
                continue
            # 절대 이동폭(bp): 라벨의 fast_move를 가격 대비로
            ext = low[first] if is_down else high[first]
            fwd = close[first + 1:min(first + 1 + 6, n)]
            move = (fwd.max() - ext) if is_down else (ext - fwd.min())
            bp = float(move / close[first] * 1e4) if close[first] > 0 else np.nan
            # local_extreme 창에서 극값과의 거리(상대, ATR 단위)
            a, b = max(0, first - W), min(n, first + W + 1)
            if is_down:
                gap_atr = float((low[first] - low[a:b].min()) / atr[first]) if atr[first] > 0 else np.nan
            else:
                gap_atr = float((high[a:b].max() - high[first]) / atr[first]) if atr[first] > 0 else np.nan
            results.setdefault(split, []).append({
                "side": side, "covered": covered, "first": first, "n_bars": len(ev),
                "max_proba": float(max(ps)), "captured": bool(max(ps) >= THRESHOLD),
                "fast_mult": float(fm[first]), "move_bp": bp,
                "atr_pct": float(atr[first] / close[first] * 1e4) if close[first] > 0 else np.nan,
                "extreme_gap_atr": gap_atr,
            })

    # 미포착 사건 -> 가장 가까운 포착 사건까지 거리(같은 side, 봉 단위)
    for split, evs in results.items():
        for side in ("bottom", "top"):
            cov = sorted(e["first"] for e in evs if e["covered"] and e["side"] == side)
            arr = np.array(cov)
            for e in evs:
                if e["covered"] or e["side"] != side or not len(arr):
                    continue
                e["dist_to_covered_bars"] = int(np.min(np.abs(arr - e["first"])))

    report = {"signal": "v_rebound_uncovered_event_capture", "asset": "ETHUSDT",
              "scope": {"purpose": "미포착 사건이 진짜인지 + 모델이 실제로 잡는지",
                        "deployed_config": DEPLOYED, "threshold": THRESHOLD, "gap": GAP,
                        "reselection_performed": False, "holdout_touched": False,
                        "live_code_changed": False},
              "splits": {}}

    for split in ("VAL", "OOS"):
        evs = results.get(split, [])
        if not evs:
            continue
        cov = [e for e in evs if e["covered"]]
        unc = [e for e in evs if not e["covered"]]
        log("")
        log(f"=== {split}: 사건 {len(evs):,}건 (포착 {len(cov):,} / 미포착 {len(unc):,}"
            f" = {len(unc)/len(evs)*100:.1f}%) ===")

        def blk(name, group):
            if not group:
                log(f"  {name}: 없음"); return {}
            capt = [e["captured"] for e in group]
            d = {"n": len(group), "model_capture_rate": round(float(np.mean(capt)), 4),
                 "max_proba_p25_50_75": q([e["max_proba"] for e in group]),
                 "fast_mult_p25_50_75": q([e["fast_mult"] for e in group]),
                 "move_bp_p25_50_75": q([e["move_bp"] for e in group]),
                 "atr_bp_p25_50_75": q([e["atr_pct"] for e in group]),
                 "n_bars_median": float(np.median([e["n_bars"] for e in group]))}
            log(f"  {name:6s} n={len(group):>5d}  **모델포착률 {pct(capt)}**  "
                f"확률(25/50/75) {d['max_proba_p25_50_75']}  "
                f"fast_mult {d['fast_mult_p25_50_75']}  "
                f"이동폭bp {d['move_bp_p25_50_75']}  ATRbp {d['atr_bp_p25_50_75']}  "
                f"사건길이중앙 {d['n_bars_median']:.0f}봉")
            return d
        r = {"n_events": len(evs), "uncovered_pct": round(len(unc) / len(evs) * 100, 1),
             "covered": blk("포착", cov), "uncovered": blk("미포착", unc)}

        dists = [e["dist_to_covered_bars"] for e in unc if "dist_to_covered_bars" in e]
        gaps = [e["extreme_gap_atr"] for e in unc if np.isfinite(e.get("extreme_gap_atr", np.nan))]
        if dists:
            near = float(np.mean(np.array(dists) <= GAP))
            r["uncovered_dist_to_covered"] = {"p25_50_75": q(dists),
                                              "within_gap_pct": round(near * 100, 1)}
            log(f"  미포착->최근접 포착사건 거리(봉) 25/50/75: {q(dists)}  "
                f"| GAP({GAP}봉) 이내: {near*100:.1f}%  <- 높으면 같은 V의 중복 의심")
        if gaps:
            r["uncovered_extreme_gap_atr"] = q(gaps)
            log(f"  미포착 첫봉이 ±30분 극값에서 떨어진 정도(ATR배수) 25/50/75: {q(gaps)}  "
                f"<- 0에 가까우면 '거의 극값인데 아깝게 놓친' 것")
        report["splits"][split] = r

    log("")
    log("=== 해석 ===")
    for split, r in report["splits"].items():
        c, u = r["covered"].get("model_capture_rate"), r["uncovered"].get("model_capture_rate")
        if c is None or u is None:
            continue
        log(f"  {split}: 모델포착률 포착사건 {c*100:.1f}% vs 미포착사건 {u*100:.1f}%  "
            f"(차이 {(u-c)*100:+.1f}%p)")
        if u < c * 0.5:
            log(f"    -> 미포착 사건을 모델이 절반도 못 잡는다. recall 이득이 장부상 숫자에 가깝다.")
        elif u >= c * 0.8:
            log(f"    -> 미포착 사건도 포착 사건과 비슷하게 잡는다. 재설계 전제가 실측으로 지지된다.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
