#!/usr/bin/env python3
"""라벨 타이트화가 "미포착=모델이 못 잡는 사건" 문제의 지렛대인지 **기전만** 확인.

## 질문 하나

2026-09-01 실측: V자반등 사건 중 38%가 9트리거 미포착인데, 그 사건들은 라벨 품질(fast_mult,
이동폭bp, ATR)이 포착 사건과 거의 같으면서도 **모델이 거의 못 잡는다**(포착률 27.9% vs 8.5%,
OOS 27.4% vs 12.6%). 즉 결과로는 진짜 V자반등인데 사전 피쳐로는 예고가 없던 사건이다.

사용자 제안: "라벨링 로직을 더 타이트하게 하면 되지 않나?"

타이트화는 `ATR_MULT`(fast_mult 문턱)와 `T_SUSTAIN`(giveback 문턱)을 조이는 것인데, 미포착 사건은
**정확히 그 축에서 포착 사건과 분포가 포개진다**. 그러면 조여도 양쪽이 비례해서 빠질 뿐 격차는
그대로 남는다 -- 이게 내 예상이다. 다만 "중앙값이 비슷"이 "꼬리까지 같다"를 뜻하진 않고,
아주 타이트한 구간(K>=2.5)은 Stage 1 격자 밖이라 확인되지 않았다.

## 그래서 격자를 다시 돌리지 않고 기전만 본다

3개 설정에서 **포착:미포착 비율**과 **모델 포착률 격차**가 어떻게 움직이는지만 본다.
경제성 최적화도, 파라미터 재선택도 하지 않는다(그 길은 Stage1~3에서 이미
"스윕->헤드라인->다음 검증에서 증발"로 세 번 끝났다).

  현행      K=1.50 T=0.20
  타이트    K=2.00 T=0.15   (Stage 1 격자 경계)
  초타이트  K=2.50 T=0.10   (격자 밖 -- 새 정보)

판정:
  - 격차가 좁혀지면 -> 타이트화가 지렛대. 그때 제대로 스윕할 값이 있다.
  - 그대로면 -> 라벨 축은 지렛대가 아니다. 손댈 곳은 피쳐이거나, 낮은 recall을 받아들이는 것.

모든 설정을 **동일 호출 빈도**에서 비교하지 않고, 여기서는 임계값을 배포값 0.60으로 고정한다 --
"같은 운영점에서 어느 라벨이 미포착 사건까지 보이게 하는가"가 질문이기 때문. 호출수 자체도 함께
찍어 해석에 쓴다.

⚠️진단 전용. HOLDOUT 미터치. 라이브 코드 변경 없음. 이 결과로 라벨을 바꾸지 않는다.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_label_tightening_mechanism_check_20260901.py
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
_spec = importlib.util.spec_from_file_location("vreb_s1_tight", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_feas = _s1._feas

AUDIT = ROOT / "scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py"
_aspec = importlib.util.spec_from_file_location("vreb_audit_tight", AUDIT)
_audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(_audit)

FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
ALL9 = _feas.ALL9
W, GAP = 6, 12
THRESHOLD = 0.60
CONTEXT_N, SEED = 18000, 20260829

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

CONFIGS = [
    {"name": "현행",     "atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12},
    {"name": "타이트",   "atr_mult": 2.00, "t_sustain": 0.15, "full_bars": 12},
    {"name": "초타이트", "atr_mult": 2.50, "t_sustain": 0.10, "full_bars": 12},
]

OUT_JSON = ROOT / "data/research/eth_v_rebound_label_grid_stage1_20260901/tightening_mechanism.json"


def log(msg: str) -> None:
    print(f"[tight] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}  (진단 전용 -- 라벨 재선택 없음)")

    _s1.VAL_END = OOS_END
    log("building frame...")
    sig, feat, eth = _s1.build_sig()
    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    ts_arr = sig["timestamp"].to_numpy()
    lo_flag = np.zeros(n, dtype=bool); hi_flag = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            lo_flag[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            hi_flag[i] = True
    trig_by_side = {}
    for side in ("bottom", "top"):
        t_ = np.any([sig[f"{side}_{nm}"].fillna(False).to_numpy()
                     for nm in ALL9 if nm != "local_extreme"], axis=0)
        trig_by_side[side] = t_ | (lo_flag if side == "bottom" else hi_flag)

    out = {}
    for cfg in CONFIGS:
        kw = {k: cfg[k] for k in ("atr_mult", "t_sustain", "full_bars")}
        name = cfg["name"]
        log("")
        log(f"=== {name}  K={kw['atr_mult']} T={kw['t_sustain']} ===")
        sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **kw)
        st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **kw)
        long = _s1.long_frame_for(sig, feat, sb, st)
        long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                         np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
        lab = long.loc[long["label"].notna()]
        tr = lab.loc[lab["split"] == "TRAIN"]
        if len(tr) < 2000 or tr["label"].nunique() < 2:
            log("  TRAIN 표본 부족 -- 스킵"); continue

        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(tr.iloc[idx][FEATURE_COLUMNS], tr.iloc[idx]["label"].to_numpy())
        log(f"  TRAIN 양성률 {tr['label'].mean():.4f} | 라벨행 {len(tr):,}")

        proba, call_n = {}, {}
        for sp in ("VAL", "OOS"):
            s = long.loc[long["split"] == sp].copy()
            s["model_proba"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]
            call_n[sp] = int((s["model_proba"] >= THRESHOLD).sum())
            for ts, side, pr in zip(s["timestamp"], s["side"], s["model_proba"]):
                proba[(ts, side)] = float(pr)

        rec = {}
        for side, is_down in (("bottom", True), ("top", False)):
            status = sb if is_down else st
            trig = trig_by_side[side]
            for ev in _audit.cluster_events(np.flatnonzero(status == "v_rebound"), GAP):
                first = int(ev[0]); ts0 = pd.Timestamp(ts_arr[first])
                sp = "TRAIN" if ts0 < TRAIN_END else ("VAL" if ts0 < VAL_END else "OOS")
                if sp == "TRAIN":
                    continue
                ps = [proba.get((pd.Timestamp(ts_arr[i]), side)) for i in ev]
                ps = [x for x in ps if x is not None]
                if not ps:
                    continue
                rec.setdefault(sp, []).append({"covered": bool(trig[ev].any()),
                                               "captured": bool(max(ps) >= THRESHOLD)})
        res = {"train_pos_rate": round(float(tr["label"].mean()), 4), "params": kw}
        for sp in ("VAL", "OOS"):
            evs = rec.get(sp, [])
            if not evs:
                continue
            cov = [e for e in evs if e["covered"]]
            unc = [e for e in evs if not e["covered"]]
            cr = float(np.mean([e["captured"] for e in cov])) if cov else float("nan")
            ur = float(np.mean([e["captured"] for e in unc])) if unc else float("nan")
            res[sp] = {"n_events": len(evs), "n_covered": len(cov), "n_uncovered": len(unc),
                       "uncovered_pct": round(len(unc) / len(evs) * 100, 1),
                       "capture_covered": round(cr, 4), "capture_uncovered": round(ur, 4),
                       "ratio_unc_over_cov": round(ur / cr, 3) if cr and cr == cr and cr > 0 else None,
                       "n_called_bars": call_n[sp]}
            r = res[sp]
            log(f"  {sp}: 사건 {len(evs):>5,} (미포착 {r['uncovered_pct']:>4.1f}%)  "
                f"호출봉 {r['n_called_bars']:>5,}  "
                f"포착률 포착 {cr*100:>5.1f}% / 미포착 {ur*100:>5.1f}%  "
                f"**비율 {r['ratio_unc_over_cov']}**")
        out[name] = res

    log("")
    log("=== 판정: 라벨을 조이면 격차가 좁혀지는가 ===")
    for sp in ("VAL", "OOS"):
        row = [(nm, out[nm][sp]) for nm in out if sp in out[nm]]
        if len(row) < 2:
            continue
        log(f"  [{sp}] " + "   ".join(
            f"{nm}: 미포착{r['uncovered_pct']:.0f}% 비율{r['ratio_unc_over_cov']}" for nm, r in row))
        base = row[0][1]["ratio_unc_over_cov"]
        tight = row[-1][1]["ratio_unc_over_cov"]
        if base and tight:
            if tight >= base * 1.5:
                log(f"    -> 비율이 {base}->{tight}로 뚜렷이 개선. **타이트화가 지렛대일 수 있다.**")
            elif tight <= base * 1.2:
                log(f"    -> 비율이 {base}->{tight}로 거의 그대로. "
                    f"**라벨 축은 지렛대가 아니다** -- 조여도 양쪽이 비례해 빠질 뿐.")
            else:
                log(f"    -> 비율 {base}->{tight}, 애매. 단독 근거로 쓰지 말 것.")

    report = {"signal": "v_rebound_label_tightening_mechanism", "asset": "ETHUSDT",
              "scope": {"purpose": "타이트화가 '미포착=모델이 못 잡음' 문제의 지렛대인지 기전 확인",
                        "threshold": THRESHOLD, "gap": GAP,
                        "reselection_performed": False, "holdout_touched": False,
                        "live_code_changed": False,
                        "note": "경제성 최적화 아님 -- 포착률 격차 기전만 본다"},
              "configs": CONFIGS, "results": out, "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
