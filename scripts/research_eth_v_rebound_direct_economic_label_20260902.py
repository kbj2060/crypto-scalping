#!/usr/bin/env python3
"""⭐라벨 근본 재설계 -- "V자반등 모양"이 아니라 **"여기서 진입하면 비용 후 이익인가"**를 가르친다.

## 왜 (지금까지 전부 실패한 이유의 공통점)

2026-09-01~02에 시도한 모든 라벨(v7b/giveback/앵커변형/bp하한×2종)은 **패턴 라벨**이다 --
"1.5×ATR 튀고 되반납 20% 이하" 같은 **모양**을 정의하고 그걸 맞히게 했다. 그 결과:

  · 분류는 된다(AUC 0.70) -- 모양은 실재하니까.
  · 그런데 모델은 **이미 완성된 모양**을 고른다(호출 소진율 121~128%, 라벨 양성 전체는 40%).
  · 진입가 앵커로 바꾸면 예측력 붕괴(AUC 0.58).
  · 진입기준 bp 하한을 음성으로 가르치면 메커니즘은 고쳐지나(소진 128→73%, 먹을폭 +83%)
    **방향은 안 생긴다** -- 하한은 '크기'를 고르는데 크기는 방향과 무관하다(FLOOR=20에서
    갭 −4.63bp, 귀무 9%).

**한 번도 안 해본 것: 경제적 결과 자체를 라벨로 쓰는 것.** 모양을 거치지 않고
"이 봉에서 이 방향으로 진입 → 트레일링 청산 → 비용 차감 후 양수인가"를 직접 가르친다.
이러면 소진·방향·비용이 라벨에 **구조적으로 내장**된다. 모델이 최적화하는 대상과
우리가 원하는 대상이 처음으로 일치한다.

## 라벨

각 (봉, 방향)마다 `open[i+1]` 진입 → 트레일링 청산 → 비용 10bp 차감한 `net_bp`를 계산.

  E0_binary   : net > 0 이면 1, 아니면 0
  E1_deadband : net > +5bp → 1, net < -5bp → 0, 그 사이는 **제외**
                (v7b가 '애매한 중간지대 제외'로 크게 개선된 이 저장소의 확립된 패턴)
  E2_robust   : 3개 셀 중 2개 이상에서 net > 0 이면 1 -- 단일 exit config 과적합 방지

셀은 **사전 지정**한다(VAL에서 고르지 않는다 -- 그러면 라벨이 평가셋을 훔쳐본다):
`SL=5.0/ARM=1.5/Trail=0.1`. 2026-09-01 9-10 경제성 게이트에서 매 봉 모델과 기존 giveback
모델이 **독립적으로 같은 조합을 골랐던** 교차수렴 지점이라 사전 선택 근거가 있다.
E2의 3셀 = 위 + `4.0/1.5/0.1` + `1.5/1.0/0.1`.

## 피쳐 -- 라벨 수정과 함께 처음 시도

  F0: Tier0 23 (현행)
  F1: Tier0 23 + 154셋 감사통과 150개
      ⚠️154셋은 wick 라벨에서만 테스트됐고(순증분 0) **새 라벨과 조합된 적이 없다.**
      "이 봉에서 뭐가 일어났나"가 아니라 "앞으로 뭐가 일어날까"를 묻는 라벨에서는
      미시구조 피쳐의 역할이 다를 수 있다.

## 판정 -- 기대값 정면비교 (개수 아님)

호출 빈도를 현행 배포판에 일치(VAL 1,693 / OOS 1,367)시키고,
**정방향 vs 뒤집기의 최고/중앙 기대값 갭** + 랜덤 부분표집 귀무분포(B=200).
중앙 갭 > 0 이고 귀무 >= 95%를 VAL/OOS 양쪽에서 만족해야 통과.

⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server via handoff.
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


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_s1 = _load("s1_econ", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_bt = _s1._bt
TIER0 = _s1.FEATURE_COLUMNS
FORWARD_BARS = _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

F154 = ROOT / "tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv"
AUDIT = ROOT / "data/research/eth_154feature_audit_20260901/report.json"

LABEL_CELL = (5.0, 1.5, 0.1)                      # 사전지정 (9-10 교차수렴 지점)
ROBUST_CELLS = [(5.0, 1.5, 0.1), (4.0, 1.5, 0.1), (1.5, 1.0, 0.1)]
DEADBAND_BP = 5.0
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
CONTEXT_N, SEED = 18000, 20260829
NULL_B, NULL_SEED = 200, 20260902
CHUNK = 40000
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_direct_economic_label_20260902/report.json"


def log(m): print(f"[econlabel] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=0.20, full_bars=12)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long = long.rename(columns={"label": "pattern_label"})
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    long = long.loc[(long["pos"] >= 0) & (long["pos"] + FORWARD_BARS + 1 < nk)].reset_index(drop=True)
    log(f"  프레임 {len(long):,}행")

    # ---------- 경제적 결과를 청크로 계산 ----------
    def net_bp_for(cell):
        sl, arm, tr = cell
        out = np.full(len(long), np.nan)
        i_all = long["pos"].to_numpy().astype(int)
        sgn_all = np.where(long["is_downside"].to_numpy() == 1, 1.0, -1.0)
        atr_all = long["atr"].to_numpy(dtype=float)
        for st_ in range(0, len(long), CHUNK):
            en = min(st_ + CHUNK, len(long))
            idx = i_all[st_:en]
            H = np.stack([h[j+1:j+1+FORWARD_BARS] for j in idx])
            L = np.stack([l[j+1:j+1+FORWARD_BARS] for j in idx])
            C = np.stack([c[j+1:j+1+FORWARD_BARS] for j in idx])
            e = o[idx + 1]
            out[st_:en] = _bt.simulate_trailing_vec(e, atr_all[st_:en], sgn_all[st_:en],
                                                    H, L, C, sl, arm, tr, True) * 1e4 - COST_BP
        return out

    log(f"  경제라벨 계산 (사전지정 셀 {LABEL_CELL}) ...")
    net_main = net_bp_for(LABEL_CELL)
    long["net_bp"] = net_main
    log(f"    net_bp 중앙 {np.nanmedian(net_main):+.2f}bp  양수비율 {float(np.nanmean(net_main>0))*100:.1f}%")
    votes = np.zeros(len(long))
    for cl in ROBUST_CELLS:
        votes += (net_bp_for(cl) > 0).astype(float)
        log(f"    robust 투표 셀 {cl} 완료")

    LABELS = {
        "E0_binary": np.where(np.isfinite(net_main), (net_main > 0).astype(float), np.nan),
        "E1_deadband": np.where(net_main > DEADBAND_BP, 1.0,
                        np.where(net_main < -DEADBAND_BP, 0.0, np.nan)),
        "E2_robust": np.where(np.isfinite(net_main), (votes >= 2).astype(float), np.nan),
    }
    for k, v in LABELS.items():
        m = np.isfinite(v)
        log(f"    {k:14s} 라벨행 {int(m.sum()):>7,} ({m.mean()*100:4.1f}%)  라벨률 {np.nanmean(v):.4f}")
    log(f"    (참고) 기존 패턴 라벨과 E0의 일치율: "
        f"{float((long['pattern_label'].to_numpy() == LABELS['E0_binary'])[long['pattern_label'].notna()].mean())*100:.1f}%")

    # ---------- 피쳐셋 ----------
    passed = [f["feature"] for f in json.loads(AUDIT.read_text())["features"] if f["verdict"] == "pass"]
    ex = pd.read_csv(F154)
    ex["timestamp"] = pd.to_datetime(ex["timestamp"]).dt.tz_localize("UTC")
    excols = [x for x in passed if x in ex.columns]
    long = long.merge(ex[["timestamp"] + excols], on="timestamp", how="left")
    FEATSETS = {"F0_tier0_23": TIER0, f"F1_tier0+154감사통과_{len(excols)}": TIER0 + excols}
    log(f"  피쳐셋: F0 {len(TIER0)}개 / F1 {len(TIER0)+len(excols)}개")

    # ---------- 평가 도구 ----------
    def build(s):
        rows = []
        for i_, isd, atr_ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(), s["atr"].to_numpy()):
            i = int(i_)
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr_),
                         "entry_price": float(o[i+1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    def duel(df):
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fw, flp, meta = [], [], []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr in TRAIL_GRID:
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, True)*1e4-COST_BP
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, True)*1e4-COST_BP
                    w = pv > 0
                    fw.append(float(pv.mean())); flp.append(float(fp.mean()))
                    meta.append({"sl": sl, "arm": arm, "trail": tr, "win_rate": float(w.mean()),
                                 "payoff": float(pv[w].mean()/-pv[~w].mean()) if w.any() and (~w).any() else None})
        fw, flp = np.array(fw), np.array(flp)
        bi = int(fw.argmax())
        return {"n": int(len(df)), "best_fwd": float(fw.max()), "best_flip": float(flp.max()),
                "gap_best": float(fw.max()-flp.max()), "med_fwd": float(np.median(fw)),
                "med_flip": float(np.median(flp)), "gap_med": float(np.median(fw)-np.median(flp)),
                "best_cell": meta[bi], "flip_at_best_fwd_cell": float(flp[bi])}

    # 현행 배포판 호출 수 = 목표 빈도
    TARGET_N = {"VAL": 1693, "OOS": 1367}
    report = {"signal": "v_rebound_direct_economic_label", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"idea": "라벨 = 트레일링 순손익 부호 (모양이 아니라 경제적 결과)",
                        "label_cell_preselected": list(LABEL_CELL),
                        "label_cell_rationale": "9-10 교차수렴 지점 -- VAL에서 고르지 않음",
                        "robust_cells": [list(x) for x in ROBUST_CELLS],
                        "deadband_bp": DEADBAND_BP, "cost_bp": COST_BP,
                        "target_calls": TARGET_N, "null_B": NULL_B,
                        "holdout_touched": False, "live_code_changed": False},
              "configs": {}}

    nrng = np.random.default_rng(NULL_SEED)
    for lname, lab in LABELS.items():
        long["y"] = lab
        for fname, cols in FEATSETS.items():
            key = f"{lname} | {fname}"
            sub = long.dropna(subset=cols).copy()
            tr_ = sub.loc[(sub["split"] == "TRAIN") & sub["y"].notna()]
            if len(tr_) < 5000 or tr_["y"].nunique() < 2:
                log(f"  {key}: 학습 불가"); continue
            rng = np.random.default_rng(SEED)
            ctx = tr_.iloc[np.sort(rng.choice(len(tr_), size=min(CONTEXT_N, len(tr_)), replace=False))]
            clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
            clf.fit(ctx[cols], ctx["y"].to_numpy())
            log("")
            log(f"########## {key}  TRAIN {len(tr_):,} 라벨률 {tr_['y'].mean():.4f} ##########")
            ent = {"train_n": int(len(tr_)), "label_rate": round(float(tr_["y"].mean()), 4)}
            for spn in ("VAL", "OOS"):
                s = sub.loc[sub["split"] == spn].copy()
                s["p"] = np.concatenate([clf.predict_proba(s[cols].iloc[k:k+20000])[:, 1]
                                         for k in range(0, len(s), 20000)])
                lb = s.loc[s["y"].notna()]
                auc = float(roc_auc_score(lb["y"], lb["p"])) if lb["y"].nunique() == 2 else None
                sel = s.nlargest(min(TARGET_N[spn], len(s)), "p")
                d = duel(build(sel))
                gaps = []
                for _ in range(NULL_B):
                    ridx = nrng.choice(len(s), size=len(sel), replace=False)
                    rd = duel(build(s.iloc[np.sort(ridx)]))
                    if rd:
                        gaps.append(rd["gap_med"])
                pg = round(float((np.array(gaps) < d["gap_med"]).mean()*100), 1) if len(gaps) >= 20 else None
                bc = d["best_cell"]
                log(f"  {spn} n={d['n']:,}  자기라벨AUC {auc:.4f}")
                log(f"    ⭐중앙 기대값  정방향 {d['med_fwd']:+7.2f}bp  뒤집기 {d['med_flip']:+7.2f}bp  "
                    f"갭 {d['gap_med']:+7.2f}bp  (귀무 {pg}%)"
                    f"{'  ✅' if (pg or 0) >= 95 and d['gap_med'] > 0 else ''}")
                log(f"       최고 기대값  정방향 {d['best_fwd']:+7.2f}bp  뒤집기 {d['best_flip']:+7.2f}bp  "
                    f"갭 {d['gap_best']:+7.2f}bp")
                log(f"       최고셀 {bc['sl']}/{bc['arm']}/{bc['trail']}  승률 {bc['win_rate']*100:.1f}%  "
                    f"손익비 {bc['payoff'] if bc['payoff'] else float('nan'):.3f}  "
                    f"같은셀 뒤집기 {d['flip_at_best_fwd_cell']:+.2f}bp")
                ent[spn] = {"auc": round(auc, 4) if auc else None,
                            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()},
                            "null_pctile_gap_med": pg}
            report["configs"][key] = ent
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    log("")
    log("=== 판정 (VAL/OOS 양쪽: 중앙갭>0 AND 귀무>=95%) ===")
    for k, v in report["configs"].items():
        if "VAL" not in v or "OOS" not in v:
            continue
        ok = all(v[s]["gap_med"] > 0 and (v[s]["null_pctile_gap_med"] or 0) >= 95 for s in ("VAL", "OOS"))
        log(f"  {'✅' if ok else '  '}{k:44s} VAL 갭{v['VAL']['gap_med']:+7.2f}({v['VAL']['null_pctile_gap_med']}%)  "
            f"OOS 갭{v['OOS']['gap_med']:+7.2f}({v['OOS']['null_pctile_gap_med']}%)")

    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
