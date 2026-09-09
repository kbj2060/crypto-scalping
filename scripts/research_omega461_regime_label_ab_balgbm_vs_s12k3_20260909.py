"""레짐 **라벨** A/B — balancedish vs S12_K3 (모델급 통제).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 결정(2026-09-09) "이제 hmm은 폐기하기로 했으니까 레짐 라벨을 비교해보자."

왜 이제야 깨끗한 비교가 되는가
-----------------------------
앞선 비교들은 모델급(HMM vs GBM)과 라벨이 섞여 있었다. HMM 을 폐기하면 두 후보는
**완전히 같은 모델급**이 된다 — 동일 HistGradientBoosting HP, 동일 136 feature_cols/medians,
동일 컷오프(≤2025-09-30), 동일 SEED, 동일 purged 5-fold OOF 프로토콜.
**남는 차이는 학습 타깃 라벨 하나뿐이다.** 이 스크립트는 그 하나를 잰다.

  balgbm : `balancedish_adx16_slope15_bb012` (ADX≥16 · EMA기울기 · BB폭 임계 규칙)
  s12k3  : `S12_K3` (er_12/er_24 효율비 + net_24 방향앵커 + slope_12, K=3 confirm)

무엇을 재는가 — 라벨 정의에 의존하지 않는 잣대만
-----------------------------------------------
각 라벨의 자체 정확도(κ)는 **비교 불가**다(다른 정답). 그래서 라벨 무관 잣대만 쓰고,
이번에는 **격차의 블록 부트스트랩 CI** 까지 낸다 — 앞선 방향축 검정에서 점추정 차이가
잡음이었던 전례가 있으므로.

  1. **변동성 분리력** (추세태그 전방변동성 ÷ chop태그 전방변동성). chop 이 제 뜻대로
     작동하는가. 이 라인에서 두 후보가 가장 크게 갈린 축이다.
  2. **방향 판별력** (bull−bear 전방수익 bp).
  3. **안정성** (flip율, 중앙 상태지속).
  4. **라벨 상호일치도 / 클래스 비중** — 두 라벨이 실제로 얼마나 다른 사건을 가리키는가.
  5. **피쳐 대비 신규성(해석용)**: 각 라벨의 재구성 κ. κ 가 1 에 가까울수록 그 레짐 정의는
     136피쳐의 결정론적 함수에 가깝고, 부모(102 base_cols 가 그 피쳐셋과 크게 겹침)에게
     **새 정보를 덜 준다**. 이 축은 balgbm 에 불리하게 작용할 수 있어 함께 본다.

블록 부트스트랩: 7일(2016봉) 블록 — 최장 호라이즌(h288=24h)의 7배. 짝지은 재표집이므로
같은 블록에서 두 라벨을 동시에 평가해 시장 상황 차이가 상쇄된다.

준수: 신규 학습 없음(두 사이드카 모두 이미 생성됨). 라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _current_labels3_thresholded  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import _debounce, scaled_label  # noqa: E402

BAL_DIR = ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909"
S12_DIR = ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909"
BAL_MODEL = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/regime_balgbm_cut2509_model.joblib"
S12_MODEL = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/regime_cut2509_model.joblib"
HMM_MODEL = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                  / "regime3_current_sensitive_hmm_wide24_2024.joblib")
ARMS = {"balgbm(balancedish)": (BAL_DIR, "regime3_balgbm_cut2509_"),
        "s12k3(S12_K3)": (S12_DIR, "regime3_s12k3_cut2509_")}
BASE_TAGS = ("2024", "2025", "2026_rebuilt")
CLASSES = ("bull", "bear", "chop")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
HORIZONS = (12, 48, 288)
BLOCK_BARS, N_BOOT = 2016, 2000
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"


def load() -> pd.DataFrame:
    parts = []
    for tag in BASE_TAGS:
        b = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{tag}.csv",
                        low_memory=False, parse_dates=["timestamp"])
        for _, (d, pref) in ARMS.items():
            s = pd.read_csv(d / f"training_features_{tag}_{pref}sidecar.csv",
                            low_memory=False, parse_dates=["timestamp"],
                            usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
            b = b.merge(s, on="timestamp", how="inner")
        parts.append(b)
    return (pd.concat(parts, ignore_index=True).sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last").reset_index(drop=True))


def vol_ratio(pred: np.ndarray, fwd: np.ndarray) -> float:
    """추세태그 전방변동성 ÷ chop태그 전방변동성."""
    tr, ch = (pred == 0) | (pred == 1), pred == 2
    if tr.sum() < 30 or ch.sum() < 30:
        return np.nan
    c = np.std(fwd[ch])
    return float(np.std(fwd[tr]) / c) if c > 0 else np.nan


def dir_spread(pred: np.ndarray, fwd: np.ndarray) -> float:
    mb, mr = pred == 0, pred == 1
    if mb.sum() < 30 or mr.sum() < 30:
        return np.nan
    return float((np.mean(fwd[mb]) - np.mean(fwd[mr])) * 1e4)


def stability(pred):
    runs, c = [], 1
    for i in range(1, len(pred)):
        if pred[i] == pred[i - 1]:
            c += 1
        else:
            runs.append(c); c = 1
    runs.append(c)
    return {"flip_rate": round(float((pred[1:] != pred[:-1]).mean()), 4),
            "median_run_bars": float(np.median(runs))}


def paired_ci(stat_fn, preds: dict, fwd: np.ndarray, blocks: np.ndarray,
              ok: np.ndarray, rng) -> dict:
    uniq = np.unique(blocks)
    names = list(preds)
    draws = {n: [] for n in names}
    diff = []
    for _ in range(N_BOOT):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(blocks == b) for b in take])
        idx = idx[ok[idx]]
        if len(idx) < 200:
            continue
        cur = {n: stat_fn(preds[n][idx], fwd[idx]) for n in names}
        for n in names:
            if np.isfinite(cur[n]):
                draws[n].append(cur[n])
        if np.isfinite(cur[names[0]]) and np.isfinite(cur[names[1]]):
            diff.append(cur[names[0]] - cur[names[1]])
    out = {}
    for n in names:
        a = np.array(draws[n]); lo, hi = np.quantile(a, [0.025, 0.975])
        out[n] = {"point": round(float(stat_fn(preds[n][ok], fwd[ok])), 4),
                  "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
    a = np.array(diff); lo, hi = np.quantile(a, [0.025, 0.975])
    out["diff_balgbm_minus_s12k3"] = {"mean": round(float(a.mean()), 4),
                                      "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                                      "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main() -> int:
    df = load()
    rng = np.random.default_rng(20260909)
    hmm_cfg = joblib.load(HMM_MODEL)["label_config"]
    s12_spec = joblib.load(S12_MODEL)["label_spec"]
    y_bal_all = _current_labels3_thresholded(df, hmm_cfg)
    y_s12_all = _debounce(scaled_label(df["close"], 12, float(s12_spec["T1_er12"]),
                                       float(s12_spec["T2_er24"])), 3)
    bal_k = joblib.load(BAL_MODEL); s12_k = joblib.load(S12_MODEL)
    print(f"[통제] 두 arm 모두 HGB {bal_k['config']} · SEED {bal_k['seed']} · "
          f"{len(bal_k['feature_cols'])}피쳐 · 컷오프 {bal_k['train_range'].split('~')[1][:10]}", flush=True)
    print(f"[차이] 학습 타깃 라벨 하나뿐 — balancedish vs S12_K3", flush=True)

    report = {"controlled": {"hp": bal_k["config"], "seed": bal_k["seed"],
                             "n_features": len(bal_k["feature_cols"]),
                             "cutoff": bal_k["train_range"]}, "windows": {}}

    for split, (s, e) in SPLITS.items():
        m = ((df["timestamp"] >= s) & (df["timestamp"] <= e)).to_numpy()
        d = df[m].reset_index(drop=True)
        close = pd.to_numeric(d["close"], errors="raise").to_numpy(np.float64)
        preds = {}
        for name, (_, pref) in ARMS.items():
            p = d[[f"{pref}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
            preds[name] = p.argmax(1)
        y_bal, y_s12 = y_bal_all[m], y_s12_all[m]
        blocks = np.arange(len(d)) // BLOCK_BARS

        print(f"\n{'='*90}\n[{split}] {s} ~ {e}  {len(d):,}봉  독립 블록 {len(np.unique(blocks))}개", flush=True)
        res = {"bars": int(len(d))}

        # --- 라벨 자체의 성질 ---
        lab_share = {"balancedish": {n: round(float((y_bal == i).mean()), 3) for i, n in enumerate(CLASSES)},
                     "S12_K3": {n: round(float((y_s12 == i).mean()), 3) for i, n in enumerate(CLASSES)}}
        lab_agree = float((y_bal == y_s12).mean())
        lab_kappa = float(cohen_kappa_score(y_bal, y_s12, labels=[0, 1, 2]))
        recon = {"balancedish": round(float(cohen_kappa_score(y_bal, preds["balgbm(balancedish)"], labels=[0, 1, 2])), 4),
                 "S12_K3": round(float(cohen_kappa_score(y_s12, preds["s12k3(S12_K3)"], labels=[0, 1, 2])), 4)}
        res.update({"label_class_shares": lab_share, "label_agreement": round(lab_agree, 4),
                    "label_kappa_between": round(lab_kappa, 4), "reconstruction_kappa": recon})
        print(f"  라벨 비중  balancedish {lab_share['balancedish']}   S12_K3 {lab_share['S12_K3']}", flush=True)
        print(f"  두 라벨 상호일치 {lab_agree*100:.1f}%  (κ={lab_kappa:.4f})", flush=True)
        print(f"  피쳐로부터의 재구성 κ  balancedish {recon['balancedish']:.4f}  "
              f"S12_K3 {recon['S12_K3']:.4f}   ← 높을수록 피쳐와 중복(신규정보 적음)", flush=True)

        # --- 안정성 ---
        res["stability"] = {n: stability(p) for n, p in preds.items()}
        print(f"\n  안정성  " + "   ".join(
            f"{n}: flip {v['flip_rate']:.4f} 지속 {v['median_run_bars']:.1f}봉"
            for n, v in res["stability"].items()), flush=True)

        # --- 라벨 무관 잣대 + 격차 CI ---
        res["forward"] = {}
        for h in HORIZONS:
            fwd = np.full(len(close), np.nan)
            fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
            ok = np.isfinite(fwd)
            vr = paired_ci(vol_ratio, preds, fwd, blocks, ok, rng)
            ds = paired_ci(dir_spread, preds, fwd, blocks, ok, rng)
            res["forward"][f"h{h}"] = {"vol_ratio": vr, "dir_spread_bp": ds}
            print(f"\n  h{h} ({h*5//60}h)", flush=True)
            print(f"    변동성 분리력  balgbm {vr['balgbm(balancedish)']['point']:.3f}"
                  f"[{vr['balgbm(balancedish)']['lo']:.3f},{vr['balgbm(balancedish)']['hi']:.3f}]   "
                  f"s12k3 {vr['s12k3(S12_K3)']['point']:.3f}"
                  f"[{vr['s12k3(S12_K3)']['lo']:.3f},{vr['s12k3(S12_K3)']['hi']:.3f}]", flush=True)
            dd = vr["diff_balgbm_minus_s12k3"]
            print(f"      Δ(balgbm−s12k3) {dd['mean']:+.3f} CI[{dd['lo']:+.3f},{dd['hi']:+.3f}] "
                  f"{'✅ 구분됨' if dd['excludes_zero'] else '❌ 구분 불가'}", flush=True)
            dd2 = ds["diff_balgbm_minus_s12k3"]
            print(f"    방향 판별력    balgbm {ds['balgbm(balancedish)']['point']:+.1f}bp   "
                  f"s12k3 {ds['s12k3(S12_K3)']['point']:+.1f}bp   "
                  f"Δ {dd2['mean']:+.1f} CI[{dd2['lo']:+.1f},{dd2['hi']:+.1f}] "
                  f"{'✅ 구분됨' if dd2['excludes_zero'] else '❌ 구분 불가'}", flush=True)
        report["windows"][split] = res

    (OUT / "label_ab_balgbm_vs_s12k3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/label_ab_balgbm_vs_s12k3.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
