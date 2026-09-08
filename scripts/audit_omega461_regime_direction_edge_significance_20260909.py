"""전제 검증 — "HMM 이 방향 정보에서 이긴다"가 통계적으로 실재하는가.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지적(2026-09-09) "방향 정보는 이미 오메가 모델이 HMM 으로 학습해서 그런 거 아니야?"

두 갈래로 나뉜다
---------------
(1) **Phase 1 라우팅 A/B 는 확실히 교란돼 있다** — 전문가들이 wide24 라우팅으로 학습됐으므로
    A arm 이 구조적으로 유리하다. 이건 계약·실험문서에 이미 명시된 한계다.
(2) **그러나 전방수익 판별력 지표는 오메가와 무관하다** — `forward_power()` 는 분류기의 태그와
    원시 종가만 쓴다. 부모 TabM 이 계산에 들어가지 않는다.

그럼에도 확인 안 하고 넘어간 것이 있다: **그 격차가 애초에 통계적으로 실재하는가.**
h48 은 4시간 전방창인데 봉은 5분 간격이라 인접 표본의 전방수익이 거의 같은 구간을 공유한다.
명목 n 은 수천이지만 독립 표본은 훨씬 적다. 이 세션 초반 TabPFN 결과를 뒤집었던 바로 그 함정이다.

이 스크립트가 하는 일
--------------------
`bull` 태그 이후 수익 − `bear` 태그 이후 수익(bp) 을 **블록 부트스트랩**으로 재표집해 CI 를 낸다.
· 블록 길이 = 2016봉(7일) — 최장 호라이즌(h288=24h)의 7배로 잡아 블록 간 전방창 공유를 없앤다
· 모델별 CI 와, 두 모델 **격차**의 CI 를 함께 낸다(짝지은 블록 재표집 — 같은 블록에서 두 모델을
  동시에 평가하므로 시장 상황 차이가 상쇄된다)
· CI 가 0 을 포함하면 "그 모델의 방향 정보가 0 과 구분 불가", 격차 CI 가 0 을 포함하면
  "두 모델의 방향 정보가 서로 구분 불가" — 후자면 하이브리드의 전제가 무너진다.

준수: 신규 학습 없음. 저장된 사이드카만 읽는다. 라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WIDE24_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
BAL_DIR = ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909"
S12_DIR = ROOT / "data/ensemble/supervised/omega461_regimegbm_cut2509_20260909"
ARMS = {"wide24_HMM": (WIDE24_DIR, "regime3_current_sensitive_wide24_", "hmm_wide24"),
        "balgbm_GBM": (BAL_DIR, "regime3_balgbm_cut2509_", None),
        "s12k3_GBM": (S12_DIR, "regime3_s12k3_cut2509_", None)}
BASE_TAGS = ("2024", "2025", "2026_rebuilt")
CLASSES = ("bull", "bear", "chop")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
HORIZONS = (12, 48, 288)
BLOCK_BARS = 2016      # 7일 -- 최장 호라이즌(288봉=24h)의 7배
N_BOOT = 2000
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"


def load() -> pd.DataFrame:
    parts = []
    for tag in BASE_TAGS:
        b = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{tag}.csv",
                        usecols=["timestamp", "close"], parse_dates=["timestamp"])
        for name, (d, pref, stem) in ARMS.items():
            fn = (f"training_features_{tag}_regime3_current_sensitive_hmm_wide24.csv" if stem
                  else f"training_features_{tag}_{pref}sidecar.csv")
            s = pd.read_csv(d / fn, low_memory=False, parse_dates=["timestamp"],
                            usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
            b = b.merge(s, on="timestamp", how="inner")
        parts.append(b)
    return (pd.concat(parts, ignore_index=True).sort_values("timestamp")
              .drop_duplicates("timestamp", keep="last").reset_index(drop=True))


def spread_bp(pred: np.ndarray, fwd: np.ndarray) -> float:
    """bull 태그 전방수익 − bear 태그 전방수익 (bp). 한쪽이라도 비면 nan."""
    mb, mr = pred == 0, pred == 1
    if not mb.any() or not mr.any():
        return np.nan
    return float((np.mean(fwd[mb]) - np.mean(fwd[mr])) * 1e4)


def main() -> int:
    df = load()
    rng = np.random.default_rng(20260909)
    report = {"block_bars": BLOCK_BARS, "n_boot": N_BOOT, "windows": {},
              "scope_note": "이 지표는 오메가 부모를 쓰지 않는다 -- 분류기 태그와 원시 종가만 사용. "
                            "따라서 '부모가 HMM 으로 학습돼서'로는 설명되지 않는다. "
                            "교란이 있는 것은 Phase 1 라우팅 A/B 쪽이다."}

    for split, (s, e) in SPLITS.items():
        m = ((df["timestamp"] >= s) & (df["timestamp"] <= e)).to_numpy()
        d = df[m].reset_index(drop=True)
        close = pd.to_numeric(d["close"], errors="raise").to_numpy(np.float64)
        preds = {}
        for name, (_, pref, _) in ARMS.items():
            p = d[[f"{pref}{c}_prob" for c in CLASSES]].to_numpy(np.float64)
            preds[name] = p.argmax(1)
        blocks = np.arange(len(d)) // BLOCK_BARS
        uniq = np.unique(blocks)
        print(f"\n{'='*84}\n[{split}] {s} ~ {e}  {len(d):,}봉  독립 블록 {len(uniq)}개"
              f"(각 {BLOCK_BARS}봉=7일)", flush=True)

        res = {"bars": int(len(d)), "n_blocks": int(len(uniq)), "horizons": {}}
        for h in HORIZONS:
            fwd = np.full(len(close), np.nan)
            fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
            ok = np.isfinite(fwd)
            point = {n: spread_bp(pr[ok], fwd[ok]) for n, pr in preds.items()}

            # 짝지은 블록 재표집 -- 같은 블록에서 세 모델을 동시에 평가
            draws = {n: [] for n in preds}
            diffs = {"HMM_minus_balgbm": [], "HMM_minus_s12k3": []}
            for _ in range(N_BOOT):
                take = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([np.flatnonzero(blocks == b) for b in take])
                idx = idx[ok[idx]]
                if len(idx) < 100:
                    continue
                cur = {}
                for n, pr in preds.items():
                    v = spread_bp(pr[idx], fwd[idx])
                    cur[n] = v
                    if np.isfinite(v):
                        draws[n].append(v)
                if np.isfinite(cur["wide24_HMM"]) and np.isfinite(cur["balgbm_GBM"]):
                    diffs["HMM_minus_balgbm"].append(cur["wide24_HMM"] - cur["balgbm_GBM"])
                if np.isfinite(cur["wide24_HMM"]) and np.isfinite(cur["s12k3_GBM"]):
                    diffs["HMM_minus_s12k3"].append(cur["wide24_HMM"] - cur["s12k3_GBM"])

            hres = {"point_bp": {k: (round(v, 2) if np.isfinite(v) else None) for k, v in point.items()},
                    "ci_bp": {}, "diff_ci_bp": {}}
            print(f"\n  h{h} ({h*5//60}h) — bull−bear 격차 (bp), 블록 부트스트랩 95% CI", flush=True)
            for n in preds:
                a = np.array(draws[n])
                lo, hi = np.quantile(a, [0.025, 0.975])
                excl = bool(lo > 0 or hi < 0)
                hres["ci_bp"][n] = {"point": round(point[n], 2), "lo": round(float(lo), 2),
                                    "hi": round(float(hi), 2), "excludes_zero": excl}
                print(f"    {n:14s} {point[n]:+8.1f}  CI[{lo:+8.1f},{hi:+8.1f}] "
                      f"{'✅ 0 배제' if excl else '❌ 0 포함'}", flush=True)
            for dn, arr in diffs.items():
                a = np.array(arr)
                lo, hi = np.quantile(a, [0.025, 0.975])
                excl = bool(lo > 0 or hi < 0)
                hres["diff_ci_bp"][dn] = {"mean": round(float(a.mean()), 2), "lo": round(float(lo), 2),
                                          "hi": round(float(hi), 2), "excludes_zero": excl}
                print(f"    Δ {dn:22s} {a.mean():+8.1f}  CI[{lo:+8.1f},{hi:+8.1f}] "
                      f"{'✅ 두 모델 구분됨' if excl else '❌ 구분 불가'}", flush=True)
            res["horizons"][f"h{h}"] = hres
        report["windows"][split] = res

    (OUT / "direction_edge_significance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/direction_edge_significance.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
