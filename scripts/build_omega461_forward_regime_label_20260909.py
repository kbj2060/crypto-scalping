"""전향(forward) 레짐 라벨 — 지연 제거 설계.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 결정(2026-09-09) 리드-랙 진단에서 기존 라벨이 **−12봉(1시간) 후행**, debounce 적용
시 −18봉으로 확인되자 "허용 안돼".

왜 라벨 정의를 바꿔야만 하는가
-----------------------------
"지금 어느 레짐인가"는 **과거로 정의**(→ 필연적 지연) 또는 **미래로 정의**(→ 예측 문제) 둘 중
하나다. 기존 balancedish / S12_K3 는 **둘 다 과거 정의**다:
    balancedish : EMA21 기울기(5봉 차분) + ADX(14)      — 전부 후행
    S12_K3      : er_12/er_24 효율비 + net_24 + slope_12 — 전부 후행
그래서 시각 t 의 태그가 t−12..t 구간을 서술한다(실측 상관 0.667). 고점 직후 bull, 저점 직후
bear 가 유지되는 것은 **정의상 필연**이며 모델·debounce 로는 못 고친다.

이 스크립트의 라벨 (전향 정의)
-----------------------------
앞으로 H봉 동안의 실제 경로로 라벨을 만든다:
    net_fwd = (close[t+H] − close[t]) / close[t]
    eff_fwd = |net_fwd| / Σ|diff(close)| over (t, t+H]        ← 경로 효율(추세성)
    bull : net_fwd > +θ  AND  eff_fwd >= e
    bear : net_fwd < −θ  AND  eff_fwd >= e
    chop : 나머지
θ, e 는 **TRAIN 구간에서만** 캘리브레이션한다 — balnobb 라벨의 TRAIN 클래스 비중에 백분위
매칭해서 비교 가능성을 유지한다(같은 클래스 균형, 다른 정의).

⚠️ 이건 라벨이 미래를 쓰는 것이고 **피쳐는 전부 인과**다(t 까지). 표준 지도학습 설정이며
미래참조가 아니다 — 단, 두 가지 경계를 반드시 지킨다:
  1. 학습 컷오프 근처에서 라벨이 컷오프 너머 H봉을 훔쳐본다 → **컷오프에서 H봉을 잘라낸다**
  2. OOF fold 경계에서도 같은 이유로 **퍼지 ≥ H** 를 준다 (여기선 2H)

⚠️ 정직한 위험 고지
------------------
이 전환은 레짐을 "결정론적 재구성"(balancedish κ 0.96 — 사실상 공식 복원)에서 **진짜 예측
문제**로 바꾼다. bull/bear 를 미래 기준으로 맞히는 것은 이 저장소가 7개 label×model 조합으로
반복 실패한 축이다(`docs/eth_omega4_6_1_accuracy_research_ideas_20260811.md` §2). 따라서
**사전 킬 기준**을 건다:

  K1. 리드-랙 최대 상관 h 가 **0 이상**이어야 한다(후행이면 목적 미달 — 설계 실패).
  K2. 전방수익 bull−bear 격차가 **7일 블록 부트스트랩에서 CI 가 0 을 배제**해야 한다.
      (기존 라벨들은 이 검정에서 전부 실패했다. 여기서도 실패하면 전향 라벨도 방향 정보가
       없다는 뜻이고, 그러면 레짐으로 방향을 잡는 축 전체가 닫힌다.)
  K3. 분류 성능이 **무작위(bal_acc 1/3)를 유의하게 상회**해야 한다.

K1 은 설계상 거의 자동 통과할 것이고, **진짜 관문은 K2**다.

준수: 학습은 이 라벨의 레짐 GBM 하나. 피쳐/HP/시드/컷오프는 기존 arm 과 동일하게 고정.
라이브 파일 미변경.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_regime3_current_hmm_wide24_20260529 import _adx, _num  # noqa: E402
from research_eth_regime_s12k3_label_train_20260902 import GBM3_HP, GBM3_MODEL_PATH, SEED  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_CSVS  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

HMM_MODEL = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                  / "regime3_current_sensitive_hmm_wide24_2024.joblib")
PREFIX = "regime3_fwd48_cut2509_"
CLASSES = ["bull", "bear", "chop"]
H = 48                      # 전향 창 4시간 — 부모 h48 계열과 맞춤
SPAN_START = pd.Timestamp("2024-01-01T00:00:00")
CUTOFF_END = pd.Timestamp("2025-09-30T23:55:00")
SPAN_END = pd.Timestamp("2026-08-30T23:55:00")
SPLITS = {"validation": ("2025-10-01", "2025-12-31 23:55:00"),
          "oos": ("2026-01-01", "2026-02-28 23:55:00")}
N_FOLDS, PURGE_BARS = 5, 2 * H
LAGS = np.arange(-288, 289, 6)
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"
SIDECAR_DIR = ROOT / "data/ensemble/supervised/omega461_fwd48_cut2509_20260909"


def balancedish_nobb(frame, cfg):
    """비교 기준(클래스 비중 매칭용) — build_omega461_balancedish_nobb_sidecar 와 동일 정의."""
    close, high, low = _num(frame, "close"), _num(frame, "high"), _num(frame, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    slope = ((ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
             ).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    adx = adx.fillna(0.0).to_numpy()
    y = np.full(len(frame), 2, dtype=np.int64)
    tr = adx >= float(cfg["trend_adx_min"])
    y[tr & (slope > float(cfg["slope_min"]))] = 0
    y[tr & (slope < -float(cfg["slope_min"]))] = 1
    y[adx < float(cfg["weak_adx_max"])] = 2
    return y


def forward_label(close: np.ndarray, train_mask: np.ndarray, target_shares: dict):
    """전향 라벨 + TRAIN 전용 임계 캘리브레이션. 마지막 H봉은 라벨 불가(=NaN 표시로 -1)."""
    n = len(close)
    net = np.full(n, np.nan)
    net[:-H] = (close[H:] - close[:-H]) / close[:-H]
    absdiff = np.abs(np.diff(close, prepend=close[0]))
    csum = np.concatenate([[0.0], np.cumsum(absdiff)])
    path = np.full(n, np.nan)
    path[:-H] = csum[H + 1:] - csum[1:n - H + 1]
    eff = np.full(n, np.nan)
    ok = np.isfinite(net) & np.isfinite(path) & (path > 0)
    eff[ok] = np.abs(net[ok]) * close[ok] / path[ok]

    valid = np.isfinite(net) & np.isfinite(eff)
    tm = train_mask & valid
    # e = 효율 하한: TRAIN 에서 '추세' 총비중(=1-chop목표)에 맞춰 백분위로 잡는다
    trend_target = 1.0 - float(target_shares["chop"])
    e = float(np.quantile(eff[tm], 1.0 - min(max(trend_target * 1.6, 0.05), 0.95)))
    cand = tm & (eff >= e)
    # θ = |net| 하한: 후보 중 상위 (trend_target / P(cand)) 비율이 남도록
    keep = trend_target / max(float(cand.sum()) / max(int(tm.sum()), 1), 1e-9)
    theta = float(np.quantile(np.abs(net[cand]), 1.0 - min(max(keep, 0.02), 0.99)))

    y = np.full(n, 2, dtype=np.int64)
    y[valid & (eff >= e) & (net > theta)] = 0
    y[valid & (eff >= e) & (net < -theta)] = 1
    y[~valid] = -1                                  # 라벨 불가(프레임 끝 H봉)
    return y, e, theta


def fit(X, y, seed=SEED):
    m = HistGradientBoostingClassifier(random_state=seed, **GBM3_HP)
    m.fit(X, y)
    return m


def derive_six(proba):
    p = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    s = np.sort(p, axis=1)
    out = {f"{PREFIX}{n}_prob": p[:, i] for i, n in enumerate(CLASSES)}
    out[f"{PREFIX}confidence"] = p.max(axis=1)
    out[f"{PREFIX}margin"] = s[:, -1] - s[:, -2]
    out[f"{PREFIX}entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1) / np.log(3.0)
    return out


def lead_lag(pred, close, step=12):
    s = np.where(pred == 0, 1.0, np.where(pred == 1, -1.0, 0.0))
    r = np.full(len(close), np.nan)
    r[:-step] = (close[step:] - close[:-step]) / close[:-step]
    cors = []
    for h in LAGS:
        a, b = (s[:len(s) - h], r[h:]) if h >= 0 else (s[-h:], r[:len(r) + h])
        m = np.isfinite(a) & np.isfinite(b)
        cors.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 else np.nan)
    cors = np.array(cors)
    i = int(np.nanargmax(cors))
    return {"peak_lag_bars": int(LAGS[i]), "peak_corr": round(float(cors[i]), 4),
            "corr_at_0": round(float(cors[np.where(LAGS == 0)[0][0]]), 4)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    src = joblib.load(GBM3_MODEL_PATH)
    cols, medians = src["feature_cols"], src["feature_medians"]
    cfg = joblib.load(HMM_MODEL)["label_config"]

    frames = [pd.read_csv(p, low_memory=False, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = (pd.concat(frames, ignore_index=True).sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last").reset_index(drop=True))
    df = df[(df["timestamp"] >= SPAN_START) & (df["timestamp"] <= SPAN_END)].reset_index(drop=True)
    df = _with_raw_state12(df)
    ts = df["timestamp"]
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64)

    y_ref = balancedish_nobb(df, cfg)
    train_raw = (ts <= CUTOFF_END).to_numpy()
    tgt = {n: float((y_ref[train_raw] == i).mean()) for i, n in enumerate(CLASSES)}
    print(f"[비중 목표] balnobb TRAIN 비중에 매칭: {[f'{k} {v:.3f}' for k, v in tgt.items()]}", flush=True)

    y, e, theta = forward_label(close, train_raw, tgt)
    # 컷오프 근처 H봉은 라벨이 컷오프 너머를 훔쳐보므로 학습에서 제외
    train_mask = train_raw.copy()
    cut_i = int(np.flatnonzero(train_raw)[-1])
    train_mask[max(cut_i - H + 1, 0): cut_i + 1] = False
    train_mask &= (y >= 0)
    print(f"[전향 라벨] H={H}봉(4h)  eff>={e:.4f}  |net|>={theta*100:.3f}%  "
          f"라벨불가 {int((y<0).sum())}봉  컷오프 퍼지 {H}봉", flush=True)
    print(f"[학습] {int(train_mask.sum()):,}봉  클래스비중 "
          f"{ {n: round(float((y[train_mask]==i).mean()),4) for i,n in enumerate(CLASSES)} }", flush=True)

    X = (df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
           .replace([np.inf, -np.inf], np.nan).fillna(pd.Series(medians)).fillna(0.0))
    full = fit(X[train_mask], y[train_mask])
    proba = full.predict_proba(X)

    tr_idx = np.flatnonzero(train_mask)
    oof = np.zeros((len(tr_idx), 3))
    pos = {v: i for i, v in enumerate(tr_idx)}
    for k, te in enumerate(np.array_split(tr_idx, N_FOLDS)):
        lo, hi = te[0] - PURGE_BARS, te[-1] + PURGE_BARS
        rows = tr_idx[(tr_idx < lo) | (tr_idx > hi)]
        oof[[pos[v] for v in te]] = fit(X.iloc[rows], y[rows], seed=SEED + k).predict_proba(X.iloc[te])
        print(f"   OOF fold {k+1}/{N_FOLDS} (퍼지 {PURGE_BARS}봉)", flush=True)
    proba[tr_idx] = oof
    print(f"\n[진단] in-sample bal_acc {balanced_accuracy_score(y[train_mask], full.predict(X[train_mask])):.4f}"
          f" | OOF {balanced_accuracy_score(y[train_mask], oof.argmax(1)):.4f}  (무작위 0.3333)", flush=True)

    six = derive_six(proba)
    out = pd.DataFrame({"timestamp": ts})
    for k, v in six.items():
        out[k] = v
    for p in TRAIN_CSVS:
        tag = p.stem.replace("training_features_", "")
        stamps = set(pd.read_csv(p, usecols=["timestamp"], parse_dates=["timestamp"])["timestamp"])
        part = out[out["timestamp"].isin(stamps)].reset_index(drop=True)
        part.to_csv(SIDECAR_DIR / f"training_features_{tag}_{PREFIX}sidecar.csv", index=False)
        print(f"   사이드카 {tag}: {len(part):,}행", flush=True)

    joblib.dump({"model_id": "omega461_fwd48_cut2509_20260909", "classes": CLASSES,
                 "feature_cols": cols, "feature_medians": medians, "model": full, "config": GBM3_HP,
                 "train_range": f"{SPAN_START}~{CUTOFF_END} (컷오프 −{H}봉 퍼지)", "seed": SEED,
                 "prefix": PREFIX,
                 "label_spec": {"family": f"forward_net_efficiency_H{H}", "horizon_bars": H,
                                "eff_min": e, "abs_net_min": theta,
                                "calibrated_on": "TRAIN only, share-matched to balnobb"},
                 "notes": "전향 레짐 라벨(미래 H봉 경로로 정의) + 인과 피쳐 예측. 지연 제거 목적."},
                OUT / "regime_fwd48_cut2509_model.joblib")

    rep = {"label": {"horizon": H, "eff_min": e, "abs_net_min": theta, "target_shares": tgt},
           "windows": {}}
    print("\n" + "=" * 88)
    for split, (s, ee) in SPLITS.items():
        m = ((ts >= s) & (ts <= ee)).to_numpy() & (y >= 0)
        pr = proba[m].argmax(1)
        ba = float(balanced_accuracy_score(y[m], pr))
        kp = float(cohen_kappa_score(y[m], pr, labels=[0, 1, 2]))
        ll = lead_lag(pr, close[m])
        runs, c = [], 1
        for i in range(1, len(pr)):
            if pr[i] == pr[i - 1]:
                c += 1
            else:
                runs.append(c); c = 1
        runs.append(c)
        rr = np.array(runs)
        rep["windows"][split] = {"bars": int(m.sum()), "bal_acc": round(ba, 4), "kappa": round(kp, 4),
                                 "lead_lag": ll, "transitions": int((pr[1:] != pr[:-1]).sum()),
                                 "median_run": float(np.median(rr)), "mean_run": round(float(rr.mean()), 1),
                                 "pred_shares": {n: round(float((pr == i).mean()), 3)
                                                 for i, n in enumerate(CLASSES)}}
        print(f"[{split}] {int(m.sum()):,}봉  bal_acc {ba:.4f} (무작위 .3333)  κ {kp:.4f}", flush=True)
        print(f"   ⭐리드-랙 최대상관 h = {ll['peak_lag_bars']:+d}봉  (상관 {ll['peak_corr']:.4f}, "
              f"h=0 상관 {ll['corr_at_0']:.4f})   ← K1: h>=0 이어야 통과", flush=True)
        print(f"   전환 {rep['windows'][split]['transitions']}회  중앙 {np.median(rr):.0f}봉  "
              f"평균 {rr.mean():.1f}봉  예측비중 {rep['windows'][split]['pred_shares']}", flush=True)

    (OUT / "fwd48_build_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    print(f"\n산출물: {OUT}/fwd48_build_report.json", flush=True)
    print("\n※ K2(전방수익 bull−bear 격차 CI)는 별도 블록 부트스트랩 스크립트에서 판정한다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
